import pandas as pd
import logging
from spacy.tokens import Doc 
from statistics import mean
from ppinat.ppiparser.ppiannotation import PPIAnnotation
from ppinot4py.computers import condition_computer
from ppinot4py.model import TimeInstantCondition
from ppinat.helpers import Log
import numpy as np
import re, math
from collections import Counter
from fastDamerauLevenshtein import damerauLevenshtein
from pyjarowinkler import distance as pyjarowinkler_d
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

logger = logging.getLogger(__name__)


class Slot():
    """
        Pair (attribute,value) that represents a condition (in the sense of 
        TimeInstantCondition) of the type attribute=value. It can also contain 
        a second pair so that the condition is att1=value1 and att2=value2.

        It is used as the result of the Matcher
    """
    def __init__(self, column1, value1, column2=None, value2=None):
        self.column1 = column1
        self.value1 = value1
        self.column2 = column2
        self.value2 = value2

    def type(self):
        if self.column2 is None:
            return self.column1
        return self.column1 + "-" + self.column2

    def text(self):
        if self.column2 is None:
            text = self.value1
        else: 
            text = self.value1 + " " + self.value2

        return text.replace("_", " ")

    def text_complete(self):
        if self.column2 is None:
            text = self.column1 + " " + self.value1
        else:         
            text = self.column1 + " " + self.value1 + " " + self.column2 + " " + self.value2

        return text.replace("_", " ")

    def to_condition(self, negation = False):
        comparison = "==" if not negation else "!="
        if self.column2 is None:
            condition = f"`{self.column1}` {comparison} '{self.value1}'"
        else:
            condition = f"`{self.column1}` {comparison} '{self.value1}' & `{self.column2}` {comparison} '{self.value2}'"

        return TimeInstantCondition(condition)

    def __repr__(self):
        if self.column2 is None:
            return "Slot: " + self.column1 + ": " + self.value1
        return "Slot: " + self.column1 + ": " + self.value1 + ", " + self.column2 + ": " + self.value2

class SlotPair(Slot):
    def __init__(self, slot1, slot2):
        super(SlotPair, self).__init__(slot1.column1, None, slot1.column2, None)
        self.slot1 = slot1
        self.slot2 = slot2
    
    def to_condition_1(self, negation = False):
        return self.slot1.to_condition(negation)
    def to_condition_2(self, negation=False):
        return self.slot2.to_condition(negation)

    def __repr__(self):
        return str(self.slot1) +", "+ str(self.slot2)

    def __hash__(self) -> int:
        return hash((self.slot1, self.slot2))
        
    def __eq__(self, __o: object) -> bool:
        return (self.slot1, self.slot2) == (__o.slot1, __o.slot2)

    def __ne__(self, __o: object) -> bool:
        return not (self == __o)


class MatchedAnnotation:
    def __init__(self, annotation: PPIAnnotation):
        self.annotation = annotation
        self.event_matches = {}
        self.aggr = None
        self.groupby = None

    def add_event_match(self, event, matched_slot):
        self.event_matches[event] = matched_slot

    def __repr__(self):
        res = "matches: " + str(self.event_matches)
        if self.aggr:
            res = res + "\n aggregation:" + self.aggr
        if self.groupby:
            res = res + "\n group-by:" + str(self.groupby)
        return res


def choose_best_similarity(similarity_values: dict, values=False):
    """
        Returns the best similarity from a dict that contains the
        name and the similarity value.

        Parameters
        ------------
        similarity_values : dict
            the dict containing the similarity values
        values : boolean (default False)
            indicates if the similarity value should be returned together with 
            the name of the attribute

    """
    if len(similarity_values) >= 1:
        #candidate_similarity = list(map(lambda x: similarity_values[x], candidate_slots))
        #best_index = candidate_similarity.index(max(candidate_similarity))            
        key = max(similarity_values.keys(), key=(lambda key: similarity_values[key]))
        if not values:
            return key
        else:
            return key, similarity_values[key]
#        elif len(similarity_values) == 1:
#            return similarity_values.keys()[0]
    else:
        return None


class SimilarityComputer:
    """
        It performs matches between text recognized by the parser and the 
        event log.
        
        Parameters
        -----------
        log : Log
            The event log
        
        nlp : nlp
            The NLP object that will be used to perform the matching

        metric_decoder
            The object that is used to perform additional decoding of metrics

        two_column_match : bool, default True
            Whether two columns will be considered for matching or if
            just one column will be used instead

        slot_threshold : int, default 100
            The max number of unique different values that a column must have
            to consider it in the matching of conditions (att1=val1)

        att_threshold : int, default 15
            The max number of unique different values that a column must have 
            to consider it in the matching of attributes
        ignored_columns : list
            The name of the columns that will be ignored during matching

    """

    def __init__(self, 
                 log: Log, 
                 nlp, 
                 metric_decoder=None,
                 two_column_match=False, 
                 two_step_match=True,
                 slot_threshold=100, 
                 att_threshold=15, 
                 ignored_columns=None,
                 weights=None):
        self.decoder = metric_decoder

        logger.info("Initializing Embeddings...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        #self.model = SentenceTransformer('all-MiniLM-L12-v2')

        logger.info("Initializing bart-large-mlni...")
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli').to(self.device)

        self.slot_threshold = slot_threshold
        self.att_threshold = att_threshold
        if ignored_columns is None:
            self.ignored_columns = []
        else:
            self.ignored_columns = ignored_columns
        self.nlp = nlp

        self.weights = weights if weights is not None else {
            "one_slot": {
                'slot_is_sim': 0.25,
                'slot_complete_is_sim': 0.2,
                'slot_emb': 0.25,
                'slot_complete_emb': 0.2,
                'att_is_sim': 0.05,
                'att_complete_is_sim': 0.05
            },
            "multi_slot": {
                'ev1_$slot_is_sim': 0.05,
                "ev1_$slot_complete_is_sim": 0.05,
                'ev1_$slot_emb': 0.07,
                'ev1_$slot_complete_emb': 0.06,
                'ev2_$slot_is_sim': 0.05,
                "ev2_$slot_complete_is_sim": 0.05,
                'ev2_$slot_emb': 0.07,
                'ev2_$slot_complete_emb': 0.06,

                "ev1_$att_is_sim": 0.01,
                "ev1_$att_complete_is_sim": 0.01,
                "ev2_$att_is_sim": 0.01,
                "ev2_$att_complete_is_sim": 0.01,
                "same_type": 0.25,
                "condition_ratio": 0.25
            }
        }

        self.id_case = log.id_case
        self.time_column = log.time_column
        self.activity = log.activity_column
        self.resource = "org:role"
        #self.log = log.as_eventlog()
        self.df: pd.DataFrame = log.as_dataframe()

        self.two_column_match = two_column_match
        self.two_step_match = two_step_match

        if two_column_match and two_step_match:
            raise ValueError('two_step_match and two_column_match cannot be True at the same time')

        #self.sanitize_typology_column()
        # self.add_extra_typology_columns()
        logger.info("Initializing slot candidates")
        self._compute_slot_candidates()

        logger.info("Initializing att candidates")
        self.atts = self._compute_att_candidates()        
        self.aggr_values = {
            'AVG': list(self.nlp.pipe(['average', 'mean', 'total average'])),
            'MIN': list(self.nlp.pipe(['minimum', 'min', 'minimal'])),
            'MAX': list(self.nlp.pipe(['maximum', 'max', 'maximal'])),
            'SUM': list(self.nlp.pipe(['sum', 'total'])),
            '%': list(self.nlp.pipe(['percentage', 'percentage of', 'ratio', 'fraction', 'fraction of'])),
        }
        self.period_values = {
            'H': list(self.nlp.pipe(['hour', 'hourly'])),
            'D': list(self.nlp.pipe(['day', 'daily'])),
            'W': list(self.nlp.pipe(['week', 'weekly'])),
            'M': list(self.nlp.pipe(['month', 'monthly'])),
            'Q': list(self.nlp.pipe(['quarter', 'quarterly'])),
#            '6M': list(self.nlp.pipe(['semester', 'semesterly'])),
            'Y': list(self.nlp.pipe(['year', 'yearly', 'annually']))
        }
        self.op_values = {
            'equal': list(self.nlp.pipe(["equals to", "=", "is"])),
            'not_equal': list(self.nlp.pipe(["different than", "not equals to", "!="])),
            'gt': list(self.nlp.pipe(["greater than", "bigger than", "larger than", ">=",  ">", "greater or equal than", "at least"])),
            'lt': list(self.nlp.pipe(["lower than", "less than", "shorter than", "smaller than", "<=", "<", "lower or equal than", "at most"]))            
        }
        logger.info("Finished similarity initialization")

    def column_candidates(self, for_slot=True):
        if for_slot:
            if self.two_step_match:
                columns = set(self.slots.keys())
            else:
                columns = { slot.column1 for slot in self.slots }
                if self.two_column_match:
                    columns.update({ slot.column2 for slot in self.slots })
        else:
            columns = self.atts

        return columns

    def all_attributes(self):
        return set(self.df.columns.values)

    def value_candidates(self, attribute):
        return self.df[attribute].dropna().unique()

    def match_annotation_to_log(self, annotation: PPIAnnotation):
        matched_annotation = MatchedAnnotation(annotation)
        events = annotation.get_events()
        if len(events) == 1:
            matched_annotation.add_event_match(events[0], self.find_most_similar_slot(events[0]))
        if len(events) == 2:
            # match with equal type
            corresp = self.find_most_similar_slots_matching_types(events[0], events[1])
            matched_annotation.add_event_match(events[0], corresp[0])
            matched_annotation.add_event_match(events[1], corresp[1])
        if annotation.get_aggregation_function():
            matched_annotation.aggr = choose_best_similarity(self.find_agg_function(annotation.get_aggregation_function()))
        if annotation.get_grouping():
            matched_annotation.groupby = choose_best_similarity(self.find_most_similar_attribute(annotation.get_grouping()))
        return matched_annotation

    def domain_of_attribute(self, attribute_name, categorical_threshold=300):
        col = self.df[attribute_name]

        if pd.api.types.is_numeric_dtype(col):
            domain = 'numeric'
        elif pd.api.types.is_string_dtype(col):
            domain = self.value_candidates(attribute_name)
            # At this moment, we consider that if the possible values is
            # higher than a threshold, then the domain is a string
            if len(domain) > categorical_threshold:
                domain = 'string'
        elif pd.api.types.is_categorical_dtype(col):
            domain = self.value_candidates(attribute_name)
        elif pd.api.types.is_datetime64_any_dtype(col):
            domain = 'datetime'
        elif pd.api.types.is_timedelta64_dtype(col):
            domain = 'timedelta'
        else:
            domain = 'string'

        return domain


    def find_most_similar_slot(self, event, delta_heuristics = 0.05) -> dict:
        compute = self._compute_single_slot_similarity(event)

        self.slot_information = {}

        for slot in compute:
            s = str(slot)
            self.slot_information[s] = {}

            slot_value = compute[slot]

            slot_value["slot_sim"] = compute_distances(slot_value["slot_sim"])
            slot_value["slot_complete_sim"] = compute_distances(slot_value["slot_complete_sim"])

            result = 0
            for w in self.weights['one_slot']:
                if self.weights['one_slot'][w] > 0:
                    param = slot_value[w] * self.weights['one_slot'][w]
                    self.slot_information[s][w] = param
                    self.slot_information[s][w+"_without_weight"] = slot_value[w]
                    result += param

            compute[slot] = result
            self.slot_information[s]["total_score"] = compute[slot]

        compute = _filter_best_similarity(compute, delta_heuristics)

        return compute


    def perc_conditions_in_cases(self, condition): 
        cs = condition_computer(self.df, self.id_case, condition, "ACTIVITY", "TRANSITION")
        num_conditions = cs.groupby(self.df[self.id_case]).sum()

        return (num_conditions > 1).sum()/(num_conditions >= 1).sum()


    def find_most_similar_slots_matching_types(self, event1, event2, delta_heuristics = 0.05) -> dict:
        compute = self._compute_multi_slot_similarity(event1, event2)

        self.slot_information = {}


        for slot in compute:
            s = str(slot)
            self.slot_information[s] = {}

            slot_value = compute[slot]

            self.slot_information[s]["condition_ratio_base"] = slot_value["condition_ratio"]
            slot_value["condition_ratio"] = (1/(1+ math.exp(-10*compute_distances(slot_value["condition_ratio"]))) - 0.5)/0.5
            slot_value["ev1_$slot_sim"] = compute_distances(slot_value["ev1_$slot_sim"])
            slot_value["ev1_$slot_complete_sim"] = compute_distances(slot_value["ev1_$slot_complete_sim"])
            slot_value["ev2_$slot_sim"] = compute_distances(slot_value["ev2_$slot_sim"])
            slot_value["ev2_$slot_complete_sim"] = compute_distances(slot_value["ev2_$slot_complete_sim"])

            result = 0
            for w in self.weights['multi_slot']:
                if self.weights['multi_slot'][w] > 0:
                    param = slot_value[w] * self.weights['multi_slot'][w]
                    self.slot_information[s][w] = param
                    self.slot_information[s][w+"_without_weight"] = slot_value[w]
                    result += param


            #condition_ratio_value = - weight_condition_ratio * (1 - (1/(1+ math.exp(-10*self.compute_distances(slot_value["condition_ratio"]))) - 0.5)/0.5)
            #condition_ratio_value = weight_condition_ratio * (1/(1+ math.exp(-10*self.compute_distances(slot_value["condition_ratio"]))) - 0.5)/0.5
            #self.slot_information[s]["condition_ratio"] = condition_ratio_value
            #self.slot_information[s]["condition_ratio_without_weight"] = self.compute_distances(slot_value["condition_ratio"])

            compute[slot] = result
            self.slot_information[s]["total_score"] = compute[slot]

        compute = _filter_best_similarity(compute, delta_heuristics)

        return compute


    def _compute_single_slot_similarity(self, event):
        # ACTIVITY = "Resolve"
        # slot_sim (event, "Resolve")-> dict of distances (framework_distance, cosine_distance, damerauLevenshtein_distance, pyjarowinkler_distance)
        # slot_complete_sim (event, "ACTIVITY RESOLVE")-> dict of distances (framework_distance, cosine_distance, damerauLevenshtein_distance, pyjarowinkler_distance)
        # att_sim (mean of distances of slot_sim for each value of ACTIVITY) -> dict of distances (framework_distance, cosine_distance, damerauLevenshtein_distance, pyjarowinkler_distance)
        # att_complete_sim (mean of distances of slot_complete_sim for each value of ACTIVITY) -> dict of distances (framework_distance, cosine_distance, damerauLevenshtein_distance, pyjarowinkler_distance)
        # single_slot (0,1) -> 1 if slot only has one column and 0 if it has two. (currently it is always 1)
        # condition_ratio (0,1) -> % of times the condition is true

        logger.info(f"compute similarity features for [{event}]")
        similarity_features = self._compute_slot_similarity_features(event, self.slots)
        logger.info(f"finished compute similarity features for [{event}]")

        result = {}

        for (key, value) in similarity_features.items():
            result[key] = {**value, "condition_ratio": self.slots_condition[key]}

        return result


    def _compute_multi_slot_similarity(self, event1, event2):
        # ev1_slot_sim -> dict of distances (framework_distance, cosine_distance, damerauLevenshtein_distance, pyjarowinkler_distance)
        # ev1_slot_complete_sim -> dict of distances (framework_distance, cosine_distance, damerauLevenshtein_distance, pyjarowinkler_distance)
        # ev1_att_sim -> dict of distances (framework_distance, cosine_distance, damerauLevenshtein_distance, pyjarowinkler_distance)
        # ev1_att_complete_sim -> dict of distances (framework_distance, cosine_distance, damerauLevenshtein_distance, pyjarowinkler_distance)
        # ev1_single_slot (0,1) -> 1 if slot only has one column and 0 if it has two. (currently it is always 1)
        # ev2_slot_sim -> dict of distances (framework_distance, cosine_distance, damerauLevenshtein_distance, pyjarowinkler_distance)
        # ev2_slot_complete_sim -> dict of distances (framework_distance, cosine_distance, damerauLevenshtein_distance, pyjarowinkler_distance)
        # ev2_att_sim -> dict of distances (framework_distance, cosine_distance, damerauLevenshtein_distance, pyjarowinkler_distance)
        # ev2_att_complete_sim -> dict of distances (framework_distance, cosine_distance, damerauLevenshtein_distance, pyjarowinkler_distance)
        # ev2_single_slot (0,1) -> 1 if slot only has one column and 0 if it has two. (currently it is always 1)
        # same_type (0,1) -> 1 if both slots have the same attribute else 0
        # condition_ratio (0,1) -> % of times the metric from/to can be defined (both occur and from is before to)

        logger.info(f"compute similarity features for [{event1}, {event2}]")
        similarity_features_1 = self._compute_slot_similarity_features(event1, self.slots)
        similarity_features_2 = self._compute_slot_similarity_features(event2, self.slots)
        logger.info(f"finished compute similarity features for [{event1}, {event2}]")

        pair_sim_values = {}
        # we add matching pairs
        for match1 in similarity_features_1.keys():
            for match2 in similarity_features_2.keys():
                # two matches cannot refer to the same event
                if match1 == match2:
                    continue
                # if matches share a type we add them with a pair_priority factor
                pair_sim_values[SlotPair(match1, match2)] = {
                    **{f"ev1_${key}": value for (key,value) in similarity_features_1[match1].items()},
                    **{f"ev2_${key}": value for (key,value) in similarity_features_2[match2].items()},
                    "same_type": 1.0 if match1.type() == match2.type() else 0.0 
                }

        result = {}

        for (key, value) in pair_sim_values.items():
            result[key] = {**value, "condition_ratio": self.slots_difference[key]}

        return result

    
    def idf(self, term, att):
        if term in self.idf_[att]:
            return self.idf_[att][term]
        else:
            return math.log(self.length_[att])


    def _compute_slot_similarity_features(self, event, slots):
        matching = [SimMatching(self.nlp, self.idf)]

        if any([x in self.weights['one_slot'] and self.weights['one_slot'][x] > 0 for x in EmbMatching.list_features()]):
            matching.append(EmbMatching(self.model, self.embeddings))
        if any([x in self.weights['one_slot'] and self.weights['one_slot'][x] > 0 for x in BartMatching.list_features()]):
            matching.append(BartMatching(self.tokenizer, self.device, self.nli_model))

        for m in matching:
            m.encode_event(event)

        res = {}

        if self.two_step_match:
            for att in list(slots.keys()):
                similarity_slot = {}

                for m in matching:
                    m.encode_att(att)

                for i, slot in enumerate(slots[att]):
                    similarity_slot[slot] = {}

                    for m in matching:
                        similarity_slot[slot].update(m.compute_feature(i, slot))


                if len(similarity_slot) > 0:
                    similarity_att = {
                        "att_sim": { m: mean([x["slot_sim"][m]]) for x in similarity_slot.values() for m in x["slot_sim"]},
                        "att_complete_sim": { m: mean([x["slot_complete_sim"][m]]) for x in similarity_slot.values() for m in x["slot_complete_sim"]},
                        "att_is_sim": mean([x["slot_is_sim"] for x in similarity_slot.values()]),
                        "att_complete_is_sim": mean([x["slot_complete_is_sim"] for x in similarity_slot.values()])
                    }
                    
                    for slot in similarity_slot:
                        res[slot] = {
                            **similarity_slot[slot],
                            **similarity_att
                        }

        else:
            res = {}

        return res


    def find_most_similar_attribute(self, text, filter_att='group', delta_heuristics=0) -> dict:
        """
            Finds the most similar attribute name for a given text. This is
            used, for instance, to match the attribute used in a group by
            construction (e.g. group by 'state').

            The current implementation only checks the similarity of the name
            of the attribute with the text provided. However, this could be
            extended to consider the names of the values as well, possibly 
            providing adequate weights.

            text : string
                text that is being matched
            filter_att : 'slot' | 'group' | 'none'
                indicates if the attribute should meet the conditions for being
                part of a slot in terms of number of changes in cases, for being
                in a group by construction in terms of the number of values of the
                attribute, or if it considers no filter at all
            delta_heuristics : double between 0 and 1 (default 0)
                delta expressed in percentage allowed to consider candidates other than those with 
                the best similarity
        """

        if filter_att == "group":
            res = self.find_most_similar_from_list(text, self.atts, delta_heuristics)
        elif filter_att == "slot":
            res = self.find_most_similar_from_list(text, self.column_candidates(), delta_heuristics)
        else:
            res = self.find_most_similar_from_list(text, self.df.columns.values, delta_heuristics)

        return res

    def find_most_similar_value(self, text, attribute_name, delta_heuristics=0) -> dict:
        """
            Finds the most similar value name for a given text in a column. 

            text : string
                text that is being matched
            attribute_name : string
                the attribute name whose values are checked
            delta_heuristics : double between 0 and 1 (default 0)
                delta expressed in percentage allowed to consider candidates other than those with 
                the best similarity
        """
        return self.find_most_similar_from_list(text, self.value_candidates(attribute_name), delta_heuristics)

    def find_most_similar_from_list(self, text, elem_list, delta_heuristics=0) -> dict:
        """
            Finds the most similar value name for a given text in a list of elements. 

            text : string
                text that is being matched
            elem_list : string
                the list of elements
            delta_heuristics : double between 0 and 1 (default 0)
                delta expressed in percentage allowed to consider candidates other than those with 
                the best similarity
        """
        res = self._compute_att_similarity_scores(text, elem_list)            
        res = _filter_best_similarity(res, delta_heuristics)

        return res


    def find_agg_function(self, text, delta_heuristics=0) -> dict:
        """
            Finds the most similar aggregation function for a given text. For
            instance, it identifies that 'mean' refers to the average aggregation
            function. The list of possible values is currently provided directly
            by the Matcher class. It could be possible to make it configurable.

            Parameters
            ----------
            text : string
                text that is being matched
            delta_heuristics : double between 0 and 1 (default 0)
                delta expressed in percentage allowed to consider candidates other than those with 
                the best similarity
        """
        return get_best_similarity(self.nlp, text, self.aggr_values.items(), delta_heuristics)

    def find_period(self, text, delta_heuristics=0) -> dict:
        """
            Finds the most similar period function for a given text. For
            instance, it identifies that 'monthly' refers to a periodicity of each 
            month. The list of possible values is currently provided directly
            by the Matcher class. It could be possible to make it configurable.

            Parameters
            ----------
            text : string
                text that is being matched
            delta_heuristics : double between 0 and 1 (default 0)
                delta expressed in percentage allowed to consider candidates other than those with 
                the best similarity
        """
        return get_best_similarity(self.nlp, text, self.period_values.items(), delta_heuristics)

    def find_op(self, text, delta_heuristics=0) -> dict:
        return get_best_similarity(self.nlp, text, self.op_values.items(), delta_heuristics)

    def _compute_att_similarity_scores(self, text, attributes):        
        text = text.lower()
        text_vector = self.nlp(preprocess_label(text))
        res = {}
        for att in attributes:
            att_low = att.lower() if callable(getattr(att, "lower", None)) else att
            if text == att_low:
                res[att] = 1.0
            else:
                att_vector = self.nlp(preprocess_label(att_low))
                if att_vector.vector_norm:
                    res[att] = get_similarity(text_vector, att_vector)
        return res

    def _compute_similarity_scores(self, event, slots):
        text_vector = self.nlp(event)
        text_vector = Doc(vocab=self.nlp.vocab, words=[word.text for word in text_vector if not (word.is_stop or word.is_punct)])
        if not text_vector.has_vector:
            text_vector = self.nlp(event)        

        if self.two_step_match:
            # slots contains a dictionary with slot as keys and similarity as value
            # atts contains a dictionary with attributes as keys and mean similarity of 
            #      all slots with that attribute as column as values
            res_temp = {
                "slots": {},
                "atts": {}
            }
            for att in slots.keys():
                att_slots = slots[att]
                similarity_att = {}
                for slot in att_slots:
                    if event == slot.text():
                        similarity_att[slot] = 1.0
                    else:
                        slot_vector = self.nlp(slot.text())
                        if slot_vector.vector_norm:
                            similarity_att[slot] = get_similarity(text_vector, slot_vector)*0.5
                        
                        slot_vector = self.nlp(slot.text_complete())
                        if slot_vector.vector_norm:
                            similarity_att[slot] = get_similarity(text_vector, slot_vector)*0.5 + (similarity_att[slot] if slot in similarity_att else 0)                        

                if len(similarity_att) > 0:
                    res_temp["atts"][att] = mean(similarity_att.values())

                    for slot in similarity_att:
                        res_temp["slots"][slot] = similarity_att[slot]
            
            res = {x: res_temp["slots"][x] + res_temp["atts"][x.column1] for x in res_temp["slots"]}

        else:
            # res is a dictionary with slots as keys and similarity as value
            res = {}

            for slot in slots:
                if event == slot.text():
                    res[slot] = 1.0
                else:
                    slot_vector = self.nlp(slot.text())
                    # check if slot can be properly vectorized
                    if slot_vector.vector_norm:
                        res[slot] = get_similarity(text_vector, slot_vector)

        return res

    # The following three methods are not used anymore because they are domain-specifc
    def add_event_class_column(self):
        # Extract event classes as combination of TYPOLOGY and STATE, e.g., "incident opened", "request closed"
        temp_df = self.df["TYPOLOGY"].str.split(".", n=1, expand=True)
        self.df["eventClass"] = temp_df[0] + " " + self.df["STATE"]
        self.df["eventClass"] = self.df["eventClass"].str.replace('_', ' ', regex=True)

    def sanitize_typology_column(self):
        # Sanitizes and splits the TYPOLOGY column into colname, e.g.,
        temp_df = self.df["TYPOLOGY"].str.split(".", n=1, expand=True)
        colname = "TYPOLOGY2"
        self.df[colname] = temp_df[0]
        self.df[colname] = self.df[colname].str.replace('_', ' ', regex=True)

    def add_extra_typology_columns(self):
        # this method is heavily based on domain knowledge, so preferably avoid
        temp_df = self.df["TYPOLOGY"].str.split(".", n=1, expand=True)
        temp_df[0] = temp_df[0].str.replace('_', ' ', regex=True)
        temp_df["eventClass"] = temp_df[0] + " " + self.df["STATE"]
        for typology in temp_df[0].unique():
            self.df.loc[temp_df["eventClass"].str.startswith(typology), typology + " status"] = temp_df["eventClass"]


    def _column_candidates(self):
        """ 
        select all categorical columns except for ignored ones
        """
        return [c for c in self.df.select_dtypes(include=["object"]).columns.tolist() if c not in self.ignored_columns]


    def _compute_att_candidates(self):
        columns = self._column_candidates()

        result = set()
        for column in columns:
            values = self.df[column].dropna().unique()
            if 1 < len(values) < self.att_threshold:
                result.add(column)

        return result        

    def _compute_slot_candidates(self):
        columns = self._column_candidates()

        if self.two_step_match:
            result = {}
            embeddings = {}
            slot_condition = {}
            slot_position = {}
            slot_difference = {}
            idf = {}
            length = {}
            number_of_cases = len(self.df[self.id_case].unique())

            for column in columns:
                values = self.df[column].dropna().unique()                
                #TODO: The following restriction makes sense for time, but for count is not so important
                change_in_case = np.any(self.df[column].groupby(self.df[self.id_case]).agg('nunique') > 1)
                if change_in_case and len(values) < self.slot_threshold:
                    #matrix = self.df.groupby([self.id_case, column])['case:id'].count().unstack().fillna(0)
                    count_conditions = self.df[column].groupby(self.df[self.id_case]).value_counts().unstack().count()
                    slot_position[column] = self.df.groupby(self.id_case).cumcount().groupby([self.df[self.id_case], self.df[column]]).first()
                    result[column] = [] # TODO Check this
                    freq = {}
                    for value in values:                        
                        slot = Slot(column, value)
                        result[column].append(slot)
                        slot_condition[slot] = count_conditions[value] / number_of_cases
                        lemmas = set([token.lemma_ for token in self.nlp(value) if not (token.is_stop or token.is_punct or token.is_space)])
                        for w in lemmas:
                            if w in freq:
                                freq[w] = freq[w] + 1
                            else:
                                freq[w] = 1

                    idf[column] = {w: math.log(len(values)/(1+freq[w])) for w in freq}
                    length[column] = len(values)

                    embeddings[column] = self.model.encode([preprocess_label(v) for v in values], convert_to_tensor=True)
                    embeddings[column+"-attrib"] = self.model.encode([column + " " + preprocess_label(v) for v in values], convert_to_tensor=True)

            for slot1 in slot_condition:
                for slot2 in slot_condition:
                    if slot1 == slot2:
                        continue
                    
                    difference = slot_position[slot2.column1].xs(slot2.value1, level=1) - slot_position[slot1.column1].xs(slot1.value1, level=1)
                    slot_difference[SlotPair(slot1, slot2)] = (difference > 0).sum() / len(difference)
            
        else:
            result = set()
            selected_columns = []
            # add candidates from all columns in log with fewer than self.slot_threshold unique values
            for column in columns:
                values = self.df[column].dropna().unique()
                change_in_case = np.any(self.df[column].groupby(self.df[self.id_case]).agg('nunique') > 1)
                if change_in_case and len(values) < self.slot_threshold and column not in self.ignored_columns:
                    selected_columns.add(column)
                    for value in values:
                        result.add(Slot(column, value))
            if self.two_column_match:
                #     add  columns capturing value combinations
                for i in range(len(selected_columns) - 1):
                    col1 = selected_columns[i]
                    for j in range(i + 1, len(selected_columns)):
                        col2 = selected_columns[j]
                        for group in self.df.groupby([col1, col2]):
                            vals = group[0]
                            result.add(Slot(col1, vals[0], col2, vals[1]))

        self.slots = result
        self.slots_condition = slot_condition
        self.slots_difference = slot_difference
        self.embeddings = embeddings
        self.idf_ = idf
        self.length_ = length


    def metric_decoder(self, text):
        return self.decoder.predict_annotation(text)       



def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def get_cosine(text1, text2):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def framework_distance(text, values_to_compare):
    return {k : mean([text.similarity(word) for word in v]) for k, v in values_to_compare}

def cosine_distance(text, values_to_compare):
    return {k : mean([get_cosine(text, word.text) for word in v]) for k, v in values_to_compare}

def damerauLevenshtein_distance(text, values_to_compare):
    return {k : max([damerauLevenshtein(text, word.text) for word in v]) for k, v in values_to_compare}

def pyjarowinkler_distance(text, values_to_compare):
    return {k : max([pyjarowinkler_d.get_jaro_distance(text, word.text, winkler=True, scaling=0.1) for word in v]) for k, v in values_to_compare}

def mean_similarities(similarities):
    return {k : mean([v[k] for v in similarities]) for k in similarities[0]}
        

def get_best_similarity(nlp, text, values_to_compare, delta_heuristics):
    text_vector = nlp(text)
    similarities_framework = framework_distance(text_vector, values_to_compare)
    # similarities_cosine = cosine_distance(text, values_to_compare)
    similarities_damerauLevenshtein = damerauLevenshtein_distance(text, values_to_compare)
    similarities_pyjarowinkler = pyjarowinkler_distance(text, values_to_compare)
    #similarities = mean_similarities([similarities_framework, similarities_cosine, similarities_damerauLevenshtein, similarities_pyjarowinkler])
    similarities_fram = {key: value * 0.75 for key, value in similarities_framework.items()}
    similarities_pydam = {key: mean([value, similarities_pyjarowinkler[key]]) * 0.25 for key, value in similarities_damerauLevenshtein.items()}
    similarities = {key: value + similarities_pydam[key] for key, value in similarities_fram.items()}
    return _filter_best_similarity(similarities, delta_heuristics)

def get_similarity(text1, text2):
    framework_distance = text1.similarity(text2)
    #cosine_distance = self.get_cosine(text1.text, text2.text)
    damerauLevenshtein_distance = damerauLevenshtein(text1.text, text2.text)
    pyjarowinkler_distance = pyjarowinkler_d.get_jaro_distance(text1.text, text2.text, winkler=True, scaling=0.1)
    return framework_distance*0.75 + 0.25*mean([damerauLevenshtein_distance, pyjarowinkler_distance])

def _filter_best_similarity(similarity_values, delta_heuristics):
    best_similarity = max(similarity_values.values(), default=0) - delta_heuristics
    similarity_values = {key: value for (key, value) in similarity_values.items() if value >= best_similarity}

    return similarity_values

def compute_distances(d):
    if type(d) is not dict:
        return d

    weight_internal_distances = 0.65
    weight_external_distances = 0.35

    return ((d["framework_distance"])) * weight_internal_distances \
    + ((d["damerauLevenshtein_distance"]  + d["pyjarowinkler_distance"]) / 3) * weight_external_distances


class _Matching:
    def encode_event(self, event):
        return

    def encode_att(self, att):
        return
    
    def compute_feature(self, i, slot):
        return {}
class EmbMatching(_Matching):
    def __init__(self, model, embeddings):
        self.model = model
        self.embeddings = embeddings

    def encode_event(self, event):
        self.event_embedding = self.model.encode(event, convert_to_tensor=True)
    
    def encode_att(self, att):
        self.cosine_scores = util.cos_sim(self.event_embedding, self.embeddings[att])
        self.cosine_scores_ext = util.cos_sim(self.event_embedding, self.embeddings[att+"-attrib"])
        
    def compute_feature(self, i, slot):
        return {
            "slot_emb": self.cosine_scores[0][i],
            "slot_complete_emb": self.cosine_scores_ext[0][i],
        }

    @staticmethod
    def list_features():
        return ["slot_emb", "slot_complete_emb"]

class BartMatching(_Matching):
    def __init__(self, tokenizer, device, nli_model):
        self.tokenizer = tokenizer
        self.device = device
        self.nli_model = nli_model


    def encode_event(self, event):
        self.event = event
    
    def compute_feature(self, i, slot):
        hypothesis3 = f'The condition is {preprocess_label(slot.text_complete())}.'
        hypo_single = f'It refers to {preprocess_label(slot.text())}'

        return {
            "bart_large_mnli_personalized": self.calculate_bart_large_mnli_personalized(self.event, hypo_single),
            "bart_large_mnli_personalized_complete": self.calculate_bart_large_mnli_personalized(self.event, hypothesis3)
        }

    def calculate_bart_large_mnli_personalized(self, premise, hypothesis):
        x = self.tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation='only_first')
        logits = self.nli_model(x.to(self.device))[0]

        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1]

        return prob_label_is_true.item()

    @staticmethod
    def list_features():
        return ["bart_large_mnli_personalized_complete", "bart_large_mnli_personalized"]


class SimMatching(_Matching):
    def __init__(self, nlp, idf):
        self.nlp = nlp
        self.idf = idf
    
    def encode_event(self, event):
        self.text_vector = self.nlp(event)
        self.text_vector = Doc(vocab=self.nlp.vocab, words=[word.text for word in self.text_vector if not (word.is_stop or word.is_punct)])
        if not self.text_vector.has_vector:
            self.text_vector = self.nlp(event)  

        self.lemmas_text = set([token.lemma_ for token in self.text_vector])
        

    def encode_att(self, att):
        self.att = att

    def compute_feature(self, i, slot):
        label = preprocess_label(slot.text())
        if len(label) == 0:
            label = slot.text()
        slot_vector = self.nlp(label)
        slot_vector_terms = [term for term in slot_vector if not (term.is_stop or term.is_punct)]

        slot_sim = get_similarity_vector(self.text_vector, slot_vector)
            
        slot_vector_complete = self.nlp(preprocess_label(slot.text_complete()))
        slot_complete_sim = get_similarity_vector(self.text_vector, slot_vector_complete)

        slot_complete_vector_terms = [term for term in slot_vector_complete if not (term.is_stop or term.is_punct)]

        lemmas_slot = set([token.lemma_ for token in slot_vector_terms])
        lemmas_slot_complete = set([token.lemma_ for token in slot_complete_vector_terms])

        idf_slot = [self.idf(term, self.att) for term in lemmas_slot]
        idf_slot_complete = [self.idf(term, self.att) for term in lemmas_slot_complete]
        idf_text = [self.idf(term, self.att) for term in self.lemmas_text]

        return {
            "slot_sim": slot_sim,
            "slot_complete_sim": slot_complete_sim,
            "slot_is_sim": _is_sim(self.idf, self.text_vector, slot_vector_terms , self.att),
            "slot_complete_is_sim": _is_sim(self.idf, self.text_vector, slot_complete_vector_terms , self.att),
            "idf_text": sum(idf_text) / len(idf_text) if len(idf_text) > 0 else 1, 
            "idf_slot": sum(idf_slot) / len(idf_slot) if len(idf_slot) > 0 else 1,
            "idf_slot_complete": sum(idf_slot_complete) / len(idf_slot_complete),
        }


def _maxsim(t, w):        
    return max([t.similarity(t2) for t2 in w]) if len(w) > 0 else 0

def _is_sim(idf, text_vector_terms, slot_vector_terms, att):
    sim_text = sum([_maxsim(term, slot_vector_terms) * idf(term.lemma_, att) for term in text_vector_terms]) if len(text_vector_terms) > 0 else 0
    idf_text = sum([idf(term.lemma_, att) for term in text_vector_terms]) if len(text_vector_terms) > 0 else 1
    sim_slot = sum([_maxsim(term, text_vector_terms) * idf(term.lemma_, att) for term in slot_vector_terms]) if len(slot_vector_terms) > 0 else 0
    idf_slot = sum([idf(term.lemma_, att) for term in slot_vector_terms]) if len(slot_vector_terms) > 0 else 1

    if idf_text == 0:
        idf_text = 1

    if idf_slot == 0:
        idf_slot = 1

    return (sim_text/idf_text + sim_slot/idf_slot)/2


def get_similarity_vector(text1, text2):
    if (text1.text == '' or text2.text == '') and (text1.text != text2.text):
        return {
            "framework_distance": 0,
            "cosine_distance": 0,
            "damerauLevenshtein_distance": 0,
            "pyjarowinkler_distance": 0
        }

        
    result = {
        "framework_distance": text1.similarity(text2) if text1.text != text2.text else 1.0,
        "cosine_distance": get_cosine(text1.text, text2.text) if text1.text != text2.text else 1.0,
        "damerauLevenshtein_distance": damerauLevenshtein(text1.text, text2.text, similarity=True) if text1.text != text2.text else 1.0,
        "pyjarowinkler_distance": pyjarowinkler_d.get_jaro_distance(text1.text, text2.text, winkler=True, scaling=0.1) if text1.text != text2.text else 1.0
    }

    return result





WORD = re.compile('\w')
CAMEL_PATTERN_1 = re.compile('(.)([A-Z][a-z]+)')
CAMEL_PATTERN_2 = re.compile('([a-z0-9])([A-Z])')

NON_ALPHANUM_PATTERN = re.compile('[^a-zA-Z]')

def _camel_to_white(label):
    label = CAMEL_PATTERN_1.sub(r'\1 \2', label)
    return CAMEL_PATTERN_2.sub(r'\1 \2', label)

def preprocess_label(label):
    label = str(label)
    label = _camel_to_white(label).lower()
    label = NON_ALPHANUM_PATTERN.sub(' ', label)
    parts = label.split()
    res = []
    for part in parts:
        clean = ''.join([i for i in part if not i.isdigit()])
        res.append(clean)
    result = ' '.join(res)
    if len(result) == 0:
        result = label
    return result
