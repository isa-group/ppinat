import logging
import re

import pandas as pd
import ppinot4py
import spacy

import ppinot4py.model as ppinot
from ppinot4py.model import TimeInstantCondition, RuntimeState, AppliesTo

import ppinat.bot.commands as commands
import ppinat.matcher.recognizers as r
from ppinat.helpers import load_log
from ppinat.matcher.similarity import SimilarityComputer
from ppinat.models.gcloud import update_models
from ppinat.ppiparser.decoder import load_decoder
from ppinat.ppiparser.ppiannotation import PPIAnnotation
from ppinat.ppiparser.transformer import (load_general_transformer,
                                          load_general_transformer_flant5,
                                          load_perfect_decoder,
                                          load_transformer,
                                          load_transformer_es)

logger = logging.getLogger(__name__)

def load_similarity(log, metrics, parsing_model, weights):
    NLP = spacy.load('en_core_web_lg')
    LOG = load_log(log, id_case="ID", time_column="DATE",
                activity_column="ACTIVITY")

    logger.info(f"Loading parser {parsing_model}... ")
    if isinstance(parsing_model, str):
        if parsing_model == "general":
            TOKEN_CLASSIFIER = './ppinat/models/GeneralClassifier'
            DECODER = load_general_transformer(TOKEN_CLASSIFIER)            
        elif parsing_model == "specific":
            update_models()
            TEXT_CLASSIFIER = './ppinat/models/TextClassification'
            TIME_MODEL = './ppinat/models/TimeModel'
            COUNT_MODEL = './ppinat/models/CountModel'
            DATA_MODEL = './ppinat/models/DataModel'
            DECODER = load_transformer(TEXT_CLASSIFIER, TIME_MODEL, COUNT_MODEL, DATA_MODEL)
        elif parsing_model == "perfect":
            DECODER = load_perfect_decoder(metrics)
        elif parsing_model == "specific_es":
            update_models("specific_es")
            TEXT_CLASSIFIER = './ppinat/models/TextClassification_es'
            TIME_MODEL = './ppinat/models/TimeModel_es'
            COUNT_MODEL = './ppinat/models/CountModel_es'
            DATA_MODEL = './ppinat/models/DataModel_es'
            DECODER = load_transformer_es(TEXT_CLASSIFIER, TIME_MODEL, COUNT_MODEL, DATA_MODEL)
        elif parsing_model == "general_flant5":
            update_models("general_flant5")
            PARSER_MODEL = './ppinat/models/GeneralParser_flant5'
            DECODER = load_general_transformer_flant5(PARSER_MODEL)
        else:
            TRAINING_FILE = 'input/parser_training/parser_training_data.json'
            PARSER_SERIAL_FILE = 'input/parser_training/parser_serialized.p'
            DECODER= load_decoder(NLP, TRAINING_FILE, PARSER_SERIAL_FILE)
    else:
        DECODER = parsing_model

    logger.info("Loading similarity computer...")
    SIMILARITY = SimilarityComputer(LOG, NLP, metric_decoder=DECODER, weights = weights)
    return SIMILARITY

def generate_weights(iss=0, emb=0, bart=0, vec=0, att=0, complete=0, multi_heur=0):
    one_slot = {
        "slot_sim": vec * (1-att) * (1 - complete),
        "slot_complete_sim": vec * (1-att) * complete,
        "slot_is_sim": iss * (1-att) * (1 - complete),
        "slot_complete_is_sim": iss * (1-att) * complete,
        "slot_emb": emb * (1-att) * (1 - complete),
        "slot_complete_emb": emb * (1-att) * (complete),
        "bart_large_mnli_personalized": bart * (1-att) * (1-complete),
        "bart_large_mnli_personalized_complete": bart * (1-att) * complete,
        "att_is_sim": att * (1 - complete),
        "att_complete_is_sim": att * (complete)
    }

    multi_slot = {
        **({f"ev1_${k}": v/2*(1-multi_heur) for k,v in one_slot.items()}),
        **({f"ev2_${k}": v/2*(1-multi_heur) for k,v in one_slot.items()}),
        "same_type": multi_heur / 2,
        "condition_ratio": multi_heur / 2
    }

    return {
        "one_slot": one_slot,
        "multi_slot": multi_slot
    }

class PPINat:
    def __init__(self):
        self.similarity = None
        self.log_configuration = None  
        self.disable_heuristics = False      

    def load_context(self, log, parsing_model='specific', matching_weights=None, raw_weights=None, disable_heuristics = False):
        if matching_weights is None and raw_weights is None:
            matching_weights = {
                "iss": 0.25,
                "emb": 0.5,
                "bart": 0.25,
                "complete": 0.5,
                "att": 0.2,
                "multi_heur": 0.25
            }

        if matching_weights is not None:
            weights = generate_weights(**matching_weights)
        elif raw_weights is not None:
            weights = raw_weights

        self.similarity = load_similarity(log, None, parsing_model, weights)
        self.log_configuration = ppinot4py.computers.LogConfiguration(id_case=self.similarity.id_case, time_column=self.similarity.time_column, activity_column=self.similarity.activity)
        self.disable_heuristics = disable_heuristics

    def parse(self, ppi) -> PPIAnnotation: 
        return self.similarity.metric_decoder(ppi)
    
    def _resolve_partials(self, ppi):
        recognized_entity = r.RecognizedEntities(None, ppi)

        agg_command = commands.ComputeMetricCommand()
        try: 
            agg_command.match_entities(recognized_entity, self.similarity, heuristics=not self.disable_heuristics)
        except Exception as e:
            logger.exception("Error while matching entities", exc_info=e, stack_info=True)

        base_measure = agg_command.partials["base_measure"][0]

        return agg_command, base_measure


    def resolve(self, ppi):
        agg_command, base_measure = self._resolve_partials(ppi)

        if len(list(agg_command.alt_match_a.keys())) > 0:
            for key in agg_command.alt_match_a.keys():
                if key not in agg_command.values and agg_command.alt_match_a[key] != []:
                    agg_command.values[key] = agg_command.alt_match_a[key][0]
        
        if len(list(base_measure.alt_match_a.keys())) > 0:
            for key in base_measure.alt_match_a.keys():
                if key not in base_measure.values and base_measure.alt_match_a[key] != []:
                    base_measure.values[key] = base_measure.alt_match_a[key][0]

        base_metric = base_measure.run(None, **base_measure.values)
        agg_command.values["base_measure"] = base_metric
        result = agg_command.run(None, **agg_command.values)

        return result.metric
    
    def compute(self, metric, time_grouper=None):
        if time_grouper is not None and isinstance(time_grouper, str):
            time_grouper = pd.Grouper(freq=time_grouper)
        return ppinot4py.measure_computer(metric, self.similarity.df, log_configuration=self.log_configuration, time_grouper=time_grouper)

    def resolve_compute(self, ppi, time_grouper=None):
        metric = self.resolve(ppi)
        return self.compute(metric, time_grouper=time_grouper)
    


class PPINatJson:
    def __init__(self):
        self.log = None
        self.id_case="ID"
        self.time_column="DATE"
        self.activity_column="ACTIVITY"

    def load_log(self, log):
        LOG = load_log(log, id_case=self.id_case, time_column=self.time_column,
                    activity_column=self.activity_column)
        
        self.log = LOG.as_dataframe()
        self.log_configuration = ppinot4py.computers.LogConfiguration(id_case=self.id_case, time_column=self.time_column, activity_column=self.activity_column)


    def resolve(self, ppi):
        if "begin" in ppi:
            from_cond = self._transform_condition(ppi["begin"]) if ppi["begin"] else TimeInstantCondition(
                            RuntimeState.START, applies_to=AppliesTo.PROCESS)
            to_cond = self._transform_condition(ppi["end"]) if ppi["end"] else TimeInstantCondition(
                            RuntimeState.END, applies_to=AppliesTo.PROCESS)

            base_metric = ppinot.TimeMeasure(
                        from_condition=from_cond,
                        to_condition=to_cond
                    )

            aggregation = self._transform_agg(ppi["aggregation"])

        elif "count" in ppi:
            count_cond = self._transform_condition(ppi["count"])
            base_metric = ppinot.CountMeasure(count_cond)
            aggregation = "AVG"
            
        other = {}

        if "group_by" in ppi and ppi["group_by"]:
            other["grouper"] = [ppinot.DataMeasure(ppi["group_by"])]

        if "filter" in ppi and ppi["filter"]:
            left, op, right = self._separate_logical_expression(ppi["filter"])
            if left == "activity":
                bm = ppinot.CountMeasure(f"`{self.activity_column}` {op} {right}")
                other["filter_to_apply"] = ppinot.DerivedMeasure(function_expression=f"ma > 0",
                                                            measure_map={"ma": bm})
            elif left in self.log.columns:
                bm = ppinot.DataMeasure(left)
                other["filter_to_apply"] = ppinot.DerivedMeasure(function_expression=f"ma {op} {right}",
                                                            measure_map={"ma": bm})
            else:
                logger.warning(f"Unknown filter: {ppi['filter']}. It will be ignored")

        metric = ppinot.AggregatedMeasure(
            base_measure=base_metric,
            single_instance_agg_function=aggregation,        
            **other
        )

        return metric

    def _transform_condition(self, cond): 
        left, op, right = self._separate_logical_expression(cond)
        if left == "activity":
            left = self.activity_column
        elif left not in self.log.columns:
            logger.warning(f"Unknown condition: {cond}")    
        
        return f"`{left}` {op} {right}"

    def _transform_agg(self, agg):
        if agg == "average":
            return "AVG"
        elif agg == "total":
            return "SUM"
        elif agg == "minimum":
            return "MIN"
        elif agg == "maximum":
            return "MAX"
        else:
            return agg.upper()

    def _separate_logical_expression(self, expression):
        # Use regular expression to find the logical operator
        match = re.search(r'(\s*[\w\s:$#]+)\s*([!=<>]+)\s*([\w\s:$#\'\"]+)\s*', expression)
        if match:
            left_side = match.group(1).strip()
            operator = match.group(2).strip()
            right_side = match.group(3).strip()
            return left_side, operator, right_side
        else:
            return None

    def compute(self, metric, time_grouper=None):
        if time_grouper is not None and isinstance(time_grouper, str):
            time_grouper = pd.Grouper(freq=time_grouper)
        return ppinot4py.measure_computer(metric, self.log, log_configuration=self.log_configuration, time_grouper=time_grouper)

    def resolve_compute(self, json_ppi, time_grouper=None):
        metric = self.resolve(ppi)
        return self.compute(metric, time_grouper=time_grouper)






