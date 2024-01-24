import gzip
import json
import logging
import ssl
import time
import urllib
from enum import Enum
from os import remove
from os.path import exists

from colorama import Fore
from ppinot4py.model import AppliesTo, RuntimeState, TimeInstantCondition

import ppinat.bot.commands as commands
import ppinat.bot.types as bottypes
import ppinat.matcher.recognizers as r
from ppinat.computer import load_similarity
from ppinat.ppiparser.ppiannotation import PPIAnnotation, text_by_tag

logger = logging.getLogger(__name__)

ATTRIBUTES = [
    "from", "to", "when", "condition", "filter", "aggregation", "data", "period", "groupby"
]

TAGS = [
    "TSE", "TEE", "TBE", "CE", "AttributeName", "GBC", "FDE", "AGR", "CCI", "AttributeValue"
]

class EvalResult(Enum):
    
    OK1 = 1 # prediction and goldstandard have values and they match
    OK2 = 2 # prediction and goldstandard have no values
    NOK1 = 3 # prediction and goldstandard have values, but they don't match
    NOK2 = 4 # prediction has value, but goldstandard doesn't
    NOK3 = 5 # prediction has no value, but goldstandard does
    P1 = 6 # prediction and goldstandard have values and they partially match

    def is_ok(self):
        return self == EvalResult.OK1 or self == EvalResult.OK2
    def is_fail(self):
        return self == EvalResult.NOK1 or self == EvalResult.NOK2 or self == EvalResult.NOK3


class Evaluation:
    def __init__(self, initial=None):
        if initial is None:
            self.value = {er: 0 for er in EvalResult}
        else:
            self.value = initial

    def add(self, eval):
        self.value = {er: self.value[er] + eval.value[er] for er in EvalResult}

    def add_value(self, value):
        self.value[value] += 1

    def true_positive(self, strict=True):
        if strict:
            true_positive = self.value[EvalResult.OK1] 
        else:
            true_positive = self.value[EvalResult.OK1] + self.value[EvalResult.P1]
        return true_positive

    def false_positive(self, strict=True):
        if strict:
            false_positive = self.value[EvalResult.NOK1] + self.value[EvalResult.NOK2] + self.value[EvalResult.P1]
        else:
            false_positive = self.value[EvalResult.NOK1] + self.value[EvalResult.NOK2]
        return false_positive

    def false_negative(self, strict=True):
        if strict:
            false_negative = self.value[EvalResult.NOK1] + self.value[EvalResult.NOK3] + self.value[EvalResult.P1]
        else:
            false_negative = self.value[EvalResult.NOK1] + self.value[EvalResult.NOK3]
        return false_negative

    def precision(self, strict=True):
        if self.true_positive(strict) == 0:
            return 0

        return float(self.true_positive(strict)) / float(self.true_positive(strict) + self.false_positive(strict))

    def recall(self, strict=True):
        if self.true_positive(strict) == 0:
            return 0
        return float(self.true_positive(strict)) / float(self.true_positive(strict) + self.false_negative())


class TestExecution:
    def __init__(self, args, dataset, parsing_model, matching_models, disable_heuristics=False):
        if not exists(dataset):
            raise RuntimeError(
                f"File provided does not exist: {dataset}")

        with open(dataset, "r") as j:
            data = json.load(j)

            self.seleted_model = parsing_model
            if self.seleted_model == "general":
                print("Using general token classification model")
            elif self.seleted_model == "specific":
                print("Using specific token classification model")
            elif self.seleted_model == "perfect":
                print("Using perfect metric decoder")
            elif self.seleted_model == "general_flant5":
                print("Using general parser from flant5")
            else:
                print("Using viterbi decoder")

            self.result = {k: InputTest() for k in matching_models}

            datasets = data["datasets"]
            for d_name in datasets:
                metrics = data["metrics"]

                print(Fore.WHITE + "Loading dataset: " + d_name + Fore.RESET)
                load_ds_start = time.time()
                log_file = load_dataset(datasets[d_name])
                load_ds_end = time.time()
                self.load_ds = load_ds_end - load_ds_start
                print(Fore.GREEN + "Loaded" + Fore.RESET)

                print(Fore.WHITE + "Loading similarity computer: " + Fore.RESET)
                load_sim_start = time.time()
                SIMILARITY = load_similarity(log_file, metrics, self.seleted_model, None)
                load_sim_end = time.time()
                self.load_sim = load_sim_end - load_sim_start
                print(Fore.GREEN + "Loaded" + Fore.RESET)

                for model_name in matching_models:
                    SIMILARITY.weights = matching_models[model_name]
                    self.result[model_name].evaluate(d_name, metrics, SIMILARITY, disable_heuristics)

            for model_name in matching_models:
                self.result[model_name].finish()
            




        
        

class InputTest:
    def __init__(self):
        self.ppi_results = {
            "good": 0,
            "partial": 0,
            "bad": 0,
            "nothing": 0
        }

        self.tagging_overall = {
            "good": 0,
            "partial": 0,
            "bad": 0
        }
        self.tagging_tag = {t: Evaluation() for t in TAGS}

        self.matching_overall = {
            "good": 0,
            "partial": 0,
            "bad": 0,
            "parser_issue": 0
        }
        self.matching_attribute = {at: Evaluation() for at in ATTRIBUTES}

        self.attribute_results = {k: {
            'good': 0,
            'partial': 0,
            'bad': 0
        } for k in ATTRIBUTES}

        self.goldstandard_atts = {k: 0 for k in ATTRIBUTES}
        self.identified_atts = {k: 0 for k in ATTRIBUTES}

        self.execution_times = {
            "parsing": [],
            "matching": []
        }

    def evaluate(self, d_name, metrics, SIMILARITY, disable_heuristics=False):

                print(Fore.WHITE + "Analyzing metrics..." + Fore.RESET)
                for m in filter(lambda x: (x["dataset"] == d_name or d_name in x["dataset"]) and ("goldstandard" in x and d_name in x["goldstandard"]), metrics):
                    result = None

                    print(f"{Fore.BLUE}Metric -----> {m['description']}{Fore.RESET}")
                    recognized_entity = r.RecognizedEntities(None, m["description"])

                    parsing_start = time.time()
                    annotation = self.evaluate_parser(SIMILARITY, recognized_entity, m["slots"])
                    parsing_end = time.time()
                    parsing_time = parsing_end - parsing_start
                    self.execution_times["parsing"].append(parsing_time)
                    
                    matching_start = time.time()
                    result = self.evaluate_matcher(SIMILARITY, recognized_entity, annotation, m["goldstandard"][d_name], m["type"], disable_heuristics)
                    matching_end = time.time()
                    matching_time = matching_end - matching_start
                    self.execution_times["matching"].append(matching_time)

                    # if args.verbosity and result is not None:
                    #     extract_summary(SIMILARITY, m, table_summary, result)

                    print()


            # if args.verbosity:
            #     excel = pd.DataFrame(table_summary)
            #     excel.to_csv("input/info.csv")
    
    def finish(self):

            print(f"{Fore.YELLOW}-------------GENERAL INFO-------------{Fore.RESET}")
            print(f"Threshold 'a': {str(bottypes.BaseMetric.threshold_a)}")
            print(f"Threshold 'b': {str(bottypes.BaseMetric.threshold_b)}")

            print_results_summary("Parsing results summary", self.tagging_overall)
            tagging_aggregation = aggregate_eval_results(self.tagging_tag)
            print_evaluation(tagging_aggregation)
            for t in self.tagging_tag:
                print(f"-- Tag {t}")
                print_evaluation(self.tagging_tag[t])

            print(f"Parsing execution time: {sum(self.execution_times['parsing'])} seconds")

            print_results_summary("Matching results summary", self.matching_overall)
            matching_aggregation = aggregate_eval_results(self.matching_attribute)
            print_evaluation(matching_aggregation)
            for t in self.matching_attribute:
                print(f"-- Attribute {t}")
                print_evaluation(self.matching_attribute[t])

            print(f"Matching execution time: {sum(self.execution_times['matching'])} seconds")

    def evaluate_matcher(self, similarity, recognized_entity, annotation, goldstandard, type, disable_heuristics=False):
        if type == annotation.get_measure_type():
        
            agg_result, agg_command = self.aggregation_eval(recognized_entity, similarity, goldstandard, disable_heuristics)

            if 'base_measure' in agg_command.partials and agg_command.partials['base_measure'] is not None:
                metric_result = self.analyse_metric(
                    agg_command.partials['base_measure'][0], recognized_entity , similarity, goldstandard, type)
            else:
                logger.error('base_measure not found in agg_command')
                metric_result = 'bad'

            if metric_result == 'good' and agg_result == 'good':
                result = 'good'
            elif metric_result == 'bad' and agg_result == 'bad':
                result = 'bad'
            else:
                result = 'partial'
        else:
            result = 'parser_issue'
            expected = goldstandard['type'] if 'type' in goldstandard else None
            print(f"{Fore.RED}Matching skipped because measure type didn't match. Found <{annotation.get_measure_type()}>, expected <{expected}>{Fore.RESET}")

        self.matching_overall[result] += 1

        return result

    def evaluate_parser(self, similarity, recognized_entity, slots):
        annotation: PPIAnnotation = similarity.metric_decoder(recognized_entity.text)

        parser_metric_result, parser_tag_result, parser_errors = evaluate_parser_metric(annotation, slots = slots)
        print_parser_results(parser_metric_result, parser_tag_result, parser_errors)

        self.tagging_overall[parser_metric_result] += 1
        for t in TAGS:
            self.tagging_tag[t].add_value(parser_tag_result[t])

        return annotation


    def analyse_metric(self, command, recognized_entity, similarity, goldstandard, type):        
        matcher_metric_result = None
        if type == "time":
            matcher_metric_result = self.time_metric_eval(command, goldstandard)

        elif type == "count":
            matcher_metric_result = self.count_metric_eval(command, goldstandard)

        elif type == "data":
            matcher_metric_result = self.data_metric_eval(command, goldstandard)

        return matcher_metric_result

    def aggregation_eval(self, recognized_entity, similarity, goldstandard, disable_heuristics=False):
        aggregation_result = None
        filter_result = None

        agg_command = commands.ComputeMetricCommand()
        try: 
            agg_command.match_entities(recognized_entity, similarity, heuristics=not disable_heuristics)
        except Exception as e:
            print(Fore.RED + "An error ocurred while matching entities" + Fore.RESET)
            logger.exception("Error while matching entities", exc_info=e, stack_info=True)
            return None, agg_command


        # Agg function evaluation
        agg_eval_func = (lambda x: x == goldstandard['aggregation']) if 'aggregation' in goldstandard else None
        agg_result, agg_pos, agg_found = param_eval(agg_eval_func, agg_command, "agg_function")
        print_results("aggregation", agg_pos, agg_found)
        self.matching_attribute["aggregation"].add_value(agg_result)

        # Filter (denominator) evaluation
        filter_eval_func = (lambda x: comparing_conditions(goldstandard["filter"], x)) if "filter" in goldstandard else None
        filter_result, filter_pos, filter_found = param_eval(filter_eval_func, agg_command, "denominator")
        print_results("filter", filter_pos, filter_found)
        self.matching_attribute["filter"].add_value(filter_result)

        period_eval_func = (lambda x: x == goldstandard["period"]["window"]) if ("period" in goldstandard) else None
        period_result, period_pos, period_found = internal_eval(period_eval_func, agg_command.period)
        print_results("period", period_pos, period_found)
        self.matching_attribute["period"].add_value(period_result)

        group_eval_func = (lambda x: x == goldstandard["groupby"]) if ("groupby" in goldstandard) else None
        groupby_result, groupby_pos, groupby_found = internal_eval(group_eval_func, agg_command.groupby)
        print_results("groupby", groupby_pos, groupby_found)
        self.matching_attribute["groupby"].add_value(groupby_result)

        if agg_result.is_ok() and filter_result.is_ok():
            aggregation_result = 'good'
        elif agg_result.is_fail() or filter_result.is_fail():
            aggregation_result = 'bad'
        else:
            aggregation_result = 'partial'

        return aggregation_result, agg_command

    def data_metric_eval(self, command, goldstandard):
        data_result, data_pos, data_found = param_eval(lambda x: comparing_attributes(goldstandard["data_condition"], x), command, "attribute")
        print_results("data attribute", data_pos, data_found)
        self.matching_attribute["data"].add_value(data_result)

        cond_result, cond_expected, cond_found = condition_eval(goldstandard, command)
        print_condition_results(cond_result, cond_expected, cond_found)
        self.matching_attribute['condition'].add_value(cond_result)

        if data_result.is_ok() and cond_result.is_ok():
            matcher_metric_result = 'good'
        elif data_result.is_fail() or cond_result.is_fail():
            matcher_metric_result = 'bad'
        else:
            matcher_metric_result = 'partial'

        return matcher_metric_result

    def count_metric_eval(self, command, goldstandard):
        when_result, when_pos, when_found = param_eval(lambda x: comparing_conditions(goldstandard["count_condition"], x), command, "when")
        print_results("when condition", when_pos, when_found)
        self.matching_attribute['when'].add_value(when_result)

        cond_result, cond_expected, cond_found = condition_eval(goldstandard, command)
        print_condition_results(cond_result, cond_expected, cond_found)
        self.matching_attribute['condition'].add_value(cond_result)

        if when_result.is_ok() and cond_result.is_ok():
            matcher_metric_result = 'good'
        elif when_result.is_fail() or cond_result.is_fail():
            matcher_metric_result = 'bad'
        else:
            matcher_metric_result = 'partial'
        
        return matcher_metric_result

    def time_metric_eval(self, command, goldstandard):
        from_result, from_eval_pos, from_found = param_eval(lambda x: comparing_conditions(goldstandard["from_condition"], x), command, "from_cond")
        to_result, to_eval_pos, to_found = param_eval(lambda x: comparing_conditions(goldstandard["to_condition"], x), command, "to_cond")
        #print_time_results(from_eval_pos, to_eval_pos, from_found, to_found)
        print_results("from", from_eval_pos, from_found)
        print_results("to", to_eval_pos, to_found)

        self.matching_attribute['from'].add_value(from_result)
        self.matching_attribute['to'].add_value(to_result)

        cond_result, cond_expected, cond_found = condition_eval(goldstandard, command)
        print_condition_results(cond_result, cond_expected, cond_found)
        self.matching_attribute['condition'].add_value(cond_result)
        
        if from_result.is_ok() and to_result.is_ok() and cond_result.is_ok():
            matcher_metric_result = 'good'
        elif from_result.is_fail() or to_result.is_fail() or cond_result.is_fail():
            matcher_metric_result = 'bad'
        else:
            matcher_metric_result = 'partial'

        return matcher_metric_result


def print_results_summary(title, overall):
    print(f"{title}:")
    print(f"{Fore.GREEN}Good: {str(overall['good'])}{Fore.RESET}")
    print(f"{Fore.YELLOW}Partial: {str(overall['partial'])}{Fore.RESET}")
    print(f"{Fore.RED}Bad: {str(overall['bad'])}{Fore.RESET}")
    if "parser_issue" in overall:
        print(f"{Fore.RED}Bad: {str(overall['parser_issue'])}{Fore.RESET}")

def print_evaluation(evaluation):
    print(f"OK1: {evaluation.value[EvalResult.OK1]} (prediction and goldstandard have values and they match)")
    print(f"OK2: {evaluation.value[EvalResult.OK2]} (prediction and goldstandard have no values)")
    print(f"P1: {evaluation.value[EvalResult.P1]} (prediction and goldstandard have values and they partially match)")
    print(f"NOK1: {evaluation.value[EvalResult.NOK1]} (prediction and goldstandard have values, but they don't match)")
    print(f"NOK2: {evaluation.value[EvalResult.NOK2]} (prediction has value, but goldstandard doesn't)")
    print(f"NOK3: {evaluation.value[EvalResult.NOK3]} (prediction has no value, but goldstandard does)")
    print(f"Precision: {evaluation.precision()} / Recall: {evaluation.recall()}")
    print(f"Precision: {evaluation.precision(strict=False)} / Recall: {evaluation.recall(strict=False)} [Allows partial]")



def extract_summary(similarity, metric, columns_values, result):
    for slot in similarity.slot_information:
        slots = slot.split(", ")
        if len(slots) == 2:
            slot1 = slots[0]
            slot2 = slots[1]
        else:
            slot1 = slots[0]
            slot2 = None
        values = {"metric": metric["description"],
                    "type": metric["type"],
                    "dataset": metric["dataset"],
                    "result": result,
                    "slot1": slot1,
                    "slot2": slot2}

        for score in similarity.slot_information[slot]:
            values[score] = similarity.slot_information[slot][score]

        columns_values.append(values)
        # for v in values:
        #    columns_values[v].append(values[v])


def aggregate_eval_results(results):
    return Evaluation({er: sum([results[k].value[er] for k in results]) for er in EvalResult})


def load_dataset(d):    
    if "url" in d and not exists(d["file"]):
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(d["url"], d["file"] + ".gz")
        decompress(d["file"] + ".gz")
    return d["file"]

def decompress(filename):
    f = gzip.open(filename)
    write_file(filename[:filename.rfind(".gz")], f.read())
    f.close()
    remove(filename)

def write_file(filename, data):
    try:
        f = open(filename, "wb")
    except IOError as e:
        print(e.errno, e.message)
    else:
        f.write(data)
        f.close()



def aggregate_results(attribute_results):
    return {k: sum([attribute_results[att][k] for att in attribute_results]) for k in ['good', 'partial', 'bad']}

def print_overall_results(attribute_results, goldstandard, identified):
    print(Fore.GREEN + "Good: " + str(attribute_results["good"]) + Fore.RESET)
    print(Fore.YELLOW + "partial: " +
        str(attribute_results["partial"]) + Fore.RESET)
    print(Fore.RED + "Bad: " + str(attribute_results["bad"]) + Fore.RESET)

    precision = attribute_results['good'] / identified if identified > 0 else -1
    recall = attribute_results['good'] / goldstandard if goldstandard > 0 else -1
    print(f"Precision: {precision} / Recall: {recall}")


def evaluate_parser_metric(annotation, slots):
    from_text = text_by_tag(annotation, "TSE")
    to_text = text_by_tag(annotation, "TEE")
    only_text = text_by_tag(annotation, "TBE")
    when_text = text_by_tag(annotation, "CE")
    data_attribute = text_by_tag(annotation, "AttributeName")
    group_text = text_by_tag(annotation, 'GBC')
    denominator_text = text_by_tag(annotation, 'FDE')
    aggregation_text = annotation.get_aggregation_function()
    conditional_text = text_by_tag(annotation, 'CCI')
    conditional_attr_text = text_by_tag(annotation, 'AttributeValue')

    tags = {
        "TSE": from_text,
        "TEE": to_text,
        "TBE": only_text,
        "CE": when_text,
        "AttributeName": data_attribute,
        "GBC": group_text,
        "FDE": denominator_text,
        "AGR": aggregation_text,
        "CCI": conditional_text,
        "AttributeValue": conditional_attr_text
    }

    tag_result = {}
    errors = []

    for tag in TAGS:
        if tag in slots:
            if slots[tag] == tags[tag]:
                tag_result[tag] = EvalResult.OK1
            elif tags[tag] is None:
                tag_result[tag] = EvalResult.NOK3
            else:
                tag_result[tag] = EvalResult.NOK1
        else:
            if tags[tag] is None:
                tag_result[tag] = EvalResult.OK2
            else:
                tag_result[tag] = EvalResult.NOK2

        if not tag_result[tag].is_ok():            
            errors.append((tag, slots[tag] if tag in slots else None, tags[tag]))

    is_ok = [tag_result[t] == EvalResult.OK1 for t in tag_result]

    if not errors:
        metric_result = 'good'
    elif any(is_ok):
        metric_result = 'partial'
    else:
        metric_result = 'bad'

    return metric_result, tag_result, errors
    
def print_parser_results(parser_metric_result, parser_tag_result, parser_errors):
    if parser_metric_result == 'good':
        print(f"{Fore.GREEN}All slots have been tagged well{Fore.RESET}")
    elif parser_metric_result == 'partial':
        result_types = [parser_tag_result[t] for t in parser_tag_result]
        print(f"{Fore.YELLOW}{result_types.count(EvalResult.OK1)}/{len(result_types) - result_types.count(EvalResult.OK2)} slots have been tagged well{Fore.RESET}")
    else:
        print(f"{Fore.RED}Bad tagged{Fore.RESET}")

    for tag, expected, found in parser_errors:
        print(f"Correct slot for {tag} is '{expected}' but found '{found}'")


def comparing_attributes(attribute, attribute_result):
    comparison = False

    if "attribute" in attribute:
        if attribute["attribute"] == attribute_result:
            comparison = True

    return comparison        

def comparing_conditions(condition, condition_result):
    comparison = False

    if "attribute" in condition and "value" in condition and "operator" in condition:
        condition_string = str(condition_result)
        if condition["operator"] in condition_string \
                and condition["value"].lower() in condition_string.lower() \
                and condition["attribute"] in condition_string:
            comparison = True
    elif "case" in condition:
        if isinstance(condition_result, TimeInstantCondition) \
                and condition_result.applies_to == AppliesTo.PROCESS:
            if condition["case"] == "begin" and condition_result.changes_to_state == RuntimeState.START:
                comparison = True
            elif condition["case"] == "end" and condition_result.changes_to_state == RuntimeState.END:
                comparison = True

    return comparison

def print_time_results(p1, p2, all1, all2):
    text = None
    c1 = p1 is not None
    c2 = p2 is not None
    if c1 and c2:
        text = f"Matched from (position {p1+1} / {len(all1)}) and to (pos {p2+1} / {len(all2)}) condition"
        if p1 == 0 and p2 == 0:
            matching_color = Fore.GREEN
        else:
            matching_color = Fore.YELLOW

        print(matching_color + text + Fore.RESET)
    elif c1:
        text = f"Matched only from (pos {p1+1} / {len(all1)}) condition [to total found {len(all2)}]"
        print(Fore.YELLOW + text + Fore.RESET)
    elif c2:
        text = f"Matched only to (pos {p2+1} / {len(all2)}) condition [from total found {len(all1)}]"
        print(Fore.YELLOW + text + Fore.RESET)
    else:
        text = f"Not matched from and to condition [from total found {len(all1)} / to total found {len(all2)}]"
        print(Fore.RED + text + Fore.RESET)

    print("Values found for from:")
    
    for i, v in enumerate(all1):
        if c1 and i == p1:
            text = Fore.GREEN + str(v.value)
        else:
            text = str(v.value)

        print(text + Fore.RESET)

    print("Values found for to:")
    for i, v in enumerate(all2):
        if c2 and i == p2:
            text = Fore.GREEN + str(v.value)
        else:
            text = str(v.value)

        print(text + Fore.RESET)


def print_results(label, p, all):
    text = None
    c = p is not None
    if c:
        if p == 0:
            matching_color = Fore.GREEN
        else:
            matching_color = Fore.YELLOW

        text = f"Matched {label} (position {p+1 if len(all) > 0 else 0} / {len(all)})"
        print(f"{matching_color}{text}{Fore.RESET}")
    else:
        print(f"{Fore.RED}Not matched {label} [total found {len(all)}]{Fore.RESET}")

    if all:
        print(f"values found for {label}:")
        for i, v in enumerate(all):
            try:
                text =  str(v.value)
            except:
                try:
                    text =  str(v.metric)
                except:
                    text = str(v)

            if i == p:
                text = Fore.GREEN + text

            print(text + Fore.RESET)


def print_condition_results(result, expected, found):
    if result == EvalResult.OK2:
        # Print nothing
        return
    if result == EvalResult.OK1:
        matching_color = Fore.GREEN
        text = "Matched condition"
    else:
        matching_color = Fore.RED
        if result == EvalResult.NOK3:
            text = f"Not matched condition. Expected: {expected['operator']} {expected['value']}"
        elif result == EvalResult.NOK2:
            text = f"Found a condition that did not exist: {found.op} {found.value.value}"
        elif result == EvalResult.NOK1:
            text = f"Wrongly matched conditions. Found: {found.op} {found.value.value}. Expected: {expected['operator']} {expected['value']}"

    print(f"{matching_color}{text}{Fore.RESET}")


def param_eval(goldstandard, command, command_param):
    if command_param in command.values:
        found_values = [command.values[command_param]]
    else:
        found_values = command.alt_match_a[command_param] if command_param in command.alt_match_a else [] + \
            command.alt_match_b[command_param] if command_param in command.alt_match_b else [
        ]

    return compute_evaluation_result(goldstandard, found_values)

def compute_evaluation_result(goldstandard, found_values):
    if goldstandard is None:
        if found_values:
            result = EvalResult.NOK2
            position = None
        else:
            result = EvalResult.OK2
            position = 0
    else:
        position = next((i for i,v in enumerate(found_values) if goldstandard(v.value)), None)

        if position is None:
            if found_values:
                result = EvalResult.NOK1
            else:
                result = EvalResult.NOK3
            
        elif position == 0:
            result = EvalResult.OK1
        else:
            result = EvalResult.P1    

    return result, position, found_values

def internal_eval(goldstandard, value):
    found_values = value[0] + value[1] if value is not None else []
    return compute_evaluation_result(goldstandard, found_values)


def condition_eval(goldstandard, command):
    result = None

    expected_condition = goldstandard["condition"] if "condition" in goldstandard else None
    found_condition = command.values["conditional_metric"] if "conditional_metric" in command.values else None

    if expected_condition is not None:
        if found_condition is None:
            result = EvalResult.NOK3
        else:
            if "operator" in expected_condition and "value" in expected_condition:
                if expected_condition["operator"] == found_condition.op \
                        and expected_condition["value"] == str(found_condition.value.value):
                    result = EvalResult.OK1
                else:
                    result = EvalResult.NOK1
            else:
                result = EvalResult.NOK1

    else:
        if found_condition is not None:
            result = EvalResult.NOK2
        else:
            result = EvalResult.OK2                

    return result, expected_condition, found_condition




def main(args):
    print(Fore.BLUE)
    print(r" _  _      ____  _     _____    _____  _____ ____  _____ ")
    print(r"/ \/ \  /|/  __\/ \ /\/__ __/  /__ __//  __// ___\/__ __/")
    print(r"| || |\ |||  \/|| | ||  / \      / \  |  \  |    \  / \  ")
    print(r"| || | \|||  __/| \_/|  | |      | |  |  /_ \___ |  | |  ")
    print(r"\_/\_/  \|\_/   \____/  \_/      \_/  \____\\____/  \_/ ")
    print(Fore.RESET)

    if args.verbosity:
        print(Fore.YELLOW + "INFO: Verbosity is on" + Fore.RESET)

    InputTest(args)
