import gzip
import json
import ssl
import urllib
from os import remove
from os.path import exists

import pandas as pd
import ppinat.bot.commands as commands
import ppinat.bot.types as t
import ppinat.matcher.recognizers as r
import spacy
from colorama import Fore
from ppinat.helpers import load_log
from ppinat.matcher.similarity import SimilarityComputer
from ppinat.ppiparser.ppiannotation import PPIAnnotation, text_by_tag
from ppinat.ppiparser.transformer import load_transformer, load_general_transformer, load_perfect_decoder
from ppinot4py.model import AppliesTo, RuntimeState, TimeInstantCondition
from ppinat.models.gcloud import update_models


def aggregate_results(attribute_results):
    return {k: sum([attribute_results[att][k] for att in attribute_results]) for k in ['good', 'regular', 'bad']}


def print_results(attribute_results, goldstandard, identified):
    print(Fore.GREEN + "Good: " + str(attribute_results["good"]) + Fore.RESET)
    print(Fore.YELLOW + "Regular: " +
          str(attribute_results["regular"]) + Fore.RESET)
    print(Fore.RED + "Bad: " + str(attribute_results["bad"]) + Fore.RESET)

    precision = attribute_results['good'] / \
        identified if identified > 0 else -1
    recall = attribute_results['good'] / \
        goldstandard if goldstandard > 0 else -1
    print(f"Precision: {precision} / Recall: {recall}")


class InputTest:
    def __init__(self, args, other=None):
        if not exists(args.filename):
            raise RuntimeError(
                f"File provided does not exist: {args.filename}")

        with open(args.filename, "r") as j:
            data = json.load(j)
            datasets = data["datasets"]
            self.metrics = data["metrics"]

            self.weights = other["weights"] if "weights" in other else None

            self.ppi_results = {
                "good": 0,
                "regular": 0,
                "bad": 0,
                "nothing": 0
            }

            self.tagging_results = {
                "good": 0,
                "regular": 0,
                "bad": 0
            }

            self.seleted_model = args.model
            if self.seleted_model == "general":
                print("Using general token classification model")
            elif self.seleted_model == "specific":
                print("Using specific token classification model")
            else:
                print("Using perfect metric decoder")

            columns_values = []

            attributes = [
                "from", "to", "when", "condition", "filter", "aggregation", "data"
            ]

            self.attribute_results = {k: {
                'good': 0,
                'regular': 0,
                'bad': 0
            } for k in attributes}

            self.goldstandard_atts = {k: 0 for k in attributes}
            self.identified_atts = {k: 0 for k in attributes}

            for d_name in datasets:
                print(Fore.WHITE + "Loading dataset: " + d_name + Fore.RESET)
                d = datasets[d_name]
                if "url" in d and not exists(d["file"]):
                    ssl._create_default_https_context = ssl._create_unverified_context
                    urllib.request.urlretrieve(d["url"], d["file"] + ".gz")
                    self.decompress(d["file"] + ".gz")
                print(Fore.GREEN + "Loaded" + Fore.RESET)

                print(Fore.WHITE + "Analysing metrics..." + Fore.RESET)
                SIMILARITY = self.similarity_charge(d["file"])
                for m in self.metrics:
                    if m["dataset"] == d_name or d_name in m["dataset"]:
                        if m["type"] == "time":
                            self.analyse_metric(
                                commands.TimeMetricCommand(), SIMILARITY, m, d_name, args, columns_values)
                        elif m["type"] == "count":
                            self.analyse_metric(
                                commands.CountMetricCommand(), SIMILARITY, m, d_name, args, columns_values)
                        elif m["type"] == "data":
                            self.analyse_metric(
                                commands.DataMetricCommand(), SIMILARITY, m, d_name, args, columns_values)

            if args.verbosity:
                excel = pd.DataFrame(columns_values)
                excel.to_csv("input/info.csv")
#                excel.to_excel("input/input_test_info.xlsx")

            print(Fore.YELLOW + "-------------GENERAL INFO-------------" + Fore.RESET)
            print("Threshold 'a': " + str(t.BaseMetric.threshold_a) +
                  "\n" + "Threshold 'b': " + str(t.BaseMetric.threshold_b))

            if self.seleted_model != "perfect":
                print("Parsing results:")
                print(Fore.GREEN + "Good: " +
                      str(self.tagging_results["good"]) + Fore.RESET)
                print(Fore.YELLOW + "Regular: " +
                      str(self.tagging_results["regular"]) + Fore.RESET)
                print(Fore.RED + "Bad: " +
                      str(self.tagging_results["bad"]) + Fore.RESET)

            print("Metric results:")
            print(Fore.GREEN + "Good: " +
                  str(self.ppi_results["good"]) + Fore.RESET)
            print(Fore.YELLOW + "Regular: " +
                  str(self.ppi_results["regular"]) + Fore.RESET)
            print(Fore.RED + "Bad: " +
                  str(self.ppi_results["bad"]) + Fore.RESET)
            print("Nothing: " + str(self.ppi_results["nothing"]))

            overall_results = aggregate_results(self.attribute_results)
            print("Attribute results:")
            print_results(overall_results, sum(
                self.goldstandard_atts.values()), sum(self.identified_atts.values()))
            for att in attributes:
                print(f"Attribute {att} results:")
                print_results(
                    self.attribute_results[att], self.goldstandard_atts[att], self.identified_atts[att])

    def comparing_attributes(self, attribute, attribute_result):
        comparison = False

        if "attribute" in attribute:
            if attribute["attribute"] == attribute_result.value:
                comparison = True

        return comparison

    def comparing_conditions(self, condition, condition_result):
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

    def show_time_results(self, c1, c2, p1, p2, all1, all2):
        text = None
        if c1 and c2:
            text = f"Matched from ({p1} / {len(all1)}) and to ({p2} / {len(all2)}) condition"
            if p1 == 0 and p2 == 0:
                matching_color = Fore.GREEN
            else:
                matching_color = Fore.YELLOW

            print(matching_color + text + Fore.RESET)
        elif c1:
            text = f"Matched only from ({p1} / {len(all1)}) condition [to total {len(all2)}]"
            print(Fore.YELLOW + text + Fore.RESET)
        elif c2:
            text = f"Matched only to ({p2} / {len(all2)}) condition [from total {len(all1)}]"
            print(Fore.YELLOW + text + Fore.RESET)
        else:
            text = f"Not matched from and to condition [from total {len(all1)} / to total {len(all2)}]"
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

        return text

    def show_results(self, label, c, p, all):
        text = None
        if c:
            if p == 0:
                matching_color = Fore.GREEN
                mode = "good"
            else:
                matching_color = Fore.YELLOW
                mode = "regular"

            text = f"Matched {label} ({p} / {len(all)})"
            print(matching_color + text + Fore.RESET)
        else:
            text = f"Not matched {label} [total {len(all)}]"
            print(Fore.RED + text + Fore.RESET)

        print(f"values found for {label}:")
        for i, v in enumerate(all):
            if c and i == p:
                text = Fore.GREEN + str(v.value)
            else:
                text = str(v.value)

            print(text + Fore.RESET)

        return text

    def show_condition_results(self, associated_condition, results):
        if associated_condition:
            matching_color = Fore.GREEN
            text = f"Matched condition"
        else:
            matching_color = Fore.RED
            if results is None:
                text = f"Not matched condition"
            else:
                text = f"Wrongly matched conditions: {results.op} {results.value.value}"

        print(matching_color + text + Fore.RESET)

    def param_eval(self, goldstandard, command, command_param):
        if command_param in command.values:
            values_list = [command.values[command_param]]
        else:
            values_list = command.alt_match_a[command_param] if command_param in command.alt_match_a else [] + \
                command.alt_match_b[command_param] if command_param in command.alt_match_b else [
            ]

        matching_value = False
        position_value = 0

        for i, v in enumerate(values_list):
            eval_result = v.value

            matching_value = goldstandard(eval_result)
            if matching_value:
                position_value = i
                break

        if matching_value:
            if position_value == 0:
                output = 'good'
            else:
                output = 'regular'
        else:
            output = 'bad'

        return {'match': matching_value, 'pos': position_value, 'output': output, 'values': values_list}

    def analyse_metric(self, command, similarity, metric, dataset_name, args, columns_values):
        recognized_entity = r.RecognizedEntities(None, metric["description"])

        try:
            command.match_entities(recognized_entity, similarity)
        except:
            return

        annotation: PPIAnnotation = similarity.metric_decoder(
            recognized_entity.text)

        if self.seleted_model != "perfect":
            print(Fore.BLUE + "Metric -----> " +
                  metric["description"] + Fore.RESET)

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

            good_tags = 0
            bad_tags = 0

            for tag in metric["slots"].keys():
                if metric["slots"][tag] == tags[tag]:
                    good_tags += 1
                else:
                    bad_tags += 1
                    print(
                        f"Correct slot for {tag} is '{metric['slots'][tag]}' but found '{tags[tag]}'")

            if good_tags == 0:
                self.tagging_results["bad"] += 1
                print(f"{Fore.RED}Bad tagged{Fore.RESET}")
            elif good_tags >= 1 and bad_tags >= 1:
                self.tagging_results["regular"] += 1
                print(
                    f"{Fore.YELLOW}{good_tags}/{good_tags + bad_tags} slots have been tagged well{Fore.RESET}")
            else:
                self.tagging_results["good"] += 1
                print(f"{Fore.GREEN}All slots have been tagged well{Fore.RESET}")

        result = None
        if("goldstandard" in metric and dataset_name in metric["goldstandard"]):
            goldstandard = metric["goldstandard"][dataset_name]
            if metric["type"] == "time":
                from_condition = goldstandard["from_condition"]
                to_condition = goldstandard["to_condition"]

                from_eval = self.param_eval(lambda x: self.comparing_conditions(
                    from_condition, x), command, "from_cond")
                to_eval = self.param_eval(lambda x: self.comparing_conditions(
                    to_condition, x), command, "to_cond")

                self.attribute_results['from'][from_eval['output']] += 1
                self.attribute_results['to'][to_eval['output']] += 1
                self.goldstandard_atts['from'] += 1
                self.goldstandard_atts['to'] += 1
                if len(from_eval['values']) > 0:
                    self.identified_atts['from'] += 1
                if len(to_eval['values']) > 0:
                    self.identified_atts['to'] += 1

                self.show_time_results(
                    from_eval['match'], to_eval['match'], from_eval['pos'], to_eval['pos'], from_eval['values'], to_eval['values'])

                cond = self.condition_eval(goldstandard, command)
                if cond is not None:
                    self.attribute_results['condition'][cond] += 1

                if from_eval['output'] == 'good' and to_eval['output'] == 'good' and (cond is None or cond == 'good'):
                    result = 'good'
                elif from_eval['output'] == 'bad' or to_eval['output'] == 'bad' or (cond == 'bad'):
                    result = 'bad'
                else:
                    result = 'regular'

            elif metric["type"] == "count":
                condition = goldstandard["count_condition"]

                when_eval = self.param_eval(
                    lambda x: self.comparing_conditions(condition, x), command, "when")
                self.attribute_results['when'][when_eval['output']] += 1
                self.goldstandard_atts['when'] += 1
                if len(when_eval['values']) > 0:
                    self.identified_atts['when'] += 1

                self.show_results("when condition",
                                  when_eval['match'], when_eval['pos'], when_eval['values'])

                cond = self.condition_eval(goldstandard, command)
                if cond is not None:
                    self.attribute_results['condition'][cond] += 1

                if when_eval['output'] == 'good' and (cond is None or cond == 'good'):
                    result = 'good'
                elif when_eval['output'] == 'bad' or (cond == 'bad'):
                    result = 'bad'
                else:
                    result = 'regular'

            elif metric["type"] == "data":
                condition = goldstandard["data_condition"]

                data_eval = self.param_eval(lambda x: self.comparing_attribute(
                    condition, x), command, "attribute")
                self.attribute_results['data'][data_eval['output']] += 1
                self.goldstandard_atts['data'] += 1
                if len(data_eval['values']) > 0:
                    self.identified_atts['data'] += 1

                self.show_results("data attribute",
                                  data_eval['match'], data_eval['pos'], data_eval['values'])

                cond = self.condition_eval(goldstandard, command)
                if cond is not None:
                    self.attribute_results['condition'][cond] += 1

                if data_eval['output'] == 'good' and (cond is None or cond == 'good'):
                    result = 'good'
                elif data_eval['output'] == 'bad' or (cond == 'bad'):
                    result = 'bad'
                else:
                    result = 'regular'

            if annotation.get_aggregation_function() is not None:
                agg_command = commands.ComputeMetricCommand()
                agg_command.match_entities(recognized_entity, similarity)

                if "aggregation" in goldstandard:
                    self.goldstandard_atts['aggregation'] += 1

                    agg_eval = self.param_eval(
                        lambda x: x == goldstandard['aggregation'], agg_command, "agg_function")
                    self.attribute_results['aggregation'][agg_eval['output']] += 1
                    if len(agg_eval['values']) > 0:
                        self.identified_atts['aggregation'] += 1
                    self.show_results(
                        "aggregation", agg_eval['match'], agg_eval['pos'], agg_eval['values'])

                    filter_eval = None
                    if "filter" in goldstandard:
                        self.goldstandard_atts["filter"] += 1
                        filter_eval = self.param_eval(lambda x: self.comparing_conditions(
                            goldstandard["filter"], x), command, "denominator")
                        self.attribute_results['filter'][filter_eval['output']] += 1
                        if len(filter_eval['values']) > 0:
                            self.identified_atts['filter'] += 1
                        self.show_results(
                            "filter", filter_eval['match'], filter_eval['pos'], filter_eval['values'])

                    if result == 'good' and agg_eval['output'] == 'good' and (filter_eval is None or cond == 'good'):
                        result = 'good'
                    elif result == 'bad' or agg_eval['output'] == 'bad' or (filter_eval == 'bad'):
                        result = 'bad'
                    else:
                        result = 'regular'
                else:
                    self.identified_atts["aggregation"] += 1
                    if any(["denominator" in x for x in [agg_command.values, agg_command.alt_match_a, agg_command.alt_match_b]]):
                        self.identified_atts["filter"] += 1

                    result = 'bad'
            else:
                if "aggregation" in goldstandard:
                    self.goldstandard_atts["aggregation"] += 1
                    if "filter" in goldstandard:
                        self.goldstandard_atts["filter"] += 1

            self.ppi_results[result] += 1

        else:
            result = "There are no defined conditions. Execution result:"
            print(Fore.YELLOW + result + Fore.RESET)
            print(command)
            self.ppi_results["nothing"] = self.ppi_results["nothing"] + 1

        self.extract_excel(args, similarity, metric, columns_values, result)
        print()

    def condition_eval(self, goldstandard, command):
        result = None
        comparison_associated = False
        if "conditional_metric" in command.values:
            self.identified_atts['condition'] += 1

        if "condition" in goldstandard:
            self.goldstandard_atts['condition'] += 1
            associated_condition = goldstandard["condition"]
            if "conditional_metric" in command.values:
                associated_condition_result = command.values["conditional_metric"]
            else:
                associated_condition_result = None

            if "operator" in associated_condition and "value" in associated_condition:
                if associated_condition_result:
                    if associated_condition["operator"] == associated_condition_result.op \
                            and associated_condition["value"] == str(associated_condition_result.value.value):
                        comparison_associated = True

            if comparison_associated:
                result = 'good'
            else:
                result = "bad"

            self.show_condition_results(
                comparison_associated, associated_condition_result)

        return result

    def extract_excel(self, args, similarity, metric, columns_values, result):
        if args.verbosity:
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

    def similarity_charge(self, log):
        NLP = spacy.load('en_core_web_lg')
        LOG = load_log(log, id_case="ID", time_column="DATE",
                       activity_column="ACTIVITY")

        if self.seleted_model == "general":
            TOKEN_CLASSIFIER = './ppinat/models/GeneralClassifier'
            DECODER = load_general_transformer(TOKEN_CLASSIFIER)

        elif self.seleted_model == "specific":
            update_models()
            TEXT_CLASSIFIER = './ppinat/models/TextClassification'
            TIME_MODEL = './ppinat/models/TimeModel'
            COUNT_MODEL = './ppinat/models/CountModel'
            DATA_MODEL = './ppinat/models/DataModel'
            DECODER = load_transformer(
                TEXT_CLASSIFIER, TIME_MODEL, COUNT_MODEL, DATA_MODEL)
        else:
            DECODER = load_perfect_decoder(self.metrics)

        SIMILARITY = SimilarityComputer(
            LOG, NLP, metric_decoder=DECODER, weights=self.weights)
        return SIMILARITY

    def decompress(self, filename):
        f = gzip.open(filename)
        self.write_file(filename[:filename.rfind(".gz")], f.read())
        f.close()
        remove(filename)

    def write_file(self, filename, data):
        try:
            f = open(filename, "wb")
        except IOError as e:
            print(e.errno, e.message)
        else:
            f.write(data)
            f.close()


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
