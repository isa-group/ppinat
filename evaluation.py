import argparse
from ppinat.input import input_test
import json
import pandas as pd
import re
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s:%(module)s:%(funcName)s:%(message)s', level=logging.INFO)

def compute_precision_recall(overall, identified, goldstandard, allow_regular=False):
    found = overall['good'] + \
        overall['regular'] if allow_regular else overall['good']

    return {
        "precision": found / identified if identified > 0 else None,
        "recall": found / goldstandard if goldstandard > 0 else None
    }


parser = argparse.ArgumentParser()
parser.add_argument('filename', metavar='FILENAME', help='the file with the test config', nargs='?', default='./config.json' )
parser.add_argument("-v", "--verbosity",
                    help="If this option is activated, you will be able to view the complete information in a CSV file called input_test_info.csv", action="store_true")


# filename = "metrics_dataset-domesticDeclarations"
#filename = "metrics_dataset-traffic-test"
# filename = "metrics_dataset"
#args = parser.parse_known_args([f"input/{filename}.json"])

args = parser.parse_args()


def create_row(attrib, evaluation, **kwargs):
    return {
        **kwargs,
        "attrib": attrib,
        **evaluation.value,
        "precision": evaluation.precision(),
        "recall": evaluation.recall(),
        "precision_partial": evaluation.precision(strict=False),
        "recall_partial": evaluation.recall(strict=False)
    }

def update_results(results, evaluations, **kwargs):
    tagging_aggregation = input_test.aggregate_eval_results(evaluations)
    results.append(create_row("GLOBAL", tagging_aggregation, **kwargs))
    for t in evaluations:
        results.append(create_row(t, evaluations[t], **kwargs))

config = {}
with open(args.filename, "r") as c:
    config = json.load(c)

datasets = config["datasets"]
parsing = config["parsing"]
matching = config["matching"]

if "hs" in matching:
    processed_matching = {}
    model_comb = [(x, y, z) for x in matching["hs"]["is"] for y in matching["hs"]["emb"] for z in matching["hs"]["bart"] if x + y + z == 1]
    for (x, y, z) in model_comb:
        for complete in matching["hs"]["complete"]:
            for att in matching["hs"]["att"]:
                one_slot = {
                    "slot_is_sim": x * (1-att) * (1 - complete),
                    "slot_complete_is_sim": x * (1-att) * complete,
                    "slot_emb": y * (1-att) * (1 - complete),
                    "slot_complete_emb": y * (1-att) * (complete),
                    "bart_large_mnli_personalized_complete": z * (1-att),
                    "att_is_sim": att * (1 - complete),
                    "att_complete_is_sim": att * (complete)
                }

                for multi_heur in matching["hs"]["multi_heur"]:
                    multi_slot = {
                        **({f"ev1_${k}": v/2*(1-multi_heur) for k,v in one_slot.items()}),
                        **({f"ev2_${k}": v/2*(1-multi_heur) for k,v in one_slot.items()}),
                        "same_type": multi_heur / 2,
                        "condition_ratio": multi_heur / 2
                    }
                    name = f"is_{x}__emb_{y}__bart_{z}__c_{complete}__att_{att}__mh_{multi_heur}"

                    processed_matching[name] = {
                        "one_slot": one_slot,
                        "multi_slot": multi_slot
                    }

    matching = processed_matching

for dataset in datasets:
    parsing_metrics_results = []
    parsing_tags_results = []
    matching_metrics_results = []
    matching_attrib_results = []

    for parsing_model in parsing:

        print(f"---------- {parsing_model}  ----------")
        test_execution = input_test.TestExecution(
            args=args, dataset=dataset, parsing_model=parsing_model, matching_models = matching)

            
        for matching_model in test_execution.result:
            test = test_execution[matching_model]

            parsing_metrics_results.append({
                "parsing_type": parsing_model,
                "matching_type": matching_model,
                **test.tagging_overall
            })
            update_results(parsing_tags_results, test.tagging_tag, parsing_type=parsing_model, matching_type=matching_model)

            matching_metrics_results.append({
                "parsing_type": parsing_model,
                "matching_type": matching_model,
                **test.matching_overall
            })
            update_results(matching_attrib_results, test.matching_attribute, parsing_type=parsing_model, matching_type=matching_model)

    dataset_name = re.findall("./input/(.*)", dataset)[0]

    df = pd.DataFrame(parsing_tags_results)
    df.to_csv(f"{dataset_name}-parsing-tags-results.csv")
    pm = pd.DataFrame(parsing_metrics_results)
    pm.to_csv(f"{dataset_name}-parsing-metrics-results.csv")
    mm = pd.DataFrame(matching_metrics_results)
    mm.to_csv(f"{dataset_name}-matching-metrics-results.csv")
    df = pd.DataFrame(matching_attrib_results)
    df.to_csv(f"{dataset_name}-matching-attrib-results.csv")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print("OVERALL PARSING RESULTS:")
        print(pm)
        print("OVERALL MATCHING RESULTS")
        print(mm)
