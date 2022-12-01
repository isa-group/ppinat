import argparse
import os
import time
from ppinat.input import input_test
import json
import pandas as pd
import re
import logging
import utils

logging.basicConfig(format='%(asctime)s %(levelname)s:%(module)s:%(funcName)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

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
    matching = utils.create_search_grid(matching["hs"])
    logger.info(f"Grid size: {len(matching)}")
else:
    for key in matching:
        if "$gen" in matching[key]:
            matching[key] = utils.generate_weights(**matching[key]["$gen"])

disable_heuristics = "disable_heuristics" in config and config["disable_heuristics"]
logger.info(f"Heuristics disabled: {disable_heuristics}")

for dataset in datasets:
    parsing_metrics_results = []
    parsing_tags_results = []
    matching_metrics_results = []
    matching_attrib_results = []

    for parsing_model in parsing:

        print(f"---------- {parsing_model}  ----------")

        try:
            test_execution = input_test.TestExecution(
                args=args, dataset=dataset, parsing_model=parsing_model, matching_models = matching, disable_heuristics=disable_heuristics)

                
            for matching_model in test_execution.result:
                test = test_execution.result[matching_model]

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
        except Exception as e:
            logger.exception(f"Failed test execution of {parsing_model} in {dataset}", exc_info=e)

    dataset_name = re.findall("./input/(.*)", dataset)[0]

    if not os.path.exists("results"):
        os.mkdir("results")

    timestr = time.strftime("%Y%m%d-%H%M%S")

    df = pd.DataFrame(parsing_tags_results)
    df.to_csv(f"results/{timestr}-{dataset_name}-parsing-tags-results.csv")
    pm = pd.DataFrame(parsing_metrics_results)
    pm.to_csv(f"results/{timestr}-{dataset_name}-parsing-metrics-results.csv")
    mm = pd.DataFrame(matching_metrics_results)
    mm.to_csv(f"results/{timestr}-{dataset_name}-matching-metrics-results.csv")
    df = pd.DataFrame(matching_attrib_results)
    df.to_csv(f"results/{timestr}-{dataset_name}-matching-attrib-results.csv")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print("OVERALL PARSING RESULTS:")
        print(pm)
        print("OVERALL MATCHING RESULTS")
        print(mm)
