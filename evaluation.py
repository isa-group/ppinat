import argparse
from ppinat.input import input_test
import json
import pandas as pd


def compute_precision_recall(overall, identified, goldstandard, allow_regular = False):
    found = overall['good'] + overall['regular'] if allow_regular else overall['good']

    return {
        "precision": found / identified if identified > 0 else None,
        "recall": found / goldstandard if goldstandard > 0 else None
    }    

parser = argparse.ArgumentParser()
parser.add_argument('filename', metavar='FILENAME', help='the file with the test', nargs='?', default='input/metrics_dataset-traffic-test.json' )
parser.add_argument("-v", "--verbosity" ,help="If this option is activated, you will be able to view the complete information in a CSV file called input_test_info.csv", action="store_true")
parser.add_argument("-m", "--model", help="If this option is activated, you will be able to choose the model to evaluate. 1: General token classification, 2: Specific token classification", type=int, default=2)

one_slot_weights = {
     "emb_is": {
        'slot_is_sim': 0.25,
        'slot_complete_is_sim': 0.2,
        'slot_emb': 0.25,
        'slot_complete_emb': 0.2,
        'att_is_sim': 0.05,
        'att_complete_is_sim': 0.05
    },
    "only_bart": {
        'bart_large_mnli_personalized_complete':0.9,
        'att_is_sim': 0.05,
        'att_complete_is_sim': 0.05
    },
    "emb_bart": {
        'slot_is_sim': 0.25,
        'slot_complete_is_sim': 0.2,
        'bart_large_mnli_personalized_complete':0.45,
        'att_is_sim': 0.05,
        'att_complete_is_sim': 0.05
    }
    # "only_emb": {
    #     'slot_emb': 0.5,
    #     'slot_complete_emb': 0.4,
    #     'att_is_sim': 0.05,
    #     'att_complete_is_sim': 0.05
    # },
    # "only_is": {
    #     'slot_is_sim': 0.5,
    #     'slot_complete_is_sim': 0.4,
    #     'att_is_sim': 0.05,
    #     'att_complete_is_sim': 0.05
    # },
    # "only_sim": {
    #     'slot_sim': 0.5,
    #     'slot_complete_sim': 0.4,
    #     'att_is_sim': 0.05,
    #     'att_complete_is_sim': 0.05
    # },
    # "emb_sim": {
    #     'slot_sim': 0.25,
    #     'slot_complete_sim': 0.2,
    #     'slot_emb': 0.25,
    #     'slot_complete_emb': 0.2,
    #     'att_is_sim': 0.05,
    #     'att_complete_is_sim': 0.05
    # },
    # "sim_is": {
    #     'slot_sim': 0.25,
    #     'slot_complete_sim': 0.2,
    #     'slot_is_sim': 0.25,
    #     'slot_complete_is_sim': 0.2,
    #     'att_is_sim': 0.05,
    #     'att_complete_is_sim': 0.05
    # },
    # "no_complete": {
    #     'slot_is_sim': 0.45,
    #     'slot_emb': 0.45,
    #     'att_is_sim': 0.1,
    # },
    # "without_atts": {
    #     'slot_is_sim': 0.25,
    #     'slot_complete_is_sim': 0.25,
    #     'slot_emb': 0.25,
    #     'slot_complete_emb': 0.25
    # },
    # "without_same": {
    #     'slot_is_sim': 0.25,
    #     'slot_complete_is_sim': 0.2,
    #     'slot_emb': 0.25,
    #     'slot_complete_emb': 0.2,
    #     'att_is_sim': 0.05,
    #     'att_complete_is_sim': 0.05
    # },
    # "without_condition": {
    #     'slot_is_sim': 0.25,
    #     'slot_complete_is_sim': 0.2,
    #     'slot_emb': 0.25,
    #     'slot_complete_emb': 0.2,
    #     'att_is_sim': 0.05,
    #     'att_complete_is_sim': 0.05
    # }

}

multi_slot_weights = {
    "emb_is" : {
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
    },
    "only_bart": {
        'ev1_$bart_large_mnli_personalized_complete': 0.25,
        'ev2_$bart_large_mnli_personalized_complete': 0.25,
        'same_type': 0.25,
        'condition_ratio': 0.25
    },
    "emb_bart" : {
        'ev1_$slot_is_sim': 0.05,
        "ev1_$slot_complete_is_sim": 0.05,
        'ev1_$bart_large_mnli_personalized_complete': 0.13,
        'ev2_$slot_is_sim': 0.05,
        "ev2_$slot_complete_is_sim": 0.05,
        'ev2_$bart_large_mnli_personalized_complete': 0.13,
        "ev1_$att_is_sim": 0.01,
        "ev1_$att_complete_is_sim": 0.01,
        "ev2_$att_is_sim": 0.01,
        "ev2_$att_complete_is_sim": 0.01,
        "same_type": 0.25,
        "condition_ratio": 0.25
    },
    "only_emb": {
        'ev1_$slot_emb': 0.12,
        'ev1_$slot_complete_emb': 0.11,
        'ev2_$slot_emb': 0.12,
        'ev2_$slot_complete_emb': 0.11,
        "ev1_$att_is_sim": 0.01,
        "ev1_$att_complete_is_sim": 0.01,
        "ev2_$att_is_sim": 0.01,
        "ev2_$att_complete_is_sim": 0.01,
        "same_type": 0.25,
        "condition_ratio": 0.25
    },
    "only_is": {
        'ev1_$slot_is_sim': 0.12,
        "ev1_$slot_complete_is_sim": 0.11,
        'ev2_$slot_is_sim': 0.12,
        "ev2_$slot_complete_is_sim": 0.11,
        "ev1_$att_is_sim": 0.01,
        "ev1_$att_complete_is_sim": 0.01,
        "ev2_$att_is_sim": 0.01,
        "ev2_$att_complete_is_sim": 0.01,
        "same_type": 0.25,
        "condition_ratio": 0.25
    },
    "only_sim": {
        'ev1_$slot_sim': 0.12,
        "ev1_$slot_complete_sim": 0.11,
        'ev2_$slot_sim': 0.12,
        "ev2_$slot_complete_sim": 0.11,
        "ev1_$att_is_sim": 0.01,
        "ev1_$att_complete_is_sim": 0.01,
        "ev2_$att_is_sim": 0.01,
        "ev2_$att_complete_is_sim": 0.01,
        "same_type": 0.25,
        "condition_ratio": 0.25
    },
    "emb_sim" : {
        'ev1_$slot_sim': 0.05,
        "ev1_$slot_complete_sim": 0.05,
        'ev1_$slot_emb': 0.07,
        'ev1_$slot_complete_emb': 0.06,
        'ev2_$slot_sim': 0.05,
        "ev2_$slot_complete_sim": 0.05,
        'ev2_$slot_emb': 0.07,
        'ev2_$slot_complete_emb': 0.06,
        "ev1_$att_is_sim": 0.01,
        "ev1_$att_complete_is_sim": 0.01,
        "ev2_$att_is_sim": 0.01,
        "ev2_$att_complete_is_sim": 0.01,
        "same_type": 0.25,
        "condition_ratio": 0.25
    },
    "sim_is" : {
        'ev1_$slot_sim': 0.05,
        "ev1_$slot_complete_sim": 0.05,
        'ev1_$slot_is_sim': 0.07,
        'ev1_$slot_complete_is_sim': 0.06,
        'ev2_$slot_sim': 0.05,
        "ev2_$slot_complete_sim": 0.05,
        'ev2_$slot_is_sim': 0.07,
        'ev2_$slot_complete_is_sim': 0.06,
        "ev1_$att_is_sim": 0.01,
        "ev1_$att_complete_is_sim": 0.01,
        "ev2_$att_is_sim": 0.01,
        "ev2_$att_complete_is_sim": 0.01,
        "same_type": 0.25,
        "condition_ratio": 0.25
    },

    "no_complete" : {
        'ev1_$slot_is_sim': 0.1,
        'ev1_$slot_emb': 0.13,
        'ev2_$slot_is_sim': 0.1,
        'ev2_$slot_emb': 0.13,
        "ev1_$att_is_sim": 0.02,
        "ev2_$att_is_sim": 0.02,
        "same_type": 0.25,
        "condition_ratio": 0.25
    },
    "without_atts": {
        'ev1_$slot_is_sim': 0.06,
        "ev1_$slot_complete_is_sim": 0.06,
        'ev1_$slot_emb': 0.07,
        'ev1_$slot_complete_emb': 0.06,
        'ev2_$slot_is_sim': 0.06,
        "ev2_$slot_complete_is_sim": 0.06,
        'ev2_$slot_emb': 0.07,
        'ev2_$slot_complete_emb': 0.06,
        "same_type": 0.25,
        "condition_ratio": 0.25
    },
    "without_same" : {
        'ev1_$slot_is_sim': 0.09,
        "ev1_$slot_complete_is_sim": 0.08,
        'ev1_$slot_emb': 0.10,
        'ev1_$slot_complete_emb': 0.09,
        'ev2_$slot_is_sim': 0.08,
        "ev2_$slot_complete_is_sim": 0.08,
        'ev2_$slot_emb': 0.1,
        'ev2_$slot_complete_emb': 0.09,
        "ev1_$att_is_sim": 0.01,
        "ev1_$att_complete_is_sim": 0.01,
        "ev2_$att_is_sim": 0.01,
        "ev2_$att_complete_is_sim": 0.01,
        "condition_ratio": 0.25
    },
    "without_condition" : {
        'ev1_$slot_is_sim': 0.09,
        "ev1_$slot_complete_is_sim": 0.08,
        'ev1_$slot_emb': 0.10,
        'ev1_$slot_complete_emb': 0.09,
        'ev2_$slot_is_sim': 0.08,
        "ev2_$slot_complete_is_sim": 0.08,
        'ev2_$slot_emb': 0.1,
        'ev2_$slot_complete_emb': 0.09,
        "ev1_$att_is_sim": 0.01,
        "ev1_$att_complete_is_sim": 0.01,
        "ev2_$att_is_sim": 0.01,
        "ev2_$att_complete_is_sim": 0.01,
        "same_type": 0.25,
    }
}
# filename = "metrics_dataset-domesticDeclarations"
#filename = "metrics_dataset-traffic-test"
# filename = "metrics_dataset"
#args = parser.parse_known_args([f"input/{filename}.json"])

args = parser.parse_args()

results = []
ppi_results = []

parsing_metrics_results = []
parsing_tags_results = []
matching_metrics_results = []
matching_attrib_results = []


def create_row(v, attrib, evaluation):
    return {
        "type": v,
        "attrib": attrib,
        **evaluation.value,
        "precision": evaluation.precision(),
        "recall": evaluation.recall(),
        "precision_partial": evaluation.precision(strict=False),
        "recall_partial": evaluation.recall(strict=False)
    }

def update_results(results, v, evaluations):
    tagging_aggregation = input_test.aggregate_eval_results(evaluations)
    results.append(create_row(v, "GLOBAL", tagging_aggregation))
    for t in evaluations:
        results.append(create_row(v, t, evaluations[t]))

for v in one_slot_weights:
    other = {
        "weights": {
            "one_slot": one_slot_weights[v],
            "multi_slot": multi_slot_weights[v]
        }
    }

    print("---------- "+ v + " -------------------")
    test = input_test.InputTest(args=args, other=other)

    parsing_metrics_results.append({
        "type": v,
        **test.tagging_overall
    })
    update_results(parsing_tags_results, v, test.tagging_tag)

    matching_metrics_results.append({
        "type": v,
        **test.matching_overall
    })
    update_results(matching_attrib_results, v, test.matching_attribute)

#print(json.dumps(results, indent=4))

df = pd.DataFrame(parsing_tags_results)
df.to_csv(f"parsing-tags-results.csv")
pm = pd.DataFrame(parsing_metrics_results)
pm.to_csv(f"parsing-metrics-results.csv")
mm = pd.DataFrame(matching_metrics_results)
mm.to_csv(f"matching-metrics-results.csv")
df = pd.DataFrame(matching_attrib_results)
df.to_csv(f"matching-attrib-results.csv")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print("OVERALL PARSING RESULTS:")
    print(pm)
    print("OVERALL MATCHING RESULTS")
    print(mm)