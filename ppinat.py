import argparse
import json
import logging
import sys

from colorama import Fore

from ppinat.computer import PPINat
from ppinat.ppiparser.ppiannotation import PPIAnnotation, text_by_tag

logger = logging.getLogger(__name__)

def print_values(values):
    for k, v in values.items():
        if isinstance(v, list):
            if len(v) > 0:
                print(f"\n{k}:")
                for i in v:
                    print(i.value)
        else:
            print(f"\n{k}:")
            print(v.value)

def process_ppi(user_input, ppinat, args):
    print(f"\n{user_input}")
    
    if args.verbose:        
        annotation: PPIAnnotation = ppinat.parse(user_input)

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

        print(f"\n{Fore.LIGHTBLUE_EX}Parsing result:{Fore.RESET}")
        if from_text: print(f"From text: {from_text}")
        if to_text: print(f"To text: {to_text}")
        if only_text: print(f"Only text: {only_text}")
        if when_text: print(f"When text: {when_text}")
        if data_attribute: print(f"Data attribute: {data_attribute}")
        if group_text: print(f"Group by: {group_text}")
        if denominator_text: print(f"Denominator: {denominator_text}")
        if aggregation_text: print(f"Aggregation: {aggregation_text}")
        if conditional_text: print(f"Conditional: {conditional_text}")
        if conditional_attr_text: print(f"Conditional attribute: {conditional_attr_text}")

        print(f"\n{Fore.LIGHTBLUE_EX}Matching result:{Fore.RESET}")
        agg_command, base_measure = ppinat._resolve_partials(user_input)

        print(f"{Fore.CYAN}Values:{Fore.RESET}")
        print_values(agg_command.values)
        print_values(base_measure.values)

        print(f"\n{Fore.CYAN}A alternatives:{Fore.RESET}")
        print_values(agg_command.alt_match_a)
        print_values(base_measure.alt_match_a)

        print(f"\n{Fore.CYAN}B alternatives:{Fore.RESET}")
        print_values(agg_command.alt_match_b)
        print_values(base_measure.alt_match_b)

    
    print(f"\n{Fore.LIGHTBLUE_EX}Result:{Fore.RESET}")
    metric = ppinat.resolve(user_input)
    print(metric)
    print(ppinat.compute(metric, time_grouper=args.time))    

def load_config(args):
    config = {}
    with open(args.config, "r") as c:
        config = json.load(c)

    parsing = config["parsing"][0]
    matching = config["matching"]
    log_file = args.log

    matching_weights = None
    raw_weights = None

    if "hs" in matching:
        print(Fore.RED + "Cannot do a grid search. Use evaluation.py for that" + Fore.RESET)
        sys.exit()
    else:
        matching_key = list(matching.keys())[0]
        if len(matching) > 1:
            logger.warning("Only the first config value will be used")

        if "$gen" in matching[matching_key]:
            matching_weights = matching[matching_key]["$gen"]
        else:
            raw_weights = matching[matching_key]

    disable_heuristics = "disable_heuristics" in config and config["disable_heuristics"]
    logger.info(f"Heuristics disabled: {disable_heuristics}")

    ppinat = PPINat()
    ppinat.load_context(log_file, parsing, matching_weights=matching_weights, raw_weights=raw_weights, disable_heuristics=disable_heuristics)

    return ppinat


parser = argparse.ArgumentParser()
parser.add_argument('PPI', nargs='?', help='The ppi that is being computed', default=None)
parser.add_argument('-f', '--file', action='store', help='File with a list of PPIs to compute', default=None)
parser.add_argument('-l', '--log', action='store', help='Indicates the log you want to use', default='./input/event_logs/DomesticDeclarations.xes' )
parser.add_argument('-c', '--config', action='store', help='The file with the config', default='./config.json' )
parser.add_argument('-t', '--time', action='store', help='Time grouper used to compute the ppi (e.g. 1M, 6M, 1Y...)', default=None)
parser.add_argument('-v', '--verbose', action='store_true', help='Prints the results of the parsing and matching')

args = parser.parse_args()

if args.verbose:
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(module)s:%(funcName)s:%(message)s', level=logging.INFO)

ppinat = load_config(args)

if args.PPI is not None:
    process_ppi(args.PPI, ppinat=ppinat, args=args)
elif args.file is not None:
    lines = []
    with open(args.file, "r") as ppis:
        lines = ppis.readlines()

    for line in lines:
        process_ppi(line, ppinat, args)    

else:
    while True:
        user_input = input(f"{Fore.BLUE}\nPlease, write the performance indicator that you want to compute:\n{Fore.RESET}")
        process_ppi(user_input, ppinat, args)


    
