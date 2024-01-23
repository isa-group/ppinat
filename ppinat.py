import argparse
import json
import logging
import ppinat.matcher.recognizers as r
import ppinat.input.input_test as input_test
from ppinat.ppiparser.ppiannotation import PPIAnnotation, text_by_tag
from colorama import Fore
import ppinat.bot.commands as commands
import utils
import ppinot4py
from ppinot4py.computers.metrics_computer import LogConfiguration

logging.basicConfig(format='%(asctime)s %(levelname)s:%(module)s:%(funcName)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('log', metavar='LOG', help='Indicates the path of log you want to use', nargs='?', default='./input/event_logs/DomesticDeclarations.xes' )
parser.add_argument('filename', metavar='FILENAME', help='The file with the config', nargs='?', default='./config.json' )
parser.add_argument('-v', '--verbose', action='store_true', help='Prints the results of the parsing and matching')

args = parser.parse_args()

config = {}
with open(args.filename, "r") as c:
    config = json.load(c)

parsing = config["parsing"][0]
matching = config["matching"]
log_file = args.log

if "hs" in matching:
    matching = utils.create_search_grid(matching["hs"])
    logger.info(f"Grid size: {len(matching)}")
else:
    matching_key = list(matching.keys())[0]
    if "$gen" in matching[matching_key]:
        matching[matching_key] = utils.generate_weights(**matching[matching_key]["$gen"])
        matching = matching[matching_key]

disable_heuristics = "disable_heuristics" in config and config["disable_heuristics"]
logger.info(f"Heuristics disabled: {disable_heuristics}")

similarity = input_test.load_similarity(log_file, None, parsing, matching)

while True:
    user_input = input(f"{Fore.BLUE}\nPlease, write the performance indicator that you want to compute:\n{Fore.RESET}")

    recognized_entity = r.RecognizedEntities(None, user_input)
    annotation: PPIAnnotation = similarity.metric_decoder(recognized_entity.text)
    
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

    agg_command = commands.ComputeMetricCommand()
    try: 
        agg_command.match_entities(recognized_entity, similarity, heuristics=not disable_heuristics)
    except Exception as e:
        print(Fore.RED + "An error ocurred while matching entities" + Fore.RESET)
        logger.exception("Error while matching entities", exc_info=e, stack_info=True)
    
    print(f"\n{user_input}")

    base_measure = agg_command.partials["base_measure"][0]
    
    if args.verbose:
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

        print(f"{Fore.CYAN}Values:{Fore.RESET}")
        utils.print_values(agg_command.values)
        utils.print_values(base_measure.values)

        print(f"\n{Fore.CYAN}A alternatives:{Fore.RESET}")
        utils.print_values(agg_command.alt_match_a)
        utils.print_values(base_measure.alt_match_a)

        print(f"\n{Fore.CYAN}B alternatives:{Fore.RESET}")
        utils.print_values(agg_command.alt_match_b)
        utils.print_values(base_measure.alt_match_b)

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
    
    print(f"\n{Fore.LIGHTBLUE_EX}Result:{Fore.RESET}")
    print(result.metric)

    print(ppinot4py.measure_computer(result.metric, similarity.df, LogConfiguration(id_case=similarity.id_case, time_column=similarity.time_column, activity_column=similarity.activity)))
