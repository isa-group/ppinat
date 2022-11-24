import datetime
import logging

import numpy as np
import pandas as pd
import ppinat.bot.base as b
import ppinat.helpers.log as log
import ppinot4py
from ppinat.matcher.similarity import SimilarityComputer
from ppinot4py.model import AppliesTo, RuntimeState, TimeInstantCondition

logger = logging.getLogger(__name__)


class Metric(b.RenderType):

    def describe(self):
        return str(self.metric)

    def _format_compute(self, result) -> str:
        def format_index(index):
            return index.strftime('%B %d, %Y, %r') if isinstance(index, datetime.datetime) else str(index)

        if isinstance(result, pd.Series):
            index_names = " | ".join(result.index.names)
            separator = "|".join(["----------" for i in result.index.names])
            logger.debug(f"Index names: {result.index.names}")

            table = f"## Result\r\n| {index_names} | Value | \n|{separator}|-----------|"

            for index, row in result.items():
                logger.debug(f"{type(index)} - {index}")
                if isinstance(index, tuple):
                    index_text = " | ".join([format_index(i) for i in index])
                else:
                    index_text = format_index(index)
                logger.debug(f"{index}, {row}")
                table = f"{table}\n|{index_text}|{row}|"

            result_text = table
        else:
            result_text = f"{result}"

        return result_text


class AggMetric(Metric):
    def __init__(self, metric, period=None, text=None, table=None):
        self.metric = metric
        self.period = period
        self.text = text
        self.table = table

    @staticmethod
    def match(text, similarity):
        return None

    def render(self, tools: b.Tools):
        result = tools.measure_computer.compute(self.metric, self.period)
        text = self._format_compute(result)
        if isinstance(result, pd.Series):
            result = result.to_frame()
        if isinstance(result, pd.DataFrame):
            table = result.to_dict('split')
            multiindex = result.index.nlevels > 1
            if multiindex:
                table['index'] = [list(i) for i in table['index']]
            table['index_names'] = list(result.index.names)
            table['types'] = {
                'index': result.index.dtype if not multiindex else [l.dtype for l in result.index.levels],
                'columns': list(result.dtypes)
            }
            table['type'] = 'AggMetric'
            self.text = text
            self.table = {"table": table}
            return self.text, self.table
        else:
            self.text = text
            return self.text


def is_datetime_like(data):
    return pd.api.types.is_datetime64_dtype(data.dtype) or pd.api.types.is_datetime64tz_dtype(data.dtype) or pd.api.types.is_timedelta64_dtype(data.dtype) or pd.api.types.is_period_dtype(data.dtype)


class BaseMetric(Metric):
    def __init__(self, metric):
        self.metric = metric
        self.HEAD = 1000

    @staticmethod
    def match(text, similarity):
        return None

    def domain_of_metric(self, similarity: SimilarityComputer, categorical_threshold=300):
        if isinstance(self.metric, ppinot4py.model.CountMeasure):
            domain = Domain('numeric')
        elif isinstance(self.metric, ppinot4py.model.TimeMeasure):
            domain = Domain('timedelta')
        elif isinstance(self.metric, ppinot4py.model.DataMeasure):
            attribute_name = self.metric.data_content_selection
            domain = Domain.domain_of_attribute(
                attribute_name, similarity, categorical_threshold)
        else:
            domain = Domain('string')

        return domain

    def render(self, tools: b.Tools):
        result = tools.measure_computer.compute(self.metric)
        text = self._format_compute(result.head(self.HEAD))

        if isinstance(result, pd.Series):
            result = result.to_frame()

        if isinstance(result, pd.DataFrame):
            table = result.head(self.HEAD).to_dict('split')
            multiindex = result.index.nlevels > 1
            if multiindex:
                table['index'] = [list(i) for i in table['index']]
            table['index_names'] = list(result.index.names)
            table['types'] = {
                'index': result.index.dtype if not multiindex else [l.dtype for l in result.index.levels],
                'columns': list(result.dtypes)
            }
            table['type'] = 'BaseMetric'

            column_names = list(result)
            column_data = column_names[len(column_names) - 1]
            data = result[column_data][~result[column_data].isna()]

            if is_datetime_like(data):
                data = data.dt.days.to_list()
            else:
                data = data.to_list()

            hist, bin_edges = np.histogram(data)

            histogram = {}
            bin_edges = [round(i, 4) for i in bin_edges.tolist()[1:]]
            for interval in bin_edges:
                histogram[interval] = hist[bin_edges.index(interval)]

            table['histogram'] = histogram

            return text, {"table": table}
        else:
            return text


class MetricComparison(b.RenderType):
    def __init__(self, text1=None, table1=None, name1=None, text2=None, table2=None, name2=None):
        self.text1 = text1
        self.table1 = table1
        self.name1 = name1
        self.text2 = text2
        self.table2 = table2
        self.name2 = name2

    def render(self, tools: b.Tools):
        if self.table1 is not None and self.table2 is not None:
            table = {"table": {}}

            table["table"]["columns"] = []
            for c in self.table1["table"]["columns"]:
                table["table"]["columns"].append(c+"_"+self.name1)
                table["table"][c+"_"+self.name1] = []
                for c_value in self.table1["table"][c]:
                    table["table"][c+"_"+self.name1].append(c_value)

            for c in self.table2["table"]["columns"]:
                table["table"]["columns"].append(c+"_"+self.name2)
                table["table"][c+"_"+self.name2] = []
                for c_value in self.table2["table"][c]:
                    table["table"][c+"_"+self.name2].append(c_value)

            table["table"]["index"] = self.table1["table"]["index"]
            table["table"]["index_names"] = self.table1["table"]["index_names"]
            table["table"]["types"] = self.table1["table"]["types"]

            table['table']['type'] = 'MetricComparison'

            return self.text1+"\n"+self.text2, table
        else:
            return self.name1 + ": " + self.text1 + " | " + self.name2 + ": " + self.text2


class LogicCondition(b.PPIBotType):
    description = "boolean condition"
    entity_name = "Condition"
    secondary_elements = [{"name": "operand", "usesFeature": {"name": "boolean_ops", "type": "phraselist",
                                                              "definition": "boolean_ops(interchangeable) disabledForAllModels",
                                                              "values": ["equals to", "different than", "=", "not equals to", "greater than", "lower than", "is", "bigger than", "shorter than", "larger than", "smaller than", ">=", "<=", "<", ">", "greater or equal than", "lower or equal than"]}},
                          {"name": 'value', "usesFeature": [{"name": "AttributeValueList", "type": "list"}, {"name": "AttributeValue"}]}]

    def __init__(self, value, op='equal'):
        self.value = value
        self.op = op

    def to_condition(self, with_map=False):
        op_dict = {
            'gt': '>',
            'lt': '<',
            'equal': '==',
            'not_equal': '!='
        }

        comparison = op_dict[self.op] if self.op in op_dict else '=='

        domain = self.value.domain
        map = {}

        if pd.api.types.is_list_like(domain) or domain == 'string':
            repr = f"'{self.value.value}'"
        elif domain == 'numeric':
            repr = f"{self.value.value}"
        elif domain == 'datetime' or domain == 'timedelta':
            if not with_map:
                repr = f"{self.value.value}"
            else:
                repr = f"c_{domain}"
                map[repr] = self.value.value
        else:
            logger.warning(f"Literal - match error: {domain} is not supported")
            repr = f"{self.value.value}"

        text = f" {comparison} {repr}"
        if not with_map:
            return text
        else:
            return text, map

    @staticmethod
    def match_condition(entity, similarity, domain=None):
        if entity is None:
            return None

        op = None
        value = None

        # Ideally, this should be improved with some semantic matching
        if "operand" in entity:
            operand = entity["operand"]
            if isinstance(operand, list):
                operand = operand[0]
            operand = operand.lower()
            if operand in ["equals to", "=", "is"]:
                op = "equal"
            elif operand in ["different than", "not equals to", "!="]:
                op = "not_equal"
            elif operand in ["greater than", "bigger than", "larger than", ">=",  ">", "greater or equal than", "at least"]:
                op = "gt"
            elif operand in ["lower than", "less than", "shorter than", "smaller than", "<=", "<", "lower or equal than", "at most"]:
                op = "lt"
            else:
                possible_op = similarity.find_op(operand)
                possible_op_sorted = [k for k, v in sorted(
                    possible_op.items(), key=lambda item: item[1], reverse=True)]
                op = possible_op_sorted[0]


        if "value" in entity:
            value = entity["value"]
            if isinstance(value, list):
                value = value[0]
            literal = pick_first_thresholds(
                Literal.match(value, similarity, domain))

        if literal is not None and op is not None:
            return LogicCondition(literal, op=op)
        else:
            return None

    @staticmethod
    def match(text, similarity, **kwargs):
        # We rely on commands to provide this type
        return None


class LogRender(b.RenderType):
    def __init__(self, detail=None):
        self.detail = detail

    def render(self, tools: b.Tools) -> str:
        if self.detail is None:
            return self._render_attribute(tools.similarity.df)
        else:
            return self._render_values(tools.similarity.df)

    def _render_attribute(self, df):
        columns = df.columns.values
        return log.attribute_options(columns, df)

    def _render_values(self, df):
        values = df[self.detail].unique()
        values_text = ", ".join(map(str, values))
        return values_text


class LogAttribute(b.PPIBotType):
    description = "name of an attribute of the log"
    entity_name = "AttributeName"
    usesFeature = {"name": "AttributeList", "type": "list"}

    def __init__(self, value):
        self.value = value

    @staticmethod
    def match(attribute_name, similarity: SimilarityComputer, filter='none'):
        if attribute_name is None:
            return None
        possible_attributes: dict = similarity.find_most_similar_attribute(
            attribute_name, filter, 0.7)
        possible_attributes_sorted = {k: v for k, v in sorted(
            possible_attributes.items(), key=lambda item: item[1], reverse=True)}

        return get_tuple_by_threshold(LogAttribute.threshold_a, LogAttribute.threshold_b, possible_attributes_sorted, lambda attribute_name: LogAttribute(attribute_name))

    @staticmethod
    def parse(text):
        return LogAttribute(text)

    def domain_of_attribute(self, similarity: SimilarityComputer, categorical_threshold=300):
        return Domain.domain_of_attribute(self.value, similarity, categorical_threshold)

def resolve_tuple_with_first(tuple):
    if tuple is None:
        return None
        
    alt_a, alt_b = tuple
    if len(tuple[0]) > 0:
        return tuple[0][0]
    elif len(tuple[1]) > 0:
        return tuple[1][0]
    else:
        return None


def get_tuple_by_threshold(threshold_a, threshold_b, possible_attributes, parser):
    soft_max_temp = 0.25
    top_k = 10
    alt_match_a = []
    alt_match_b = []

    if len(possible_attributes) > 0:
        try:
            values = np.array([pv.cpu() for pv in possible_attributes.values()][:top_k]) / soft_max_temp
        except:
            values = np.array(list(possible_attributes.values())
                                [:top_k]) / soft_max_temp
        softmax = np.exp(values) / np.sum(np.exp(values))
        rel_a = np.max(softmax) * (1 - threshold_a)
        rel_b = np.max(softmax) * (1 - threshold_b)

        for k, v in zip(possible_attributes, softmax):
            if v > rel_a:
                alt_match_a.append(parser(k))
            if v > rel_b and v < rel_a:
                alt_match_b.append(parser(k))

    return alt_match_a, alt_match_b


def pick_first_thresholds(alt_match):
    if alt_match is None or not isinstance(alt_match, tuple):
        return alt_match

    if len(alt_match[0]) > 0:
        return alt_match[0][0]
    elif len(alt_match[1]) > 0:
        return alt_match[1][0]
    else:
        return None


class Period(b.PPIBotType):
    description = "time frequency (e.g. monthly, or weekly)"
    entity_name = "Period"
    secondary_elements = [{"name": "Value", "usesFeature": {"name": "number", "type": "prebuilt"}},
                          {"name": '"Time unit"', "usesFeature": {"name": "periods", "type": "phraselist",
                                                                  "definition": "periods(interchangeable) disabledForAllModels",
                                                                  "values": ["weekly", "yearly", "quarterly", "daily", "hourly", "monthly", "annually", "month", "year",
                                                                             "week", "day", "hour", "quarter", "semester"]}}]

    def __init__(self, value):
        self.value = value

    @staticmethod
    def match(period_text, similarity: SimilarityComputer):
        if period_text is None:
            return None

        period = {
            "time_unit": period_text,
            "value": 1
        }

        return Period.match_period(period, similarity)

    @staticmethod
    def match_period(period, similarity: SimilarityComputer):
        if period is None:
            return None

        possible_periods: dict = similarity.find_period(
            period["time_unit"], 0.7)
        possible_periods_sorted = {k: v for k, v in sorted(
            possible_periods.items(), key=lambda item: item[1], reverse=True)}
        try:
            value = int(period["value"])
        except:
            value = 1

        return get_tuple_by_threshold(LogAttribute.threshold_a, LogAttribute.threshold_b, possible_periods_sorted, lambda period_name: Period(f"{value}{period_name}"))

    @staticmethod
    def parse(text):
        return Period(text)


class AggFunction(b.PPIBotType):
    description = "aggregation function"
    entity_name = "AggFunction"
    usesFeature = {"name": "AggValues", "type": "phraselist",
                   "definition": "AggValues(interchangeable) disabledForAllModels",
                   "values": ["average", "max", "min", "sum", "maximum", "minimum", "mean", "median", "maximal", "minimal"]}

    def __init__(self, value):
        self.value = value

    @staticmethod
    def match(agg_text, similarity: SimilarityComputer, **args):
        if agg_text is None:
            return None

        possible_agg: dict = similarity.find_agg_function(agg_text, 0.7)
        possible_agg_sorted = {k: v for k, v in sorted(
            possible_agg.items(), key=lambda item: item[1], reverse=True)}

        return get_tuple_by_threshold(AggFunction.threshold_a, AggFunction.threshold_b, possible_agg_sorted, lambda agg_name: AggFunction(agg_name))

    @staticmethod
    def parse(text):
        return AggFunction(text)


class InstantCondition(b.PPIBotType):
    entity_name = "InstantCondition"

    def __init__(self, value):
        self.value = value

    @staticmethod
    def match(text, similarity):
        return None

    @staticmethod
    def detect_negative(text, similarity):
        doc = similarity.nlp(text)
        return [tok for tok in doc if tok.dep_ == 'neg']

    @staticmethod
    def modify_operator_negation(cond, text, similarity):
        if text:
            negation_tokens = InstantCondition.detect_negative(text, similarity)
            if negation_tokens:
                if type(cond) is tuple:
                    for instants in cond:
                        for instant in instants:
                            instant.value.changes_to_state.name = instant.value.changes_to_state.name.replace("==", "!=")
                elif type(cond) is InstantCondition:
                    cond.value.changes_to_state.name = cond.value.changes_to_state.name.replace("==", "!=")

    @staticmethod
    def detect_negative_pair(pair, text1, text2, similarity):
        InstantCondition.modify_operator_negation(pair[0], text1, similarity)
        InstantCondition.modify_operator_negation(pair[1], text2, similarity)

    @staticmethod
    def match_from_text(text, similarity: SimilarityComputer):
        #InstantCondition.threshold_a = 0.6
        #InstantCondition.threshold_b = 0.3
        logger.info(f"Instant Condition - Matching log for {text}")
        possible_pairs = similarity.find_most_similar_slot(text, delta_heuristics=0.5)
        possible_pairs_sorted = {k: v for k, v in sorted(possible_pairs.items(), key=lambda item: item[1], reverse=True)}
        cond = get_tuple_by_threshold(InstantCondition.threshold_a, InstantCondition.threshold_b, possible_pairs_sorted, lambda match_name: (InstantCondition(match_name.to_condition())))
        #logger.info(f"matching: {match}")
        return cond


    @staticmethod
    def match_special_pair(similarity: SimilarityComputer, text1, type):
        result = []
        logger.info(f"Instant Condition - Matching special pair for {text1}, {type}")

        if len(type) > 0:
            #InstantCondition.threshold_a = 1.3
            #InstantCondition.threshold_b = 1.2
            possible_match: dict = (
                similarity.find_most_similar_slot(text1, delta_heuristics=0.7))
            possible_match_sorted = {k: v for k, v in sorted(possible_match.items(
            ), key=lambda item: item[1], reverse=True)}  # Here i have all the possible matches cond1
            cond1 = get_tuple_by_threshold(InstantCondition.threshold_a, InstantCondition.threshold_b,
                                           possible_match_sorted, lambda match_name: (InstantCondition(match_name.to_condition())))

            InstantCondition.modify_operator_negation(cond1, text1, similarity)

            for t in type:
                if t == 'negation':
                    cond2 = get_tuple_by_threshold(InstantCondition.threshold_a, InstantCondition.threshold_b, possible_match_sorted, lambda match_name: (
                        InstantCondition(match_name.to_condition(negation=True))))
                    result.append((cond1, cond2))
                elif t == 'begin':
                    result.append((InstantCondition(TimeInstantCondition(
                        RuntimeState.START, applies_to=AppliesTo.PROCESS)), cond1))
                elif t == 'end':
                    result.append((cond1, InstantCondition(TimeInstantCondition(
                        RuntimeState.END, applies_to=AppliesTo.PROCESS))))
        
        #return list(map(lambda x: InstantCondition.detect_negative_pair(x, text1, None, similarity), result))
        return result


    @staticmethod
    def match_pair(similarity: SimilarityComputer, text1, text2=None, type=None):
        result = ()
        logger.info(f"Instant Condition - Matching pair for {text1}, {text2}")

        if text2 is not None:
            #InstantCondition.threshold_a = 1.8
            #InstantCondition.threshold_b = 1.5
            possible_pairs: dict = (similarity.find_most_similar_slots_matching_types(
                text1, text2, delta_heuristics=0.7))
            possible_pairs_sorted = {k: v for k, v in sorted(
                possible_pairs.items(), key=lambda item: item[1], reverse=True)}

            logger.debug(f"Instant Condition - matching: ${possible_pairs}")
            cond1 = get_tuple_by_threshold(InstantCondition.threshold_a, InstantCondition.threshold_b, possible_pairs_sorted, lambda match_name: (InstantCondition(match_name.slot1.to_condition())))
            cond2 = get_tuple_by_threshold(InstantCondition.threshold_a, InstantCondition.threshold_b, possible_pairs_sorted, lambda match_name: (InstantCondition(match_name.slot2.to_condition())))
            result = cond1, cond2

        InstantCondition.detect_negative_pair(result, text1, text2, similarity)
        return result


class FromInstantCondition(InstantCondition):
    entity_name = "FromInstantCondition"


class ToInstantCondition(InstantCondition):
    entity_name = "ToInstantCondition"


class Domain(b.PPIBotType):
    def __init__(self, value, attribute=None):
        self.value = value
        self.attribute = attribute

        if attribute is not None:
            values = [{"name": v, "canonicalForm": v, "synonyms": []}
                      for v in value]
            self.dynamic_lists = [
                {
                    "listEntityName": "AttributeValueList",
                    "requestLists": values
                }
            ]

    @staticmethod
    def domain_of_attribute(attribute_name, similarity: SimilarityComputer, categorical_threshold=300):
        domain = similarity.domain_of_attribute(
            attribute_name, categorical_threshold)
        domain_attribute = None

        if pd.api.types.is_list_like(domain):
            domain_attribute = attribute_name

        return Domain(domain, attribute=domain_attribute)


class Literal(b.PPIBotType):
    description = "literal value"

    def __init__(self, value, domain=None):
        self.value = value

        if isinstance(domain, Domain):
            domain = domain.value

        self.domain = domain

    @staticmethod
    def match(text, similarity, domain=None, **args):
        if text is None:
            return None

        if domain is None:
            return Literal(text)

        if isinstance(domain, Domain):
            domain = domain.value

        if pd.api.types.is_list_like(domain):
            possible_literal: dict = similarity.find_most_similar_from_list(
                text, domain, 0.7)
            possible_literal_sorted = {k: v for k, v in sorted(
                possible_literal.items(), key=lambda item: item[1], reverse=True)}

            return get_tuple_by_threshold(Literal.threshold_a, Literal.threshold_b, possible_literal_sorted, lambda literal_name: Literal(literal_name, domain))

        elif domain == 'datetime':
            try:
                value = pd.to_datetime(text)
            except:
                logger.warning(
                    f"Literal - match error: {text} is not a datetime", exc_info=True)
                return None

        elif domain == 'numeric':
            try:
                value = pd.to_numeric(text)
            except:
                logger.warning(
                    f"Literal - match error: {text} is not a numeric variable", exc_info=True)
                return None

        elif domain == 'timedelta':
            try:
                value = pd.to_timedelta(text)
            except:
                logger.warning(
                    f"Literal - match error: {text} is not a timedelta", exc_info=True)
                return None

        elif domain == 'string':
            value = str(text)

        else:
            logger.warning(f"Literal - match error: {domain} is not supported")
            return None

        return Literal(value, domain)

    @staticmethod
    def parse(text, domain=None):
        return Literal(text, domain)


class LogValue(b.PPIBotType):
    description = "value of an attribute of the log"
    entity_name = "AttributeValue"
    usesFeature = {"name": "AttributeValueList", "type": "list"}

    def __init__(self, value):
        self.value = value

    @staticmethod
    def match(text, similarity: SimilarityComputer, attribute):
        if text is None:
            return None

        possible_values: dict = similarity.find_most_similar_value(
            text, attribute.value, 0.7)
        print(f"LogValue - match: {attribute.value} / {possible_values}")
        possible_values_sorted = {k: v for k, v in sorted(
            possible_values.items(), key=lambda item: item[1], reverse=True)}
        return get_tuple_by_threshold(LogValue.threshold_a, LogValue.threshold_b, possible_values_sorted, lambda value_name: LogValue(value_name))

    @staticmethod
    def parse(text):
        return LogValue(text)


class VariableSave(b.RenderType):
    entity_patternany_name = "VariableName"

    def __init__(self, name, value, variables):
        self.name = name
        self.value = value
        self.variables = variables

    def render(self, tools):
        text = f"The metric has been saved as {self.name}"
        channel = {"variables": self.variables.names()}
        return text, channel


class DescribeLog(b.RenderType):
    def render(self, tools: b.Tools) -> str:
        df = tools.similarity.df
        event_number = df.shape[0]
        cases_number = df.drop_duplicates(subset=['ID']).shape[0]
        avg_events_per_case = round(event_number / cases_number, 2)
        attributes_names = df.columns.tolist()
        activity_names = df.drop_duplicates(subset=['ACTIVITY'])[
            'ACTIVITY'].tolist()
        first_event_date = df.iloc[0]['DATE']
        last_event_date = df.iloc[-1]['DATE']
        text = f"- There are {event_number} events in the log."
        text += f"\n- There are {cases_number} cases in the log."
        text += f"\n- The average number of cases per event is {avg_events_per_case}."
        text += f"\n- The are {len(attributes_names)} attributes of the log: {attributes_names}."
        text += f"\n- The activities of the log are: {activity_names}."
        text += f"\n- The first event date is {first_event_date}."
        text += f"\n- The last event date is {last_event_date}."
        return text


class FractionMetric(Metric):
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator
        self.metric = None

    def render(self, tools: b.Tools):
        result_num = tools.measure_computer.compute(self.numerator)
        result_num = self._format_compute(result_num)

        result_den = tools.measure_computer.compute(self.denominator)
        result_den = self._format_compute(result_den)

        num_days = result_num.split('days')[0]
        den_days = result_den.split('days')[0]

        self.metric = f"The fraction of {self.numerator} over {self.denominator}"

        return f"{float(num_days) / float(den_days)}"
