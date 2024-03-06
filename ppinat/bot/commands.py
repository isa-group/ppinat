import logging

from ppinat.matcher.recognizers import RecognizedEntities
from ppinat.matcher.similarity import SimilarityComputer
from ppinat.ppiparser.ppiannotation import PPIAnnotation, text_by_tag

import ppinat.helpers.log as log
import pandas as pd
import ppinot4py.model as ppinot
from ppinot4py.model import TimeInstantCondition, RuntimeState, AppliesTo

import ppinat.bot.types as t
import ppinat.bot.base as b

logger = logging.getLogger(__name__)


def groupby_hints(similarity):
    group_by_candidates = set(similarity.column_candidates(for_slot=False))
    all_columns = set(similarity.df.columns.values)
    rest_columns = all_columns - group_by_candidates
    if len(rest_columns) > 0:
        rest_columns_text = ", ".join(rest_columns)
        extra = f"""You could also group by
                {rest_columns_text}. However, all of them have many
                different values, just one value or are of a type that
                is not suitable for grouping.\n\n You can ask me *show me the values of attribute X*
                if you want more details about any of them."""
    else:
        extra = ""
    alternatives = log.attribute_options(group_by_candidates, similarity.df)
    return f" The best options may be:\n{alternatives}\n\n{extra}"


def instant_condition_hints(similarity, *args):
    columns = set(similarity.column_candidates())
    all_columns = set(similarity.df.columns.values)
    rest_columns = all_columns - columns
    if len(rest_columns) > 0:
        rest_columns_text = ", ".join(rest_columns)
        extra = f"""You can also use any of these other attributes:
                {rest_columns_text}. However, all of them have many
                different values, or they do not change during the case
                so they cannot be used for time conditions.\n\n You can ask me *show me the values of attribute X* if you want more details
                about any of them."""
    else:
        extra = ""
    columns_text = log.attribute_options(columns, similarity.df)
    return f" You can define the condition based on any of the following attributes:\n{columns_text}\n\n{extra}"


def value_hints(similarity, attribute):
    domain = domain_of_attribute(similarity, attribute)
    return value_condition_hints(similarity, domain)


def value_condition_hints(similarity, domain):
    conditions = conditions_for_domain(domain)
    conditions_text = ", ".join(
        [f"[{condition.command_name}]({condition.description})" for condition in conditions])
    return f"The possible values for the condition are: {conditions_text}. {domain_hints(similarity, domain)}"


def domain_hints(similarity, domain):
    if pd.api.types.is_list_like(domain.value):
        return f"The set of possible values is the following: {', '.join(domain.value)}"
    else:
        return f"The value should be a {domain.value}"


def conditions_for_domain(domain):
    conditions = [EqualsToCommand, NotEqualsToCommand]
    sortable_domains = {"numeric", "datetime", "timedelta"}
    if domain in sortable_domains:
        conditions += [GreaterThanCommand, LowerThanCommand]
    return conditions


class _ConditionsCommand(b.PPIBotCommand):
    required_context = t.Domain
    output = t.LogicCondition

    parameters = {
        "value": b.Parameter(
            question="Which is the value?",
            fail="I'm sorry, didn't get that.",
            # entities might be missing
            param_type=t.Literal,
            entities=["AttributeValue", "AttributeValueList", "number"],
            match_options={
                "domain": b.ValueOfContext()
            },
            hints=domain_hints,
            friendly_name="the value"
        )
    }

    def run_description(self, context, value):
        return None


class EqualsToCommand(_ConditionsCommand):
    command_name = "Equals to"
    description = "Compares two values and checks whether they are the same. Example: 'equals to closed'"
    phrases = [
        "=",
        "= {@AttributeValue = 235}",
        "equal to {@AttributeValue = 23.78}",
        "equal to something",
        "is {@AttributeValue = 7}",
        "equals to {@AttributeValue = activity x}",
        "equals to {@AttributeValue = closed}",
        "equals to {@AttributeValue = opened}",
        "equals to {@AttributeValue = provider3}",
        "is some value",
        "use equal to",
        "use equals to"
    ]

    def run(self, context, value):
        return t.LogicCondition(value, op='equal')


class NotEqualsToCommand(_ConditionsCommand):
    command_name = "Not Equals to"
    description = "Compares two values and checks whether they are not the same. Example: 'not equal to 100'"
    output = t.LogicCondition
    phrases = [
        "!= {@AttributeValue=activity y}",
        "not equal to",
        "not equal to {@AttributeValue=32}",
        "not equal to {@AttributeValue=provider3}",
        "not equals to",
        "not equals to {@AttributeValue=worker}",
        "different than {@AttributeValue=send fine}",
        "use different than",
        "use not equal to",
        "use not equals to"
    ]

    def run(self, context, value):
        return t.LogicCondition(value, op='not_equal')


class GreaterThanCommand(_ConditionsCommand):
    command_name = "Greater than"
    description = "Compares two values and checks whether the first is greater than the second. Example: 'first_attribute greater than second_attribute'"
    output = t.LogicCondition
    phrases = [
        "> {@AttributeValue=6}",
        "greater than",
        "more than",
        "bigger than",
        "greater than {@AttributeValue=32}",
        "more than {@AttributeValue=45}",
        "use greater than",
        "bigger than {@AttributeValue=32}",
        "larger than {@AttributeValue=42}",
        "greater than a value"
    ]

    def run(self, context, value):
        return t.LogicCondition(value, op='gt')


class LowerThanCommand(_ConditionsCommand):
    command_name = "Lower than"
    description = "Compares two values and checks whether the first is lower than the second. Example: 'first_attribute lower than second_attribute'"
    output = t.LogicCondition
    phrases = [
        "< {@AttributeValue=6}",
        "lower than",
        "less than",
        "smaller than",
        "lower than {@AttributeValue=32}",
        "less than {@AttributeValue=45}",
        "use less than",
        "smaller than {@AttributeValue=32}",
        "shorter than {@AttributeValue=42}",
        "less than a value"
    ]

    def run(self, context, value):
        return t.LogicCondition(value, op='lt')


def domain_of_attribute(similarity, attribute, categorical_threshold=300):
    return attribute.domain_of_attribute(similarity, categorical_threshold)


class StartCaseConditionCommand(b.PPIBotCommand):
    command_name = "Start Case Condition"
    description = "Sets as condition the beginning of the case."
    required_context = None
    output = t.InstantCondition
    phrases = [
        "the beginning of the case",
        "when the process starts",
        "the beginning of the process",
        "the beginning of the instance",
        "the start of the process",
        "when the case begins"
    ]

    def run(self, context):
        return t.InstantCondition(ppinot.TimeInstantCondition(ppinot.RuntimeState.START, ppinot.AppliesTo.PROCESS))

    def run_description(self, context):
        return f"Ok, we'll use the beginning of the case"


class EndCaseConditionCommand(b.PPIBotCommand):
    command_name = "End Case Condition"
    description = "Sets as condition the end of the case."
    required_context = None
    output = t.InstantCondition
    phrases = [
        "the end of the case",
        "when the process finishes",
        "the end of the process",
        "the end of the instance",
        "the end of the process",
        "when the case is over",
        "at the end of the case"
    ]

    def run(self, context):
        return t.InstantCondition(ppinot.TimeInstantCondition(ppinot.RuntimeState.END, ppinot.AppliesTo.PROCESS))

    def run_description(self, context):
        return f"Ok, we'll use the end of the case"


class BaseInstantConditionCommand(b.PPIBotCommand):
    command_name = "Base Instant Condition"
    description = "Specifies a condition for an attribute. Example 'state equals to closed'."
    required_context = None
    output = t.InstantCondition
    phrases = [
        "{@AttributeName=state}",
        "{@AttributeName=activity} {@Condition={@operand=equals to} {@AttributeValue={@value=activity x}}}",
        "{@AttributeName=resolutor} {@Condition={@operand=equals to} {@AttributeValue={@value=provider3}}",
        "{@AttributeName=size} {@Condition={@operand=equal to} {@value=23.78}}",
        "{@AttributeName=state} {@Condition={@operand=equals to} {@AttributeValue={@value=closed}}}",
        "{@AttributeName=state} {@Condition={@operand=equals to} {@AttributeValue={@value=opened}}}",
        "{@AttributeName=state} {@Condition={@operand=is} {@AttributeValue={@value=fixed}}}",
        "when {@AttributeName=activity} {@Condition={@operand=is} {@AttributeValue={@value=prepare invoice}}}",
        "{@AttributeName=payment} {@Condition={@operand==} {@value=235}}",
        "{@AttributeName=priority} {@Condition={@operand=equals to} {@value=7}}",
    ]

    parameters = {
        "attribute": b.Parameter(
            question="You need to pick an attribute.",
            fail="Sorry, I couldn't get the attribute.",
            param_type=t.LogAttribute,
            friendly_name="the attribute",
            hints=instant_condition_hints
        ),
        "cond": b.Parameter(
            question="You need to specify the condition for the attribute.",
            fail="I'm sorry, didn't get that.",
            entities=["Condition"],
            param_type=t.LogicCondition,
            hints=value_hints,
            context=domain_of_attribute,
            match_options={
                "attribute": b.ValueOf("attribute")
            },
            friendly_name="the condition"
        )
    }

    def match_entities(self, entities: RecognizedEntities, similarity, **args):
        attribute = self.parameters["attribute"].extract_entity(entities)
        matched_attribute = self.match(
            "attribute", attribute, similarity) if attribute is not None else False

        if matched_attribute:
            domain = domain_of_attribute(similarity, self.get("attribute"))
            condition = self.parameters["cond"].extract_entity(entities)
            matched_condition = t.LogicCondition.match_condition(
                condition, similarity, domain)
            if matched_condition is not None:
                self.save("cond", matched_condition)

        return False

    def run(self, context, attribute, cond):
        condition = f"`{attribute.value}` {cond.to_condition()}"
        return t.InstantCondition(TimeInstantCondition(condition))

    def run_description(self, context, attribute, cond):
        return f"Ok, we'll use the condition {attribute.value} {cond.to_condition()}"


def cyclic_precondition(context, similarity: SimilarityComputer, **values):
    perc_threshold = 0.05
    perc_from_cond = similarity.perc_conditions_in_cases(
        (values["from_cond"].value))
    perc_to_cond = similarity.perc_conditions_in_cases(
        (values["to_cond"].value))
    return perc_from_cond >= perc_threshold and perc_to_cond >= perc_threshold


def aggregation_cyclic_precondition(context, similarity, **values):
    if values["is_cyclic"] is not None and values["is_cyclic"].value == "Yes":
        return True
    else:
        return False


def metric_precondition_false(context, similarity, **values):
    return False


def conditional_metric_friendly_name(operand):
    if operand == "equal":
        return "equal to"
    elif operand == "not_equal":
        return "not equal to"
    elif operand == "gt":
        return "greater than"
    elif operand == "lt":
        return "less than"


class TimeMetricCommand(b.PPIBotCommand):
    command_name = "Time command"
    description = "Calculates the time elapsed between two given time instants in each case."
    intent_names = ["Metric"]
    intent_filter = {
        # "metric_type": "time",
        "AggFunction": None
    }
    phrases = "phrases/time.txt"

    required_context = None
    confirm = True
    output = t.BaseMetric

    parameters = {
        "from_cond": b.Parameter(
            question=f"When do you want to start measuring time?",
            fail="Sorry, let's start over",
            param_type=t.FromInstantCondition,
            friendly_name="the beggining condition",
            hints=instant_condition_hints
        ),
        "to_cond": b.Parameter(
            question=f"When do you want to end measuring time?",
            fail="Sorry, let's start over",
            param_type=t.ToInstantCondition,
            friendly_name="the end condition",
            hints=instant_condition_hints
        ),
        "is_cyclic": b.Parameter(
            question="Is the time metric cyclic?",
            fail="Sorry, I didn't get that",
            param_type=t.Literal,
            options=["Yes", "No"],
            precondition=cyclic_precondition,
            match_options={
                "from_cond": b.ValueOf("from_cond"),
                "to_cond": b.ValueOf("to_cond")
            }
        ),
        "agg_metric_cycle": b.Parameter(
            question="How do you want to aggregate cyclic times?",
            fail="Sorry, I didn't get that",
            param_type=t.AggFunction,
            options=["average", "sum", "max", "min"],
            precondition=aggregation_cyclic_precondition,
            match_options={
                "is_cyclic": b.ValueOf("is_cyclic"),
            }
        ),
        "conditional_metric": b.Parameter(
            question="",
            fail="Sorry, I didn't get that",
            param_type=t.LogicCondition,
            precondition=metric_precondition_false
        )
    }

    def match_entities(self, result, similarity, heuristics=True, **args):
        # We use the metric decoder instead of the entities provided by the recognizer
        if hasattr(result, "text"):
            annotation: PPIAnnotation = similarity.metric_decoder(result.text)
        else:
            annotation = PPIAnnotation = similarity.metric_decoder(
                result["text"])

        from_text = text_by_tag(annotation, "TSE")
        to_text = text_by_tag(annotation, "TEE")
        only_text = text_by_tag(annotation, "TBE")
        conditional_text = text_by_tag(annotation, "CCI")
        conditional_attribute_text = text_by_tag(annotation, "AttributeValue")

        if conditional_text is not None and conditional_attribute_text is not None:
            entity = {
                "operand": conditional_text,
                "value": conditional_attribute_text
            }
            matched_condition = t.LogicCondition.match_condition(
                entity, similarity)
            self.save("conditional_metric", matched_condition)

        if only_text is not None:
            logger.info(f"TimeMetricCommand - Matching for [only:{only_text}]")
            self.only_text_time_metric = True

            if heuristics:
                result = t.InstantCondition.match_special_pair(similarity, only_text, type=['negation', 'begin', 'end'])
                (cond, cond_neg) = result[0]
                (start, cond2_beg) = result[1]
                (cond1_end, end) = result[2]
            
                cond1 = ([start] + cond[0], cond[1])
                cond2 = (cond[0] + cond_neg[0] + [end], cond[1] + cond_neg[1])

                self.save("from_cond", cond1, True)
                self.save("to_cond", cond2, True)
            else:
                cond = t.InstantCondition.match(only_text, similarity)
                self.save_or_unknown("to_cond", cond, only_text,
                                    save_alternatives=True)

            return True

        if from_text is not None and to_text is not None:
            logger.info(
                f"TimeMetricCommand - Matching for [from:{from_text}] and [to:{to_text}]")
            (cond1, cond2) = t.InstantCondition.match_pair(
                similarity, from_text, to_text)

            self.save_or_unknown("from_cond", cond1,
                                 from_text, save_alternatives=True)
            self.save_or_unknown("to_cond", cond2, to_text,
                                 save_alternatives=True)

            return True

        elif from_text is not None:
            logger.info(f"TimeMetricCommand - Matching for [from:{from_text}]")
            if heuristics:
                (cond, end) = t.InstantCondition.match_special_pair(similarity, from_text, type=['end'])[0]
                self.save_or_unknown("from_cond", cond, from_text, save_alternatives=True)
                self.save("to_cond", ([end],[]), save_alternatives=True)
            else:
                cond = t.InstantCondition.match(from_text, similarity)
                self.save_or_unknown("from_cond", cond, from_text, save_alternatives=True)

        elif to_text is not None:
            logger.info(f"TimeMetricCommand - Matching for [to:{to_text}]")
            if heuristics:
                (start, cond) = t.InstantCondition.match_special_pair(similarity, to_text, type=['begin'])[0]
                self.save("from_cond", ([start],[]), save_alternatives=True)
                self.save_or_unknown("to_cond", cond, to_text, save_alternatives=True)
            else:
                cond = t.InstantCondition.match(to_text, similarity)
                self.save_or_unknown("to_cond", cond, to_text, save_alternatives=True)

        return False

    def run(self, context, from_cond, to_cond, is_cyclic=None, agg_metric_cycle=None, conditional_metric=None):
        if is_cyclic != None and is_cyclic.value == "Yes":
            metric = t.BaseMetric(ppinot.TimeMeasure(
                from_condition=from_cond.value,
                to_condition=to_cond.value,
                time_measure_type="CYCLIC",
                single_instance_agg_function=agg_metric_cycle.value
            ))
            if conditional_metric is not None:
                mmap = {"ma": metric.metric}
                cond_repr, cond_map = conditional_metric.to_condition(
                    with_map=True)
                mmap.update(cond_map)
                return t.BaseMetric(ppinot.DerivedMeasure(function_expression=f"ma {cond_repr}",
                                                          measure_map=mmap))
            return metric

        else:
            metric = t.BaseMetric(ppinot.TimeMeasure(
                from_condition=from_cond.value,
                to_condition=to_cond.value,
            ))

            if conditional_metric is not None:
                mmap = {"ma": metric.metric}
                cond_repr, cond_map = conditional_metric.to_condition(
                    with_map=True)
                mmap.update(cond_map)
                return t.BaseMetric(ppinot.DerivedMeasure(function_expression=f"ma {cond_repr}",
                                                          measure_map=mmap))
            return metric

    def run_description(self, context, from_cond, to_cond, is_cyclic=None, agg_metric_cycle=None, conditional_metric=None):
        if conditional_metric is not None:
            conditional_operand = conditional_metric_friendly_name(
                conditional_metric.op)
            return f"We are measuring time from {from_cond.value} to {to_cond.value} {conditional_operand} {conditional_metric.value.value}"
        else:
            return f"We are measuring time from { from_cond.value } to { to_cond.value }"

    def conditional_alernatives_filter(self, param_name, alt_parsed):
        if hasattr(self, "only_text_time_metric") and param_name == "to_cond":
            if "==" in str(self.get("from_cond").value) and "!=" in alt_parsed:
                return True
            elif "==" not in str(self.get("from_cond").value) and "==" in alt_parsed:
                return True
            elif "==" not in alt_parsed and "!=" not in alt_parsed:
                return True
            else:
                return False
        else:
            return True


class CountMetricCommand(b.PPIBotCommand):
    command_name = "Count command"
    description = "Calculates the number of times a given event occurs in each case."
    intent_names = ["Metric"]
    phrases = "phrases/count.txt"
    intent_filter = {
        #        "metric_type": "count",
        "AggFunction": None
    }
    required_context = None
    confirm = True
    output = t.BaseMetric

    parameters = {
        "when": b.Parameter(
            question="When do you want to count?",
            fail="Sorry, let's start over",
            param_type=t.InstantCondition,
            hints=instant_condition_hints
        ),
        "conditional_metric": b.Parameter(
            question="",
            fail="Sorry, I didn't get that",
            param_type=t.LogicCondition,
            precondition=metric_precondition_false
        )
    }

    def match_entities(self, result, similarity, heuristics=True, **args):
        # We use the metric decoder instead of the entities provided by the recognizer
        if hasattr(result, "text"):
            annotation: PPIAnnotation = similarity.metric_decoder(result.text)
        else:
            annotation: PPIAnnotation = similarity.metric_decoder(
                result["text"])

        when_text = text_by_tag(annotation, "CE")
        conditional_text = text_by_tag(annotation, "CCI")
        conditional_attribute_text = text_by_tag(annotation, "AttributeValue")
        aggregation = annotation.get_aggregation_function()

        logger.info(f"CountMetricCommand - Matching entities")


        if conditional_text is not None and conditional_attribute_text is not None:
            entity = {
                "operand": conditional_text,
                "value": conditional_attribute_text
            }
            matched_condition = t.LogicCondition.match_condition(
                entity, similarity)
            self.save("conditional_metric", matched_condition)
        else:
            if heuristics and aggregation is not None:
                agg_match = t.resolve_tuple_with_first(t.AggFunction.match(aggregation, similarity))
                if agg_match.value == "%":
                    conditional_metric = t.LogicCondition(t.Literal(0, 'numeric'), op='gt')
                    self.save("conditional_metric", conditional_metric)

        if when_text is not None:
            if heuristics:
                if t.InstantCondition.detect_negative(when_text, similarity):
                    matched_condition = t.LogicCondition(t.Literal(0, 'numeric'), op='equal')
                    self.save("conditional_metric", matched_condition)

            cond = t.InstantCondition.match_from_text(when_text, similarity)
            self.save_or_unknown("when", cond, when_text,
                                 save_alternatives=True)
            return True


        return False

    def run(self, context, when, conditional_metric=None):

        metric = t.BaseMetric(ppinot.CountMeasure(when.value))
        if conditional_metric is not None:
            mmap = {"ma": metric.metric}
            cond_repr, cond_map = conditional_metric.to_condition(
                with_map=True)
            mmap.update(cond_map)
            return t.BaseMetric(ppinot.DerivedMeasure(function_expression=f"ma {cond_repr}",
                                                      measure_map=mmap))
        return metric

    def run_description(self, context, when, conditional_metric=None):
        if conditional_metric is not None:
            conditional_operand = conditional_metric_friendly_name(
                conditional_metric.op)
            return f"We are measuring when {when.value} {conditional_operand} {conditional_metric.value.value}"
        return f"We are measuring when { when.value }"


def data_attribute_hints(similarity):
    all_columns = set(similarity.df.columns.values)
    return f"""The attributes you can choose from are: {', '.join(all_columns)}. You can ask
           *show me the values of attribute X* if you want more details about any of them."""


class DataMetricCommand(b.PPIBotCommand):
    command_name = "Data metric"
    description = "Returns the value of an attribute in each case."
    required_context = None
    intent_names = ["Metric"]
    intent_filter = {
        "AggFunction": None
    }

    confirm = True
    output = t.BaseMetric
    phrases = "phrases/data.txt"

    parameters = {
        "attribute": b.Parameter(
            question="From which attribute do you want to get its value?",
            fail="Sorry, let's start over",
            param_type=t.LogAttribute,
            hints=data_attribute_hints
        ),
        "attribute_value": b.Parameter(
            question="Which is the value?",
            fail="Sorry, let's start over",
            param_type=t.LogValue,
            precondition=metric_precondition_false
        ),
        "conditional_metric": b.Parameter(
            question="",
            fail="Sorry, I didn't get that",
            param_type=t.LogicCondition,
            precondition=metric_precondition_false
        )
    }

    def match_entities(self, result, similarity, **args):
        if hasattr(result, "text"):
            annotation: PPIAnnotation = similarity.metric_decoder(result.text)
        else:
            annotation: PPIAnnotation = similarity.metric_decoder(
                result["text"])

        attribute_name = text_by_tag(annotation, "AttributeName")
        conditional_attribute_text = text_by_tag(annotation, "AttributeValue")

        logger.info(f"DataMetricCommand - Matching entities")


        if attribute_name is not None:
            attr = t.LogAttribute.match(attribute_name, similarity)
            self.save_or_unknown("attribute", attr, attribute_name,
                                 save_alternatives=True)           
            if conditional_attribute_text is not None and self.get("attribute", get_alternatives=True) is not None:
                attribute_value = t.LogValue.match(conditional_attribute_text, similarity, self.get("attribute", get_alternatives=True))
                self.save_or_unknown("attribute_value", attribute_value, conditional_attribute_text, save_alternatives=True)

                if self.get('attribute_value') is not None:
                    entity = {
                        "operand": "equal",
                        "value": '"' + self.get('attribute_value').value + '"'
                    }
                    matched_condition = t.LogicCondition.match_condition(
                        entity, similarity)
                    self.save("conditional_metric", matched_condition)
            
            return True

        return False

    def run(self, context, attribute, conditional_metric=None, attribute_value=None):
        metric = t.BaseMetric(ppinot.DataMeasure(attribute.value))
        if conditional_metric is not None:
            mmap = {"ma": metric.metric}
            cond_repr, cond_map = conditional_metric.to_condition(
                with_map=True)
            mmap.update(cond_map)
            return t.BaseMetric(ppinot.DerivedMeasure(function_expression=f"ma {cond_repr}",
                                                      measure_map=mmap))
        return metric

    def run_description(self, context, attribute, conditional_metric=None, attribute_value=None):
        return f"We are returning the value of { attribute.value }"


class ConditionalMetricCommand(b.PPIBotCommand):
    command_name = "Conditional metric"
    description = "Computes a derived metric by applying a condition to one metric. Example: 'I want to define a conditional metric'"
    required_context = None
    output = t.BaseMetric
    phrases = [
        "i want to define a conditional metric",
        "i want to define a condition to a metric",
        "add a condition to a metric",
        "add a condition to this metric",
        "define a conditional metric"
    ]

    parameters = {
        "metric": b.Parameter(
            question="Which is the metric you want to apply the condition to?",
            fail="Sorry, let's start over",
            param_type=t.BaseMetric,
            load_from_context=lambda context: not isinstance(
                context, ppinot.model.DerivedMeasure)
        ),
        "cond": b.Parameter(
            question="Which is the condition for the metric?",
            fail="I'm sorry, didn't get that.",
            entities=["Condition"],
            param_type=t.LogicCondition,
            hints=lambda similarity, metric: value_condition_hints(
                similarity, metric.domain_of_metric(similarity)),
            context=lambda sim, metric: metric.domain_of_metric(sim),
            match_options={
                "metric": b.ValueOf("metric")
            }
        )
    }

    def run(self, context, metric, cond):
        mmap = {"ma": metric.metric}
        cond_repr, cond_map = cond.to_condition(with_map=True)
        mmap.update(cond_map)
        return t.BaseMetric(ppinot.DerivedMeasure(function_expression=f"ma {cond_repr}",
                                                  measure_map=mmap))

    def run_description(self, context, metric, cond):
        return f"We are measuring {metric.metric} { cond.to_condition() }"


class ComputeMetricCommand(b.PPIBotCommand):
    command_name = "Compute metric"
    description = "Computes an aggregated metric over several cases. Example: 'average time between <condition> and <condition>'."
    intent_names = ["Metric", "Time command", "Count command", "Data metric"]
    required_context = None
    output = t.AggMetric
    phrases = [
        "i want to compute a metric",
        "i want to aggregate this metric",
        "aggregate the results using the {@AggFunction=min}",
        "compute the {@AggFunction=average} of this",
        "compute the {@AggFunction=sum}"
    ]

    parameters = {
        "base_measure": b.Parameter(
            question="What do you want to measure?",
            fail="Sorry, metric type not supported",
            param_type=t.BaseMetric,
            friendly_name="the base measure",
            load_from_context=True
        ),
        "agg_function": b.Parameter(
            question="How do you want to aggregate the results?",
            fail="Sorry, this option is not available",
            param_type=t.AggFunction,
            options=["AVG", "SUM", "MAX", "MIN"],
            friendly_name="the aggregation function"
        ),
        "denominator": b.Parameter(
            question="What do you want to measure?",
            fail="Sorry, metric type not supported",
            param_type=t.BaseMetric,
            friendly_name="the base measure of denominator",
        )
    }

    def __init__(self, context=None, expected=None, situation=None, intent=None):
        super(ComputeMetricCommand, self).__init__(
            context, expected, situation, intent)
        self.groupby = None
        self.period = None

    @classmethod
    def match_intent_condition(cls, intent, entities):
        if isinstance(entities, PPIAnnotation):
            return entities.extract_entity("aggregation") is not None
        else:
            if intent == cls.command_name:
                return True
            else:
                return entities.extract_entity("AggFunction") is not None

    def match_entities(self, result, similarity, heuristics=True, **args):
        if self.intent == self.command_name:
            return super().match_entities(result, similarity)
        else:
            annotation: PPIAnnotation = similarity.metric_decoder(result.text)
            logger.info(f"ComputeMetricCommand - Matching entities for annotation: {annotation}")

            self.metric_type = annotation.get_measure_type()
            if self.metric_type == "count" or self.metric_type == "count_es":
                command_type = CountMetricCommand
                default_agg = 'SUM'
            elif self.metric_type == "time" or self.metric_type == "time_es":
                command_type=TimeMetricCommand
                default_agg = 'AVG'
            elif self.metric_type == "data" or self.metric_type == "data_es":
                command_type=DataMetricCommand
                default_agg = 'SUM'
            else:
                command_type = None

            if command_type is not None:
                found_command, _ = self.save_partial(
                    "base_measure",
                    command_type=command_type,
                    context=None,
                    entities=result,
                    similarity=similarity,
                    heuristics = heuristics,
                )
                logger.info(f"ComputeMetricCommand - Base command found ({found_command})")

            matched_agg = self.match("agg_function", annotation.get_aggregation_function(), similarity)
            if not matched_agg and heuristics:
                self.save("agg_function", t.AggFunction.parse(default_agg))


            # groupby_text = annotation.get_chunk_by_tag("GBC")
            if text_by_tag(annotation, "GBC"):
                self.groupby = t.LogAttribute.match(text_by_tag(annotation, "GBC"), similarity)

                if len(self.groupby[0]) > 0:
                    if self.groupby[0][0].value == similarity.time_column:
                        self.period = t.Period.match(text_by_tag(annotation, "GBC"), similarity)
                        self.groupby = None
                else:
                    self.period = t.Period.match(text_by_tag(annotation, "GBC"), similarity)

            denominator_text = text_by_tag(annotation, "FDE")
            if denominator_text is not None:
                cond = t.InstantCondition.match_from_text(
                    denominator_text, similarity)
                self.save_or_unknown("denominator", cond,
                                     denominator_text, save_alternatives=True)
            # else:
            #     self.save("denominator", None)

        # We always need to confirm the base measure
        return False

    def run_description(self, context, base_measure, agg_function, denominator=None):
        agg_metric = self.run(context, base_measure, agg_function, denominator)
        return f'The result of the percentage is:' if agg_function.value == "%" else f"The result of {agg_metric.metric} is:"

    def run(self, context, base_measure, agg_function, denominator=None):
        if agg_function is None:
            raise RuntimeError("Aggregated function not matched")

        if base_measure is None:
            raise RuntimeError(
                "A base measure is needed to build the aggregation")

        group_value = None
        if self.groupby is not None:
            if len(self.groupby) > 0:
                if len(self.groupby[0]) > 0:
                    group_value = self.groupby[0][0].value
                elif len(self.groupby[1]) > 0:
                    group_value = self.groupby[1][0].value
            
        if group_value is not None:
            dm = ppinot.DataMeasure(group_value)
            dm.id = group_value
            grouper = [dm]
        else:
            grouper = []

        if denominator is not None:
            metric = t.BaseMetric(ppinot.CountMeasure(denominator.value))
            mmap = {"ma": metric.metric}
            conditional_metric = t.LogicCondition(
                t.Literal(0, domain="numeric"), "gt")
            cond_repr, cond_map = conditional_metric.to_condition(
                with_map=True)
            mmap.update(cond_map)
            filter = t.BaseMetric(ppinot.DerivedMeasure(function_expression=f"ma {cond_repr}",
                                                        measure_map=mmap))
            return t.AggMetric(
                metric=ppinot.AggregatedMeasure(
                    base_measure=base_measure.metric,
                    single_instance_agg_function="AVG" if agg_function.value == "%" else agg_function.value,
                    grouper=grouper,
                    filter_to_apply=filter.metric
                ),
                period=None
            )

        if self.metric_type == "count":
            return t.AggMetric(
                metric=ppinot.AggregatedMeasure(
                    base_measure=base_measure.metric,
                    single_instance_agg_function="AVG" if agg_function.value == "%" else agg_function.value,
                    grouper=grouper
                ),
                period=None
            )
        elif self.metric_type == "time":
            metric = t.AggMetric(
                metric=ppinot.AggregatedMeasure(
                    base_measure=base_measure.metric,
                    single_instance_agg_function="SUM" if agg_function.value == "%" else agg_function.value,
                    grouper=grouper
                ),
                period=None
            )

            if agg_function.value != "%":
                return metric

            total_metric = t.BaseMetric(ppinot.TimeMeasure(
                from_condition=t.InstantCondition(TimeInstantCondition(
                    RuntimeState.START, applies_to=AppliesTo.PROCESS)).value,
                to_condition=t.InstantCondition(TimeInstantCondition(
                    RuntimeState.END, applies_to=AppliesTo.PROCESS)).value,
                single_instance_agg_function="SUM"
            ))
            total_metric = t.AggMetric(
                metric=ppinot.AggregatedMeasure(
                    base_measure=total_metric.metric,
                    single_instance_agg_function="SUM",
                    grouper=grouper
                ),
                period=None
            )
            return t.FractionMetric(
                numerator=metric.metric,
                denominator=total_metric.metric
            )
        elif self.metric_type == "data":
            return t.AggMetric(
                metric=ppinot.AggregatedMeasure(
                    base_measure=base_measure.metric,
                    single_instance_agg_function= "AVG" if agg_function.value == "%" else agg_function.value,
                    grouper=grouper
                ),
                period=None
            )


def groupby_precondition(context, similarity, **values):
    metric: t.AggMetric = context
    return len(metric.metric.grouper) > 0


class GroupByCommand(b.PPIBotCommand):
    command_name = "Group by"
    description = "Groups metrics obtained by a given attribute. Example: 'group by priority'"
    required_context = t.AggMetric
    output = t.AggMetric
    warn_size = 35
    phrases = [
        "group by attribute {@AttributeName=activity}",
        "group by {@AttributeName=priority}",
        "group by some value",
        "i want to group by {@AttributeName=priority}",
        "i want to group by some attribute",
        "i want to group by some column of the log",
        "i want to group by {@AttributeName=type of resource}",
        "use group by"
    ]

    parameters = {
        "groupby": b.Parameter(
            question="Which attribute do you want to use to group by?",
            fail="Sorry, I couldn't get the attribute.",
            param_type=t.LogAttribute,
            entities=["AttributeName", "AttributeList"],
            hints=groupby_hints,
            friendly_name="the attribute to group by"
        ),
        "add": b.Parameter(
            question="""You are already grouping by one or more
                        attributes, do you want to **add** this attribute to
                        the list of group by or do you want to **replace** the
                        list with this new value?""",
            fail="Sorry, I couldn't get the response",
            param_type=t.Literal,
            precondition=groupby_precondition,
            options=["add", "replace"],
            match_options={
                "domain": "string"
            }
        )
    }

    def confirm_message(self, tools: b.Tools, context, groupby, add=None):
        num_values = len(tools.similarity.value_candidates(groupby.value))
        if num_values > self.warn_size:
            return f"""The attribute {groupby.value} has {num_values} values.
                   Are you sure you want to group by this attribute?"""
        else:
            return None

    def run_description(self, context, groupby, add=None):
        return f"I'm grouping by {groupby.value}"

    def run(self, context, groupby, add=None):
        if groupby is not None:
            dm = ppinot.DataMeasure(groupby.value)
            dm.id = groupby.value
            grouper = [dm]

            if add is not None and add.value == 'add':
                grouper = context.metric.grouper + grouper

            context.metric.grouper = grouper

        else:
            logger.warning("No value for groupby")

        return context


class PeriodicityCommand(b.PPIBotCommand):
    command_name = "Periodicity"
    description = "Displays the metrics obtained with a given periodicity. Example: 'yearly'"
    required_context = t.AggMetric
    output = t.AggMetric
    warn_size = 100
    phrases = "phrases/periodicity.txt"

    parameters = {
        "period": b.Parameter(
            question=f"Which frequency of aggregation do you want to apply? Some examples are: monthly, weekly, daily, yearly",
            fail=f"Sorry, I couldn't get the frequency.",
            param_type=t.Period,
            friendly_name="the frequency of aggregation"
        )
    }

    def match_entities(self, entities, similarity, **args):

        time_unit = entities.extract_entity("Time unit", "Period")
        value = entities.extract_entity("Value", "Period")

        if time_unit is not None:
            period = {
                "time_unit": time_unit.value,
                "value": 1
            }

            if value is not None:
                period["value"] = value.value

            self.save("period", self.parameters["period"].param_type.match_period(
                period, similarity), True)

        return self.get("period") is not None

    def confirm_message(self, tools: b.Tools, context, period):
        values = tools.similarity.df.groupby(tools.similarity.id_case)[
            tools.similarity.time_column].last()

        num_values = len(pd.date_range(start=min(values),
                                       end=max(values), freq=period.value))
        if num_values > self.warn_size:
            return f"""The aggregation frequency chosen leads to {num_values} values.
                   Are you sure you want to aggregate using this frequency?"""
        else:
            return None

    def run_description(self, context, period):
        return f"I'm aggregating every {period.value}"

    def run(self, context, period):
        if period is not None:
            context.period = pd.Grouper(freq=period.value)
        else:
            logger.warning("No value for period")

        return context


class SaveMetric(b.PPIBotCommand):
    command_name = "Save metric"
    description = "Saves a metric as a variable with the given name. Example: 'save as x'"
    required_context = t.AggMetric
    output = None
    variables_access = True
    phrases = [
        "save {@VariableName}",
        "save as {@VariableName}",
        "save metric as {@VariableName}"
    ]

    parameters = {
        "name": b.Parameter(
            question=f"Which is the name of the metric?",
            fail=f"Sorry, I cannot use that name.",
            param_type=t.Literal,
            entities=["VariableName"],
            match_options={
                "domain": t.Domain("string")
            },
            friendly_name="the name of the metric"
        )
    }

    def run_description(self, context, name):
        return None

    async def run(self, context, variables, name):
        variables.set_variable(name.value, context)

        return t.VariableSave(name.value, context, variables)


class LoadMetric(b.PPIBotCommand):
    command_name = "Load metric"
    description = "Loads the metric with the given name. Example: 'load metric x'"
    required_context = None
    output = t.AggMetric
    variables_access = True
    phrases = [
        "load metric {@VariableName}",
        "load {@VariableName}"
    ]

    parameters = {
        "name": b.Parameter(
            question=f"Which is the name of the metric?",
            fail=f"Sorry, I cannot use that name.",
            param_type=t.Literal,
            entities=["VariableName"],
            match_options={
                "domain": t.Domain("string")
            },
            friendly_name="the name of the metric"
        )
    }

    @ staticmethod
    def help_filter(situation, variables):
        return len(variables.names()) > 0

    def run_description(self, context, name):
        return f"Loading metric {name.value}"

    async def run(self, context, variables, name):
        if name.value in variables.names():
            return variables.get_variable(name.value)
        else:
            return b.ErrorType(message=f"Metric {name.value} could not be found")


class ShowAttributesCommand(b.PPIBotCommand):
    command_name = "Show log attributes"
    description = "Displays all attributes available in the event log. Example: 'show log attributes'"
    required_context = None
    output = None
    phrases = [
        "show me the attributes",
        "tell me the attributes of the log",
        "tell me the columns of the dataset",
        "which are the attributes of the log?",
        "which are the attributes?",
        "which attributes can i pick?"
    ]

    parameters = {}

    def run_description(self, context):
        return "The attributes of the log are:"

    def run(self, context):
        return t.LogRender()


class ShowValuesCommand(b.PPIBotCommand):
    command_name = "Show attribute values"
    description = "Displays the specific values of a given event log attribute. Example: 'show attribute values'"
    required_context = None
    output = None
    warn_size = 90
    phrases = [
        "show me details for {@AttributeName=resource}",
        "show me the values of {@AttributeName=typology}",
        "tell me the values of {@AttributeName=activities}",
        "tell me the values of one attribute",
        "tell me which are the {@AttributeName=roles}",
        "which are the options for {@AttributeName=amount}?",
        "which are the values for {@AttributeName=priority}?",
        "which are the values of {@AttributeName=activity}?"
    ]

    parameters = {
        "attribute": b.Parameter(
            question="For which attribute do you want to see its values",
            fail="Sorry, I couldn't get the attribute.",
            param_type=t.LogAttribute,
            friendly_name="the attribute"
        )
    }

    def confirm_message(self, tools: b.Tools, context, attribute):
        num_values = len(tools.similarity.value_candidates(attribute.value))
        if num_values > self.warn_size:
            return f"""The attribute {attribute.value} has {num_values} values.
                   Showing all of them it is going to take me a while.
                   Are you sure you want to do it?"""
        else:
            return None

    def run_description(self, context, attribute):
        return f"The values of {attribute.value} are:"

    def run(self, context, attribute):
        return t.LogRender(detail=attribute.value)


class NoneCommand:
    command_name = "None"
    description = "Represents user requests not acknowledged by the bot."
    phrases = [
        "2 hours meeting with ladawn padilla at 5 on tuesday",
        "add coffee on august 12th at 2 pm with herman for one hour",
        "add ultimate frisbee on sunday at 2 pm for 2 hours",
        "add work to calendar friday from 8 for an hour",
        "appending group meeting next monday at 11am for an hour",
        "at 10 am do i have a 30 minutes meeting",
        "create a calendar appointment at 3:30 tomorrow for half an hour",
        "create a event with eden roth at 4pm today for 30 mins",
        "create appointment for 30 minutes",
        "please schedule an all hands meeting for 2 hours",
        "schedule appointment yearly check up friday march one 10 am one hour",
        "schedule lunch with pura at noon on wednesday may 20 minutes"
    ]


class DescribeLogCommand(b.PPIBotCommand):
    command_name = "Describe log"
    parameters = {}
    description = "Get the event log information"
    phrases = [
        "describe log",
        "get log information",
        "log information",
        "describe event log",
        "get event log information",
    ]

    def run_description(self, context):
        return "The principal features of the log are:"

    def run(self, context):
        return t.DescribeLog()


def compare_precondition(context, similarity, **values):
    if isinstance(context, t.AggMetric):
        return False
    else:
        return True


class CompareMetrics(b.PPIBotCommand):
    command_name = "Compare metrics"
    description = "Allows to compare two metrics that have been saved or the current \
    metric with one of the saved metrics. The values of both metrics shall be displayed \
    in a table and a graph or textually, depending on the case. Both metrics must have the \
    same 'group by' and 'periodicity'. Example: 'compare x with y'"
    required_context = None
    output = None
    variables_access = True
    phrases = [
        "compare {@VariableName} with {@VariableName}",
        "compare this with {@VariableName}",
        "compare {@VariableName} with this",
        "compare metrics",
        "I want to compare two metrics"
    ]

    parameters = {
        "name1": b.Parameter(
            question=f"Which is the name of the first metric?",
            fail=f"Sorry, I cannot use that name.",
            param_type=t.Literal,
            entities=["VariableName"],
            match_options={
                "domain": t.Domain("string")
            },
            friendly_name="the name of the first metric"
        ),
        "name2": b.Parameter(
            question=f"Which is the name of the second metric?",
            fail=f"Sorry, I cannot use that name.",
            entities=["VariableName"],
            param_type=t.Literal,
            match_options={
                "domain": t.Domain("string")
            },
            precondition=compare_precondition,
            friendly_name="the name of the second metric"
        )
    }

    def match_entities(self, entities: RecognizedEntities, **args):
        variables = entities.extract_entity("VariableName", all_values=True)
        if variables is None:
            return False

        variables = list(filter(lambda x: "THIS" != x.upper(),
                                map(lambda x: x.value, variables)))

        if len(variables) >= 2:
            self.save("name2", variables[1])
        else:
            self.save("name2", None)

        self.save("name1", variables[0])
        return False

    def run_description(self, context, name1, name2):
        name1 = name1.value if isinstance(name1, t.Literal) else name1
        name2 = name2.value if isinstance(name2, t.Literal) else name2
        if name2 is None:
            return f"Comparing {name1} with the current metric"
        else:
            return f"Comparing metrics {name1} and {name2}"

    async def run(self, context, variables, name1, name2):
        name1 = name1.value if isinstance(name1, t.Literal) else name1
        name2 = name2.value if isinstance(name2, t.Literal) else name2
        compare = t.MetricComparison()
        try:
            v1 = variables.get_variable(name1)
            compare.name1 = name1
            compare.text1 = v1.text
            compare.table1 = v1.table

            if name2 is None:
                v2 = context
                compare.name2 = "current"
            else:
                v2 = variables.get_variable(name2)
                compare.name2 = name2
            compare.text2 = v2.text
            compare.table2 = v2.table

            return compare

        except Exception as e:
            logger.error(e, exc_info=True)
            return b.ErrorType(message="The metrics could not be compared")


class ShowActivitiesCommand(b.PPIBotCommand):
    command_name = "Show activities"
    description = "Displays the name of the activities of the log. Example: 'show activities'"
    required_context = None
    output = None
    warn_size = 90
    phrases = [
        "show activities",
        "show all activities",
        "which are the activities of the log?",
        "show activity"
    ]

    parameters = {}

    def confirm_message(self, tools: b.Tools, context):
        num_values = len(tools.similarity.value_candidates("ACTIVITY"))
        if num_values > self.warn_size:
            return f"""There are {num_values} activities.
                   Showing all of them it is going to take me a while.
                   Are you sure you want to do it?"""
        else:
            return None

    def run_description(self, context):
        return "The name of the activities are:"

    def run(self, context):
        return t.LogRender(detail="ACTIVITY")
