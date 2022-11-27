import logging
from enum import Enum
from typing import Dict

from ppinat.helpers.computer import MeasureComputer
from ppinat.matcher.similarity import SimilarityComputer
from ppinat.matcher.recognizers import RecognizedEntities
import copy


logger = logging.getLogger(__name__)

class Tools():
    """
        Convenience class that groups the main elements of PPIBot.

        Parameters
        -----------
        similarity : SimilarityComputer
            The class that performs matches between text and the event log

        command_recognizer : CommandRecognizer
            The class that finds the command that matches the text received and
            runs that command.

        measure_computer : MeasureComputer
            The class that computes the metric with the information of the event log.
    """

    def __init__(self, similarity, command_recognizer, measure_computer):
        self.similarity: SimilarityComputer = similarity
        self.command_recognizer: CommandRecognizer = command_recognizer
        self.measure_computer: MeasureComputer = measure_computer


class ValueOf():
    """
        ValueOf is used when specifying the match_options dict of a parameter to
        assign the value of a previous parameter to a key of the match_options.
        This match_options dict will be sent to the functions performed in the parameter
        like hints, match or context.        

        Parameters
        -----------
        value : str
            The name of the parameter
    """

    def __init__(self, value):
        self.value = value


class ValueOfContext():
    """
        ValueOfContext is used when specifying the match_options dict of a parameter to
        assign the value of the context to a key of the match_options.
        This match_options dict will be sent to the functions performed in the parameter
        like hints, match or context.        
    """
    pass


class Parameter():
    """
        Parameter is the class used to configure the parameters of a command. It is not
        intended to be extended, but just used as is as a container of configuration of 
        the command. It also has several methods that perform actions mainly related
        to the management of the match_options dict.

        Parameters
        ----------
        question : str
            The text that will be sent to ask the user the value of this parameter.

        fail : str
            The text that will be sent to the user if the chatbot is not able to 
            understand the user utterance.

        param_type : PPIBotType
            Class that extends PPIBotType that represents the type of the parameter.
            This class is the one that contains the text matcher.

        hints : function that returns str (default None)
            This function must return a string with hints about how to fill this
            parameter that will be sent to the user together with the question.
            The function receives a first parameter with the SimilarityComputer and 
            the keys of match_options if any.

        options : list (default None)
            A list with the options that are offered to the user from which she can 
            choose a value.

        confirm : boolean (default False)
            If True, the bot will ask the user for a confirmation about what the bot 
            has understood.

        use_command : PPIBotCommand (default None)
            If this parameter is not None, the command will be used to take over the
            conversation. The result of the command will be assigned to the parameter.

        match_options : dict (default None)
            This is a dict that will be sent to the functions performed in the 
            parameter like hints, match or precondition. You can use ValueOf("a")
            and ValueOfContext() to refer to the values of a previous parameter or
            to the value of the context, respectively.

        precondition : function that returns boolean (default None)
            This function must return a boolean that specifies if the parameter is 
            mandatory (True) or it can be ignored (False). The function receives a 
            first parameter with the context and the keys of match_options if any.

        context : callable or any (default None)
            If context is callable, the function is called with the SimilarityComputer
            and the keys of match_options if any. The result is used as the context
            of the command used to fill the parameter. If context is any other value
            this value will be directly used as context.

        entities : list (default None)
            A list of the entities with which this parameter can be matched. If the first
            element of the list is not present in the entities, it will try the next one.

        load_from_context : callable or boolean (default None)
            This function receives a parameter with the context and must return a boolean or
            it can be a boolean itself. A value of True means that this parameter will 
            automatically take the value of the context if there is no valid entity for 
            this parameter and the context is of the right type. Otherwise it will do 
            nothing. The value will be assigned during the match_entities phase of the 
            command. This means that it will not be loaded if the command overwrites 
            the match_entities function.
    """

    def __init__(
        self,
        question,
        fail,
        param_type,
        hints=None,
        options=None,
        confirm=False,
        use_command=None,
        match_options=None,
        precondition=None,
        context=None,
        entities=None,
        friendly_name=None,
        load_from_context=False
    ):
        self.question = question
        self.fail = fail
        self.param_type = param_type
        self._hints = hints
        self.options = options
        self.confirm = confirm
        self.use_command = use_command
        self.match_options = match_options
        self.precondition = precondition
        self.context = context
        self.entities = entities if entities is not None else []
        self.friendly_name = friendly_name
        self.load_from_context = load_from_context

    ENTITY_THRESHOLD = 0.5

    def hints(self, similarity, context, values):
        if self._hints is None:
            hints_text = None
        else:
            options = self._replace_match_options(context, values)
            hints_text = f"{self._hints(similarity, **options)}"

        return hints_text

    def _replace_match_options(self, context, values):
        if self.match_options is None:
            return {}
        else:
            return {k: ((None if v.value not in values else values[v.value]) if isinstance(v, ValueOf) else (context if isinstance(v, ValueOfContext) else v)) for (k, v) in self.match_options.items()}

    def match(self, value, similarity, context, values):
        if self.match_options is None:
            match = self.param_type.match(value, similarity)
        else:
            options = self._replace_match_options(context, values)
            match = self.param_type.match(value, similarity, **options)

        return match

    def resolve_context(self, similarity, context, values):
        if self.context is None:
            return None
        elif callable(self.context):
            options = self._replace_match_options(context, values)
            return self.context(similarity, **options)
        else:
            return self.context

    def _get_entity_value(self, entity, entities): 
        value = None
        recognized_entity = entities.extract_entity(entity)
        if recognized_entity is not None and recognized_entity.score > self.ENTITY_THRESHOLD:
            value = recognized_entity.value

        return value

    def _extract_entity_in_entities(self, entities):
        value = None
        for entity in self.entities:
            value = self._get_entity_value(entity, entities)
            if value is not None:
                break
        return value

    def _extract_entity_in_type(self, entities):
        value = None
        if hasattr(self.param_type, "entity_name"):
            value = self._get_entity_value(self.param_type.entity_name, entities)
        return value

    def extract_entity(self, entities):
        value = self._extract_entity_in_entities(entities)
        
        if value is None:
            value = self._extract_entity_in_type(entities)

        return value


    def evaluate_precondition(self, context, similarity, values):
        if self.precondition is None:
            return True
        else:
            options = self._replace_match_options(context, values)
            return self.precondition(context, similarity, **options)
            
    def evaluate_load_from_context(self, context):
        if self.load_from_context is None:
            return False
        elif callable(self.load_from_context):
            return self.load_from_context(context)
        else:
            return self.load_from_context

    def __str__(self):
        return f"{self.param_type} [{self.question}]"


class PPIBotCommand():
    confirm = False
    required_context = None
    output = None
    variables_access = False
    parameters: Dict[str, Parameter] = {}

    def __init__(self, context=None, expected_output=None, situation=None, intent=None):
        self.context = context
        self.expected_output = expected_output
        self.situation = situation if situation is not None else []
        self.intent = intent
        self.values = {}
        self.partials = {}
        self.alt_match_a = {}
        self.alt_match_b = {}
        self.unknown = {}

    def load_context(self, context):
        # TODO: The returned value of context is not taken into account yet
        if self.required_context is not None and not isinstance(context, self.required_context):
            logger.warning("Context for command not valid")
            return False
        else:
            self.context = context
            return True

    def save_or_unknown(self, param_name, value, text, save_alternatives=False):
        if value is None or value == ([],[]):
            self.add_unknown_parameters(param_name, text)
        else:
            self.unknown.pop(param_name, None)
            self.save(param_name, value, save_alternatives)

    def save(self, param_name, value, save_alternatives=False):
        self.alt_match_a[param_name] = []
        self.alt_match_b[param_name] = []
        if save_alternatives:
            if len(set([str(v.value) for v in value[0]])) == 1:
                self.values.update({param_name: value[0][0]})
            else:
                for alt in value[0]:
                    if str(alt.value) not in [str(a.value) for a in self.alt_match_a[param_name]]:
                        self.alt_match_a[param_name].append(alt)
            for alt in value[1]:
                if str(alt.value) not in [str(b.value) for b in self.alt_match_b[param_name]]:
                    self.alt_match_b[param_name].append(alt)
        else:
            self.values.update({param_name: value})

    def save_partial(self, param_name, command_type, context, entities, similarity, heuristics=True):
        command = resolve_command(command_type, context, entities, similarity, expected=self.parameters[param_name].param_type, heuristics=heuristics)
        self.partials.update({param_name: (command, context)})
        return (command, context)

    def get(self, param_name, get_alternatives=False):
        result = self.values[param_name] if param_name in self.values else None
        if not get_alternatives or result is not None:
            return result
        else:            
            found_values = self.alt_match_a[param_name] if param_name in self.alt_match_a else [] + \
                self.alt_match_b[param_name] if param_name in self.alt_match_b else [
            ]

            return found_values[0] if len(found_values) > 0 else None

    def add_unknown_parameters(self, param_name, value):
        if type(value) is str:
            self.unknown[param_name] = value

        
    def is_unknown(self, param_name):
        return (param_name in self.unknown) and (param_name not in self.values)

    def match(self, param_name, value, similarity: SimilarityComputer):
        if value is not None:
            matched = self.parameters[param_name].match(
                value, similarity, self.context, self.values)
            if matched is None or matched == ([], []):
                self.add_unknown_parameters(param_name, value)
            else:
                self.unknown.pop(param_name, None)
                if isinstance(matched, tuple):
                    self.save(param_name, matched, True)
                else:
                    self.save(param_name, matched)

        return self.get(param_name) is not None

    def _match_entity_to_parameter(self, param_name, entities: RecognizedEntities, similarity: SimilarityComputer, context):
        matched = False
        parameter = self.parameters[param_name]
        
        entity = parameter.extract_entity(entities)
        # Whether the entity has been recognised
        if entity is not None:
            matched = self.match(param_name, entity, similarity)

        if not matched and parameter.evaluate_load_from_context(context) and isinstance(context, parameter.param_type):
            self.save(param_name, context)
            matched = True

        return matched

    def match_entities(self, entities: RecognizedEntities, similarity: SimilarityComputer, context=None, **args):
        matched = True

        for param_name in self.parameters:
            matched = matched and self._match_entity_to_parameter(param_name, entities, similarity, context)

        return matched

    @classmethod
    def match_intent_condition(cls, intent, entities):
        match = True
        if hasattr(cls, "intent_filter"):
            for key in cls.intent_filter:
                entity_key = entities.extract_entity(key)
                if entity_key is None: 
                    if cls.intent_filter[key] is not None:
                        match = False
                        break
                elif cls.intent_filter[key] != entity_key.value:
                    match = False
                    break

        return match

    @staticmethod
    def help_filter(situation, variables):
        return True

    def _unmatched_parameters(self):
        return [p for p in self.parameters if p not in self.values]

    def next_unmatched_parameter(self, similarity: SimilarityComputer = None):
        unmatched_parameters = self._unmatched_parameters()
        for candidate in unmatched_parameters:
            if self.parameters[candidate].evaluate_precondition(self.context, similarity, self.values):
                return candidate

        return None
        # return unmatched_parameters[0] if len(unmatched_parameters) > 0 else None

    async def run_command(self, variables_accesor=None, turn_context=None):
        if not self.variables_access:
            return self.run(self.context, **self.values)
        else:
            variables = await variables_accesor.get(turn_context, Variables)

            result = await self.run(self.context, variables, **self.values)

            await variables_accesor.set(turn_context, variables)

            return result

    def run_command_description(self):
        return self.run_description(self.context, **self.values)

    def needs_confirmation(self):
        return self.confirm or hasattr(self, 'confirm_message')

    def confirm_message_command(self, tools):
        message = None

        if hasattr(self, 'confirm_message'):
            message = self.confirm_message(tools, self.context, **self.values)
        elif self.confirm:
            message = f"{self.run_command_description()}, is that correct?"

        return message

    def reset(self):
        self.values = {}
        self.partials = {}
        self.alt_match_a = {}

    def conditional_alernatives_filter(self, param_name, alt_parsed):
        return True

    def __str__(self):
        return f"{self.command_name} {self.context} [{self.values}]"

class Variables():
    def __init__(self):
        self.content = {}

    def set_variable(self, name, value):
        self.content[name] = copy.deepcopy(value)

    def get_variable(self, name):
        return copy.deepcopy(self.content[name])

    def names(self):
        return list(self.content.keys())


class ExitCommand(PPIBotCommand):
    command_name = "Cancel"
    parameters = {}
    description = "Cancels the current running action. Example: 'cancel'"
    phrases = [
        "cancel",
        "exit",
        "i don't want to do this",
        "no, i want to do something else",
        "quit"
    ]

    @staticmethod
    def help_filter(situation, variables):
        return DialogSituation.MAIN not in situation

    def run_description(self, context):
        return None

    def run(self, context):
        return ExitType()


class HelpCommand(PPIBotCommand):
    command_name = "Help"
    parameters = {}
    description = "Displays the bot's current suggestions. Example: 'help'"
    variables_access = True
    phrases = [
        "help",
        "help me",
        "what can i do?",
        "what can i say?",
        "which are my options?"
    ]

    @staticmethod
    def help_filter(situation, variables):
        return DialogSituation.HELP_COMMAND not in situation

    def run_description(self, context):
        return None

    async def run(self, context, variables):
        situation = self.situation
        situation.append(DialogSituation.HELP_COMMAND)
        return HelpType(context=self.context, expected=self.expected_output, situation=situation, variables=variables)
class PPIBotType():
    threshold_a = 0.2
    threshold_b = 0.2

class UndoCommand(PPIBotCommand):
    command_name = "Undo"
    description = "Undo the last command. Example: 'undo'"
    required_context = PPIBotType
    output = None
    phrases = [
        "undo",
        "back"
    ]

    parameters = {}

    def run_description(self, context):
        return None

    def run(self, context):
        return Undo(context)

class RenderType(PPIBotType):
    def render(self, tools):
        return str(self)


class ExitType(RenderType):
    def render(self, tools):
        return "Ok"


class ErrorType(RenderType):
    def __init__(self, message):
        self.message = message

    def render(self, tools):
        return self.message


class HelpType(RenderType):
    def __init__(self, context=None, expected=None, situation=None, variables=None):
        self.context = context
        self.expected = expected
        self.situation = situation if situation is not None else []
        self.variables = variables

    def _format(self, command_list):
        if command_list is not None:
            return ", ".join(map(self._format_command, filter(lambda x: x.help_filter(self.situation, self.variables), command_list)))
        else:
            return ""

    def _format_command(self, command):
        return f"[{command.command_name}]({command.description})"

    def _format_type(self, type_def):
        if hasattr(type_def, "description"):
            return type_def.description
        else:
            return str(type_def)

    def render(self, tools):

        if self.expected is not None:
            commands_expected = commands_list(
                output_expected=self.expected, include_output_none=False, context=self.context, include_context_none=True)
            commands_output_none = commands_list(
                output_expected=None, context=self.context, include_context_none=True)
            text = f"You need to provide a {self._format_type(self.expected)}."

            if len(commands_expected) > 0:
                text = text + \
                    f" You can do that using {self._format(commands_expected)}."
                if len(commands_output_none) > 0:
                    text = text + \
                        f" You can also use these commands to get additional information: {self._format(commands_output_none)}"
            else:
                if len(commands_output_none) > 0:
                    text = text + \
                        f" You can use these commands to get additional information: {self._format(commands_output_none)}"
        else:
            commands_context = commands_list(output_expected=RenderType, include_output_none=True,
                                             context=self.context, include_context_none=False) if self.context is not None else []
            commands_none = commands_list(
                output_expected=RenderType, include_output_none=True, context=None)
            if len(commands_context) > 0:
                text = f"Some of the commands you can use are: {self._format(commands_context)}."
                if len(commands_none) > 0:
                    text = text + \
                        f" You can also use {self._format(commands_none)}"
            else:
                text = f"Some of the commands you can use are: {self._format(commands_none)}."
        
        text += " (move the pointer over each of them for more details)"
        return text

class Undo(RenderType):
    def __init__(self, context):
        self.context = context

    def render(self,context):
        text = f"Undoing the last action"
        return text

def _all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in _all_subclasses(c)])


def all_commands():
    return _all_subclasses(PPIBotCommand)


def all_types():
    return _all_subclasses(PPIBotType)


def commands_list(output_expected='none', include_output_none=True, context='none', include_context_none=False):
    """
    Returns the set of commands that meet certain conditions
    
    The conditions use to determine the set of commands are based on the values
    of two attributes of the command: 'output', which is the type that the
    command returns that will be saved into the context (note that the command
    can return other types only to be rendered and in that case output will
    be None), and 'required_context', which is the type that the command 
    requires to be executed. 

    Parameters
    ----------
    output_expected : 'none' | PPIBotType | None (default 'none')        
    include_output_none : boolean (default True)
    context : 'none' | PPIBotType | None (default 'none')
    include_context_none : boolean (default False)
    """
    commands = {c for c in all_commands() if hasattr(c, "command_name")}

    if output_expected == 'none' and context == 'none':
        return commands

    if output_expected != 'none':
        if output_expected is None:
            output_filter = {c for c in commands if c.output is None}
        else:
            output_filter = {c for c in commands if c.output is not None and issubclass(
                c.output, output_expected)}
            if include_output_none:
                output_filter = output_filter.union(
                    {c for c in commands if c.output is None})
    else:
        output_filter = set(commands)

    if context != 'none':
        if context is None:
            context_filter = {
                c for c in commands if c.required_context is None}
        else:
            context_filter = {c for c in commands if c.required_context is not None and isinstance(
                context, c.required_context)}
            if include_context_none:
                context_filter = context_filter.union(
                    {c for c in commands if c.required_context is None})
    else:
        context_filter = set(commands)

    return output_filter.intersection(context_filter)

def is_valid_in_main(command):
    """
        Returns if the command is valid to be used in the main dialog

        In the main dialog, we do not allow commands with output that are not 
        renderizable, i.e., we only allow commands either with no output or whose
        output is renderizable.
    """
    return command.output is None or issubclass(command.output, RenderType)    

def is_type_renderizable(type):
    """ 
        Returns if the given type is renderizable

        We assume a type is renderizable if it has a render method.
    """
    return callable(getattr(type, "render", None))

def _command_does_not_match_expected(command, expected):
    command_output = command.output
    return expected is not None and command_output is not None and not issubclass(command_output, expected)

def command_matches_expected(command, expected):
    command_output = command.output    
    return expected is None or (issubclass(command_output, expected) if command_output is not None else False)

def _command_does_not_match_context(command, context):
    required_context = command.required_context
    return required_context is not None and (context is None or not isinstance(context, required_context))

def _commands_for_intent(intent, entities) -> PPIBotCommand:
    potential_commands = [c for c in all_commands() if (hasattr(c, "intent_names") and intent in c.intent_names) or 
                                                       (hasattr(c, "command_name") and intent == c.command_name)]
    return list(filter(lambda c: c.match_intent_condition(intent, entities), potential_commands))

class DialogSituation(Enum):
    MAIN = 0,
    COMMAND = 1,
    PARAM = 2,
    HELP_COMMAND = 3

def resolve_command(
    intent, 
    context, 
    entities: RecognizedEntities = None, 
    similarity: SimilarityComputer = None, 
    expected = None,
    situation: list = None,
    heuristics = True
) -> PPIBotCommand:
    """
    Resolves the command that applies given an intent and a context.

    It instantiates the command that applies given an intent and a context (including the
    expected output and the situation of the dialog). It also performes the matching
    of the parameters of the command with the entities provided. To do so, a 
    SimilarityComputer is used.

    The intent can also be directly a PPIBotCommand class. In that case, it is directly
    instantiated.

    Parameters
    ----------
    intent : str | PPIBotCommand class
        The intent that will be used to choose the command (or the command class)
    context: PPIBotType | None
        The current context of the conversation
    entities : RecognizedEntities (default None)
        The entities that have been recognized. They may be used to choose the 
        command and to fill the parameters.
    similarity : SimilarityComputer (default None)
        The SimilarityComputer used to match the entities
    expected : PPIBotType | None (default None)
        The expected output that the command must provide. 
    situation: list | None (default None)
        The situation of the current dialog as a list. Values are of type DialogSituation.
    """
    if entities is not None and similarity is None:
        raise ValueError("if entities is not None, similarity must be provided")

    if intent is None:
        return None

    if isinstance(intent, str):
        command_cls = _commands_for_intent(intent, entities)
    elif issubclass(intent, PPIBotCommand):
        # This is used when intent is actually a command because of the use_command option
        command_cls = [intent]
    else:
        command_cls = []

    command = None

    for c in command_cls:
        if _command_does_not_match_expected(c, expected):
            logger.warn(f"command choosen {c.command_name} does not match {expected}")
        elif _command_does_not_match_context(c, context):
            logger.warn(f"command choosen {c.command_name} requires a context different than {context}")
        else:
            command = c(context, expected, situation, intent=intent)
            break

    if command is not None:
        if entities is not None:
            command.match_entities(entities, similarity, context=context, heuristics=heuristics)
        
        logger.debug(f"choosen {command.command_name} with {entities}")
    
    return command

def update_context(command, command_result, current_context):
    """
    Determines which is the new context after the execution of a command

    The change of the context is the command_result if it matches the type
    of the command output. Otherwise, it remains the same current context.

    Parameters:
    -----------
    command : PPIBotCommand
        The command executed
    command_result : PPIBotType
        The output of the command executed
    current_context : Any
        The current context that will be returned if the context does
        not change.
    """
    new_context = current_context

    if command is not None and command.output is not None:
        if isinstance(command_result, ErrorType):
            logger.error(
                f"Error returned by command: {command_result.message}")
        elif isinstance(command_result, ExitType):
            pass
        elif isinstance(command_result, command.output):
            new_context = command_result
        else:
            logger.warning(
                f"Type returned by command {command_result} does not match its output {command.output}")

    return new_context


class CommandRecognizer():
    INTENT_THRESHOLD = 0.25

    def __init__(self, recognizer, decoder):
        self.recognizer = recognizer
        self.decoder = decoder

    async def recognize(self, step_context, context, expected = None):
        intent, entities = None, None

        if hasattr(context, 'dynamic_lists'):
            dynamic_lists = context.dynamic_lists
        else:
            dynamic_lists = None
            
        recognizer_result = await self.recognizer.recognize(step_context.context, dynamic_lists=dynamic_lists)

        if recognizer_result is not None:
            intent, score = recognizer_result.get_top_scoring_intent()
            
            # Threshold is only used for a warning, but it tries to go with the top intent anyway.
            if score < self.INTENT_THRESHOLD:                
                logger.warn(f"top intent {intent} with a score {score} < {self.INTENT_THRESHOLD}")

            entities = self.recognizer.extract_entities(recognizer_result)
            logger.info(f"recognized intent: {intent} with score {score} and params {entities}")

        # if intent is None and step_context.result is not None:
        #     logger.warn(f"Intent not recognized. Falling back to decoder")
        #     intent, entities = self._decoder_fallback(step_context, context, expected)

        return intent, entities

    def _decoder_fallback(self, step_context, context, expected):
        intent = None
        result = self.decoder.predict_annotation(step_context.result)
        if result is not None:
            logger.debug(f"Decoder Found PPI: ${result}")
            
            # if expected is None:
            #     intent = "Compute metric"
            # else:
            intent = "Metric"

        return intent, result