import collections
import logging

logger = logging.getLogger()

RecognizedParam = collections.namedtuple('RecognizedParam', ['value', 'score'])

class RecognizedEntities:
    def __init__(self, entities, text=None):
        super().__init__()
        self.entities = entities
        self.text = text

    def extract_entity(self, param_name, sub_level=None, entities=None, all_values=False):
        if entities is None:
            entities = self.entities

        result = None

        if sub_level is not None:
            level = entities.get(sub_level, [])
            if len(level) > 0:
                return self.extract_entity(param_name, entities=level[0], all_values=all_values)
        else:
            # parameter_entities = entities.get("$instance", {}).get(
            #     param_name, []
            # )

            parameter_entities = entities.get(param_name, [])

            if len(parameter_entities) > 0:
                if not all_values:
                    result = self._get_param(parameter_entities[0])
                else:
                    result = list(map(self._get_param, parameter_entities))

        return result
    

    def _get_param(self, parameter_entity):
        score = parameter_entity["score"] if "score" in parameter_entity else 1.0
        return RecognizedParam(value=parameter_entity, score=score)

