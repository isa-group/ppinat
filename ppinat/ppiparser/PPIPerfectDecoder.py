from .ppiannotation import PPIAnnotation
import ppinat.ppiparser.Tags as Tags


class PPIPerfectDecoder:
    def __init__(self, metrics):
        self.metrics = metrics

    def predict_annotation(self, input_string) -> PPIAnnotation:
        slots, type = [(s["slots"], s["type"])
                       for s in self.metrics if s["description"] == input_string][0]
        annotation = PPIAnnotation(input_string, type)
        annotation.add_word_tag("", Tags.START_TAG)
        for chunk in slots:
            annotation.add_word_tag(slots[chunk], chunk)
        annotation.add_word_tag("", Tags.END_TAG)

        return annotation
