import collections
import ppinat.ppiparser.Tags as Tags
from .Chunk import Chunk

TIME_MEASURE_STR = "time"
COUNT_MEASURE_STR = "count"
FRACTION_MEASURE_STR = "fraction"
UNKNOWN_MEASURE_STR = "unknown_measure_type"

RecognizedParam = collections.namedtuple('RecognizedParam', ['value', 'score'])


class PPIAnnotation:

    def __init__(self, description, measure_type=None):
        self.description = description
        self.measure_type = measure_type
        self.chunks = []
        self.current_chunk = None

    def add_word_tag(self, word, tag):
        if tag != Tags.SAME_CHUNK_TAG:
            self.current_chunk = Chunk(tag)
            self.chunks.append(self.current_chunk)
        if word:
            self.current_chunk.append_word(word)

    def get_text_from_chunks(self):
        return " ".join(self.get_word_sequence()).strip()

    def get_word_sequence(self):
        return [c.text() for c in self.chunks]

    def get_tag_sequence(self):
        return [c.tag for c in self.chunks]

    def has_tag(self, tag):
        return tag in self.get_tag_sequence()

    def get_chunk_by_tag(self, tag):
        for chunk in self.chunks:
            if chunk.tag == tag:
                return chunk
        return None

    def get_measure_type(self):
        if self.measure_type is not None:
            return self.measure_type
        tags = set(self.get_tag_sequence())
        if not tags.isdisjoint(Tags.TIME_TAGS):
            return TIME_MEASURE_STR
        if not tags.isdisjoint(Tags.COUNT_TAGS):
            return COUNT_MEASURE_STR
        if not tags.isdisjoint(Tags.FRACTION_TAGS):
            return FRACTION_MEASURE_STR
        return UNKNOWN_MEASURE_STR

    def get_aggregation_function(self):
        agr_chunk = [self.get_chunk_by_tag(t) for t in Tags.ALL_AGGREGATION_TAGS if self.get_chunk_by_tag(t) is not None]
        if len(agr_chunk) > 0:
            return agr_chunk[0].text()
        return None

    def get_events(self):
        return [chunk.text() for chunk in self.chunks if chunk.tag in Tags.EVENT_TAGS]

    def get_grouping(self):
        gbc_chunk = self.get_chunk_by_tag(Tags.GROUP_BY_TAG)
        if gbc_chunk:
            return gbc_chunk.text()
        return None

    # This function makes PPIAnnotation behave like RecognizedEntities from LUIS
    def extract_entity(self, key):
        result = None
        if key == "metric_type":
            result = RecognizedParam(value=self.get_measure_type(), score=1.0)
        elif key == "aggregation":
            result = RecognizedParam(self.get_aggregation_function(), score=1.0)
        elif key == "grouping":
            result = RecognizedParam(self.get_grouping(), score=1.0)
        elif key == "events":
            result = RecognizedParam(self.get_events(), score=1.0)
        else:
            return None

        return result if (result is not None and result.value is not None) else None

    def __repr__(self):
        return str(self.chunks)


class ViterbiState(PPIAnnotation):

    def __init__(self, cwk, probability, first_chunk=None):
        self.probability = probability
        self.cwk = cwk
        self.chunks = []
        self.chunks.append(Chunk(Tags.START_TAG))
        self.measure_type = None
        if first_chunk:
            self.chunks.append(first_chunk)

    def __copy__(self):
        new_instance = ViterbiState(self.cwk, self.probability)
        new_instance.chunks = []
        new_instance.chunks.extend(self.chunks)
        return new_instance

    def copy_and_extend(self, prob, new_chunk):
        new_instance = self.__copy__()
        new_instance.multiply_probability(prob)
        new_instance.chunks.append(new_chunk)
        return new_instance

    def multiply_probability(self, new_prob):
        self.probability = self.probability * new_prob

    def get_last_chunk(self):
        return self.chunks[-1]

    def remove_last_chunk(self):
        self.chunks.pop()

    def close_annotation(self):
        self.chunks.append(Chunk(Tags.END_TAG))

    def __repr__(self):
        return str(self.chunks) + " prob: " + str(self.probability)


def text_by_tag(ppi_annotation: PPIAnnotation, tag):
    text = None
    if ppi_annotation is not None:
        chunk = ppi_annotation.get_chunk_by_tag(tag) 
        text = chunk.text() if chunk is not None else None

    return text

