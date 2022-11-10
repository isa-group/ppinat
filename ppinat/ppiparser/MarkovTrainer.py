import csv
import json
from . import Tags
from .ppiannotation import PPIAnnotation
from .markovmodel.SemanticModel import SemanticModel
from .markovmodel.LexicalizationModel import LexicalizationModel


def train_semantic_model(training_data):
    model = SemanticModel()
    model.add_state(Tags.START_TAG)
    model.set_initial(Tags.START_TAG)
    for annotation in training_data:
        tags = annotation.get_tag_sequence()
        for i in range(len(tags) - 1):
            model.increment_transition(tags[i], tags[i + 1])
    return model


def train_lexicalization_model(training_data):
    model = LexicalizationModel()
    model.add_tag_state(Tags.START_TAG)
    model.set_initial(Tags.START_TAG)
    for annotation in training_data:
        tags = annotation.get_tag_sequence()
        model.increment_transition(Tags.START_TAG, tags[1])
        for chunk in annotation.chunks:
            model.add_tag_state(chunk.tag)
            model.increment_transition(chunk.tag, chunk.text())
    return model


def read_training_data(filepath):
    result = []
    with open(filepath) as json_file:
        data = json.load(json_file)
        for ppi in data['data']:
            annotation = PPIAnnotation(ppi['description'], ppi['type'])
            annotation.add_word_tag("", Tags.START_TAG)
            for chunk in ppi['annotation']:
                annotation.add_word_tag(chunk['text'], chunk['tag'])
            annotation.add_word_tag("", Tags.END_TAG)
            result.append(annotation)
    return result


# OUTDATED, use json version
def read_training_data_from_csv(filepath):
    result = []
    with open(filepath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            word = row[0]
            tag = row[1]
            # start of new annotation
            if tag == Tags.START_TAG or word == Tags.START_WORD:
                annotation = PPIAnnotation("")
                annotation.add_word_tag("", Tags.START_TAG)
                result.append(annotation)
            # end of annotation
            elif tag == Tags.END_TAG or word == Tags.END_WORD:
                annotation.add_word_tag("", Tags.END_TAG)
            # regular tag
            else:
                annotation.add_word_tag(word, tag)
            annotation.description = " ".join(annotation.get_word_sequence())
    return result


def transform_training_data(filepath_in, filepath_out):
    annotations = read_training_data_from_csv(filepath_in)
    print('input loaded')
    data = [annotation_to_json(annotation) for annotation in annotations]
    result = {"data": data}
    with open(filepath_out, 'w') as outfile:
        json.dump(result, outfile, indent=4)
    print('transformation completed to', filepath_out)


def annotation_to_json(annotation):
    return {"description": annotation.get_text_from_chunks(), "type": annotation.get_measure_type(),
            "annotation": chunks_to_dict(annotation)}


def chunks_to_dict(annotation):
    result = []
    for chunk in annotation.chunks:
        if chunk.tag not in [Tags.START_TAG, Tags.END_TAG]:
            if chunk.tag in Tags.EVENT_TAGS:
                result.append(
                    {"text": chunk.text(), "tag": chunk.tag, "type:": "slot"})
            else:
                result.append({"text": chunk.text(), "tag": chunk.tag})
    return result
