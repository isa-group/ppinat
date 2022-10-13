import ppinat.ppiparser.Tags as Tags
import ppinat.ppiparser.Tags_list as Tags_list

from .ppiannotation import PPIAnnotation

class PPIDecoder:
    def __init__(self, model, text_model, tokenizer):
        self.model = model
        self.text_model = text_model
        self.tokenizer = tokenizer

    def predict_annotation(self, input_string) -> PPIAnnotation:
        
        words = input_string.split()
        tokens  = self.tokenizer(words, return_tensors='pt', truncation=True, is_split_into_words=True)
        
        metric_type = self.text_model(**tokens)["logits"].argmax(-1).tolist()[0]
        if metric_type == 0:
            type = "time"
        elif metric_type == 1:
            type = "count"
        else:
            type = "data"

        predictions = self.model[type](**tokens)["logits"].argmax(-1).tolist()[0][1:-1]

        predictions_decoded = [Tags_list.TAGS_LIST[type][i] for i in predictions]
        predictions_decoded_cleaned = self.clean_prediction_tags(predictions_decoded)

        chunks = self.generate_chunks(words, predictions_decoded_cleaned)

        annotation = PPIAnnotation(input_string, type)
        annotation.add_word_tag("", Tags.START_TAG)
        for chunk in chunks:
            annotation.add_word_tag(chunk[0], chunk[1])
        annotation.add_word_tag("", Tags.END_TAG)

        return annotation

    def clean_prediction_tags(self, predictions_decoded):
        predictions_decoded_cleaned = []
        for prediction in predictions_decoded:
            import re
            regex =  re.search(r'^[BI]-(.*)',prediction)
            if (regex):
                predictions_decoded_cleaned.append(regex.group(1))
            else:
                predictions_decoded_cleaned.append(prediction)            
        return predictions_decoded_cleaned

    def generate_chunks(self, words, predictions_decoded_cleaned):
        chunks = []
        for i in range(len(words)):
            if i == 0:
                chunks.append((words[i], predictions_decoded_cleaned[i]))
            else:
                if predictions_decoded_cleaned[i] == predictions_decoded_cleaned[i-1]:
                    chunks[-1] = (chunks[-1][0] + " " + words[i], chunks[-1][1])
                else:
                    chunks.append((words[i], predictions_decoded_cleaned[i]))
        return chunks
