from .ppiannotation import PPIAnnotation
import ppinat.ppiparser.Tags as Tags
import ppinat.ppiparser.Tags_list as Tags_list

class PPIDecoder_flant5:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict_annotation(self, input_string) -> PPIAnnotation:
        available_tags = Tags_list.TAGS_LIST['flant5']
        prompt = f"Sentence: {input_string}\nAvailable Tags: {', '.join(available_tags)}"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=1000)
        parsing_result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        annotation = PPIAnnotation(input_string)
        annotation.add_word_tag("", Tags.START_TAG)
        for slot in parsing_result.split('; '):
            try:
                value, tag = slot.split(': ')
                annotation.add_word_tag(value, tag)
            except ValueError:
                pass
        annotation.add_word_tag("", Tags.END_TAG)
        return annotation
