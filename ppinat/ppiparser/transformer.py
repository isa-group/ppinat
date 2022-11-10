from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification
from .PPIDecoder import PPIDecoder
from .PPIPerfectDecoder import PPIPerfectDecoder
from transformers import AutoTokenizer  

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def load_general_transformer(general_classifier_model):
    model = AutoModelForTokenClassification.from_pretrained(general_classifier_model)
    return PPIDecoder(model, tokenizer)

def load_transformer(text_model, time_model, count_model, data_model):
    text_model = AutoModelForSequenceClassification.from_pretrained(text_model)
    time_model = AutoModelForTokenClassification.from_pretrained(time_model)
    count_model = AutoModelForTokenClassification.from_pretrained(count_model)
    data_model = AutoModelForTokenClassification.from_pretrained(data_model)
    model = {"time": time_model, "count": count_model, "data": data_model} 
    return PPIDecoder(model, tokenizer, text_model)

def load_perfect_decoder(metrics):
    return PPIPerfectDecoder(metrics)