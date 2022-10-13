from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification
from .PPIDecoder import PPIDecoder
from transformers import AutoTokenizer  

def load_transformer(text_model, time_model, count_model, data_model):
    text_model = AutoModelForSequenceClassification.from_pretrained(text_model)
    time_model = AutoModelForTokenClassification.from_pretrained(time_model)
    count_model = AutoModelForTokenClassification.from_pretrained(count_model)
    data_model = AutoModelForTokenClassification.from_pretrained(data_model)
    model = {"time": time_model, "count": count_model, "data": data_model}

    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return PPIDecoder(model, text_model, tokenizer)