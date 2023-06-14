from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from .PPIDecoder import PPIDecoder
from .PPIPerfectDecoder import PPIPerfectDecoder
from .PPIDecoder_flant5 import PPIDecoder_flant5

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

def load_transformer_es(text_model, time_model, count_model, data_model):
    tokenizer_es = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
    text_model = AutoModelForSequenceClassification.from_pretrained(text_model)
    time_model = AutoModelForTokenClassification.from_pretrained(time_model)
    count_model = AutoModelForTokenClassification.from_pretrained(count_model)
    data_model = AutoModelForTokenClassification.from_pretrained(data_model)
    model = {"time_es": time_model, "count_es": count_model, "data_es": data_model} 
    return PPIDecoder(model, tokenizer_es, text_model, lang="es")

def load_perfect_decoder(metrics):
    return PPIPerfectDecoder(metrics)

def load_general_transformer_flant5(flant5_model):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained(flant5_model)
    return PPIDecoder_flant5(model, tokenizer)