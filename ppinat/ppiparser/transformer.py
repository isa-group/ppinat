from transformers import AutoModelForTokenClassification
from .PPIDecoder import PPIDecoder
from transformers import AutoTokenizer  

def load_transformer(model):
    model = AutoModelForTokenClassification.from_pretrained(model)
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return PPIDecoder(model, tokenizer)