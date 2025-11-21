from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline

import torch
import torch.nn.functional as F

MODEL_NAME = "ProsusAI/finbert"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer


def load_classification_model():
    model_classify = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    return model_classify


def load_model_embed():
    model_embed = AutoModel.from_pretrained(MODEL_NAME).to(device)
    return model_embed


tokenizer = load_tokenizer()
model_classification = load_classification_model()
model_embed = load_model_embed()


def load_classification_pipeline():
    # build pipeline
    finbert = pipeline(
        "sentiment-analysis",
        model=model_classification,
        tokenizer=tokenizer,
    )
    return finbert


def enc_device(enc):
    for k, v in enc.items(): 
       enc[k] = v.to(device) 


def embed(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

    enc_device(enc)

    with torch.no_grad():
        out = model_embed(**enc)
        return out.last_hidden_state[:, 0, :]
    
    
def cosine(a, b):
    return F.cosine_similarity(a, b).item()