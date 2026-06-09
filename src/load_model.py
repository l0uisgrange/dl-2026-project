from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"]["name"], num_labels=cfg["model"]["num_labels"]
    )
    return model, tokenizer
