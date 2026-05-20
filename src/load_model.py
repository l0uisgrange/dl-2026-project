from transformers import AutoTokenizer, AutoModelForSequenceClassification
def load_model(cfg):
    tokenizer=AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)

    model=AutoModelForSequenceClassification.from_pretrained(
        model_name=cfg["model_name"],
        num_labels=cfg["num_labels"]
    )
    return model,tokenizer