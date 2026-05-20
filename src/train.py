import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer,TrainingArguments,DataCollatorWithPadding,EarlyStoppingCallback
    


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
def tokenized_dataset(datasets,tokenizer,max_length):
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )
    tokenized_dataset = datasets.map(tokenize_batch, batched=True)
    tokenized_dataset = tokenized.remove_columns(["text"]) 
    tokenized_dataset.set_format(type="torch") 
    return tokenized_dataset

def train(cfg):

    model,tokenizer = load_model(cfg)
    dataset=load_dataset(cfg)
    tokenized = tokenized_dataset(dataset, tokenizer, cfg["training"]["max_length"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        learning_rate=cfg["training"]["learning_rate"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        num_train_epochs=cfg["training"]["nums_epochs"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=cfg["training"]["logging_steps"],
        load_best_model_at_end=cfg["training"]["best_model"],
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=cfg["seed"],
    )

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])],
        )

    train_result = trainer.train(resume_from_checkpoint=cfg["resume_from_checkpoint"])
    eval_result = trainer.evaluate()

    trainer.save_model(cfg["training"]["output_dir"])
    tokenizer.save_pretrained(cfg["training"]["output_dir"])