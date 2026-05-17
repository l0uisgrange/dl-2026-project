"""
code used to evaluate our datasets on the existing model tomh/toxigen_roberta
"""

import os
import glob
import torch
import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

MODEL = "tomh/toxigen_roberta"
DEVICE = 0 if torch.cuda.is_available() else -1


def load_dataset():
    """
    load all hate and neutral datasets from data directory
    """
    files = glob.glob(os.path.join("data/", "*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True).dropna(subset=["text", "mode"])
    df["mode"] = pd.to_numeric(df["mode"], errors="coerce").astype(int)
    df["toxicity_ai"] = pd.to_numeric(df["toxicity_ai"], errors="coerce").fillna(1).astype(int)
    print(
        f"[data] {len(df)} examples | {df["mode"][df["mode"] == 1].sum()} hate | {len(df["mode"][df["mode"] == 0])} neutral")
    return df


def run_inference(df):
    """
    infer MODEL onto the given dataframe
    """
    print(f"[model] loading {MODEL}...")
    classifier = pipeline("text-classification", model=MODEL, device=DEVICE, truncation=True, max_length=512)

    preds, scores = [], []
    for i, text in enumerate(df["text"].tolist()):
        r = classifier(text)[0]
        pred = 1 if r["label"] == "LABEL_1" else 0
        score = r["score"] if pred == 1 else 1 - r["score"]
        preds.append(pred)
        scores.append(score)
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(df)}]")

    df["pred"] = preds
    df["score"] = scores
    return df


def report(df):
    """
    prints classification_report global and for each target_group
    """
    print(classification_report(df["mode"], df["pred"], labels=[0, 1], target_names=["neutral", "hate"], zero_division=0))

    # group specific report
    for group in df["target_group"].unique():
        sub = df[df["target_group"] == group]
        print(f"\n{group} (n={len(sub)}, macro F1={f1_score(sub["mode"], sub.pred, average='macro', zero_division=0):.3f})")
        print(classification_report(sub["mode"], sub["pred"], labels=[0, 1], target_names=["neutral", "hate"], zero_division=0))



def plot(df):
    """
    prepares plot for report and presentation
    """
    groups    = df["target_group"].unique()
    f1_scores = [f1_score(df[df.target_group==g]["mode"], df[df.target_group==g].pred, average="macro", zero_division=0) for g in groups]
    overall   = f1_score(df["mode"], df["pred"], average="macro", zero_division=0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("tomh/toxigen_roberta pre-finetuning")

    axes[0].bar(groups, f1_scores)
    axes[0].axhline(overall, label=f"overall f1 = {overall:.2f}")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("f1 per group")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=15)

    cm = confusion_matrix(df["mode"], df["pred"])
    sns.heatmap(cm, annot=True, fmt="d", ax=axes[1], xticklabels=["neutral", "hate"], yticklabels=["neutral", "hate"])
    axes[1].set_title("confusion matrix")
    axes[1].set_ylabel("true")
    axes[1].set_xlabel("predicted")

    axes[2].hist(df[df["mode"]==1]["score"], bins=20, alpha=0.6, color="red",  label="hate")
    axes[2].hist(df[df["mode"]==0]["score"], bins=20, alpha=0.6, color="blue", label="neutral")
    axes[2].set_title("score distribution")
    axes[2].set_xlabel("toxicity score")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("evaluation.png", dpi=150, bbox_inches="tight")
    print("[done] evaluation.png")


if __name__ == "__main__":
    df = load_dataset()
    df = run_inference(df)
    report(df)
    plot(df)
    df.to_csv("pre-finetuning-dataset.csv", index=False)
