"""
bench.py — Evaluate both fine-tuned models on the Swiss minority dataset
and generate rich visualisation plots.

Usage: python -m src.bench
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS = {
    "ToxiGen (baseline)": "./outputs/bert_toxigen",
    "ToxiGen+ (swiss)": "./outputs/bert_toxigen_plus",
}
MODEL_KEYS = list(MODELS.keys())

DATA_PATH_SWISS_VAL = "./data/dataset_toxigen_plus_val.csv"
DATA_PATH_TOXIGEN_VAL = "./data/dataset_toxigen_val.csv"
SWISS_GROUPS = {"asylum_seekers", "cross_border_workers", "portuguese"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 256
BATCH_SIZE = 32
OUT_DIR = "./assets"

os.makedirs(OUT_DIR, exist_ok=True)

GROUP_PALETTE = {
    "asylum_seekers": "#4C72B0",
    "cross_border_workers": "#DD8452",
    "portuguese": "#55A868",
    "muslim": "#C44E52",
    "asian": "#8172B2",
    "chinese": "#937860",
    "mexican": "#DA8BC3",
    "women": "#8C8C8C",
    "middle_east": "#CCB974",
    "latino": "#64B5CD",
    "native_american": "#E377C2",
    "black": "#7F7F7F",
    "lgbtq": "#BCBD22",
    "jewish": "#17BECF",
    "mental_dis": "#AEC7E8",
    "physical_dis": "#FFBB78",
}


# ── data ──────────────────────────────────────────────────────────────────────


def load_data():
    # Swiss groups: val split of toxigen_plus (unseen during training)
    swiss = pd.read_csv(DATA_PATH_SWISS_VAL).dropna(subset=["text", "labels"])
    swiss["labels"] = pd.to_numeric(swiss["labels"], errors="coerce").astype(int)
    swiss = swiss[swiss["target_group"].isin(SWISS_GROUPS)].reset_index(drop=True)

    # ToxiGen groups: val split of toxigen (unseen during training)
    toxigen_val = pd.read_csv(DATA_PATH_TOXIGEN_VAL).dropna(subset=["text", "labels"])
    toxigen_val["labels"] = pd.to_numeric(toxigen_val["labels"], errors="coerce").astype(int)

    df = pd.concat([swiss, toxigen_val], ignore_index=True)
    df = df.dropna(subset=["target_group"]).reset_index(drop=True)
    print(
        f"[data] {len(df)} examples  hate={df['labels'].sum()}  neutral={(df['labels']==0).sum()}"
    )
    print(f"       groups: {sorted(df['target_group'].unique())}")
    return df


def swiss_only(df):
    return df[df["target_group"].isin(SWISS_GROUPS)].reset_index(drop=True)


# ── inference ─────────────────────────────────────────────────────────────────


def run_inference(df, model_dir):
    tokenizer = AutoTokenizer.from_pretrained("tomh/toxigen_roberta", use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained("tomh/toxigen_roberta")
    model.load_state_dict(load_safetensors(f"{model_dir}/model.safetensors", device="cpu"))
    model = model.to(DEVICE)
    model.eval()

    preds, scores = [], []
    texts = df["text"].tolist()
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        preds.extend(torch.argmax(probs, dim=-1).cpu().tolist())
        scores.extend(probs[:, 1].cpu().tolist())

    out = df.copy()
    out["pred"] = preds
    out["hate_score"] = scores
    # certainty = max(p_hate, p_neutral)
    out["certainty"] = out["hate_score"].apply(lambda s: max(s, 1 - s))
    return out


# ── per-model plots (Swiss groups only) ───────────────────────────────────────


def plot_model(df_swiss, model_name, tag):
    groups = sorted(df_swiss["target_group"].unique())
    colors = [GROUP_PALETTE.get(g, "#999") for g in groups]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(model_name, fontsize=14, fontweight="bold", y=1.01)

    # 1. Macro F1 / Precision / Recall per Swiss group
    ax = axes[0, 0]
    metrics = {
        "Macro F1": [
            f1_score(
                df_swiss[df_swiss.target_group == g]["labels"],
                df_swiss[df_swiss.target_group == g]["pred"],
                average="macro",
                zero_division=0,
            )
            for g in groups
        ],
        "Precision": [
            precision_score(
                df_swiss[df_swiss.target_group == g]["labels"],
                df_swiss[df_swiss.target_group == g]["pred"],
                average="macro",
                zero_division=0,
            )
            for g in groups
        ],
        "Recall": [
            recall_score(
                df_swiss[df_swiss.target_group == g]["labels"],
                df_swiss[df_swiss.target_group == g]["pred"],
                average="macro",
                zero_division=0,
            )
            for g in groups
        ],
    }
    x, width = np.arange(len(groups)), 0.25
    for k, (label, vals) in enumerate(metrics.items()):
        ax.bar(x + k * width, vals, width, label=label)
    ax.set_xticks(x + width)
    ax.set_xticklabels([g.replace("_", " ") for g in groups], rotation=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Macro F1 / Precision / Recall (Swiss groups)")
    ax.legend(fontsize=8)
    ax.set_ylabel("Score")

    # 2. Confusion matrix
    ax = axes[0, 1]
    cm = confusion_matrix(df_swiss["labels"], df_swiss["pred"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        ax=ax,
        xticklabels=["neutral", "hate"],
        yticklabels=["neutral", "hate"],
        cmap="Blues",
    )
    ax.set_title("Confusion matrix (Swiss groups)")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")

    # 3. Hate-score distribution by true label
    ax = axes[1, 0]
    ax.hist(
        df_swiss[df_swiss["labels"] == 0]["hate_score"],
        bins=30,
        alpha=0.6,
        color="#4C72B0",
        label="neutral (true)",
    )
    ax.hist(
        df_swiss[df_swiss["labels"] == 1]["hate_score"],
        bins=30,
        alpha=0.6,
        color="#C44E52",
        label="hate (true)",
    )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="threshold 0.5")
    ax.set_title("Hate-score distribution by true label")
    ax.set_xlabel("P(hate)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # 4. ROC curve per Swiss group
    ax = axes[1, 1]
    for g, c in zip(groups, colors):
        sub = df_swiss[df_swiss["target_group"] == g]
        if sub["labels"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(sub["labels"], sub["hate_score"])
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{g.replace('_',' ')} (AUC={auc(fpr,tpr):.2f})")
    fpr_all, tpr_all, _ = roc_curve(df_swiss["labels"], df_swiss["hate_score"])
    ax.plot(fpr_all, tpr_all, "k--", lw=1.5, label=f"overall (AUC={auc(fpr_all,tpr_all):.2f})")
    ax.plot([0, 1], [0, 1], "gray", linestyle=":", lw=1)
    ax.set_title("ROC curve per Swiss group")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"bench_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


# ── all-16-groups comparison ───────────────────────────────────────────────────


def plot_all_groups_comparison(results_full):
    """Macro F1 on all 16 groups for both models, sorted by baseline F1."""
    all_groups = sorted(results_full[MODEL_KEYS[0]]["target_group"].unique())

    baseline_f1s = [
        f1_score(
            results_full[MODEL_KEYS[0]][results_full[MODEL_KEYS[0]].target_group == g]["labels"],
            results_full[MODEL_KEYS[0]][results_full[MODEL_KEYS[0]].target_group == g]["pred"],
            average="macro",
            zero_division=0,
        )
        for g in all_groups
    ]
    order = np.argsort(baseline_f1s)
    groups_sorted = [all_groups[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 6))
    x, width, colors = np.arange(len(groups_sorted)), 0.35, ["#4C72B0", "#DD8452"]

    for k, (name, df) in enumerate(results_full.items()):
        f1s = [
            f1_score(
                df[df.target_group == g]["labels"],
                df[df.target_group == g]["pred"],
                average="macro",
                zero_division=0,
            )
            for g in groups_sorted
        ]
        bars = ax.bar(x + k * width, f1s, width, label=name, color=colors[k], alpha=0.85)
        for bar, v in zip(bars, f1s):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
            )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(
        [g.replace("_", " ") for g in groups_sorted], rotation=35, ha="right", fontsize=9
    )
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 across all 16 groups: baseline vs. ToxiGen+")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "bench_all_groups.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


# ── certainty / confidence plots ───────────────────────────────────────────────


def plot_certainty(results_full):
    """
    2 subplots per model (side by side):
      - Certainty distribution split by correct / incorrect prediction
      - Calibration curve (reliability diagram)
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Classification certainty & calibration", fontsize=13, fontweight="bold")

    for col, (name, df) in enumerate(results_full.items()):
        correct = df["pred"] == df["labels"]
        incorrect = ~correct

        # ── certainty distribution: correct vs wrong
        ax = axes[0, col]
        ax.hist(
            df[correct]["certainty"],
            bins=30,
            alpha=0.65,
            color="#55A868",
            label=f"correct (n={correct.sum()})",
        )
        ax.hist(
            df[incorrect]["certainty"],
            bins=30,
            alpha=0.65,
            color="#C44E52",
            label=f"wrong   (n={incorrect.sum()})",
        )
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{name}\nCertainty: correct vs. wrong")
        ax.set_xlabel("Certainty  max(P(hate), P(neutral))")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

        # ── calibration curve
        ax = axes[1, col]
        prob_true, prob_pred = calibration_curve(
            df["labels"], df["hate_score"], n_bins=10, strategy="uniform"
        )
        ax.plot(prob_pred, prob_true, "o-", color="#4C72B0", lw=2, label="model")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.set_title(f"{name}\nCalibration curve (reliability diagram)")
        ax.set_xlabel("Mean predicted P(hate)")
        ax.set_ylabel("Fraction of hate samples")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "bench_certainty.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


# ── error analysis: confidence of misclassified samples ───────────────────────


def plot_error_analysis(results_full):
    """
    For each model: scatter of hate_score vs true label, coloured by correct/wrong,
    + bar chart of error rate per confidence bucket.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Error analysis by confidence", fontsize=13, fontweight="bold")

    bins = np.linspace(0, 1, 11)
    labels_bins = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins) - 1)]

    for col, (name, df) in enumerate(results_full.items()):
        correct = (df["pred"] == df["labels"]).astype(int)

        # scatter: hate_score vs true label
        ax = axes[0, col]
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(df))
        sc = ax.scatter(
            df["hate_score"],
            df["labels"] + jitter,
            c=correct,
            cmap="RdYlGn",
            alpha=0.25,
            s=6,
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"{name}\nPrediction confidence vs. true label")
        ax.set_xlabel("P(hate)")
        ax.set_ylabel("True label (jittered)")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["neutral", "hate"])
        plt.colorbar(sc, ax=ax, label="correct")

        # error rate per confidence bucket
        ax = axes[1, col]
        df2 = df.copy()
        df2["bucket"] = pd.cut(
            df2["hate_score"], bins=bins, labels=labels_bins, include_lowest=True
        )
        err_rate = df2.groupby("bucket", observed=True).apply(
            lambda x: (x["pred"] != x["labels"]).mean()
        )
        ax.bar(range(len(err_rate)), err_rate.values, color="#C44E52", alpha=0.8)
        ax.set_xticks(range(len(err_rate)))
        ax.set_xticklabels(err_rate.index, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Error rate")
        ax.set_title(f"{name}\nError rate per P(hate) bucket")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "bench_errors.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


# ── comparison plot (Swiss groups only) ───────────────────────────────────────


def plot_comparison(results_swiss):
    groups = sorted(results_swiss[MODEL_KEYS[0]]["target_group"].unique())
    x, width, colors = np.arange(len(groups)), 0.35, ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (name, df) in enumerate(results_swiss.items()):
        f1s = [
            f1_score(
                df[df.target_group == g]["labels"],
                df[df.target_group == g]["pred"],
                average="macro",
                zero_division=0,
            )
            for g in groups
        ]
        bars = ax.bar(x + k * width, f1s, width, label=name, color=colors[k], alpha=0.85)
        for bar, v in zip(bars, f1s):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([g.replace("_", " ") for g in groups], rotation=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 per minority group: baseline vs. ToxiGen+")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "bench_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


# ── main ───────────────────────────────────────────────────────────────────────


def main():
    df_full = load_data()
    results_full = {}
    results_swiss = {}

    for name, model_dir in MODELS.items():
        print(f"\n[model] {name}  ({model_dir})")
        result_full = run_inference(df_full, model_dir)
        results_full[name] = result_full
        results_swiss[name] = swiss_only(result_full)

        tag = "baseline" if "baseline" in name.lower() else "swiss"
        print(
            classification_report(
                results_swiss[name]["labels"],
                results_swiss[name]["pred"],
                labels=[0, 1],
                target_names=["neutral", "hate"],
                zero_division=0,
            )
        )
        plot_model(results_swiss[name], name, tag)

    plot_comparison(results_swiss)
    plot_all_groups_comparison(results_full)
    plot_certainty(results_full)
    plot_error_analysis(results_full)
    print("\n[done] all plots saved to", OUT_DIR)


if __name__ == "__main__":
    main()
