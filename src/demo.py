"""
demo.py — CLI entry point for training and evaluation.

Commands:
  train-toxigen       Fine-tune on ToxiGen-16k (baseline)
  train-toxigen-plus  Fine-tune on ToxiGen+ (swiss context)
  compare             Run inference on held-out eval data and compare both models

Usage:
  python -m src.demo train-toxigen
  python -m src.demo train-toxigen-plus
  python -m src.demo compare
"""

import argparse
import sys

import pandas as pd
import torch
import yaml
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── constants ─────────────────────────────────────────────────────────────────

CONFIGS = {
    "toxigen": "./config/config_toxigen.yaml",
    "toxigen-plus": "./config/config.yaml",
}

MODELS = {
    "ToxiGen (baseline)": "./outputs/bert_toxigen",
    "ToxiGen+ (swiss)": "./outputs/bert_toxigen_plus",
}

DATA_PATH_SWISS_VAL = "./data/dataset_toxigen_plus_val.csv"
DATA_PATH_TOXIGEN_VAL = "./data/dataset_toxigen_val.csv"
SWISS_GROUPS = {"asylum_seekers", "cross_border_workers", "portuguese"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 256
BATCH_SIZE = 32


# ── training ──────────────────────────────────────────────────────────────────


def cmd_train(config_key: str):
    from src.train import train

    config_path = CONFIGS[config_key]
    print(f"[train] loading config from {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


# ── evaluation ────────────────────────────────────────────────────────────────


def load_eval_data() -> pd.DataFrame:
    # Swiss groups: val split of toxigen_plus (unseen during training)
    swiss = pd.read_csv(DATA_PATH_SWISS_VAL).dropna(subset=["text", "labels"])
    swiss["labels"] = pd.to_numeric(swiss["labels"], errors="coerce").astype(int)
    swiss = swiss[swiss["target_group"].isin(SWISS_GROUPS)].reset_index(drop=True)

    # ToxiGen groups: val split of toxigen (unseen during training)
    toxigen_val = pd.read_csv(DATA_PATH_TOXIGEN_VAL).dropna(subset=["text", "labels"])
    toxigen_val["labels"] = pd.to_numeric(toxigen_val["labels"], errors="coerce").astype(int)

    df = pd.concat([swiss, toxigen_val], ignore_index=True).dropna(subset=["target_group"])
    print(f"[data] {len(df)} examples  hate={df['labels'].sum()}  neutral={(df['labels'] == 0).sum()}")
    print(f"       groups: {sorted(df['target_group'].unique())}")
    return df


def run_inference(df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
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
        done = min(i + BATCH_SIZE, len(texts))
        if done % 1000 == 0 or done == len(texts):
            print(f"  [{done}/{len(texts)}] processed")

    out = df.copy()
    out["pred"] = preds
    out["hate_score"] = scores
    return out


def print_report(df: pd.DataFrame, model_name: str):
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  MODEL: {model_name}")
    print(sep)

    y_true, y_pred = df["labels"], df["pred"]
    print("\n  OVERALL")
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["neutral", "hate"], zero_division=0
    )
    print("\n".join("    " + line for line in report.splitlines()))
    print(f"    Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"    Macro F1 : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")

    print("\n  PER GROUP")
    for group in sorted(df["target_group"].unique()):
        sub = df[df["target_group"] == group]
        macro_f1 = f1_score(sub["labels"], sub["pred"], average="macro", zero_division=0)
        acc = accuracy_score(sub["labels"], sub["pred"])
        print(f"\n    [{group}]  n={len(sub)}  macro F1={macro_f1:.3f}  acc={acc:.3f}")
        rep = classification_report(
            sub["labels"],
            sub["pred"],
            labels=[0, 1],
            target_names=["neutral", "hate"],
            zero_division=0,
        )
        print("\n".join("      " + line for line in rep.splitlines()))


def cmd_compare():
    df = load_eval_data()
    for model_name, model_dir in MODELS.items():
        print(f"\n[model] loading {model_name} from {model_dir} ...")
        result = run_inference(df, model_dir)
        print_report(result, model_name)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="ToxiGen hate speech detection — train or compare models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train-toxigen", help="Fine-tune on ToxiGen-16k (baseline)")
    subparsers.add_parser("train-toxigen-plus", help="Fine-tune on ToxiGen+ (swiss context)")
    subparsers.add_parser("compare", help="Infer on held-out eval data and compare both models")

    args = parser.parse_args()

    if args.command == "train-toxigen":
        cmd_train("toxigen")
    elif args.command == "train-toxigen-plus":
        cmd_train("toxigen-plus")
    elif args.command == "compare":
        cmd_compare()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
