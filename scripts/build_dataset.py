from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def sample_df(filename, n=500):
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    return df.sample(n=n, random_state=42).reset_index(drop=True)


def save_splits(df, name, seed=42, test_size=0.10):
    train, val = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["labels"])
    train.to_csv(DATA_DIR / f"dataset_{name}_train.csv", index=False)
    val.to_csv(DATA_DIR / f"dataset_{name}_val.csv", index=False)
    df.to_csv(DATA_DIR / f"dataset_{name}.csv", index=False)
    print(f"[{name}] total={len(df)}  train={len(train)}  val={len(val)}")


# ---- LOAD TOXIGEN (deduplicated) ----
toxigen = load_dataset("toxigen/toxigen-data", name="train")
toxigen = toxigen["train"].to_pandas()
toxigen = toxigen.rename(columns={"generation": "text", "prompt_label": "labels", "group": "target_group"})
toxigen = toxigen[["text", "labels", "target_group"]]

# Deduplicate by text to avoid cross-split contamination
toxigen = toxigen.drop_duplicates(subset=["text"]).reset_index(drop=True)

toxigen_shuffled = toxigen.sample(frac=1, random_state=42).reset_index(drop=True)

# ---- TOXIGEN (16k, no swiss) ----
toxigen_16k = toxigen_shuffled.iloc[:16000].copy()
save_splits(toxigen_16k, "toxigen")

# ---- SWISS DATA ----
files = {
    "hate_asylum_seekers": "hate_asylum_seekers.csv",
    "hate_cross_border_workers": "hate_cross_border_workers.csv",
    "hate_portuguese": "hate_portuguese.csv",
    "neutral_asylum_seekers": "neutral_asylum_seekers.csv",
    "neutral_border_workers": "neutral_cross_border_workers.csv",
    "neutral_portuguese": "neutral_portuguese.csv",
}

dfs = [sample_df(f) for f in files.values()]
swiss_df = pd.concat(dfs, ignore_index=True)[["text", "mode", "target_group"]]
swiss_df = swiss_df.rename(columns={"mode": "labels"})
swiss_df["labels"] = swiss_df["labels"].astype(str).str.strip()
swiss_df = swiss_df[swiss_df["labels"].isin(["0", "1"])].copy()
swiss_df["labels"] = swiss_df["labels"].astype(int)

# ---- TOXIGEN+ (13k toxigen + 3k swiss) ----
toxigen_13k = toxigen_shuffled.iloc[:13000].copy()
df_plus = pd.concat([swiss_df, toxigen_13k], ignore_index=True)
df_plus = df_plus.sample(frac=1, random_state=42).reset_index(drop=True)
save_splits(df_plus, "toxigen_plus")

# ---- HELDOUT (rows after 16k, never seen during training) ----
heldout = toxigen_shuffled.iloc[16000:].copy()
heldout.to_csv(DATA_DIR / "dataset_toxigen_heldout.csv", index=False)
print(f"[heldout] rows={len(heldout)}")
