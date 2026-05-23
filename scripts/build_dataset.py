import pandas as pd
from datasets import load_dataset
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"


# ---- FUNCTION TO SAMPLE CSV SAFELY ----
def sample_df(filename, n=500):
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    return df.sample(n=n, random_state=42).reset_index(drop=True)



files = {
    "hate_asylum_seekers": "hate_asylum_seekers.csv",
    "hate_cross_border_workers": "hate_cross_border_workers.csv",
    "hate_portuguese": "hate_portuguese.csv",
    "neutral_asylum_seekers": "neutral_asylum_seekers.csv",
    "neutral_border_workers": "neutral_cross_border_workers.csv",
    "neutral_portuguese": "neutral_portuguese.csv",
}

dfs = [sample_df(f) for f in files.values()]
df = pd.concat(dfs, ignore_index=True)

# keep only needed columns (safer)
df = df[["text", "mode","target_group"]]


# ---- LOAD TOXIGEN ----
toxigen = load_dataset(
    "toxigen/toxigen-data",
    name="train",
)

toxigen = toxigen["train"].to_pandas()



toxigen = toxigen.rename(columns={
    "generation": "text",
    "prompt_label": "mode",
    "group": "target_group",
})

toxigen = toxigen[["text", "mode", "target_group"]]

toxigen_shuffled = toxigen.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

toxigen_13000 = toxigen_shuffled.iloc[:13000]

toxigen_16000 = toxigen_shuffled.iloc[:16000]

# ---- MERGE ----
df = pd.concat([df, toxigen_13000], ignore_index=True)


# ---- SAVE SAFELY ----
OUTPUT_PATH = BASE_DIR / "data" / "dataset_toxigen_swiss_context.csv"
df.to_csv(OUTPUT_PATH, index=False)
OUTPUT_PATH = BASE_DIR / "data" / "dataset_toxigen_16k.csv"
toxigen_16000.to_csv(OUTPUT_PATH, index=False)

print("Saved to:", OUTPUT_PATH)