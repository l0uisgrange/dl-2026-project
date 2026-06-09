import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split


def merge_datasets(datasets, percentages, final_size, shuffle=True, random_state=42):
    if not abs(sum(percentages.values()) - 1.0) < 1e-6:
        raise ValueError("Percentages must sum to 1.0")

    sampled_dfs = []
    for name, df in datasets.items():
        if name not in percentages:
            raise ValueError(f"Missing percentage for dataset: {name}")
        n_samples = int(final_size * percentages[name])
        sampled = df.sample(n=n_samples, replace=(n_samples > len(df)), random_state=random_state)
        sampled_dfs.append(sampled)

    final_dataset = pd.concat(sampled_dfs, ignore_index=True)
    if shuffle:
        final_dataset = final_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return final_dataset


annotated_data = load_dataset("toxigen/toxigen-data", name="annotated", use_auth_token=True)


def prepare_datasets(cfg) -> DatasetDict:
    df = pd.read_csv(cfg["data"]["file"])
    df = df[[cfg["data"]["text_col"], cfg["data"]["label_col"]]].copy()
    df = df.dropna(subset=[cfg["data"]["text_col"], cfg["data"]["label_col"]])
    df = df[df[cfg["data"]["text_col"]].str.len() > 0]

    df[cfg["data"]["label_col"]] = pd.to_numeric(df[cfg["data"]["label_col"]], errors="coerce")
    df = df.dropna(subset=[cfg["data"]["label_col"]])
    df[cfg["data"]["label_col"]] = df[cfg["data"]["label_col"]].astype(int) - 1

    train_df, test_df = train_test_split(df, test_size=cfg["data"]["test_size"], random_state=cfg["seed"])
    train_df, valid_df = train_test_split(train_df, test_size=cfg["data"]["valid_size"], random_state=cfg["seed"])

    rename = {cfg["data"]["text_col"]: "text", cfg["data"]["label_col"]: "labels"}
    train_df = train_df.rename(columns=rename)
    valid_df = valid_df.rename(columns=rename)
    test_df = test_df.rename(columns=rename)

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(valid_df),
            "test": Dataset.from_pandas(test_df),
        }
    )
