import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict , load_dataset

import pandas as pd

def merge_datasets(
    datasets,
    percentages,
    final_size,
    shuffle=True,
    random_state=42
):
    """
    Merge multiple datasets into one dataset using specified percentages.

    Parameters
    ----------
    datasets : dict
        Dictionary of name -> pandas DataFrame

    percentages : dict
        Dictionary of name -> percentage contribution
        Percentages must sum to 1.0

    final_size : int
        Number of rows in final dataset

    shuffle : bool
        Whether to shuffle final dataset

    random_state : int
        Random seed

    Returns
    -------
    pandas.DataFrame
        Final merged dataset
    """

    # Validate percentages
    total_percentage = sum(percentages.values())

    if not abs(total_percentage - 1.0) < 1e-6:
        raise ValueError("Percentages must sum to 1.0")

    sampled_dfs = []

    for name, df in datasets.items():

        if name not in percentages:
            raise ValueError(f"Missing percentage for dataset: {name}")

        n_samples = int(final_size * percentages[name])

        # Sample with replacement if needed
        sampled = df.sample(
            n=n_samples,
            replace=(n_samples > len(df)),
            random_state=random_state
        )

        sampled_dfs.append(sampled)

    # Merge datasets
    final_dataset = pd.concat(sampled_dfs, ignore_index=True)

    # Shuffle final dataset
    if shuffle:
        final_dataset = final_dataset.sample(
            frac=1,
            random_state=random_state
        ).reset_index(drop=True)

    return final_dataset
    
annotated_data = load_dataset("toxigen/toxigen-data", name="annotated", use_auth_token=True)
def prepare_datasets(cfg) -> DatasetDict:
    df = pd.read_csv(cfg["data"]["file"])

    # Keep only required columns
    df = df[[cfg["data"]["text_col"], cfg["data"]["label_col"]]].copy()

    # Clean text
    df = df.dropna(subset=[cfg["data"]["text_col"], cfg["data"]["label_col"]])
    
    df = df[df[cfg["data"]["text_col"]].str.len() > 0]

    # Convert score to numeric (1–5)
    df[ cfg["data"]["label_col"]] = pd.to_numeric(df[ cfg["data"]["label_col"]], errors="coerce")
    df = df.dropna(subset=[cfg["data"]["label_col"]])
    df[ cfg["data"]["label_col"]] = df[cfg["data"]["label_col"]].astype(int)-1

    # Train / test split
    train_df, test_df = train_test_split(
        df,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["seed"]
    )
    # Train / validation split
    train_df, valid_df = train_test_split(
        train_df,
        test_size=cfg["data"]["valid_size"],
        random_state=cfg["seed"]
    )

    # Rename for HuggingFace
    train_df = train_df.rename(columns={cfg["data"]["text_col"]: "text", cfg["data"]["label_col"]: "labels"})
    valid_df = valid_df.rename(columns={cfg["data"]["text_col"]: "text", cfg["data"]["label_col"]: "labels"})
    test_df = test_df.rename(columns={cfg["data"]["text_col"]: "text", cfg["data"]["label_col"]: "labels"})

    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(valid_df),
        "test": Dataset.from_pandas(test_df),
    })