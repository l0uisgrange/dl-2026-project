import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

def create_training_dataset():

    return 

    
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