from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk


def load_dataset(config):
    """
    Load dataset using YAML config.
    Expects train_path and val_path keys for CSV datasets.
    """

    cfg = config["dataset"]
    dataset_type = cfg["dataset_type"]

    # =====================================================
    # CSV DATASET (pre-split train/val files)
    # =====================================================
    if dataset_type == "csv":
        train_path = Path(cfg["train_path"])
        val_path = Path(cfg["val_path"])

        if not train_path.exists():
            raise FileNotFoundError(f"Train dataset not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Val dataset not found: {val_path}")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        return DatasetDict(
            {
                "train": Dataset.from_pandas(train_df, preserve_index=False),
                "validation": Dataset.from_pandas(val_df, preserve_index=False),
            }
        )

    # =====================================================
    # HF DATASET
    # =====================================================
    elif dataset_type == "hf":
        dataset_path = Path(cfg["dataset_path"])
        return load_from_disk(dataset_path)

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
