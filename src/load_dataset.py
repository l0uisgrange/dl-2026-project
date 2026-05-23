from pathlib import Path

import pandas as pd

from datasets import Dataset, load_from_disk


def load_dataset(config):
    """
    Load dataset using YAML config.
    """

    cfg = config["dataset"]

    dataset_type = cfg["dataset_type"]
    dataset_path = Path(cfg["dataset_path"])

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    # =====================================================
    # CSV DATASET
    # =====================================================
    if dataset_type == "csv":

        df = pd.read_csv(dataset_path)

        dataset = Dataset.from_pandas(
            df,
            preserve_index=False,
        )

        return dataset

    # =====================================================
    # HF DATASET
    # =====================================================
    elif dataset_type == "hf":

        dataset = load_from_disk(dataset_path)

        return dataset

    else:
        raise ValueError(
            f"Unsupported dataset type: {dataset_type}"
        )