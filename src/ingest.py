"""Data ingestion utilities for loading datasets into pandas DataFrames."""

from pathlib import Path

import pandas as pd

SUPPORTED_FORMATS = {".csv", ".json", ".xlsx", ".xls", ".parquet"}


def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """Load a dataset from disk into a pandas DataFrame based on its file extension."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"No such file: {file_path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Supported formats: {sorted(SUPPORTED_FORMATS)}"
        )

    if suffix == ".csv":
        df = pd.read_csv(path, **kwargs)
    elif suffix == ".json":
        df = pd.read_json(path, **kwargs)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, **kwargs)
    else:  # .parquet
        df = pd.read_parquet(path, **kwargs)

    if df.empty:
        raise ValueError(f"Loaded dataset from {file_path} is empty.")

    return df
