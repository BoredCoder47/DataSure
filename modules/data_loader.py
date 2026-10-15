
from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def load_dataset(file) -> pd.DataFrame:
    """Load a CSV or Excel dataset from a path or uploaded file."""

    filename = getattr(file, "name", str(file))
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Upload a CSV or Excel file."
        )

    if extension == ".csv":
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("The dataset contains no columns.")

    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return basic dataset statistics."""

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "memory_usage_bytes": int(
            df.memory_usage(index=True, deep=True).sum()
        ),
        "column_types": {
            str(column): str(dtype)
            for column, dtype in df.dtypes.items()
        },
    }
