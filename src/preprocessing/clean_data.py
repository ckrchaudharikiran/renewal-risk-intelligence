"""Data cleaning module."""

import pandas as pd


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase snake_case."""
    df.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]
    return df


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from string columns."""
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def _convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert likely date columns to datetime when parseable."""
    for col in df.columns:
        if "date" in col or col.endswith("_at"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal missing value handling for numeric and text data."""
    numeric_cols = df.select_dtypes(include=["number"]).columns
    object_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(0)
    for col in object_cols:
        df[col] = df[col].fillna("")

    return df


def clean_data(datasets: dict) -> dict:
    """Clean each tabular dataset and return a new datasets dictionary."""
    cleaned: dict = {}

    for name, data in datasets.items():
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            df = _standardize_columns(df)
            df = _strip_string_columns(df)
            df = _convert_date_columns(df)
            df = _handle_missing_values(df)
            cleaned[name] = df
        else:
            cleaned[name] = data

    return cleaned
