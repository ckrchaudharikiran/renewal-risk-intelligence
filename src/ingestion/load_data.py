"""Data ingestion utilities."""

from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path("data/raw")


def _load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from the raw data directory."""
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path}")
    return pd.read_csv(file_path)


def _load_text(filename: str) -> str:
    """Load a text-based file from the raw data directory."""
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required file: {file_path}")
    return file_path.read_text(encoding="utf-8")


def load_accounts() -> pd.DataFrame:
    """Load account-level master data."""
    return _load_csv("accounts.csv")


def load_usage() -> pd.DataFrame:
    """Load product usage metrics."""
    return _load_csv("usage_metrics.csv")


def load_support() -> pd.DataFrame:
    """Load support ticket records."""
    return _load_csv("support_tickets.csv")


def load_nps() -> pd.DataFrame:
    """Load NPS response data."""
    return _load_csv("nps_responses.csv")


def load_csm_notes() -> str:
    """Load CSM notes as raw text."""
    return _load_text("csm_notes.txt")


def load_changelog() -> str:
    """Load product changelog as raw markdown text."""
    return _load_text("changelog.md")


def load_all_data() -> dict[str, object]:
    """Load all raw datasets and return them in a dictionary."""
    return {
        "accounts": load_accounts(),
        "usage": load_usage(),
        "support": load_support(),
        "nps": load_nps(),
        "csm_notes": load_csm_notes(),
        "changelog": load_changelog(),
    }
