"""Entity resolution module for customer/account records."""

import logging
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

def match_account_names(
    source_names: list[str],
    target_names: list[str],
    threshold: int = 80,
) -> dict[str, Optional[str]]:
    """Match source account names to target account names using fuzzy search."""
    matches: dict[str, Optional[str]] = {}
    normalized_targets = [str(name).strip() for name in target_names if pd.notna(name)]

    for source in source_names:
        source_clean = str(source).strip()
        if not source_clean:
            matches[source] = None
            continue

        best = process.extractOne(
            source_clean,
            normalized_targets,
            scorer=fuzz.token_sort_ratio,
        )
        if best and best[1] >= threshold:
            matches[source] = best[0]
        else:
            matches[source] = None

    return matches


def _get_account_name_column(df: pd.DataFrame) -> Optional[str]:
    """Return a likely account-name column from a dataset."""
    candidates = ["account_name", "account", "customer_name", "company_name"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def resolve_accounts(datasets: dict, threshold: int = 80) -> dict:
    """Map datasets to a consistent account_id using fuzzy account-name matching."""
    if "accounts" not in datasets or not isinstance(datasets["accounts"], pd.DataFrame):
        raise ValueError("datasets must include an 'accounts' DataFrame.")

    accounts_df = datasets["accounts"].copy()
    if "account_id" not in accounts_df.columns:
        raise ValueError("'accounts' dataset must contain 'account_id'.")

    account_name_col = _get_account_name_column(accounts_df)
    if not account_name_col:
        raise ValueError(
            "'accounts' dataset must contain an account name column "
            "(e.g., account_name, account, customer_name, company_name)."
        )

    canonical = (
        accounts_df[[account_name_col, "account_id"]]
        .dropna(subset=[account_name_col, "account_id"])
        .copy()
    )
    canonical[account_name_col] = canonical[account_name_col].astype(str).str.strip()
    canonical = canonical[canonical[account_name_col] != ""]

    canonical_name_to_id = dict(zip(canonical[account_name_col], canonical["account_id"]))
    target_names = list(canonical_name_to_id.keys())

    resolved: dict = {"accounts": accounts_df}

    for name, dataset in datasets.items():
        if not isinstance(dataset, pd.DataFrame) or name == "accounts":
            resolved[name] = dataset
            continue

        df = dataset.copy()
        if "account_id" in df.columns:
            # Dataset already has a direct account identifier.
            resolved[name] = df
            continue

        source_col = _get_account_name_column(df)
        if not source_col:
            logger.warning("No account-name column found in dataset '%s'. Skipping.", name)
            resolved[name] = df
            continue

        unique_source_names = (
            df[source_col].dropna().astype(str).str.strip().unique().tolist()
        )
        name_matches = match_account_names(unique_source_names, target_names, threshold)

        df["_matched_account_name"] = (
            df[source_col]
            .astype(str)
            .str.strip()
            .map(name_matches)
        )
        df["account_id"] = df["_matched_account_name"].map(canonical_name_to_id)

        unmatched = df[df["account_id"].isna()]
        if not unmatched.empty:
            unmatched_count = unmatched.shape[0]
            logger.warning(
                "Dataset '%s' has %s unmatched records at threshold=%s.",
                name,
                unmatched_count,
                threshold,
            )

        df = df.drop(columns=["_matched_account_name"])
        resolved[name] = df

    return resolved
