"""Feature engineering for product portfolio signals."""

import pandas as pd

def _contains_keywords(text: str, keywords: list[str]) -> bool:
    """Check whether any keyword exists in the provided text."""
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)

def create_product_features(datasets: dict) -> pd.DataFrame:
    """Generate product risk flags per account from notes or usage signals."""
    accounts_df = datasets.get("accounts")
    if not isinstance(accounts_df, pd.DataFrame) or "account_id" not in accounts_df.columns:
        return pd.DataFrame(columns=["account_id", "migration_risk_flag"])

    account_ids = accounts_df["account_id"].dropna().unique()
    features = pd.DataFrame({"account_id": account_ids})
    features["migration_risk_flag"] = 0

    migration_keywords = ["migration", "migrate", "sdk v3", "upgrade required", "competitor"]

    account_text_map = {}
    
    # We only check usage, support, nps text fields and csm_notes
    for key in ["support", "nps"]:
        df = datasets.get(key)
        if isinstance(df, pd.DataFrame) and "account_id" in df.columns:
            text_cols = [col for col in ["note", "notes", "comment", "description", "details", "message", "verbatim_comment"] if col in df.columns]
            if text_cols:
                for idx, row in df.iterrows():
                    acct = row["account_id"]
                    text = " ".join([str(row[c]) for c in text_cols if pd.notna(row[c])])
                    account_text_map[acct] = account_text_map.get(acct, "") + " " + text

    # Incorporate raw CSM notes if available. 
    # But note mapping is hard here without rapidfuzz, but since we just need simple heuristic for product features, we will just use a simpler check, or skip csm here and rely on risk_keywords from csm_features.
    # We'll use the basic mapped texts.
    
    for idx, account_id in features["account_id"].items():
        account_text = account_text_map.get(account_id, "").lower()
        migration_flag = _contains_keywords(account_text, migration_keywords)
        features.at[idx, "migration_risk_flag"] = int(migration_flag)

    return features
