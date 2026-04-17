"""Feature engineering for NPS and sentiment signals."""

import pandas as pd

def create_nps_features(nps_df: pd.DataFrame) -> pd.DataFrame:
    """Create per-account NPS features."""
    if nps_df.empty or "account_id" not in nps_df.columns or "nps_score" not in nps_df.columns:
        return pd.DataFrame(columns=["account_id", "nps_score_normalized", "detractor_flag"])

    df = nps_df.copy()
    df["nps_score"] = pd.to_numeric(df["nps_score"], errors="coerce")
    df = df.dropna(subset=["nps_score"])

    if df.empty:
        return pd.DataFrame(columns=["account_id", "nps_score_normalized", "detractor_flag"])

    grouped = df.groupby("account_id", as_index=False).agg(avg_nps_score=("nps_score", "mean"))
    
    # Normalize score (assuming standard 0-10 scale)
    grouped["nps_score_normalized"] = grouped["avg_nps_score"] / 10.0
    grouped["nps_score_normalized"] = grouped["nps_score_normalized"].clip(0, 1)
    
    # Detractor flag (NPS <= 6 is detractor)
    grouped["detractor_flag"] = (grouped["avg_nps_score"] <= 6).astype(int)

    return grouped[["account_id", "nps_score_normalized", "detractor_flag"]]
