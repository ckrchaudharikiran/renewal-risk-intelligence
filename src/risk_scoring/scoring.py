"""Risk scoring module."""

import pandas as pd

FEATURE_WEIGHTS = {
    "usage": 0.25,
    "support": 0.20,
    "nps": 0.15,
    "csm": 0.25,
    "product": 0.15
}

def _risk_level_from_score(score: float) -> str:
    """Map a risk score to High/Medium/Low based on configured thresholds."""
    if score > 0.65:
        return "High"
    if score > 0.35:
        return "Medium"
    return "Low"

def calculate_risk_score(features_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate weighted renewal risk score and risk level per account."""
    if "account_id" not in features_df.columns:
        raise ValueError("features_df must include 'account_id'.")

    df = features_df.copy()
    
    # Fill missing columns with 0 to prevent errors
    expected_cols = [
        "usage_trend_score", "low_usage_flag",
        "p1_ticket_ratio", "open_ticket_flag",
        "nps_score_normalized", "detractor_flag",
        "sentiment_score_normalized", "churn_flag",
        "migration_risk_flag"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Calculate sub-scores per category (0 to 1)
    
    # Usage: low_usage_flag increases risk, lower usage_trend_score increases risk
    df["subscore_usage"] = (df["low_usage_flag"] + (1.0 - df["usage_trend_score"].fillna(0.5))) / 2.0
    
    # Support: p1_ticket_ratio and open_ticket_flag
    df["subscore_support"] = (df["p1_ticket_ratio"].fillna(0.0) + df["open_ticket_flag"]) / 2.0
    
    # NPS: detractor_flag and lower nps score increases risk
    df["subscore_nps"] = (df["detractor_flag"] + (1.0 - df["nps_score_normalized"].fillna(1.0))) / 2.0
    
    # CSM: churn_flag and lower sentiment score increases risk
    df["subscore_csm"] = (df["churn_flag"] + (1.0 - df["sentiment_score_normalized"].fillna(1.0))) / 2.0
    
    # Product: migration_risk_flag
    df["subscore_product"] = df["migration_risk_flag"].fillna(0.0)
    
    # Weighted sum
    df["risk_score"] = (
        df["subscore_usage"] * FEATURE_WEIGHTS["usage"] +
        df["subscore_support"] * FEATURE_WEIGHTS["support"] +
        df["subscore_nps"] * FEATURE_WEIGHTS["nps"] +
        df["subscore_csm"] * FEATURE_WEIGHTS["csm"] +
        df["subscore_product"] * FEATURE_WEIGHTS["product"]
    )
    
    # Normalize final score to 0-1 (clip just in case)
    df["risk_score"] = df["risk_score"].clip(0.0, 1.0).round(4)
    df["risk_level"] = df["risk_score"].apply(_risk_level_from_score)

    return df
