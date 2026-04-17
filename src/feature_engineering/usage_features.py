"""Feature engineering for usage metrics."""

import pandas as pd
import numpy as np

def create_usage_features(usage_df: pd.DataFrame) -> pd.DataFrame:
    """Create normalized usage features per account."""
    if usage_df.empty or "account_id" not in usage_df.columns:
        return pd.DataFrame(columns=["account_id", "usage_trend_score", "low_usage_flag"])

    df = usage_df.copy()
    df["usage_value"] = pd.to_numeric(df["usage_value"], errors="coerce").fillna(0)
    df["usage_date"] = pd.to_datetime(df["usage_date"], errors="coerce")
    df = df.dropna(subset=["usage_date"]).sort_values("usage_date")

    if df.empty:
        return pd.DataFrame(columns=["account_id", "usage_trend_score", "low_usage_flag"])

    # Aggregate by account
    def calculate_trend(group):
        if len(group) < 2:
            return 0.5  # Neutral trend if only 1 data point
        recent_val = group.iloc[-1]["usage_value"]
        old_val = group.iloc[0]["usage_value"]
        if old_val == 0:
            return 1.0 if recent_val > 0 else 0.5
        change = (recent_val - old_val) / old_val
        # Normalize change to 0-1 roughly. Change is typically between -1 and +1.
        # -1 (100% drop) -> 0.0, 0 -> 0.5, +1 (100% increase) -> 1.0
        return max(0.0, min(1.0, (change + 1) / 2.0))

    trends = df.groupby("account_id").apply(calculate_trend).reset_index(name="usage_trend_score")
    
    # Calculate low usage flag based on latest usage. Say, lower 25th percentile of all latest usages.
    latest_usage = df.groupby("account_id")["usage_value"].last().reset_index()
    if len(latest_usage) > 1:
        threshold = latest_usage["usage_value"].quantile(0.25)
        latest_usage["low_usage_flag"] = (latest_usage["usage_value"] <= threshold).astype(int)
    else:
        latest_usage["low_usage_flag"] = 0

    features = trends.merge(latest_usage[["account_id", "low_usage_flag"]], on="account_id")
    return features
