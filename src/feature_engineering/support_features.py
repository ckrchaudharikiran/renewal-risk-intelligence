"""Feature engineering for support ticket signals."""

import pandas as pd

def create_support_features(support_df: pd.DataFrame) -> pd.DataFrame:
    """Create normalized support features per account."""
    if support_df.empty or "account_id" not in support_df.columns:
        return pd.DataFrame(columns=["account_id", "p1_ticket_ratio", "open_ticket_flag"])

    df = support_df.copy()
    
    # Required columns for internal logic. We assume ticket_priority and ticket_status exist
    if "ticket_priority" not in df.columns or "ticket_status" not in df.columns:
        return pd.DataFrame(columns=["account_id", "p1_ticket_ratio", "open_ticket_flag"])

    features = (
        df.groupby("account_id", as_index=False)
        .agg(
            total_ticket_count=("account_id", "size"),
            p1_ticket_count=("ticket_priority", lambda x: (x == "P1").sum()),
            open_ticket_flag=(
                "ticket_status",
                lambda x: int(x.str.contains("open", case=False, na=False).any()),
            )
        )
    )

    # Calculate p1_ticket_ratio (0 to 1)
    features["p1_ticket_ratio"] = features["p1_ticket_count"] / features["total_ticket_count"]
    features["p1_ticket_ratio"] = features["p1_ticket_ratio"].fillna(0.0)

    # Drop intermediate columns to keep only the requested standard features
    return features[["account_id", "p1_ticket_ratio", "open_ticket_flag"]]
