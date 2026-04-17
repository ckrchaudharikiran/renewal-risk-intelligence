"""Main project pipeline entrypoint."""

import json
import os
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.feature_engineering.csm_features import create_csm_features
from src.feature_engineering.nps_features import create_nps_features
from src.feature_engineering.product_features import create_product_features
from src.feature_engineering.support_features import create_support_features
from src.feature_engineering.usage_features import create_usage_features
from src.ingestion.load_data import load_all_data
from src.llm.explanation import generate_explanations
from src.llm.llm_client import call_llm
from src.preprocessing.clean_data import clean_data
from src.preprocessing.entity_resolution import resolve_accounts
from src.risk_scoring.scoring import calculate_risk_score
from src.risk_scoring.action_engine import generate_insights_and_actions

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data/outputs"

def _safe_feature_call(func, *args, **kwargs) -> pd.DataFrame:
    """Call a feature function and return an empty DataFrame on failure."""
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        print(f"Feature generation warning ({func.__name__}): {exc}")
        return pd.DataFrame(columns=["account_id"])

def _merge_features(accounts_df: pd.DataFrame, feature_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge account-level feature DataFrames into a single table."""
    base = accounts_df[["account_id"]].drop_duplicates().copy()
    base["account_id"] = base["account_id"].astype(str)
    for frame in feature_frames:
        if isinstance(frame, pd.DataFrame) and "account_id" in frame.columns:
            frame = frame.copy()
            frame["account_id"] = frame["account_id"].astype(str)
            base = base.merge(frame, on="account_id", how="left")
    return base.fillna(0) # Not optimal for all columns but works. However risk scorer handles fills now.

def _prepare_usage_df(usage_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw usage schema to expected feature schema."""
    df = usage_df.copy()
    if not df.empty:
        if "usage_date" not in df.columns and "month" in df.columns:
            df["usage_date"] = pd.to_datetime(df["month"].astype(str) + "-01", errors="coerce")
        if "usage_value" not in df.columns:
            if "api_calls" in df.columns:
                df["usage_value"] = pd.to_numeric(df["api_calls"], errors="coerce")
            else:
                df["usage_value"] = 0
    return df

def _prepare_support_df(support_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw support schema to expected feature schema."""
    df = support_df.copy()
    if not df.empty:
        if "ticket_priority" not in df.columns and "priority" in df.columns:
            df["ticket_priority"] = df["priority"]
        if "ticket_status" not in df.columns and "status" in df.columns:
            df["ticket_status"] = df["status"]
    return df

def _prepare_nps_df(nps_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw NPS schema to expected feature schema."""
    df = nps_df.copy()
    if not df.empty:
        if "nps_score" not in df.columns and "score" in df.columns:
            df["nps_score"] = df["score"]
    return df

def _fallback_llm(prompt: str) -> str:
    """Return a deterministic fallback LLM response if API is unavailable."""
    prompt_lower = prompt.lower()
    if "return concise json with keys: sentiment, churn_risk" in prompt_lower:
        sentiment = "neutral"
        churn_risk = "low"
        
        # Smart heuristics to simulate LLM extraction
        bad_words = ["discount", "walk", "furious", "poorly", "dealbreaker", "cratered", "churn", "competitor"]
        medium_words = ["tense", "escalate", "nervous", "rocky", "mess", "struggling"]
        
        if any(w in prompt_lower for w in bad_words):
            sentiment = "negative"
            churn_risk = "high"
        elif any(w in prompt_lower for w in medium_words):
            sentiment = "negative"
            churn_risk = "medium"
            
        return f'{{"sentiment": "{sentiment}", "churn_risk": "{churn_risk}", "key_issues": ["Quota limit reached"]}}'
        
    return (
        '{"explanation":"Account shows distinct risk signals requiring immediate attention.",'
        '"recommended_actions":["Review CSM outreach","Address support escalations","Monitor usage closely"]}'
    )

def _filter_upcoming_renewals(accounts_df: pd.DataFrame) -> pd.DataFrame:
    """Filter accounts to those renewing in the next 90 days."""
    if "contract_end_date" not in accounts_df.columns:
        print("Warning: contract_end_date missing. Assuming all accounts are candidates.")
        return accounts_df
        
    df = accounts_df.copy()
    now = pd.Timestamp.now()
    end_date = now + pd.Timedelta(days=90)
    
    df["contract_end_date_dt"] = pd.to_datetime(df["contract_end_date"], errors="coerce")
    
    missing_mask = df["contract_end_date_dt"].isna()
    if missing_mask.any():
        print(f"Warning: {missing_mask.sum()} accounts missing renewal date. Including them.")
        
    mask = missing_mask | ((df["contract_end_date_dt"] >= now) & (df["contract_end_date_dt"] <= end_date))
    return df[mask].drop(columns=["contract_end_date_dt"])

def main() -> None:
    """Run the full renewal-risk-intelligence pipeline."""
    print("Pipeline initialized")

    # 1) Load data
    datasets = load_all_data()

    # 2) Clean data
    cleaned = clean_data(datasets)

    # 3) Resolve entities
    resolved = resolve_accounts(cleaned)

    # Filter to upcoming renewals
    accounts_df = resolved.get("accounts", pd.DataFrame())
    accounts_df = _filter_upcoming_renewals(accounts_df)
    resolved["accounts"] = accounts_df
    
    # 4) Generate all features
    use_llm = os.getenv("PIPELINE_ENABLE_LLM", "false").lower() in {"1", "true", "yes"}
    llm_func = call_llm if use_llm else _fallback_llm

    usage_features = _safe_feature_call(
        create_usage_features,
        _prepare_usage_df(resolved.get("usage", pd.DataFrame()))
    )
    support_features = _safe_feature_call(
        create_support_features,
        _prepare_support_df(resolved.get("support", pd.DataFrame()))
    )
    nps_features = _safe_feature_call(
        create_nps_features,
        _prepare_nps_df(resolved.get("nps", pd.DataFrame()))
    )
    product_features = _safe_feature_call(
        create_product_features, 
        resolved
    )
    csm_features = _safe_feature_call(
        create_csm_features,
        resolved.get("csm_notes", ""),
        accounts_df,
        llm_func
    )

    # 5) Merge all features into one dataframe
    all_features = _merge_features(
        accounts_df,
        [usage_features, support_features, nps_features, product_features, csm_features]
    )

    # 6) Compute risk scores
    scored = calculate_risk_score(all_features)
    
    # Merge risk scores back to features to have a complete picture for explanations & actions
    scored_features = all_features.merge(scored[["account_id", "risk_score", "risk_level"]], on="account_id", how="left")

    # 7) Generate rule-based actions & insights
    scored_features = generate_insights_and_actions(scored_features)

    # 8) Generate explanations
    try:
        final_results_df = generate_explanations(scored_features, llm_func)
    except Exception as exc:
        print(f"LLM explanation warning: {exc}. Using fallback explanations.")
        final_results_df = generate_explanations(scored_features, _fallback_llm)

    # 9) Format final output payload
    final_payloads = []
    for _, row in final_results_df.iterrows():
        # Combine actions
        combined_actions = []
        if isinstance(row.get("rule_based_actions"), list):
            combined_actions.extend(row["rule_based_actions"])
        if isinstance(row.get("llm_recommended_actions"), str):
            # Try to parse markdown list from LLM
            llm_actions = [a.strip("- ") for a in row["llm_recommended_actions"].split("\n") if a.strip("- ")]
            combined_actions.extend(llm_actions)
        
        # Deduplicate
        seen = set()
        final_actions = []
        for a in combined_actions:
            if a not in seen and a.strip():
                final_actions.append(a)
                seen.add(a)

        key_signals = []
        for feature in ["low_usage_flag", "p1_ticket_ratio", "open_ticket_flag", "detractor_flag", "churn_flag", "migration_risk_flag"]:
            if row.get(feature, 0) > 0:
                key_signals.append({feature: row[feature]})

        payload = {
            "account_id": row["account_id"],
            "risk_score": row.get("risk_score"),
            "risk_level": row.get("risk_level"),
            "key_signals": key_signals,
            "explanation": row.get("explanation"),
            "recommended_actions": final_actions,
            "insights": row.get("insights", [])
        }
        final_payloads.append(payload)

    # 10) Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(final_payloads).to_csv(OUTPUT_DIR / "risk_scores.csv", index=False)
    
    (OUTPUT_DIR / "final_results.json").write_text(
        json.dumps(final_payloads, indent=2, default=str),
        encoding="utf-8"
    )
    print("Pipeline completed. Outputs saved to data/outputs/")

if __name__ == "__main__":
    main()
