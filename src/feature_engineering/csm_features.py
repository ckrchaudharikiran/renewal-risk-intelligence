"""Feature engineering for CSM engagement signals."""

import json
import re
from pathlib import Path
from typing import Callable
import pandas as pd
from rapidfuzz import process, fuzz

from src.llm.prompts import extract_csm_signals_prompt

CACHE_PATH = Path("data/processed/csm_llm_cache.json")

def _load_cache() -> dict:
    """Load cached CSM feature extraction results from disk."""
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

def _save_cache(cache: dict) -> None:
    """Persist CSM feature extraction cache to disk."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")

def _split_notes(csm_text: str) -> list[str]:
    """Split raw CSM notes into individual blocks."""
    blocks = re.split(r'\n---\n', csm_text)
    return [b.strip() for b in blocks if b.strip()]

def _map_note_to_account(note: str, accounts_df: pd.DataFrame):
    """Extract account ID from note using regex and fuzzy matching."""
    if accounts_df is None or accounts_df.empty or "account_id" not in accounts_df.columns:
        return None

    # 1. Try regex for account id (e.g. "acct 1001", "#1007", "account 1016", "(1009)")
    match = re.search(r'(?:acct|account|#|\()\s*(\d{4})', note, flags=re.IGNORECASE)
    if match:
        acct_id = match.group(1)
        if acct_id in accounts_df['account_id'].astype(str).values:
            return acct_id

    # 2. Try fuzzy match for account name
    # Heuristic: the account name is usually on the first or second line
    lines = note.split('\n')
    header = " ".join(lines[:2])[:150]
    
    if "account_name" not in accounts_df.columns:
        return None

    name_to_id = {str(row['account_name']): str(row['account_id']) 
                  for _, row in accounts_df.dropna(subset=['account_name']).iterrows()}
    
    if name_to_id:
        result = process.extractOne(header, name_to_id.keys(), scorer=fuzz.token_set_ratio)
        if result and result[1] > 80:
            return name_to_id[result[0]]
            
    return None

def create_csm_features(csm_text: str, accounts_df: pd.DataFrame, llm_client: Callable[[str], str]) -> pd.DataFrame:
    """Extract account-level CSM risk signals using an LLM with local caching."""
    notes = _split_notes(csm_text)
    cache = _load_cache()
    records = []

    for note in notes:
        if note.startswith('=== CSM Call Notes'):
            continue
            
        account_id = _map_note_to_account(note, accounts_df)
        if not account_id:
            continue
            
        cache_key = f"{account_id}::{note}"
        if cache_key in cache:
            parsed = cache[cache_key]
        else:
            prompt = extract_csm_signals_prompt(note)
            try:
                llm_response = llm_client(prompt)
                payload = json.loads(llm_response)
                
                sentiment = str(payload.get("sentiment", "neutral")).lower()
                churn_risk = str(payload.get("churn_risk", "low")).lower()
                key_issues_raw = payload.get("key_issues", [])
                
                sentiment_score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
                sentiment_score = sentiment_score_map.get(sentiment, 0.0)
                churn_flag = 1 if churn_risk in {"medium", "high"} else 0
                
                if isinstance(key_issues_raw, list):
                    risk_keywords = [str(issue).strip() for issue in key_issues_raw if str(issue).strip()]
                else:
                    risk_keywords = [str(key_issues_raw)]
                    
                parsed = {
                    "sentiment_score": sentiment_score,
                    "churn_flag": churn_flag,
                    "risk_keywords": risk_keywords
                }
                cache[cache_key] = parsed
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
                continue
                
        records.append({"account_id": account_id, **parsed})

    _save_cache(cache)
    
    if not records:
        return pd.DataFrame(columns=["account_id", "sentiment_score_normalized", "churn_flag", "risk_keywords"])
        
    df = pd.DataFrame(records)
    
    # Aggregate multiple notes per account
    grouped = df.groupby("account_id").agg({
        "sentiment_score": "mean",
        "churn_flag": "max",
        "risk_keywords": lambda x: list(set(sum(x, [])))
    }).reset_index()
    
    # Normalize sentiment_score (-1 to 1) -> (0 to 1)
    grouped["sentiment_score_normalized"] = (grouped["sentiment_score"] + 1) / 2.0
    grouped = grouped.drop(columns=["sentiment_score"])
    
    return grouped

def build_csm_features(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Use create_csm_features")
