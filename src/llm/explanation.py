"""LLM explanation generation utilities."""

import json
import pandas as pd
from src.llm.prompts import generate_explanation_prompt

def _parse_explanation_response(response_text: str) -> tuple[str, str]:
    """Parse LLM response into explanation and recommended actions."""
    explanation = "Could not generate explanation."
    recommended_actions = "Review account signals."

    try:
        payload = json.loads(response_text)
        explanation = str(payload.get("explanation", explanation)).strip()
        actions = payload.get("recommended_actions", recommended_actions)
        if isinstance(actions, list):
            recommended_actions = "; ".join(str(item).strip() for item in actions if str(item).strip())
        else:
            recommended_actions = str(actions).strip() or recommended_actions
        return explanation, recommended_actions
    except Exception:
        pass

    return explanation, recommended_actions

def generate_explanations(features_df: pd.DataFrame, llm_client) -> pd.DataFrame:
    """Generate LLM explanations and recommended actions per account."""
    if "account_id" not in features_df.columns:
        raise ValueError("features_df must include 'account_id'.")

    df = features_df.copy()
    explanations = []
    actions = []

    for _, row in df.iterrows():
        feature_payload = row.to_dict()
        prompt = generate_explanation_prompt(json.dumps(feature_payload, default=str, indent=2))
        response_text = llm_client(prompt)
        explanation, recommended_actions = _parse_explanation_response(response_text)
        explanations.append(explanation)
        actions.append(recommended_actions)

    df["explanation"] = explanations
    df["llm_recommended_actions"] = actions
    return df

def generate_risk_explanation(customer_context: str) -> str:
    """Backward-compatible helper returning raw LLM prompt for context."""
    return generate_explanation_prompt(customer_context)
