"""Prompt templates for LLM-based explanations."""

def extract_csm_signals_prompt(note_text: str) -> str:
    """Build a prompt to extract structured CSM signals from notes."""
    return (
        "You are an analyst extracting renewal-risk signals from CSM notes.\n"
        "Return concise JSON with keys: sentiment, churn_risk, key_issues, competitor_mentions.\n"
        "Rules:\n"
        "- sentiment: positive | neutral | negative\n"
        "- churn_risk: low | medium | high\n"
        "- key_issues: list of short phrases\n"
        "- competitor_mentions: list of competitor names, empty if none\n\n"
        f"CSM notes:\n{note_text}"
    )

def generate_explanation_prompt(account_features: str) -> str:
    """Build a prompt to explain account risk and recommend actions."""
    return (
        "You are a customer success analyst.\n"
        "Given account features, explain why the account is at renewal risk and what to do next.\n"
        "Return concise JSON with keys: 'explanation', 'recommended_actions'.\n"
        "For 'explanation', provide a markdown string explaining the risk.\n"
        "For 'recommended_actions', provide a markdown list of 3-5 action items (or a string).\n"
        "- Focus on evidence from provided features.\n\n"
        f"Account features:\n{account_features}"
    )

def build_risk_explanation_prompt(customer_context: str) -> str:
    """Backward-compatible wrapper for explanation prompt generation."""
    return generate_explanation_prompt(customer_context)
