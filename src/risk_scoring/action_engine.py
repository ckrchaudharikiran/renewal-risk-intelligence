"""Action and insights generation module."""

import pandas as pd

def generate_insights_and_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Generate rule-based actions and killer insights."""
    insights = []
    actions = []
    
    for _, row in df.iterrows():
        acct_insights = []
        acct_actions = []
        
        # Rule-based actions
        if row.get("migration_risk_flag", 0) == 1:
            acct_actions.append("Assign migration specialist")
        if row.get("p1_ticket_ratio", 0) > 0.5 or row.get("open_ticket_flag", 0) == 1:
            acct_actions.append("Escalate support immediately")
        if row.get("low_usage_flag", 0) == 1:
            acct_actions.append("Drive adoption workshop")
        
        # Risk keywords matching for actions
        risk_keywords = row.get("risk_keywords", [])
        if isinstance(risk_keywords, list):
            kw_str = " ".join(risk_keywords).lower()
            if "budget" in kw_str or "discount" in kw_str or "price" in kw_str:
                acct_actions.append("Prepare discount strategy")
            
        # Killer Insights
        # Silent churn: high NPS + low usage
        nps_normalized = row.get("nps_score_normalized", 0)
        if nps_normalized >= 0.8 and row.get("low_usage_flag", 0) == 1:
            acct_insights.append({
                "insight_type": "Silent Churn",
                "description": "High NPS but low usage. Account might be churning silently."
            })
            
        # Migration risk
        if row.get("migration_risk_flag", 0) == 1 and row.get("open_ticket_flag", 0) == 1:
            acct_insights.append({
                "insight_type": "Migration + Support Issues",
                "description": "Account shows migration risk compounded by open support tickets."
            })
            
        # Executive escalation
        if isinstance(risk_keywords, list):
            kw_str = " ".join(risk_keywords).lower()
            if any(x in kw_str for x in ["cto", "vp", "ciso", "cro", "executive"]):
                acct_insights.append({
                    "insight_type": "Executive Escalation",
                    "description": "Executive mention detected in CSM notes."
                })
                
        insights.append(acct_insights)
        actions.append(acct_actions)
        
    df["insights"] = insights
    df["rule_based_actions"] = actions
    return df
