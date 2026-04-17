"""Streamlit app for renewal risk intelligence."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

OUTPUT_DIR = Path("data/outputs")
FINAL_RESULTS_PATH = OUTPUT_DIR / "final_results.json"

st.set_page_config(page_title="Renewal Risk Intelligence", layout="wide")

def _load_outputs() -> list[dict]:
    """Load final results JSON."""
    if not FINAL_RESULTS_PATH.exists():
        return []
    
    try:
        return json.loads(FINAL_RESULTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

st.title("Renewal Risk Intelligence Dashboard")
st.caption("Identify at-risk accounts, view insights, and take action.")

data = _load_outputs()
if not data:
    st.warning("No output files found in data/outputs/. Run the pipeline first.")
    st.stop()

# Convert to DF for filtering
df = pd.DataFrame(data)

risk_levels = ["All"] + sorted(df["risk_level"].dropna().unique().tolist())
col1, col2 = st.columns([1, 3])
with col1:
    selected_level = st.selectbox("Filter by risk level", options=risk_levels)

filtered = df.copy()
if selected_level != "All":
    filtered = filtered[filtered["risk_level"] == selected_level]

st.subheader("Account Risk Dashboard")

# Display a high-level table
display_cols = ["account_id", "risk_level", "risk_score"]
st.dataframe(filtered[display_cols], use_container_width=True)

st.subheader("Detailed Account Insights")
for _, row in filtered.iterrows():
    acct_id = row["account_id"]
    r_level = row["risk_level"]
    r_score = row["risk_score"]
    
    with st.expander(f"Account {acct_id} | Risk: {r_level} | Score: {r_score}"):
        
        # Two columns layout inside expander
        ecol1, ecol2 = st.columns(2)
        
        with ecol1:
            st.markdown("### 💡 Explanation")
            st.markdown(row.get("explanation", "No explanation available."))
            
            st.markdown("### ⚡ Key Signals")
            signals = row.get("key_signals", [])
            if signals:
                for s in signals:
                    for k, v in s.items():
                        st.markdown(f"- **{k}**: {v}")
            else:
                st.markdown("No major risk signals.")
        
        with ecol2:
            st.markdown("### 🎯 Recommended Actions")
            actions = row.get("recommended_actions", [])
            if actions:
                for a in actions:
                    st.markdown(f"- {a}")
            else:
                st.markdown("No actions recommended.")
            
            st.markdown("### 🔎 Killer Insights")
            insights = row.get("insights", [])
            if insights:
                for ins in insights:
                    st.info(f"**{ins.get('insight_type', 'Insight')}**: {ins.get('description', '')}")
            else:
                st.markdown("_No cross-signal insights detected._")
