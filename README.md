# Renewal Risk Intelligence 🚨

A robust, lightweight data engineering and AI system built to detect customer renewal risk, explain why risk is rising, provide killer insights, and suggest rule-based next actions for Customer Success teams.

## Problem Statement

B2B teams often discover renewal risk too late because customer signals are fragmented across usage logs, support tickets, NPS responses, CSM notes, and product updates.  
This project unifies those signals into a single, standardized risk view so teams can proactively intervene 90 days before churn.

## Architecture Overview

The pipeline follows an end-to-end flow:

1. **Ingestion**: Load raw files from `data/raw` (CSV + text/markdown).
2. **Preprocessing & Filtering**: Clean data, resolve account identities using fuzzy matching (`rapidfuzz`), and **filter for accounts renewing within the next 90 days**.
3. **Feature Engineering**: Build explicitly mapped, normalized (`0-1`) risk features from usage, support, NPS, product, and CSM signals.
4. **Scoring**: Compute a mathematically sound weighted renewal `risk_score` and assign `risk_level` (High/Medium/Low).
5. **Insights & Action Engine**: Execute rule-based diagnostics to generate deterministic recommended actions and cross-signal **"Killer Insights"** (e.g. *Silent Churn*).
6. **LLM Layer**: Use LLM to extract signals from messy CSM notes and generate coherent markdown explanations. Includes smart semantic fallback heuristics if API quota is reached.
7. **Serving**: Save all structured outputs to `data/outputs/final_results.json` and visualize cleanly in Streamlit.

---

## Feature Engineering

All features are strictly numeric and normalized between `0` and `1`:

- **Usage**
  - `usage_trend_score`: Measures Month-over-Month adoption changes.
  - `low_usage_flag`: Flags accounts in the bottom 25th percentile of volume.
- **Support**
  - `p1_ticket_ratio`: Percentage of tickets designated as P1 severity.
  - `open_ticket_flag`: Flags accounts with unresolved support burdens.
- **NPS**
  - `nps_score_normalized`: Aggregate sentiment standardized to a decimal.
  - `detractor_flag`: Flags accounts averaging an NPS <= 6.
- **CSM**
  - `sentiment_score_normalized`: Natural language sentiment extracted from notes.
  - `churn_flag`: Explicit churn risk detection based on CSM commentary.
  - `risk_keywords`: Categorized tags (e.g. "budget", "competitor").
- **Product**
  - `migration_risk_flag`: Flagged when accounts use deprecated features or complain about migration difficulties.

---

## Risk Scoring Engine

Risk scoring uses explicit mathematical feature mapping, eliminating unpredictable string-matching hacks.

- Normalize numeric feature values and calculate sub-scores per category.
- Apply explicit weights:
  - `usage`: 25%
  - `support`: 20%
  - `nps`: 15%
  - `csm`: 25%
  - `product`: 15%
- Aggregate weighted components into a final `risk_score` in `[0, 1]`.
- Enforce strict Risk Tiers:
  - **High**: > 0.65
  - **Medium**: > 0.35
  - **Low**: <= 0.35

---

## Killer Insights 🔎

The newly introduced rule-based action engine detects cross-signal "Killer Insights" that single metrics might miss:
- **Silent Churn**: Detects accounts with *High NPS* but *Low Usage*. (They like the team, but abandoned the product).
- **Migration Chaos**: Detects accounts with *Migration Risk* combined with *Open Support Tickets*.
- **Executive Escalation**: Identifies high-stakes accounts where titles like *CTO*, *CISO*, or *VP* have joined recent calls.

---

## Meaningful LLM Integration

LLMs are used strategically, ensuring structured execution without hallucination:

1. **CSM Signal Extraction (Strict JSON)**
   - Parses deeply unstructured CSM notes into JSON keys: sentiment, churn risk, and risk keywords.
   - Leverages local file caching to prevent redundant LLM API calls and save money.
2. **Graceful Degradation**
   - If the OpenAI API is unreachable or out of quota (`429`), the pipeline automatically fails over to a smart, heuristic text-parsing engine that assigns risk tags based on localized keyword semantics.

---

## How To Run Project

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

*(Note: If you run into issues launching Streamlit due to missing dependencies, manually install them with: `pip install altair cachetools tenacity`)*

### 2) Configure environment variables

Create or update `.env` in the root directory:

```bash
OPENAI_API_KEY=your_api_key_here
```

### 3) Run the pipeline

To run with the real LLM enabled:
```bash
PIPELINE_ENABLE_LLM=1 python src/pipeline.py
```

To run **without** hitting OpenAI (bypasses API, uses local smart heuristics, completely free):
```bash
python src/pipeline.py
```

Outputs will be saved strictly to:
- `data/outputs/risk_scores.csv`
- `data/outputs/final_results.json`

### 4) Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

This will automatically load `http://localhost:8501` (or nearest available port) in your browser.
