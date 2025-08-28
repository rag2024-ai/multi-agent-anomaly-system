# Multi‑Agent Anomaly Correlation & Impact — Full Demo

This repository is a **ready-to-run demo** of a multi-agent system that:
- Generates **synthetic news** aligned with anomalies detected in your internal dataset
- Curates & embeds news
- Detects **external anomalies** from news (statistical + ML / Isolation Forest with persistence)
- Detects **internal anomalies** from your dataset (statistical + ML / Isolation Forest with persistence)
- **Correlates** external ↔ internal anomalies
- Estimates **impact** (Interrupted Time Series)
- Presents results in a **Streamlit UI**

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Place your dataset at `data/input/Adjusted_Retail_Sales_Data_with_Anomalies.csv`

**Accepted column names (auto-mapped):**
- date: `date` or `Date`
- units: `units` or `Units_Sold`
- sales: `sales` or `Sales_Amount`
- optional: `region`/`Region`, `category`/`Category`, `sku`/`SKU`

### Run the full pipeline (Filesystem mode)
```bash
bash scripts/run_all_fs.sh
```

### Launch the UI
```bash
streamlit run ui/app.py
```
Open http://localhost:8501

## Outputs
- Topics (newline‑delimited JSON):
  - `data/outputs/topics/news.curated.jsonl`
  - `data/outputs/topics/news.anomalies.jsonl`
  - `data/outputs/topics/internal.anomalies.jsonl`
- Reports (JSON):
  - `data/outputs/reports/correlations.json`
  - `data/outputs/reports/impact.json`
- Models (persisted):
  - `data/outputs/models/internal_iforest.joblib`
  - `data/outputs/models/external_iforest.joblib`

## Notes
- We pin **NumPy 1.26.x** to avoid ABI conflicts some native deps have with NumPy 2.
- External anomaly ML detector is **unsupervised (Isolation Forest)**.
- Internal anomaly ML detector uses **Isolation Forest** on `[units, sales, avg_price]`.
- If you change the dataset schema, adjust mapping in `agents/common/text.py::standardize_columns`.
