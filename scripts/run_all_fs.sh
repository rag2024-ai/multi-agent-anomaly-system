#!/usr/bin/env bash
set -euo pipefail
PY=${PYTHON:-python}

echo "▶ Generating synthetic news from dataset anomalies..."
$PY -m agents.news_generate --mode fs

echo "▶ Curating news and building embeddings..."
$PY -m agents.news_curation --mode fs

echo "▶ Detecting external anomalies (stat+ML)..."
$PY -m agents.external_anomaly --mode fs

echo "▶ Detecting internal anomalies (stat+ML)..."
$PY -m agents.internal_anomaly --mode fs

echo "▶ Correlating internal ↔ external anomalies..."
$PY -m agents.correlate --mode fs

echo "▶ Estimating impact with ITS..."
$PY -m agents.impact --mode fs

echo "✅ Done. Open Streamlit with: streamlit run ui/app.py"

