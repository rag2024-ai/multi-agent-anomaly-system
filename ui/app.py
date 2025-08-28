import streamlit as st, pandas as pd, json
from pathlib import Path

out_dir = Path("data/outputs")

st.sidebar.title("Multi-Agent Anomaly Demo")
choice = st.sidebar.radio("View", ["Curated News", "External Anomalies", "Internal Anomalies", "Correlations", "Impact Analysis"])

def df_jsonl(path):
    try:
        return pd.read_json(path, lines=True)
    except Exception:
        return pd.DataFrame()

if choice == "Curated News":
    st.subheader("Curated News Articles")
    p = out_dir/"topics/news.curated.jsonl"
    if not p.exists():
        st.warning("No curated news yet. Run: bash scripts/run_all_fs.sh")
    else:
        st.dataframe(df_jsonl(p))

elif choice == "External Anomalies":
    st.subheader("External Anomalies")
    p = out_dir/"topics/news.anomalies.jsonl"
    if not p.exists():
        st.warning("No external anomalies yet. Run the pipeline.")
    else:
        st.dataframe(df_jsonl(p))

elif choice == "Internal Anomalies":
    st.subheader("Internal Anomalies")
    p = out_dir/"topics/internal.anomalies.jsonl"
    if not p.exists():
        st.warning("No internal anomalies yet.")
    else:
        st.dataframe(df_jsonl(p))

elif choice == "Correlations":
    st.subheader("Correlations")
    p = out_dir/"reports/correlations.json"
    if not p.exists():
        st.warning("No correlations report yet.")
    else:
        data = json.load(open(p))
        st.json(data)

elif choice == "Impact Analysis":
    st.subheader("Impact Analysis")
    p = out_dir/"reports/impact.json"
    if not p.exists():
        st.warning("No impact report yet.")
    else:
        data = json.load(open(p))
        st.json(data)
