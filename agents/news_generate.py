#!/usr/bin/env python
import argparse, yaml, pandas as pd, numpy as np, random
from agents.common.io import FSBus, ensure_dir
from agents.common.text import standardize_columns, clean_text, infer_event_type, infer_region, infer_category

TOPIC_RAW = "raw.news.html"

def detect_simple_anomalies(g: pd.DataFrame):
    g = g.sort_values("date").copy()
    g["price"] = g["sales"] / g["units"].replace(0, np.nan)
    def rolling_z(x, w=14):
        r = (x - x.rolling(w, min_periods=7).mean()) / (x.rolling(w, min_periods=7).std().replace(0,1))
        return r
    def mad_z(x, w=14):
        med = x.rolling(w, min_periods=7).median()
        mad = (x.rolling(w, min_periods=7).apply(lambda v: np.median(np.abs(v - np.median(v))), raw=True)).replace(0,1e-6)
        return 0.6745*(x - med)/mad
    g["zu"] = rolling_z(g["units"])
    g["zs"] = rolling_z(g["sales"])
    g["zp"] = mad_z(g["price"])
    g["atype"] = np.where(np.abs(g["zp"])>=2.5, "Price jump",
                  np.where((np.abs(g["zu"])>=2.0)|(np.abs(g["zs"])>=2.0), "KPI anomaly", None))
    return g[g["atype"].notna()][["date","region","category","sku","atype","zu","zs","zp"]]

def synth_title(cat, reg, evt):
    stems = {
        "RegulatoryChange":[
            "New tariff policy impacts {} in {}",
            "{} sector faces compliance overhaul across {}",
            "{} duties revised across {}"
        ],
        "LaborStrike":[
            "Labor strike disrupts {} logistics in {}",
            "Union action halts shipments for {} in {}"
        ],
        "SupplyChainDisruption":[
            "Port congestion delays {} deliveries in {}",
            "Global shipping backlog hits {} in {}"
        ],
        "ProductRecall":[
            "Safety recall announced for {} across {}",
            "Quality issue triggers {} product recall in {}"
        ],
        "WeatherDisaster":[
            "Severe weather impacts {} distribution in {}",
            "Flooding disrupts {} warehouses in {}"
        ],
        "GeneralEvent":[
            "{} market watch across {}"
        ]
    }
    tmpl = random.choice(stems.get(evt, stems["GeneralEvent"]))
    try:
        return tmpl.format(cat, reg)
    except:
        return f"{evt} affects {cat} in {reg}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fs","kafka"], default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config/config.yaml","r"))
    bus = FSBus(cfg["paths"]["topics_dir"])

    df = pd.read_csv("data/input/Adjusted_Retail_Sales_Data_with_Anomalies.csv")
    df = standardize_columns(df)

    anomalies = []
    for (reg, cat), g in df.groupby(["region","category"]):
        an = detect_simple_anomalies(g)
        if not an.empty:
            anomalies.append(an)
    an_all = pd.concat(anomalies, ignore_index=True) if anomalies else pd.DataFrame(columns=["date","region","category","sku","atype","zu","zs","zp"])

    random.seed(42)
    count = 0
    for _, r in an_all.iterrows():
        evt = random.choice(["RegulatoryChange","LaborStrike","SupplyChainDisruption","ProductRecall","WeatherDisaster"])
        title = synth_title(r["category"], r["region"], evt)
        summary = f"{title}. Analysts expect short-term volatility. Category={r['category']}, Region={r['region']}."
        item = {
            "news_id": f"news-{str(pd.to_datetime(r['date']).date())}-{random.randint(1000,9999)}",
            "published_at": pd.to_datetime(r["date"]).isoformat(),
            "title": title,
            "summary": summary,
            "html": f"<html><body><h1>{title}</h1><p>{summary}</p></body></html>",
            "region": infer_region(title, r["region"]),
            "categories": [infer_category(summary, r["category"])],
            "event_type": infer_event_type(summary),
            "source_score": round(random.uniform(0.6,0.95),2)
        }
        bus.produce(TOPIC_RAW, item)
        count += 1

    print(f"Wrote {count} synthetic raw news → {cfg['paths']['topics_dir']}/{TOPIC_RAW}.jsonl")

if __name__ == "__main__":
    main()
