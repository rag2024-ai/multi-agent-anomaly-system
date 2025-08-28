#!/usr/bin/env python
import argparse, yaml, numpy as np, pathlib, joblib
from dateutil import parser as dtp
from agents.common.io import FSBus
from agents.common.vectors import Vectorizer

TOPIC_CUR = "news.curated"
TOPIC_EXT = "news.anomalies"

def ewma_z(series, alpha=0.3):
    mu=0.0; dev=1e-6; z=0.0
    for x in series:
        mu = alpha*x + (1-alpha)*mu
        dev = 0.9*dev + 0.1*abs(x-mu)
        z = (x-mu)/max(dev,1e-6)
    return float(z)

def novelty(vec, hist):
    if not hist: return 1.0
    H = np.vstack(hist)
    sims = (H @ vec) / (np.linalg.norm(H,axis=1)*np.linalg.norm(vec)+1e-9)
    return float(1.0 - float(sims.max()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fs","kafka"], default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config/config.yaml","r"))
    bus = FSBus(cfg["paths"]["topics_dir"])
    cur = bus.read_all(TOPIC_CUR)
    if not cur:
        print("No curated news found."); return

    vec = Vectorizer(cfg["paths"]["models_dir"])
    texts = [c["title"]+" "+c["summary"] for c in cur]
    Z = vec.transform(texts)

    model_path = pathlib.Path(cfg["paths"]["models_dir"]) / "external_iforest.joblib"
    clf = joblib.load(model_path) if model_path.exists() else None

    by_key = {}
    for item, z in zip(cur, Z):
        d = dtp.parse(item["published_at"]).date()
        key = (item["region"], item["categories"][0])
        by_key.setdefault(key, {}).setdefault(d, []).append((item, z))

    emitted=0
    for key, daymap in by_key.items():
        days = sorted(daymap.keys())
        hist_vecs=[]; counts=[]
        for d in days:
            items = daymap[d]
            counts.append(len(items))
            for item, z in items:
                b = ewma_z(counts[-min(30,len(counts)):], alpha=cfg["external_anomaly"]["ewma_alpha"])
                n = novelty(z, hist_vecs[-cfg["external_anomaly"]["recent_k"]:] if hist_vecs else [])
                s = len(items)
                b_term = max(0.0, min(1.0, b/3.0))
                s_term = min(1.0, s / max(1, cfg["external_anomaly"]["min_support"]))
                stat_score = 0.35*b_term + 0.45*n + 0.20*s_term

                ml_score = None
                if clf is not None:
                    import numpy as np
                    feat = np.array([[b, n, s, item.get("source_score",0.7)]])
                    pred = clf.decision_function(feat) if hasattr(clf,"decision_function") else -clf.score_samples(feat)
                    ml_score = float(1/(1+np.exp(-pred)))

                final = stat_score
                det = "statistical"
                if ml_score is not None:
                    final = 0.5*stat_score + 0.5*ml_score
                    det = "both" if stat_score>=cfg["external_anomaly"]["threshold"] else "ml"

                if final >= cfg["external_anomaly"]["threshold"]:
                    bus.produce(TOPIC_EXT, {
                        "anomaly_id": f"ext_{item['news_id']}",
                        "news_id": item["news_id"],
                        "published_at": item["published_at"],
                        "region": key[0],
                        "categories": [key[1]],
                        "event_type": item["event_type"],
                        "detector": det,
                        "signals": {"burst_z": round(b,2), "novelty": round(n,3), "support": s},
                        "score": round(final,3)
                    })
                    emitted += 1
            for _, z in items: hist_vecs.append(z)

    print(f"Emitted {emitted} external anomalies → {cfg['paths']['topics_dir']}/{TOPIC_EXT}.jsonl")

if __name__ == "__main__":
    main()
