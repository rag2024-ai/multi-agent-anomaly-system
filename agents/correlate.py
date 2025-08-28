#!/usr/bin/env python
import argparse, yaml, json, pathlib
from dateutil import parser as dtp
from annoy import AnnoyIndex
from agents.common.io import FSBus
from agents.common.vectors import Vectorizer
from agents.common.rules import direction_bonus

TOPIC_CUR="news.curated"; TOPIC_EXT="news.anomalies"; TOPIC_INT="internal.anomalies"

def sig_internal(a: dict) -> str:
    m=a.get("metrics",{}); reg=a.get("region","EU"); cat=a.get("category","Electronics")
    return f"{a['type']} {reg} {cat} units_z:{m.get('z_units',0):.2f} price_z:{m.get('mad_z_price',0):.2f}"

def time_align(t_ext, t_int, max_days=7):
    d=(t_int - t_ext).days
    if d<0: return max(0.0, 1.0 + d/2.0)
    if d<=max_days: return (max_days - d)/max_days
    return 0.0

def geo_match(r_ext, r_int): return 1.0 if (r_ext or "")[:1].upper()==(r_int or "")[:1].upper() else 0.0
def cat_match(c_ext, c_int):
    s=set([x.lower() for x in c_ext or []])
    return 1.0 if (c_int or "").lower() in s else (0.5 if any((c_int or "").lower() in x for x in s) else 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fs","kafka"], default=None)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config/config.yaml","r"))
    bus = FSBus(cfg["paths"]["topics_dir"])
    cur = bus.read_all(TOPIC_CUR); ext = bus.read_all(TOPIC_EXT); itn = bus.read_all(TOPIC_INT)
    if not (cur and ext and itn):
        print("Missing inputs for correlation."); return

    vec = Vectorizer(cfg["paths"]["models_dir"])
    d = vec.transform(["probe"]).shape[1]
    ann = AnnoyIndex(d, "angular")
    ann.load(str(pathlib.Path(cfg["paths"]["models_dir"]) / "annoy.index"))

    results=[]
    for a in itn:
        v_int = vec.transform([sig_internal(a)])[0]
        ids, dists = ann.get_nns_by_vector(v_int.tolist(), 100, include_distances=True)
        cands=[]
        t_int = dtp.parse(a["date"]).date()
        for ix, dist in zip(ids, dists):
            n = cur[ix]
            t_score = time_align(dtp.parse(n["published_at"]).date(), t_int, max_days=7)
            g_score = geo_match(n.get("region"), a.get("region"))
            c_score = cat_match(n.get("categories"), a.get("category"))
            text_score = max(0.0, 1.0 - dist/2.0)
            score = 0.30*t_score + 0.20*g_score + 0.20*c_score + 0.30*text_score
            score += direction_bonus(n.get("event_type","GeneralEvent"), a.get("category","Electronics"), a.get("metrics",{}))
            if score>=0.65:
                cands.append({"news_id": n["news_id"], "overall": round(score,3),
                              "subs":{"time":round(t_score,3),"geo":g_score,"cat":c_score,"text":round(text_score,3)}})
        cands=sorted(cands, key=lambda x: x["overall"], reverse=True)[:args.topk]
        results.append({"internal_anomaly_id": a["anomaly_id"], "top_matches": cands})

    outp = pathlib.Path(cfg["paths"]["outputs"]) / "reports" / "correlations.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(outp,"w"), indent=2)
    print(f"Wrote correlations → {outp}")

if __name__ == "__main__":
    main()
