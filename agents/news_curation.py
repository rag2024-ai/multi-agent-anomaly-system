#!/usr/bin/env python
import argparse, yaml, pathlib
from agents.common.io import FSBus
from agents.common.text import clean_text, infer_event_type
from agents.common.vectors import Vectorizer
from annoy import AnnoyIndex

TOPIC_RAW = "raw.news.html"
TOPIC_CUR = "news.curated"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fs","kafka"], default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config/config.yaml","r"))
    bus = FSBus(cfg["paths"]["topics_dir"])
    raw = bus.read_all(TOPIC_RAW)
    if not raw:
        print("No raw news found. Run agents/news_generate.py first."); return

    texts=[]; curated=[]
    for r in raw:
        text = clean_text((r.get("title","")+" "+r.get("summary","")).strip())
        if not text: continue
        curated.append({
            "news_id": r["news_id"],
            "published_at": r["published_at"],
            "title": r.get("title",""),
            "summary": r.get("summary",""),
            "region": r.get("region","EU"),
            "categories": r.get("categories",["Electronics"]),
            "event_type": infer_event_type(text),
            "source_score": r.get("source_score",0.7),
            "text": text
        })
        texts.append(text)

    vec = Vectorizer(cfg["paths"]["models_dir"], n_components=256)
    Z = vec.fit(texts)

    d = Z.shape[1]
    idx = AnnoyIndex(d, "angular")
    for i, v in enumerate(Z):
        idx.add_item(i, v.tolist())
    idx.build(10)
    idx.save(str(pathlib.Path(cfg["paths"]["models_dir"]) / "annoy.index"))

    for i, item in enumerate(curated):
        item["vec_index"] = i
        bus.produce(TOPIC_CUR, item)

    print(f"Curated {len(curated)} items → {cfg['paths']['topics_dir']}/{TOPIC_CUR}.jsonl")

if __name__ == "__main__":
    main()
