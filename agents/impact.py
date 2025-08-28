#!/usr/bin/env python
import argparse, yaml, json, pathlib, numpy as np, pandas as pd
from dateutil import parser as dtp

def its_effect(series, t0_idx, horizon=7):
    y=np.array(series,dtype=float); x=np.arange(len(y))
    pre=slice(0,t0_idx); post=slice(t0_idx, min(len(y), t0_idx+horizon))
    I=(x>=t0_idx).astype(float)
    X=np.c_[np.ones_like(x), x, I, (x-t0_idx)*I]
    beta,*_=np.linalg.lstsq(X[pre], y[pre], rcond=None)
    yhat=X@beta
    eff=y[post]-yhat[post]
    delta=float(eff.sum()); rel=float(delta/(yhat[post].sum()+1e-9))*100.0
    return {"effect_abs": round(delta,2), "effect_pct": round(rel,2)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fs","kafka"], default=None)
    ap.add_argument("--horizon", type=int, default=7)
    args = ap.parse_args()

    cfg=yaml.safe_load(open("config/config.yaml","r"))
    corr_path=pathlib.Path(cfg["paths"]["outputs"]) / "reports" / "correlations.json"
    if not corr_path.exists(): print("Run correlate first."); return
    corr=json.load(open(corr_path,"r"))

    df=pd.read_csv("data/input/Adjusted_Retail_Sales_Data_with_Anomalies.csv")
    from agents.common.text import standardize_columns
    df=standardize_columns(df)

    results=[]
    for entry in corr:
        if not entry["top_matches"]: continue
        anom_id=entry["internal_anomaly_id"]
        sku, date_str = anom_id.rsplit("_",1)
        t0 = dtp.parse(date_str).date()
        g=df[df["sku"]==sku].sort_values("date")
        if g.empty: continue
        window = g[(g["date"] >= (pd.Timestamp(t0)-pd.Timedelta(days=28))) & (g["date"] <= (pd.Timestamp(t0)+pd.Timedelta(days=args.horizon)))]
        if len(window)<14: continue
        series = window["sales"].to_list()
        t0_idx = (window["date"]==pd.Timestamp(t0)).idxmax() - window.index.min()
        eff = its_effect(series, t0_idx, horizon=args.horizon)
        results.append({
            "internal_anomaly_id": anom_id,
            "best_cause": entry["top_matches"][0],
            "impact": eff,
            "metric": "sales",
            "window_start": str(window["date"].min().date()),
            "window_end": str(window["date"].max().date())
        })

    outp=pathlib.Path(cfg["paths"]["outputs"]) / "reports" / "impact.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(outp,"w"), indent=2)
    print(f"Wrote impact → {outp}")

if __name__ == "__main__":
    main()
