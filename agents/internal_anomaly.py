#!/usr/bin/env python
import argparse, yaml, pandas as pd, numpy as np, pathlib, joblib
from agents.common.io import FSBus
from agents.common.text import standardize_columns

TOPIC_INT = "internal.anomalies"

class Rolling:
    def __init__(self, w=28):
        self.w=w; self.buf=[]
    def push(self, x: float):
        self.buf.append(float(x))
        if len(self.buf)>self.w: self.buf.pop(0)
    def z(self):
        arr=np.array(self.buf,dtype=float)
        if len(arr)<7: return 0.0
        mu=arr.mean(); sd=arr.std(ddof=1) or 1e-6
        return float((arr[-1]-mu)/sd)
    def mad_z(self):
        arr=np.array(self.buf,dtype=float)
        if len(arr)<7: return 0.0
        med=np.median(arr); mad=np.median(np.abs(arr-med)) or 1e-6
        return float(0.6745*(arr[-1]-med)/mad)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fs","kafka"], default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config/config.yaml","r"))
    bus = FSBus(cfg["paths"]["topics_dir"])
    df = pd.read_csv("data/input/Adjusted_Retail_Sales_Data_with_Anomalies.csv")
    df = standardize_columns(df).sort_values("date")

    model_path = pathlib.Path(cfg["paths"]["models_dir"]) / "internal_iforest.joblib"
    clf = joblib.load(model_path) if model_path.exists() else None

    emitted=0
    for (sku, reg, cat), g in df.groupby(["sku","region","category"]):
        ru, rs, rp = Rolling(cfg["internal_anomaly"]["window"]), Rolling(cfg["internal_anomaly"]["window"]), Rolling(cfg["internal_anomaly"]["window"])
        for _, row in g.iterrows():
            units=float(row["units"]); sales=float(row["sales"]); price=sales/max(units,1e-6)
            ru.push(units); rs.push(sales); rp.push(price)
            zu, zs, zp = ru.z(), rs.z(), rp.mad_z()

            a_type=None
            if abs(zp)>=cfg["internal_anomaly"]["mad_th"]:
                a_type="Price jump"
            elif abs(zu)>=cfg["internal_anomaly"]["z_th"] or abs(zs)>=cfg["internal_anomaly"]["z_th"]:
                a_type="KPI anomaly"
            det = "statistical" if a_type else None

            if clf is not None:
                import numpy as np
                feat = np.array([[units, sales, price]])
                pred = clf.predict(feat)[0]  # -1 anomaly, 1 normal
                if pred==-1:
                    if not a_type: a_type="ML anomaly"
                    det = "ml" if det is None else "both"

            if a_type:
                anom = {
                    "anomaly_id": f"{sku}_{str(row['date'].date())}",
                    "date": str(row["date"].date()),
                    "sku": sku, "category": cat, "region": reg, "type": a_type,
                    "detector": det,
                    "metrics": {"units": units, "sales": sales, "avg_price": price,
                                "z_units": zu, "z_sales": zs, "mad_z_price": zp}
                }
                bus.produce(TOPIC_INT, anom); emitted+=1

    print(f"Emitted {emitted} internal anomalies → {cfg['paths']['topics_dir']}/{TOPIC_INT}.jsonl")

if __name__ == "__main__":
    main()
