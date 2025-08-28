EXPECT = {
    ("RegulatoryChange","Electronics"): {"price":"+","units":"-"},
    ("SupplyChainDisruption","Electronics"): {"price":"+","units":"-"},
    ("LaborStrike","Electronics"): {"units":"-","sales":"-"},
    ("ProductRecall","Electronics"): {"units":"-","sales":"-"}
}

def direction_bonus(evt: str, cat: str, metrics: dict) -> float:
    exp = EXPECT.get((evt, cat), {})
    b=0.0
    if exp.get("price")=="+":
        if metrics.get("mad_z_price",0)>0 or metrics.get("z_price",0)>0: b+=0.05
    if exp.get("units")=="-" and metrics.get("z_units",0)<0: b+=0.05
    return b
