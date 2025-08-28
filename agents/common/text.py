import re
import pandas as pd

WS = re.compile(r"\s+")

def clean_text(s: str) -> str:
    s = (s or "").replace("\u00A0"," ")
    s = WS.sub(" ", s).strip()
    return s

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    # date
    if "date" in cols:
        df["date"] = pd.to_datetime(df[cols["date"]], errors="coerce")
    elif "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        raise ValueError("Missing date/Date column")

    # units
    if "units" in cols:
        df["units"] = pd.to_numeric(df[cols["units"]], errors="coerce")
    elif "units_sold" in cols:
        df["units"] = pd.to_numeric(df[cols["units_sold"]], errors="coerce")
    else:
        raise ValueError("Missing units/Units_Sold column")

    # sales
    if "sales" in cols:
        df["sales"] = pd.to_numeric(df[cols["sales"]], errors="coerce")
    elif "sales_amount" in cols:
        df["sales"] = pd.to_numeric(df[cols["sales_amount"]], errors="coerce")
    else:
        raise ValueError("Missing sales/Sales_Amount column")

    # optional
    df["region"] = df[cols.get("region","region")] if "region" in cols else "EU"
    df["category"] = df[cols.get("category","category")] if "category" in cols else "Electronics"
    df["sku"] = df[cols.get("sku","sku")] if "sku" in cols else "SKU_1"
    df["avg_price"] = df["sales"] / df["units"].replace(0, pd.NA)
    return df

def infer_event_type(t: str) -> str:
    t = (t or "").lower()
    if any(k in t for k in ["tariff","regulation","policy","duty","compliance"]):
        return "RegulatoryChange"
    if any(k in t for k in ["strike","walkout","labor","union"]):
        return "LaborStrike"
    if any(k in t for k in ["port","shipping","logistics","congestion","suez"]):
        return "SupplyChainDisruption"
    if any(k in t for k in ["recall","defect","safety"]):
        return "ProductRecall"
    if any(k in t for k in ["storm","flood","earthquake","cyclone"]):
        return "WeatherDisaster"
    return "GeneralEvent"

def infer_region(title: str, fallback="EU") -> str:
    t = (title or "").lower()
    if any(k in t for k in ["eu","europe","germany","france","italy"]): return "EU"
    if any(k in t for k in ["united states","us "," usa","america"]): return "NA"
    if any(k in t for k in ["india","china","apac","japan","korea"]): return "APAC"
    if any(k in t for k in ["brazil","latam","mexico","argentina"]): return "LATAM"
    return fallback

def infer_category(text: str, fallback="Electronics") -> str:
    t = (text or "").lower()
    if any(k in t for k in ["laptop","smartphone","semiconductor","electronics","device"]): return "Electronics"
    if any(k in t for k in ["apparel","clothing","fashion"]): return "Apparel"
    if any(k in t for k in ["grocery","food","beverage"]): return "Grocery"
    return fallback
