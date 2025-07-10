#!/usr/bin/env python
"""
Build first-wave features for Lahn Inc. sales data:
  • is_holiday_<country>   (public holidays in the transaction’s country)
  • profit_per_unit        (Profit ÷ Order_Quantity)
  • margin_pct             (Profit ÷ Revenue)
The script reads the cleaned parquet and writes an enriched parquet
so DVC can cache and version it.
"""
import pandas as pd, numpy as np, pathlib, holidays, sys, json

INFILE  = pathlib.Path("data/processed/sales_clean.parquet")
OUTFILE = pathlib.Path("data/feature/sales_feat_v1.parquet")

COUNTRY_NAME_TO_ISO = {
    "United States": "US",
    "USA": "US",
    "United Kingdom": "GB",
    "Canada": "CA",
    "Australia": "AU",
    "France": "FR",
    "Germany": "DE",
    "Mexico": "MX",
    # add others if they appear
}

def main():
    df = pd.read_parquet(INFILE)

    # ---------------- profit metrics -----------------
    df["profit_per_unit"] = df["Profit"] / df["Order_Quantity"]
    df["margin_pct"]      = df["Profit"] / df["Revenue"]

    # ---------------- holiday flags ------------------
    # cache holiday calendars by iso code
    cal_cache = {}
    def is_hol(row):
        iso = COUNTRY_NAME_TO_ISO.get(row["Country"])
        if not iso:
            return False
        if iso not in cal_cache:
            cal_cache[iso] = holidays.country_holidays(iso)
        return row["Date"].normalize() in cal_cache[iso]

    df["is_holiday_local"] = df.apply(is_hol, axis=1)

    # ---------- write enriched parquet ---------------
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTFILE, index=False)
    print(f"wrote {OUTFILE} with {df.shape[1]} columns and {df.shape[0]} rows")

if __name__ == "__main__":
    main()
