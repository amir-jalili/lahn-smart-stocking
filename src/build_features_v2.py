#!/usr/bin/env python
"""
Stage fe_v2  –  Lag features + RFM
Input : data/feature/sales_feat_v1.parquet
Output: data/feature/sales_feat_v2.parquet
"""

import pandas as pd, numpy as np, pathlib, datetime as dt

INFILE  = pathlib.Path("data/feature/sales_feat_v1.parquet")
OUTFILE = pathlib.Path("data/feature/sales_feat_v2.parquet")

def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["Product", "Date"]).copy()

    # 1-day and 7-day lags of Order_Quantity
    df["lag1_qty"] = df.groupby("Product")["Order_Quantity"].shift(1)
    df["lag7_qty"] = df.groupby("Product")["Order_Quantity"].shift(7)

    # 28-day centred rolling mean of Revenue
    roll = (
        df.groupby("Product")["Revenue"]
          .rolling(window=28, min_periods=7, center=True)
          .mean()
          .reset_index(level=0, drop=True)
    )
    df["rolling28_rev"] = roll
    return df

def add_rfm(df: pd.DataFrame) -> pd.DataFrame:
    # define pseudo-customer key (Country+State).  Adjust if a real Customer_ID exists later.
    cust_key = df["Country"].astype(str) + "|" + df["State"].astype(str)
    df = df.assign(cust_key=cust_key)

    snapshot = df["Date"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("cust_key")
          .agg(
              recency_days = ("Date", lambda x: (snapshot - x.max()).days),
              frequency    = ("Date", "nunique"),
              monetary     = ("Profit", "sum"),
          )
    )
    rfm["monetary"] = rfm["monetary"].round(2)

    # merge back row-level
    df = df.merge(rfm, on="cust_key", how="left")
    return df.drop(columns="cust_key")

def main():
    df = pd.read_parquet(INFILE)
    df = add_lags(df)
    df = add_rfm(df)

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTFILE, index=False)
    print(f"fe_v2 wrote {OUTFILE} → {df.shape}")

if __name__ == "__main__":
    main()
