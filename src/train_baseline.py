#!/usr/bin/env python
"""
Train a baseline next-day demand model.
Input : data/feature/sales_feat_v2.parquet
Output: models/baseline_lgbm.joblib          (trained model)
        metrics/baseline_metrics.json        (RMSE, MAE)
"""
import json, pathlib, joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import (
mean_squared_error,
mean_absolute_error,
root_mean_squared_error)
from sklearn.model_selection import train_test_split

# ---------- paths ----------
FE = pathlib.Path("data/feature/sales_feat_v2.parquet")
MODEL_OUT   = pathlib.Path("models/baseline_lgbm.joblib")
METRIC_OUT  = pathlib.Path("metrics/baseline_metrics.json")

def main():
    df = pd.read_parquet(FE)

    # ---- target: next-day qty ----
    df = df.sort_values(["Product", "Date"])
    df["target_next_qty"] = df.groupby("Product")["Order_Quantity"].shift(-1)
    df = df.dropna(subset=["target_next_qty"])          # last day of each product lost

    X = df.drop(columns=["target_next_qty", "Date"])    # keep numeric + categorical encodings
    y = df["target_next_qty"]

    # convert all object columns to pandas 'category'
    obj_cols = X.select_dtypes(include="object").columns
    for c in obj_cols:
        X[c] = X[c].astype("category")

    # simple calendar hold-out: last 20% rows as test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )

    model = LGBMRegressor(
        n_estimators=400, learning_rate=0.05, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # ---- metrics ----
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)

    METRIC_OUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_OUT)
    with open(METRIC_OUT, "w") as fp:
        json.dump({"rmse": rmse, "mae": mae}, fp, indent=2)

    print(f"Baseline RMSE={rmse:.2f}, MAE={mae:.2f}")

if __name__ == "__main__":
    main()
