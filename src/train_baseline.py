#!/usr/bin/env python
"""
Train a baseline next-day demand model.

Inputs
------
data/feature/sales_feat_v2.parquet     ← feature table produced by fe_v2

Outputs
-------
models/baseline_lgbm.joblib            ← trained LightGBM model
metrics/baseline_metrics.json          ← RMSE & MAE
metrics/baseline_feature_importance.csv← gain-based feature importance
"""
import json, pathlib, joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,        # scikit-learn ≥1.4
)
from sklearn.model_selection import train_test_split

# ---------- paths ----------
FE         = pathlib.Path("data/feature/sales_feat_v2.parquet")
MODEL_OUT  = pathlib.Path("models/baseline_lgbm.joblib")
METRIC_OUT = pathlib.Path("metrics/baseline_metrics.json")
IMP_OUT    = pathlib.Path("metrics/baseline_feature_importance.csv")

def main() -> None:
    df = pd.read_parquet(FE)

    # ---- target: next-day qty ----
    df = df.sort_values(["Product", "Date"])
    df["target_next_qty"] = df.groupby("Product")["Order_Quantity"].shift(-1)
    df = df.dropna(subset=["target_next_qty"])

    X = df.drop(columns=["target_next_qty", "Date"])
    y = df["target_next_qty"]

    # convert all object columns to pandas 'category'
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")

    # simple calendar hold-out: last 20 % rows as test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )

    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ---- metrics ----
    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)

    # ---- persist artefacts ----
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRIC_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    with open(METRIC_OUT, "w") as fp:
        json.dump({"rmse": rmse, "mae": mae}, fp, indent=2)

    # ---- feature importance ----
    importances = model.booster_.feature_importance(importance_type="gain")
    feat_imp = (
        pd.Series(importances, index=model.booster_.feature_name())
          .sort_values(ascending=False)
          .reset_index()
          .rename(columns={"index": "feature", 0: "gain"})
    )
    IMP_OUT.parent.mkdir(parents=True, exist_ok=True)
    feat_imp.to_csv(IMP_OUT, index=False)

    print(f"Baseline RMSE={rmse:.2f}, MAE={mae:.2f}")

if __name__ == "__main__":
    main()
