#!/usr/bin/env python
"""
Baseline next-day demand model using CatBoostRegressor.

Input : data/feature/sales_feat_v2.parquet
Outputs:
  models/cb_baseline.cbm                 – trained model
  metrics/cb_metrics.json                – RMSE, MAE
  metrics/cb_feature_importance.csv      – CatBoost’s “PredictionValuesChange”
"""

import json, pathlib, joblib, pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

# ---------- paths ----------
FE        = pathlib.Path("data/feature/sales_feat_v2.parquet")
MODEL_OUT = pathlib.Path("models/cb_baseline.cbm")
MET_OUT   = pathlib.Path("metrics/cb_metrics.json")
IMP_OUT   = pathlib.Path("metrics/cb_feature_importance.csv")

def main() -> None:
    df = pd.read_parquet(FE).sort_values(["Product", "Date"])
    df["target_next_qty"] = df.groupby("Product")["Order_Quantity"].shift(-1)
    df = df.dropna(subset=["target_next_qty"])

    X = df.drop(columns=["target_next_qty", "Date"])
    y = df["target_next_qty"]

    # categorical feature indices for CatBoost
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_idx  = [X.columns.get_loc(c) for c in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    test_pool  = Pool(X_test,  y_test,  cat_features=cat_idx)

    model = CatBoostRegressor(
        iterations=800,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        random_seed=42,
        early_stopping_rounds=50,
        verbose=False,
    )
    model.fit(train_pool, eval_set=test_pool)

    preds = model.predict(test_pool)
    rmse  = root_mean_squared_error(y_test, preds)
    mae   = mean_absolute_error(y_test, preds)

    # save artefacts
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    MET_OUT.parent.mkdir(parents=True, exist_ok=True)
    IMP_OUT.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(MODEL_OUT)
    with open(MET_OUT, "w") as fp:
        json.dump({"rmse": rmse, "mae": mae}, fp, indent=2)

    fi = (
        pd.DataFrame({
            "feature": model.feature_names_,
            "importance": model.get_feature_importance(type="PredictionValuesChange")
        })
        .sort_values("importance", ascending=False)
    )
    fi.to_csv(IMP_OUT, index=False)

    print(f"CatBoost RMSE={rmse:.2f}, MAE={mae:.2f}")

if __name__ == "__main__":
    main()
