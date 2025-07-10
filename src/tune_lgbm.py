#!/usr/bin/env python
"""
Tune a LightGBM next-day demand model with time-series CV.

Inputs
------
data/feature/sales_feat_v2.parquet      ← engineered features (fe_v2)

Outputs
-------
models/best_lgbm.joblib                 ← best tuned model
metrics/tuned_metrics.json              ← RMSE & MAE on hold-out
metrics/tuned_feature_importance.csv    ← gain-based feature importance
"""

import json, pathlib, joblib, optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split

# ---------------- paths ----------------
FE        = pathlib.Path("data/feature/sales_feat_v2.parquet")
MODEL_OUT = pathlib.Path("models/best_lgbm.joblib")
MET_OUT   = pathlib.Path("metrics/tuned_metrics.json")
IMP_OUT   = pathlib.Path("metrics/tuned_feature_importance.csv")

N_TRIALS  = 50           # you can raise later if compute budget allows
N_SPLITS  = 5            # time-series CV folds

# ------------- load & prep -------------
df = pd.read_parquet(FE).sort_values(["Product", "Date"])
df["target_next_qty"] = df.groupby("Product")["Order_Quantity"].shift(-1)
df = df.dropna(subset=["target_next_qty"])

X = df.drop(columns=["target_next_qty", "Date"])
y = df["target_next_qty"]

for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category")

# -------- objective for Optuna ---------
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

def objective(trial: optuna.trial.Trial) -> float:
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves":         trial.suggest_int("num_leaves", 31, 255),
        "max_depth":          trial.suggest_int("max_depth", -1, 16),
        "min_child_samples":  trial.suggest_int("min_child_samples", 5, 100),
        "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "n_jobs":      -1,
    }

    rmses = []
    for train_idx, valid_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

        mdl = LGBMRegressor(**params)
        mdl.fit(X_tr, y_tr)
        preds = mdl.predict(X_val)
        rmses.append(root_mean_squared_error(y_val, preds))

    return sum(rmses) / len(rmses)        # average fold RMSE

# ------------- run study ---------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
print("Best params:", best_params)

# -- refit best model on full training cut, test on last 20 % --
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=False
)
best_model = LGBMRegressor(**best_params, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)

preds = best_model.predict(X_test)
rmse = root_mean_squared_error(y_test, preds)
mae  = mean_absolute_error(y_test, preds)
print(f"TUNED RMSE={rmse:.2f}, MAE={mae:.2f}")

# ----------- save artefacts ------------
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
MET_OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, MODEL_OUT)
with open(MET_OUT, "w") as fp:
    json.dump({"rmse": rmse, "mae": mae}, fp, indent=2)

importances = best_model.booster_.feature_importance(importance_type="gain")
feat_imp = (
    pd.Series(importances, index=best_model.booster_.feature_name())
      .sort_values(ascending=False)
      .reset_index()
      .rename(columns={"index": "feature", 0: "gain"})
)
IMP_OUT.parent.mkdir(parents=True, exist_ok=True)
feat_imp.to_csv(IMP_OUT, index=False)
