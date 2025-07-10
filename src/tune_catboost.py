#!/usr/bin/env python
"""
Optuna-tuned CatBoostRegressor for next-day demand.

Input
-----
data/feature/sales_feat_v2.parquet

Outputs
-------
models/cb_best.cbm
metrics/cb_tuned_metrics.json
metrics/cb_tuned_feature_importance.csv
"""
import json, pathlib, optuna, pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

FE        = pathlib.Path("data/feature/sales_feat_v2.parquet")
MODEL_OUT = pathlib.Path("models/cb_best.cbm")
MET_OUT   = pathlib.Path("metrics/cb_tuned_metrics.json")
IMP_OUT   = pathlib.Path("metrics/cb_tuned_feature_importance.csv")
N_TRIALS  = 50

# ---------- load & prep ----------
df = pd.read_parquet(FE).sort_values(["Product", "Date"])
df["target_next_qty"] = df.groupby("Product")["Order_Quantity"].shift(-1)
df = df.dropna(subset=["target_next_qty"])
X = df.drop(columns=["target_next_qty", "Date"])
y = df["target_next_qty"]

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
cat_idx  = [X.columns.get_loc(c) for c in cat_cols]

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 6, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "loss_function": "RMSE",
        "random_seed": 42,
        "early_stopping_rounds": 50,
        "verbose": False,
    }

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.20, shuffle=False
    )
    tr_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)

    model = CatBoostRegressor(**params)
    model.fit(tr_pool, eval_set=val_pool)

    preds = model.predict(val_pool)
    return root_mean_squared_error(y_val, preds)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
best_params = study.best_params
print("Best params:", best_params)

# ---- refit on full training cut, evaluate on final hold-out ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=False
)
train_pool = Pool(X_train, y_train, cat_features=cat_idx)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_idx)

best_model = CatBoostRegressor(**best_params, loss_function="RMSE",
                               random_seed=42, verbose=False)
best_model.fit(train_pool)

preds = best_model.predict(test_pool)
rmse = root_mean_squared_error(y_test, preds)
mae  = mean_absolute_error(y_test, preds)
print(f"TUNED CatBoost RMSE={rmse:.2f}, MAE={mae:.2f}")

# ---- save artefacts ----
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
best_model.save_model(MODEL_OUT)

MET_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(MET_OUT, "w") as fp:
    json.dump({"rmse": rmse, "mae": mae}, fp, indent=2)

IMP_OUT.parent.mkdir(parents=True, exist_ok=True)
feat_imp = (
    pd.DataFrame({
        "feature": best_model.feature_names_,
        "importance": best_model.get_feature_importance(
            type="PredictionValuesChange")
    }).sort_values("importance", ascending=False)
)
feat_imp.to_csv(IMP_OUT, index=False)
