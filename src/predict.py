#!/usr/bin/env python
"""
CLI prediction for the CatBoost demand-model.

Usage examples
--------------
# one JSON row
echo '[{"Product":"Trail-900 Black, 48", ...}]' | python src/predict.py

# CSV file
python src/predict.py data/feature/sales_feat_v2.parquet
"""
import sys, json, pandas as pd
from catboost import CatBoostRegressor, Pool

# ---------- load model ----------
model = CatBoostRegressor()
model.load_model("models/cb_baseline.cbm")

# ---------- read input ----------
if sys.stdin.isatty():                      # file path given
    df = pd.read_json(sys.argv[1])
else:                                       # piped JSON
    df = pd.read_json(sys.stdin)

# ---------- align columns ----------
df = df.reindex(model.feature_names_, axis=1)  # ensure same order

# cat feature indices
cat_idx = [i for i, c in enumerate(df.columns)
           if df[c].dtype.name in ("object", "category")]

pool = Pool(df, cat_features=cat_idx)
preds = model.predict(pool)

print(json.dumps(preds.tolist(), indent=2))
