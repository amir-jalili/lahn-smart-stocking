#!/usr/bin/env python
"""
Predict next-day demand from JSON rows on stdin.
Example:
  cat sample_input.json | python src/predict.py
"""
import sys, json, pandas as pd
from catboost import CatBoostRegressor

model = CatBoostRegressor()
model.load_model("models/cb_baseline.cbm")     # champion model

df = pd.read_json(sys.stdin)
preds = model.predict(df)
print(json.dumps(preds.tolist(), indent=2))
