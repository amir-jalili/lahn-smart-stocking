# Smart Stocking Decisions for Lahn Inc.



````markdown
# Lahn Smart Stocking – Reproducible Demand-Forecast Pipeline

This repository contains the full workflow for forecasting next-day product demand (Applications of AI in Business, SS-2025).  
The champion model is a **CatBoost Regressor** tracked with **DVC**.

---

## 1 · Reproduce the full pipeline

```bash
# Clone and enter the project
git clone git@github.com:amir-jalili/lahn-smart-stocking.git
cd lahn-smart-stocking

# Create & activate a Python 3.10 virtual environment
python -m venv venv
source venv/bin/activate        # (Windows: venv\Scripts\activate)

# Install all project dependencies
pip install -r requirements.txt

# Fetch version-controlled data + trained CatBoost model
dvc pull

# Rebuild every stage to verify reproducibility
dvc repro                        # should finish with “workspace is clean”
````

---

## 2 · Quick prediction sanity-check

```bash
echo '[{
  "Product": "Trail-900 Black, 48",
  "Unit_Cost": 1200, "Unit_Price": 1999, "Order_Quantity": 5,
  "Profit": 3980, "Cost": 6000, "Revenue": 9980,
  "Month": "January", "Customer_Gender": "M",
  "Country": "United States", "State": "California",
  "Product_Category": "Bikes", "Sub_Category": "Road Bikes",
  "Age_Group": "Adults", "Customer_Age": 35,
  "Day": 15, "Year": 2015,
  "lag1_qty": 4,  "lag7_qty": 3,  "rolling28_rev": 8000,
  "yearmo": "2015-01", "recency_days": 28, "frequency": 6, "monetary": 2400,
  "dow": 3, "week": 3, "is_holiday_local": false,
  "profit_per_unit": 796, "margin_pct": 0.4
}]' | python src/predict.py
```

Expected output (example):

```json
[
  2.80
]
```

---

## 3 · Key results

| Model                   | RMSE     | MAE  |
| ----------------------- | -------- | ---- |
| **CatBoost (champion)** | **7.98** | 6.44 |
| LightGBM baseline       | 8.08     | 6.41 |

RMSE improved ≈ 1 % over LightGBM and ≈ 31 % over a naïve mean predictor.



