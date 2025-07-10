# Smart Stocking Decisions for Lahn Inc.

unzip -q lahn-smart-stocking.zip -d test_submission
cd test_submission
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
dvc pull                 # downloads raw data + CatBoost model
dvc repro                # must finish with “workspace is clean”

# quick prediction check
echo '[{"Product":"Trail-900 Black, 48","Unit_Cost":1200,"Unit_Price":1999,"Order_Quantity":5,"Profit":3980,"Cost":6000,"Revenue":9980,"Month":"January","Customer_Gender":"M","Country":"United States","State":"California","Product_Category":"Bikes","Sub_Category":"Road Bikes","Age_Group":"Adults","Customer_Age":35,"Day":15,"Year":2015,"lag1_qty":4,"lag7_qty":3,"rolling28_rev":8000,"yearmo":"2015-01","recency_days":28,"frequency":6,"monetary":2400,"dow":3,"week":3,"is_holiday_local":false,"profit_per_unit":796,"margin_pct":0.4}]' \
| python src/predict.py

