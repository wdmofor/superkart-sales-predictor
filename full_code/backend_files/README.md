# SuperKart Sales Prediction API

Flask REST API that serves the tuned Random Forest pipeline predicting
`Product_Store_Sales_Total`. The serialized pipeline includes the preprocessing
(one-hot encoding), so callers send raw model features only.

## Endpoints

| Method | Path | Body | Response |
| ------ | ---- | ---- | -------- |
| GET | `/` | none | Health status and model name |
| POST | `/v1/predict` | JSON object with the 10 features | `Predicted_Product_Store_Sales_Total` and `warnings` |
| POST | `/v1/predictbatch` | multipart form, CSV file in field `file` | `predictions` keyed by 1-based row number, plus `errors` and `warnings` |

## Input features

`Product_Weight`, `Product_Sugar_Content`, `Product_Allocated_Area`, `Product_MRP`,
`Store_Size`, `Store_Location_City_Type`, `Store_Type`, `Product_Id_char`,
`Store_Age_Years`, `Product_Type_Category`.

Unsupported categorical values (for example a `Store_Type` that was not in the
training data) are reported as row-level errors and are not predicted.
Values outside the training range, and product-prefix and category pairs that never occur in training, are returned as warnings.

## Example

```bash
curl -X POST http://127.0.0.1:7860/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"Product_Weight": 12.66, "Product_Sugar_Content": "Low Sugar",
       "Product_Allocated_Area": 0.027, "Product_MRP": 117.08,
       "Store_Size": "Medium", "Store_Location_City_Type": "Tier 2",
       "Store_Type": "Supermarket Type2", "Product_Id_char": "FD",
       "Store_Age_Years": 16, "Product_Type_Category": "Non Perishables"}'
```
