"""SuperKart sales-prediction web interface (Streamlit)."""

import os
import re

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("SUPERKART_API_URL", "http://localhost:7860").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 60
REFERENCE_YEAR = 2025
PRODUCT_ID_PATTERN = re.compile(r"[A-Z]{2}\d{1,4}")

PRODUCT_TYPES = [
    "Baking Goods",
    "Breads",
    "Breakfast",
    "Canned",
    "Dairy",
    "Frozen Foods",
    "Fruits and Vegetables",
    "Hard Drinks",
    "Health and Hygiene",
    "Household",
    "Meat",
    "Others",
    "Seafood",
    "Snack Foods",
    "Soft Drinks",
    "Starchy Foods",
]
PERISHABLE_PRODUCT_TYPES = {
    "Breads",
    "Breakfast",
    "Dairy",
    "Frozen Foods",
    "Fruits and Vegetables",
    "Meat",
    "Seafood",
    "Starchy Foods",
}
SUGAR_CONTENT_OPTIONS = ["Low Sugar", "Regular", "No Sugar"]
STORE_SIZE_OPTIONS = ["High", "Medium", "Small"]
CITY_TYPE_OPTIONS = ["Tier 1", "Tier 2", "Tier 3"]
STORE_TYPE_OPTIONS = ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"]
BATCH_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


class BackendError(Exception):
    """The prediction API rejected the request."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or []


def product_type_category(product_type):
    return "Perishables" if product_type in PERISHABLE_PRODUCT_TYPES else "Non Perishables"


def build_model_features(raw, reference_year=REFERENCE_YEAR):
    """Convert form inputs into the 10 features expected by the API."""
    product_id = str(raw["Product_Id"]).strip().upper()
    if not PRODUCT_ID_PATTERN.fullmatch(product_id):
        raise ValueError("Product_Id must be two letters followed by 1 to 4 digits, for example FD6114.")
    product_type = raw["Product_Type"]
    if product_type not in PRODUCT_TYPES:
        raise ValueError(f"Unsupported Product_Type '{product_type}'.")
    establishment_year = int(raw["Store_Establishment_Year"])
    if establishment_year > reference_year:
        raise ValueError("Store_Establishment_Year cannot be later than the reference year.")
    return {
        "Product_Weight": float(raw["Product_Weight"]),
        "Product_Sugar_Content": raw["Product_Sugar_Content"],
        "Product_Allocated_Area": float(raw["Product_Allocated_Area"]),
        "Product_MRP": float(raw["Product_MRP"]),
        "Store_Size": raw["Store_Size"],
        "Store_Location_City_Type": raw["Store_Location_City_Type"],
        "Store_Type": raw["Store_Type"],
        "Product_Id_char": product_id[:2],
        "Store_Age_Years": reference_year - establishment_year,
        "Product_Type_Category": product_type_category(product_type),
    }


def _parse_response(response):
    try:
        body = response.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    if response.status_code >= 400:
        raise BackendError(body.get("error", f"The API returned HTTP {response.status_code}."), body.get("details"))
    return body


def request_single_prediction(features):
    response = requests.post(f"{API_URL}/v1/predict", json=features, timeout=REQUEST_TIMEOUT_SECONDS)
    return _parse_response(response)


def request_batch_prediction(csv_bytes, filename):
    response = requests.post(
        f"{API_URL}/v1/predictbatch",
        files={"file": (filename, csv_bytes, "text/csv")},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    return _parse_response(response)


def check_api_status():
    response = requests.get(f"{API_URL}/", timeout=5)
    return _parse_response(response)


def render_single_prediction():
    with st.form("single_prediction"):
        left, right = st.columns(2)
        with left:
            product_id = st.text_input("Product_Id", value="FD6114", help="Two letters followed by 1 to 4 digits")
            product_type = st.selectbox("Product_Type", PRODUCT_TYPES, index=PRODUCT_TYPES.index("Snack Foods"))
            product_weight = st.number_input("Product_Weight", min_value=1.0, max_value=30.0, value=12.66, step=0.01)
            sugar_content = st.selectbox("Product_Sugar_Content", SUGAR_CONTENT_OPTIONS)
            allocated_area = st.number_input(
                "Product_Allocated_Area", min_value=0.001, max_value=0.5, value=0.027, step=0.001, format="%.3f"
            )
            product_mrp = st.number_input("Product_MRP", min_value=1.0, max_value=500.0, value=117.08, step=0.01)
        with right:
            store_type = st.selectbox("Store_Type", STORE_TYPE_OPTIONS, index=STORE_TYPE_OPTIONS.index("Supermarket Type2"))
            store_size = st.selectbox("Store_Size", STORE_SIZE_OPTIONS, index=STORE_SIZE_OPTIONS.index("Medium"))
            city_type = st.selectbox("Store_Location_City_Type", CITY_TYPE_OPTIONS, index=CITY_TYPE_OPTIONS.index("Tier 2"))
            establishment_year = st.number_input(
                "Store_Establishment_Year", min_value=1900, max_value=REFERENCE_YEAR, value=2009, step=1
            )
        submitted = st.form_submit_button("Predict sales")

    if not submitted:
        return
    raw = {
        "Product_Id": product_id,
        "Product_Type": product_type,
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar_content,
        "Product_Allocated_Area": allocated_area,
        "Product_MRP": product_mrp,
        "Store_Type": store_type,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Establishment_Year": establishment_year,
    }
    try:
        features = build_model_features(raw)
        body = request_single_prediction(features)
    except ValueError as exc:
        st.error(str(exc))
    except BackendError as exc:
        st.error(str(exc))
        for detail in exc.details:
            st.write(f"Row {detail.get('row')}: {detail.get('message')}")
    except requests.RequestException as exc:
        st.error(f"Could not reach the prediction API at {API_URL}: {exc}")
    else:
        st.metric("Predicted Product_Store_Sales_Total", f"{body['Predicted_Product_Store_Sales_Total']:,.2f}")
        for warning in body.get("warnings", []):
            st.warning(warning["message"])
        with st.expander("Features sent to the API"):
            st.json(features)


def summarize_batch_response(body):
    """Turn a batch API response into the tables and counts shown in the batch tab."""
    predictions = pd.DataFrame(
        [
            {"Row": int(row), "Predicted_Product_Store_Sales_Total": value}
            for row, value in body.get("predictions", {}).items()
        ],
        columns=["Row", "Predicted_Product_Store_Sales_Total"],
    )
    return {
        "rows_received": body.get("rows_received", 0),
        "rows_predicted": body.get("rows_predicted", len(predictions)),
        "predictions": predictions,
        "errors": pd.DataFrame(body.get("errors", []), columns=["row", "message"]),
        "warnings": pd.DataFrame(body.get("warnings", []), columns=["row", "message"]),
    }


def render_batch_prediction():
    st.write("Upload a CSV with these columns (one product-store row per line):")
    st.code(", ".join(BATCH_COLUMNS))
    uploaded = st.file_uploader("Batch CSV", type=["csv"])
    if uploaded is None or not st.button("Predict batch"):
        return
    try:
        body = request_batch_prediction(uploaded.getvalue(), uploaded.name)
    except BackendError as exc:
        st.error(str(exc))
        return
    except requests.RequestException as exc:
        st.error(f"Could not reach the prediction API at {API_URL}: {exc}")
        return

    summary = summarize_batch_response(body)
    st.write(f"Rows received: {summary['rows_received']}, predicted: {summary['rows_predicted']}")
    if not summary["predictions"].empty:
        st.dataframe(summary["predictions"], hide_index=True)
    if not summary["errors"].empty:
        st.subheader("Rows that could not be predicted")
        st.dataframe(summary["errors"], hide_index=True)
    if not summary["warnings"].empty:
        st.subheader("Warnings")
        st.dataframe(summary["warnings"], hide_index=True)


def main():
    st.set_page_config(page_title="SuperKart Sales Predictor", layout="wide")
    st.title("SuperKart Sales Predictor")
    st.caption("Predicts Product_Store_Sales_Total with the selected pipeline served by the API.")

    with st.sidebar:
        st.write("Prediction API")
        st.code(API_URL)
        if st.button("Check API status"):
            try:
                status = check_api_status()
                st.success(f"{status.get('status', 'unknown')}: {status.get('model', '')}")
            except (requests.RequestException, BackendError) as exc:
                st.error(f"API unavailable: {exc}")

    single_tab, batch_tab = st.tabs(["Single product", "Batch CSV"])
    with single_tab:
        render_single_prediction()
    with batch_tab:
        render_batch_prediction()


if __name__ == "__main__":
    main()
