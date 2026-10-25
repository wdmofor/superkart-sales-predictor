"""Frontend tests: input transformation, option lists, and error handling."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

DEPLOY_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = DEPLOY_DIR / "full_code" / "frontend_files"
DATA_DIR = DEPLOY_DIR / "data"

TEMPLATE_PAYLOAD = {
    "Product_Weight": 12.66,
    "Product_Sugar_Content": "Low Sugar",
    "Product_Allocated_Area": 0.027,
    "Product_MRP": 117.08,
    "Store_Size": "Medium",
    "Store_Location_City_Type": "Tier 2",
    "Store_Type": "Supermarket Type2",
    "Product_Id_char": "FD",
    "Store_Age_Years": 16,
    "Product_Type_Category": "Non Perishables",
}


@pytest.fixture(scope="module")
def frontend():
    spec = importlib.util.spec_from_file_location("superkart_frontend_app", FRONTEND_DIR / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def raw_form(frontend):
    return {
        "Product_Id": "FD6114",
        "Product_Type": "Snack Foods",
        "Product_Weight": 12.66,
        "Product_Sugar_Content": "Low Sugar",
        "Product_Allocated_Area": 0.027,
        "Product_MRP": 117.08,
        "Store_Type": "Supermarket Type2",
        "Store_Size": "Medium",
        "Store_Location_City_Type": "Tier 2",
        "Store_Establishment_Year": 2009,
    }


@pytest.fixture(scope="module")
def training_frame():
    frame = pd.read_csv(DATA_DIR / "SuperKart.csv")
    frame["Product_Sugar_Content"] = frame["Product_Sugar_Content"].replace({"reg": "Regular"})
    return frame


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def test_form_inputs_reproduce_template_example(frontend, raw_form):
    assert frontend.build_model_features(raw_form) == TEMPLATE_PAYLOAD


def test_product_id_is_normalised_to_upper_case(frontend, raw_form):
    raw_form["Product_Id"] = " nc1234 "
    features = frontend.build_model_features(raw_form)
    assert features["Product_Id_char"] == "NC"


@pytest.mark.parametrize("product_id", ["F6114", "FD12345", "FDX1", "", "123456"])
def test_invalid_product_id_is_rejected(frontend, raw_form, product_id):
    raw_form["Product_Id"] = product_id
    with pytest.raises(ValueError, match="Product_Id"):
        frontend.build_model_features(raw_form)


def test_future_establishment_year_is_rejected(frontend, raw_form):
    raw_form["Store_Establishment_Year"] = 2030
    with pytest.raises(ValueError, match="later than the reference year"):
        frontend.build_model_features(raw_form)


def test_unsupported_product_type_is_rejected(frontend, raw_form):
    raw_form["Product_Type"] = "Unknown Type"
    with pytest.raises(ValueError, match="Unsupported Product_Type"):
        frontend.build_model_features(raw_form)


@pytest.mark.parametrize(
    ("product_type", "category"),
    [
        ("Dairy", "Perishables"),
        ("Fruits and Vegetables", "Perishables"),
        ("Snack Foods", "Non Perishables"),
        ("Others", "Non Perishables"),
    ],
)
def test_product_type_maps_to_category(frontend, product_type, category):
    assert frontend.product_type_category(product_type) == category


def test_option_lists_match_training_data(frontend, training_frame):
    assert set(frontend.SUGAR_CONTENT_OPTIONS) == set(training_frame["Product_Sugar_Content"].unique())
    assert set(frontend.STORE_SIZE_OPTIONS) == set(training_frame["Store_Size"].unique())
    assert set(frontend.CITY_TYPE_OPTIONS) == set(training_frame["Store_Location_City_Type"].unique())
    assert set(frontend.STORE_TYPE_OPTIONS) == set(training_frame["Store_Type"].unique())
    assert set(frontend.PRODUCT_TYPES) == set(training_frame["Product_Type"].unique())


def test_every_perishable_type_is_a_known_product_type(frontend):
    assert frontend.PERISHABLE_PRODUCT_TYPES <= set(frontend.PRODUCT_TYPES)


def test_backend_error_carries_row_details(frontend, monkeypatch):
    details = [{"row": 1, "message": "Store_Type 'X' is not supported"}]
    fake = FakeResponse(400, {"error": "Input validation failed.", "details": details})
    monkeypatch.setattr(frontend.requests, "post", lambda *args, **kwargs: fake)
    with pytest.raises(frontend.BackendError) as excinfo:
        frontend.request_single_prediction(TEMPLATE_PAYLOAD)
    assert str(excinfo.value) == "Input validation failed."
    assert excinfo.value.details == details


def test_non_json_error_body_still_raises_backend_error(frontend, monkeypatch):
    fake = FakeResponse(500, ValueError("no json"))
    monkeypatch.setattr(frontend.requests, "post", lambda *args, **kwargs: fake)
    with pytest.raises(frontend.BackendError, match="HTTP 500"):
        frontend.request_single_prediction(TEMPLATE_PAYLOAD)


def test_successful_response_is_returned(frontend, monkeypatch):
    fake = FakeResponse(200, {"Predicted_Product_Store_Sales_Total": 3000.5, "warnings": []})
    monkeypatch.setattr(frontend.requests, "post", lambda *args, **kwargs: fake)
    body = frontend.request_single_prediction(TEMPLATE_PAYLOAD)
    assert body["Predicted_Product_Store_Sales_Total"] == 3000.5


BATCH_BODY = {
    "rows_received": 3,
    "rows_predicted": 2,
    "predictions": {"1": 2901.33, "3": 1500.0},
    "errors": [{"row": 2, "message": "Store_Type 'X' is not supported"}],
    "warnings": [{"row": 3, "message": "Store_Age_Years = 40 is outside the training range [16, 38]"}],
}


def test_batch_summary_lists_predictions_errors_and_warnings(frontend):
    summary = frontend.summarize_batch_response(BATCH_BODY)
    assert summary["rows_received"] == 3
    assert summary["rows_predicted"] == 2
    assert summary["predictions"].to_dict("records") == [
        {"Row": 1, "Predicted_Product_Store_Sales_Total": 2901.33},
        {"Row": 3, "Predicted_Product_Store_Sales_Total": 1500.0},
    ]
    assert list(summary["errors"]["row"]) == [2]
    assert list(summary["warnings"]["row"]) == [3]


def test_batch_summary_keeps_columns_when_sections_are_empty(frontend):
    body = {"rows_received": 1, "rows_predicted": 0, "predictions": {}, "errors": [{"row": 1, "message": "bad"}]}
    summary = frontend.summarize_batch_response(body)
    assert summary["predictions"].empty
    assert list(summary["predictions"].columns) == ["Row", "Predicted_Product_Store_Sales_Total"]
    assert len(summary["errors"]) == 1
    assert summary["warnings"].empty
    assert list(summary["warnings"].columns) == ["row", "message"]


def test_batch_summary_derives_prediction_count_when_missing(frontend):
    summary = frontend.summarize_batch_response({"predictions": {"1": 10.0}})
    assert summary["rows_predicted"] == 1
    assert summary["rows_received"] == 0
