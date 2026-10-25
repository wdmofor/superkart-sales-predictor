"""Backend API tests. Run after the notebook has written the model artifacts."""

import importlib.util
import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

DEPLOY_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = DEPLOY_DIR / "full_code" / "backend_files"
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
def client():
    spec = importlib.util.spec_from_file_location("superkart_backend_app", BACKEND_DIR / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.superkart_api.test_client()


def _batch_bytes():
    return (DATA_DIR / "Batch_Data_SuperKart.csv").read_bytes()


def _post_batch(client, raw_bytes, filename="batch.csv"):
    return client.post(
        "/v1/predictbatch",
        data={"file": (BytesIO(raw_bytes), filename)},
        content_type="multipart/form-data",
    )


def test_health_endpoint_reports_ok(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_single_prediction_returns_positive_value(client):
    response = client.post("/v1/predict", json=TEMPLATE_PAYLOAD)
    body = response.get_json()
    assert response.status_code == 200
    assert isinstance(body["Predicted_Product_Store_Sales_Total"], float)
    assert body["Predicted_Product_Store_Sales_Total"] > 0
    assert body["warnings"] == []


def test_out_of_range_value_is_a_warning_not_an_error(client):
    payload = {**TEMPLATE_PAYLOAD, "Store_Age_Years": 100}
    response = client.post("/v1/predict", json=payload)
    body = response.get_json()
    assert response.status_code == 200
    assert any("Store_Age_Years" in warning["message"] for warning in body["warnings"])


def test_unseen_category_pair_is_a_warning_not_an_error(client):
    payload = {**TEMPLATE_PAYLOAD, "Product_Id_char": "DR", "Product_Type_Category": "Perishables"}
    response = client.post("/v1/predict", json=payload)
    body = response.get_json()
    assert response.status_code == 200
    assert body["Predicted_Product_Store_Sales_Total"] > 0
    assert len(body["warnings"]) == 1
    assert "does not occur in the training data" in body["warnings"][0]["message"]


def test_seen_category_pair_has_no_warning(client):
    payload = {**TEMPLATE_PAYLOAD, "Product_Id_char": "DR", "Product_Type_Category": "Non Perishables"}
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    assert response.get_json()["warnings"] == []


@pytest.mark.parametrize(
    "column",
    ["Store_Type", "Product_Id_char", "Store_Age_Years"],
)
def test_missing_feature_returns_400(client, column):
    payload = {key: value for key, value in TEMPLATE_PAYLOAD.items() if key != column}
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 400
    assert column in response.get_json()["error"]


@pytest.mark.parametrize(
    ("column", "value", "expected_text"),
    [
        ("Store_Type", "Supermarket Type3", "not supported"),
        ("Product_Sugar_Content", "reg", "not supported"),
        ("Product_Weight", -1.0, "must be greater than 0"),
        ("Product_MRP", 0, "must be greater than 0"),
        ("Store_Age_Years", -3, "must be 0 or greater"),
        ("Product_Allocated_Area", "abc", "must be a finite number"),
        ("Store_Size", "", "missing"),
    ],
)
def test_invalid_single_value_returns_400_with_row_detail(client, column, value, expected_text):
    response = client.post("/v1/predict", json={**TEMPLATE_PAYLOAD, column: value})
    body = response.get_json()
    assert response.status_code == 400
    assert body["details"][0]["row"] == 1
    assert expected_text in body["details"][0]["message"]


def test_non_json_single_request_returns_400(client):
    response = client.post("/v1/predict", data="not json", content_type="text/plain")
    assert response.status_code == 400


def test_batch_predicts_valid_rows_and_reports_unseen_store_rows(client):
    response = _post_batch(client, _batch_bytes())
    body = response.get_json()
    assert response.status_code == 200
    assert body["rows_received"] == 10
    assert body["rows_predicted"] == 8
    assert sorted(int(row) for row in body["predictions"]) == [1, 2, 3, 4, 6, 7, 8, 9]
    assert [error["row"] for error in body["errors"]] == [5, 10]
    assert all("Supermarket Type3" in error["message"] for error in body["errors"])


def test_batch_flags_unseen_category_pair_on_its_row(client):
    body = _post_batch(client, _batch_bytes()).get_json()
    flagged_rows = [
        warning["row"] for warning in body["warnings"] if "does not occur in the training data" in warning["message"]
    ]
    assert flagged_rows == [8]
    assert 8 in [int(row) for row in body["predictions"]]


def test_batch_and_single_predictions_agree(client):
    batch = pd.read_csv(DATA_DIR / "Batch_Data_SuperKart.csv").head(1)
    record = json.loads(batch.to_json(orient="records"))[0]
    single = client.post("/v1/predict", json=record).get_json()["Predicted_Product_Store_Sales_Total"]
    batch_body = _post_batch(client, _batch_bytes()).get_json()
    assert batch_body["predictions"]["1"] == pytest.approx(single, abs=0.01)


def test_batch_without_file_returns_400(client):
    response = client.post("/v1/predictbatch", data={}, content_type="multipart/form-data")
    assert response.status_code == 400


def test_batch_header_only_returns_400(client):
    header = _batch_bytes().splitlines()[0] + b"\n"
    response = _post_batch(client, header)
    assert response.status_code == 400
    assert "no data rows" in response.get_json()["error"]


def test_batch_missing_columns_returns_400(client):
    response = _post_batch(client, b"Product_Weight,Product_MRP\n12.5,100\n")
    assert response.status_code == 400
    assert "Missing required column" in response.get_json()["error"]


def test_batch_unreadable_file_returns_400(client):
    response = _post_batch(client, b"")
    assert response.status_code == 400
