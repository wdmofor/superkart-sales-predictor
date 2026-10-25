"""SuperKart sales-prediction REST API (Flask)."""

import json
import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
# The saved pipeline includes the fitted preprocessing, so validated raw feature frames go straight into predict.
MODEL = joblib.load(BASE_DIR / "superkart_model.joblib")

# Column lists, allowed categories, and training ranges saved by the notebook.
with open(BASE_DIR / "superkart_metadata.json", encoding="utf-8") as metadata_file:
    METADATA = json.load(metadata_file)

FEATURE_COLUMNS = METADATA["feature_columns"]
NUMERIC_COLUMNS = METADATA["numeric_columns"]
CATEGORICAL_COLUMNS = METADATA["categorical_columns"]
CATEGORY_VALUES = METADATA["category_values"]
TRAINING_RANGES = METADATA["training_ranges"]
# Prefix/category combinations present in training; unseen combinations trigger a warning, not a rejection.
SEEN_CATEGORY_PAIRS = {tuple(pair) for pair in METADATA["observed_category_pairs"]}
STRICTLY_POSITIVE_COLUMNS = ("Product_Weight", "Product_Allocated_Area", "Product_MRP")
NON_NEGATIVE_COLUMNS = ("Store_Age_Years",)

superkart_api = Flask(__name__)
# Flask refuses request bodies over 5 MB with HTTP 413.
superkart_api.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


class InputError(ValueError):
    """The request does not contain the expected feature columns."""


def _is_blank(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return isinstance(value, float) and math.isnan(value)


def _as_finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _check_record(record):
    """Return (clean_values, problems, notes) for one input row."""
    clean, problems, notes = {}, [], []
    # Categories must match a training value after trimming spaces; blanks and unknown values are rejected.
    for column in CATEGORICAL_COLUMNS:
        value = record[column]
        if _is_blank(value):
            problems.append(f"{column} is missing")
            continue
        if isinstance(value, str):
            value = value.strip()
        if value in CATEGORY_VALUES[column]:
            clean[column] = value
        else:
            allowed = ", ".join(CATEGORY_VALUES[column])
            problems.append(f"{column} '{value}' is not supported; expected one of: {allowed}")
    # Numbers must be finite; the three measurements must be positive and the store age cannot be negative.
    for column in NUMERIC_COLUMNS:
        number = _as_finite_number(record[column])
        if number is None:
            problems.append(f"{column} must be a finite number")
        elif column in STRICTLY_POSITIVE_COLUMNS and number <= 0:
            problems.append(f"{column} must be greater than 0")
        elif column in NON_NEGATIVE_COLUMNS and number < 0:
            problems.append(f"{column} must be 0 or greater")
        else:
            clean[column] = number
            # Values outside the training range are still predicted but flagged, because tree leaves cannot extrapolate.
            low, high = TRAINING_RANGES[column]
            if not low <= number <= high:
                notes.append(
                    f"{column} = {number:g} is outside the training range "
                    f"[{low:g}, {high:g}]; the prediction may be less reliable"
                )
    # Only check the pair when both parts passed validation.
    pair = (clean.get("Product_Id_char"), clean.get("Product_Type_Category"))
    if None not in pair and pair not in SEEN_CATEGORY_PAIRS:
        notes.append(
            f"Product_Id_char '{pair[0]}' with Product_Type_Category '{pair[1]}' "
            "does not occur in the training data; the prediction may be less reliable"
        )
    return clean, problems, notes


def _prepare_frame(frame):
    """Validate rows. Returns (features_for_valid_rows, errors, warnings); row numbers are 1-based."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise InputError(f"Missing required column(s): {', '.join(missing)}")

    warnings, errors, valid_rows, valid_row_numbers = [], [], [], []
    # A missing required column stops the whole request; extra columns are ignored with a warning.
    unexpected = [str(column) for column in frame.columns if column not in FEATURE_COLUMNS]
    if unexpected:
        warnings.append({"row": None, "message": f"Ignored unexpected column(s): {', '.join(unexpected)}"})

    # Each row is checked on its own, so one bad row is reported without stopping the others.
    records = frame[FEATURE_COLUMNS].to_dict(orient="records")
    for row_number, record in enumerate(records, start=1):
        clean, problems, notes = _check_record(record)
        warnings.extend({"row": row_number, "message": note} for note in notes)
        if problems:
            errors.append({"row": row_number, "message": "; ".join(problems)})
        else:
            valid_rows.append(clean)
            valid_row_numbers.append(row_number)

    # Keep the input row numbers as the index so each prediction maps back to its source row.
    features = pd.DataFrame(valid_rows, index=valid_row_numbers, columns=FEATURE_COLUMNS)
    return features, errors, warnings


@superkart_api.get("/")
def health():
    return jsonify(
        status="ok",
        model=METADATA["model_name"],
        endpoints=["POST /v1/predict", "POST /v1/predictbatch"],
    )


@superkart_api.post("/v1/predict")
def predict_single():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send one JSON object containing the 10 model features."), 400
    try:
        # A single request goes through the same validation as one batch row.
        features, errors, warnings = _prepare_frame(pd.DataFrame([payload]))
    except InputError as exc:
        return jsonify(error=str(exc)), 400
    if errors:
        return jsonify(error="Input validation failed.", details=errors, warnings=warnings), 400
    # The request is valid, so the single validated row goes to the model.
    prediction = MODEL.predict(features)[0]
    return jsonify(Predicted_Product_Store_Sales_Total=round(float(prediction), 2), warnings=warnings)


@superkart_api.post("/v1/predictbatch")
def predict_batch():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify(error="Upload a CSV file in the multipart form field named 'file'."), 400
    # A file that cannot be parsed is a client error (400), not a server error.
    try:
        frame = pd.read_csv(uploaded.stream)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        return jsonify(error=f"The uploaded file is not a readable CSV: {exc}"), 400
    if frame.empty:
        return jsonify(error="The uploaded CSV contains no data rows."), 400
    try:
        features, errors, warnings = _prepare_frame(frame)
    except InputError as exc:
        return jsonify(error=str(exc)), 400

    predictions = {}
    if not features.empty:
        values = MODEL.predict(features)
        # JSON object keys must be strings, so the row numbers are converted here.
        predictions = {str(row): round(float(value), 2) for row, value in zip(features.index, values)}
    return jsonify(
        rows_received=len(frame),
        rows_predicted=len(predictions),
        predictions=predictions,
        errors=errors,
        warnings=warnings,
    )
