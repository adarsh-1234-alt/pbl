"""Shared data preparation for the Kaggle diabetes-prediction DNN.

Stage 1 of the project: one place that defines the feature set, cleans the raw
CSV, fits the scaler and turns a row into the exact matrix the network expects.
`train_dnn.py` uses it now; the Streamlit app will import it later, which keeps
training and serving in lock-step.

Design choices
--------------
* Numeric columns are standardised with a scikit-learn ``StandardScaler`` that is
  fitted on the training split only and saved as ``scaler.pkl``.
* The two binary flags stay 0/1.
* Categorical columns are one-hot encoded with a **fixed** level list. Dropping
  one level per group avoids a redundant column; unseen values fall back to the
  dropped reference level instead of producing a silent all-zero row.
* Missing values are filled with training medians (numeric) or 0 (binary). The
  Kaggle file has no gaps, but the app must not break if a field is omitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]
TARGET = "diabetes"

NUMERIC = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
BINARY = ["hypertension", "heart_disease"]
CATEGORICAL = ["gender", "smoking_history"]

# Levels are listed in a stable order so the column layout never changes.
GENDER_LEVELS = ["Female", "Male", "Other"]
SMOKING_LEVELS = ["No Info", "current", "ever", "former", "never", "not current"]
GENDER_REFERENCE = "Female"
SMOKING_REFERENCE = "No Info"

DISPLAY_NAMES = {
    "age": "Age",
    "bmi": "BMI",
    "HbA1c_level": "HbA1c level",
    "blood_glucose_level": "Blood glucose",
    "hypertension": "Hypertension",
    "heart_disease": "Heart disease",
    "gender_Male": "Gender: male",
    "gender_Other": "Gender: other",
    "smoking_history_current": "Smoking: current",
    "smoking_history_ever": "Smoking: ever",
    "smoking_history_former": "Smoking: former",
    "smoking_history_never": "Smoking: never",
    "smoking_history_not current": "Smoking: not current",
}


def feature_columns() -> list[str]:
    """Final model input order (numeric, binary, then one-hot levels)."""
    columns = list(NUMERIC) + list(BINARY)
    columns += [f"gender_{level}" for level in GENDER_LEVELS
                if level != GENDER_REFERENCE]
    columns += [f"smoking_history_{level}" for level in SMOKING_LEVELS
                if level != SMOKING_REFERENCE]
    return columns


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the CSV and fail early with a readable message on schema drift."""
    frame = pd.read_csv(path)
    needed = RAW_FEATURES + [TARGET]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise categories and numeric dtypes without changing row order."""
    out = frame[RAW_FEATURES].copy()

    for column in NUMERIC + BINARY:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    gender = out["gender"].astype(str).str.strip().str.lower()
    out["gender"] = gender.map({
        "female": "Female",
        "male": "Male",
    }).fillna("Other")

    smoking = out["smoking_history"].astype(str).str.strip().str.lower()
    smoking = smoking.replace({
        "": "No Info",
        "nan": "No Info",
        "none": "No Info",
        "no info": "No Info",
        "no information": "No Info",
        "unknown": "No Info",
    })
    out["smoking_history"] = smoking.where(
        smoking.isin(SMOKING_LEVELS), "No Info")

    return out


def fit_preprocessor(train_frame: pd.DataFrame) -> dict:
    """Fit medians + scaler on the training rows and return the bundle."""
    cleaned = clean_frame(train_frame)
    medians = {column: float(cleaned[column].median()) for column in NUMERIC}

    filled = cleaned.copy()
    for column in NUMERIC:
        filled[column] = filled[column].fillna(medians[column])

    scaler = StandardScaler().fit(filled[NUMERIC].to_numpy(dtype=float))
    columns = feature_columns()

    return {
        "version": 1,
        "target": TARGET,
        "raw_features": RAW_FEATURES,
        "numeric": NUMERIC,
        "binary": BINARY,
        "categorical": CATEGORICAL,
        "columns": columns,
        "display_names": {column: DISPLAY_NAMES[column] for column in columns},
        "median": medians,
        "gender_levels": GENDER_LEVELS,
        "smoking_levels": SMOKING_LEVELS,
        "gender_reference": GENDER_REFERENCE,
        "smoking_reference": SMOKING_REFERENCE,
        "scaler": scaler,
    }


def transform_frame(frame: pd.DataFrame, preprocessor: dict) -> np.ndarray:
    """Apply the saved preprocessing to raw rows and return float32 features."""
    cleaned = clean_frame(frame)

    filled = cleaned.copy()
    for column in NUMERIC:
        filled[column] = filled[column].fillna(preprocessor["median"][column])

    numeric = preprocessor["scaler"].transform(
        filled[NUMERIC].to_numpy(dtype=float))
    binary = filled[BINARY].fillna(0).to_numpy(dtype=np.float32)

    parts = [numeric.astype(np.float32), binary]
    for level in GENDER_LEVELS:
        if level == preprocessor["gender_reference"]:
            continue
        parts.append((filled["gender"] == level).to_numpy(np.float32)[:, None])
    for level in SMOKING_LEVELS:
        if level == preprocessor["smoking_reference"]:
            continue
        parts.append(
            (filled["smoking_history"] == level).to_numpy(np.float32)[:, None])

    matrix = np.hstack(parts).astype(np.float32)
    expected = len(preprocessor["columns"])
    if matrix.shape[1] != expected:
        raise ValueError(
            f"Preprocessing produced {matrix.shape[1]} columns, expected {expected}")
    return matrix


def save_preprocessor(preprocessor: dict, json_path: str | Path,
                      scaler_path: str | Path) -> None:
    """Write the portable JSON bundle and the scaler pickle separately."""
    payload = {key: value for key, value in preprocessor.items()
               if key != "scaler"}
    Path(json_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    joblib.dump(preprocessor["scaler"], scaler_path)


def load_preprocessor(json_path: str | Path,
                      scaler_path: str | Path) -> dict:
    """Rebuild the bundle written by :func:`save_preprocessor`."""
    preprocessor = json.loads(Path(json_path).read_text(encoding="utf-8"))
    preprocessor["scaler"] = joblib.load(scaler_path)
    return preprocessor
