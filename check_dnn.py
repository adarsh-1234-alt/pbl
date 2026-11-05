"""Load the Stage 1 artefacts and prove they are internally consistent.

Run after ``train_dnn.py`` (and in CI before deploying):

    python check_dnn.py

It checks the model input width, the saved feature order, the scaler, the JSON
metadata and the single-row path the Streamlit app will use. Exits non-zero on
the first broken contract.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

import dnn_data

BASE = Path(__file__).resolve().parent
MODEL = BASE / "dnn_model.keras"
PREPROCESS = BASE / "dnn_preprocess.json"
SCALER = BASE / "scaler.pkl"
METRICS = BASE / "dnn_metrics.json"
IMPORTANCE = BASE / "feature_importance.json"
BACKGROUND = BASE / "dnn_background.csv"

REQUIRED = [MODEL, PREPROCESS, SCALER, METRICS, IMPORTANCE, BACKGROUND]


def fail(message: str) -> None:
    print(f"FAIL  {message}")
    raise SystemExit(1)


def main() -> int:
    missing = [path.name for path in REQUIRED if not path.exists()]
    if missing:
        fail(f"missing artefacts: {', '.join(missing)}")

    model = tf.keras.models.load_model(MODEL)
    preprocessor = dnn_data.load_preprocessor(PREPROCESS, SCALER)
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    importance = json.loads(IMPORTANCE.read_text(encoding="utf-8"))
    background = pd.read_csv(BACKGROUND)

    columns = preprocessor["columns"]
    if len(columns) != len(set(columns)):
        fail("duplicate column names in dnn_preprocess.json")
    if int(model.input_shape[-1]) != len(columns):
        fail(f"model expects {model.input_shape[-1]} inputs, "
             f"preprocessor defines {len(columns)}")
    expected_background = dnn_data.RAW_FEATURES + [dnn_data.TARGET]
    if list(background.columns) != expected_background:
        fail("dnn_background.csv does not match RAW_FEATURES + target")
    if not 0.0 < float(metrics["threshold"]) < 1.0:
        fail("threshold in dnn_metrics.json is outside (0, 1)")
    if metrics["features"]["columns"] != columns:
        fail("metrics feature order differs from the preprocessor")

    entries = importance["features"]
    if len(entries) != len(columns):
        fail("feature_importance.json does not cover every column")
    shares = sum(float(entry["share"]) for entry in entries)
    if abs(shares - 1.0) > 1e-6:
        fail(f"SHAP shares sum to {shares:.6f}, expected 1.0")
    if [entry["mean_abs_shap"] for entry in entries] != sorted(
            (entry["mean_abs_shap"] for entry in entries), reverse=True):
        fail("feature importance is not sorted by mean |SHAP|")

    matrix = dnn_data.transform_frame(background, preprocessor)
    if not np.isfinite(matrix).all():
        fail("transformed background contains NaN or infinity")

    probabilities = model.predict(matrix, verbose=0).ravel()
    if probabilities.shape != (len(background),):
        fail("model output shape is not (rows,)")
    if not np.all((probabilities >= 0) & (probabilities <= 1)):
        fail("model output is not a probability")

    single = dnn_data.transform_frame(background.head(1), preprocessor)
    if not np.allclose(single[0], matrix[0], atol=1e-6):
        fail("single-row preprocessing differs from batch preprocessing")

    threshold = float(metrics["threshold"])
    auc = float(metrics["test"]["auc"])
    print("Stage 1 artefact check")
    print(f"  model      : {MODEL.name} "
          f"({model.count_params():,} params, {len(columns)} inputs)")
    print(f"  split      : {metrics['split']['train']:,} train / "
          f"{metrics['split']['validation']:,} val / "
          f"{metrics['split']['test']:,} test")
    print(f"  test AUC   : {auc:.4f}")
    print(f"  threshold  : {threshold:.3f} ({metrics['threshold_rule']})")
    print(f"  background : {len(background):,} rows, mean risk "
          f"{probabilities.mean():.1%}")
    print(f"  top SHAP   : " + ", ".join(
        f"{entry['display']} {entry['share']:.1%}" for entry in entries[:3]))
    print("\n  sample predictions")
    for position in range(min(3, len(background))):
        row = background.iloc[position]
        decision = "positive" if probabilities[position] >= threshold else "negative"
        print(f"    {row['gender']:<6} age {row['age']:.0f} | "
              f"HbA1c {row['HbA1c_level']:.1f} | glucose "
              f"{row['blood_glucose_level']:.0f} -> "
              f"{probabilities[position]:.3f} ({decision})")

    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
