"""Stage 1 — train, evaluate and export the diabetes-risk DNN.

Dataset: Kaggle ``diabetes_prediction_dataset.csv`` (100,000 rows, 8 features).
Outputs (all written next to this script unless ``--outdir`` is given):

* ``dnn_model.keras``     — best Keras model (weights + architecture)
* ``scaler.pkl``          — StandardScaler fitted on the training split
* ``dnn_preprocess.json`` — feature order, category levels, medians
* ``dnn_metrics.json``    — accuracy / AUC / classification report + split info
* ``feature_importance.json`` — global mean |SHAP| per feature
* ``dnn_background.csv``  — raw background sample for later local explanations
* ``dnn_training_curve.png`` / ``dnn_roc_curve.png`` — report figures

Usage
-----
    python train_dnn.py                     # full run
    python train_dnn.py --quick             # tiny smoke run, no SHAP
    python train_dnn.py --epochs 60 --shap-samples 300
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

import dnn_data

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "diabetes_prediction_dataset.csv"
FALLBACK_DATA = Path(r"C:\Users\adars\OneDrive\Desktop\DS\DS"
                     r"\diabetes_prediction_dataset.csv")

SEED = 42


# ── helpers ──────────────────────────────────────────────────────────────────
def jsonable(value):
    """Recursively convert numpy/pandas scalars so json.dump always works."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    return value


def metrics_at(y_true, probabilities, threshold):
    """Classification metrics with an explicit decision threshold."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(
            y_true, predictions, output_dict=True, zero_division=0),
        "predicted_positive": int(predictions.sum()),
    }


def build_model(n_features, learning_rate=1e-3):
    """A deliberately small MLP: enough capacity for 13 inputs, cheap to load."""
    inputs = tf.keras.Input(shape=(n_features,), name="measurements")
    x = tf.keras.layers.Dense(
        64, activation="relu", kernel_initializer="he_normal", name="dense_1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="batch_norm")(x)
    x = tf.keras.layers.Dropout(0.30, name="dropout_1")(x)
    x = tf.keras.layers.Dense(
        32, activation="relu", kernel_initializer="he_normal", name="dense_2")(x)
    x = tf.keras.layers.Dropout(0.20, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="risk")(x)

    model = tf.keras.Model(inputs, outputs, name="diabetes_dnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model


def compute_shap(model, x_train, x_test, n_background, n_explain, nsamples):
    """Global mean |SHAP| from KernelSHAP (model-agnostic, no TF graph tricks)."""
    rng = np.random.default_rng(SEED)
    background_idx = rng.choice(
        len(x_train), size=min(n_background, len(x_train)), replace=False)
    explain_idx = rng.choice(
        len(x_test), size=min(n_explain, len(x_test)), replace=False)
    background = x_train[background_idx]
    explain = x_test[explain_idx]

    def predict_fn(matrix):
        return model.predict(np.asarray(matrix, dtype=np.float32),
                             verbose=0).ravel()

    explainer = shap.KernelExplainer(predict_fn, background)
    try:
        values = explainer.shap_values(
            explain, nsamples=nsamples, l1_reg=0.0, silent=True)
    except TypeError:                       # older shap without `silent`
        values = explainer.shap_values(explain, nsamples=nsamples, l1_reg=0.0)

    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, 0]
    return values, background, explain


def save_training_plot(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for axis, key, title in ((axes[0], "loss", "Binary cross-entropy"),
                             (axes[1], "auc", "Validation AUC")):
        axis.plot(history.history.get(key, []), label="train", lw=1.8)
        axis.plot(history.history.get(f"val_{key}", []), label="validation", lw=1.8)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("epoch", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_roc_plot(y_test, probabilities, threshold, out_path):
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)

    fig, axis = plt.subplots(figsize=(5.2, 4.4))
    axis.plot(fpr, tpr, color="#1f6feb", lw=2,
              label=f"DNN (AUC {auc:.3f})")
    axis.plot([0, 1], [0, 1], color="#9aa3ad", ls="--", lw=1)
    predictions = probabilities >= threshold
    tp = int(((y_test == 1) & predictions).sum())
    fp = int(((y_test == 0) & predictions).sum())
    axis.scatter([fp / max(1, (y_test == 0).sum())],
                 [tp / max(1, (y_test == 1).sum())],
                 color="#d64545", zorder=5, s=45,
                 label=f"threshold {threshold:.2f}")
    axis.set_xlabel("False-positive rate", fontsize=9)
    axis.set_ylabel("True-positive rate", fontsize=9)
    axis.set_title("Test ROC curve", fontsize=10)
    axis.tick_params(labelsize=8)
    axis.legend(fontsize=8, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None,
                        help="CSV path (defaults to the file next to this script)")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0,
                        help="stratified row cap for a fast smoke run")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--shap-samples", type=int, default=200,
                        help="test rows explained")
    parser.add_argument("--shap-background", type=int, default=100)
    parser.add_argument("--shap-nsamples", type=int, default=200)
    parser.add_argument("--quick", action="store_true",
                        help="4k rows, 3 epochs, no SHAP (pipeline check)")
    args = parser.parse_args()

    if args.quick:
        args.limit = args.limit or 4000
        args.epochs = 3
        args.no_shap = True

    tf.keras.utils.set_random_seed(SEED)

    data_path = Path(args.data) if args.data else (
        DEFAULT_DATA if DEFAULT_DATA.exists() else FALLBACK_DATA)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")

    outdir = Path(args.outdir) if args.outdir else BASE_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    frame = dnn_data.load_dataset(data_path)
    if args.limit and args.limit < len(frame):
        frame, _ = train_test_split(
            frame, train_size=args.limit, stratify=frame[dnn_data.TARGET],
            random_state=SEED)
        frame = frame.reset_index(drop=True)
    print(f"Dataset : {len(frame):,} rows -> {data_path}")

    y_all = frame[dnn_data.TARGET].astype(int).to_numpy()
    train_frame, temp_frame, y_train, y_temp = train_test_split(
        frame, y_all, test_size=0.30, stratify=y_all, random_state=SEED)
    val_frame, test_frame, y_val, y_test = train_test_split(
        temp_frame, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

    preprocessor = dnn_data.fit_preprocessor(train_frame)
    x_train = dnn_data.transform_frame(train_frame, preprocessor)
    x_val = dnn_data.transform_frame(val_frame, preprocessor)
    x_test = dnn_data.transform_frame(test_frame, preprocessor)
    feature_names = preprocessor["columns"]

    print(f"Split   : train {len(x_train):,} | val {len(x_val):,} | "
          f"test {len(x_test):,}")
    print(f"Features: {len(feature_names)} -> {', '.join(feature_names)}")
    print(f"Target  : {y_all.sum():,} positives "
          f"({y_all.mean():.2%})")

    counts = np.bincount(y_train, minlength=2)
    class_weight = {
        0: float(len(y_train) / (2 * counts[0])),
        1: float(len(y_train) / (2 * counts[1])),
    }
    print(f"Weights : 0 -> {class_weight[0]:.3f} | 1 -> {class_weight[1]:.3f}")

    model = build_model(len(feature_names))
    model.summary()

    model_path = outdir / "dnn_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=10,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4,
            min_lr=1e-5, verbose=1),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    val_proba = model.predict(x_val, verbose=0).ravel()
    test_proba = model.predict(x_test, verbose=0).ravel()

    fpr, tpr, thresholds = roc_curve(y_val, val_proba)
    youden = tpr - fpr
    best_index = int(np.argmax(youden))
    tuned_threshold = float(np.clip(thresholds[best_index], 0.02, 0.98))

    test_auc = float(roc_auc_score(y_test, test_proba))
    val_auc = float(roc_auc_score(y_val, val_proba))
    at_default = metrics_at(y_test, test_proba, 0.50)
    at_tuned = metrics_at(y_test, test_proba, tuned_threshold)
    val_at_tuned = metrics_at(y_val, val_proba, tuned_threshold)

    # ── artefacts that the app will load ────────────────────────────────────
    dnn_data.save_preprocessor(
        preprocessor, outdir / "dnn_preprocess.json", outdir / "scaler.pkl")
    model.save(model_path)

    background_n = min(300, len(train_frame))
    train_frame.sample(background_n, random_state=SEED).to_csv(
        outdir / "dnn_background.csv", index=False)

    importance_payload = {
        "method": "KernelSHAP (model-agnostic)",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_units": "predicted probability of diabetes",
        "features": [],
    }
    shap_seconds = None
    if not args.no_shap:
        print("\nSHAP    : computing global importance "
              f"({args.shap_samples} rows, {args.shap_background} background)…")
        shap_started = time.time()
        values, background, explained = compute_shap(
            model, x_train, x_test,
            args.shap_background, args.shap_samples, args.shap_nsamples)
        shap_seconds = time.time() - shap_started

        mean_abs = np.abs(values).mean(axis=0)
        mean_signed = values.mean(axis=0)
        total = float(mean_abs.sum()) or 1.0
        order = np.argsort(-mean_abs)
        importance_payload.update({
            "n_background": int(len(background)),
            "n_explained": int(len(explained)),
            "nsamples": int(args.shap_nsamples),
            "seconds": round(shap_seconds, 1),
            "features": [
                {
                    "feature": feature_names[i],
                    "display": preprocessor["display_names"][feature_names[i]],
                    "mean_abs_shap": float(mean_abs[i]),
                    "mean_shap": float(mean_signed[i]),
                    "share": float(mean_abs[i] / total),
                }
                for i in order
            ],
        })
        for entry in importance_payload["features"][:5]:
            print(f"  {entry['display']:<22} mean|SHAP| "
                  f"{entry['mean_abs_shap']:.4f}  share {entry['share']:.1%}")

    (outdir / "feature_importance.json").write_text(
        json.dumps(jsonable(importance_payload), indent=2), encoding="utf-8")

    best_epoch = int(getattr(callbacks[1], "best_epoch", 0) or 0)
    metrics_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "path": str(data_path),
            "rows": int(len(frame)),
            "positives": int(y_all.sum()),
            "prevalence": float(y_all.mean()),
        },
        "split": {
            "train": int(len(x_train)),
            "validation": int(len(x_val)),
            "test": int(len(x_test)),
        },
        "features": {
            "count": len(feature_names),
            "columns": feature_names,
            "display": [preprocessor["display_names"][c] for c in feature_names],
        },
        "model": {
            "architecture": "64 -> 32 -> 1 (ReLU, dropout, batch norm)",
            "params": int(model.count_params()),
            "class_weight": class_weight,
            "epochs_requested": int(args.epochs),
            "epochs_run": int(len(history.history["loss"])),
            "best_epoch": best_epoch,
            "batch_size": int(args.batch_size),
        },
        "threshold": tuned_threshold,
        "threshold_rule": "Youden's J on the validation split",
        "validation": {
            "auc": val_auc,
            "at_default": metrics_at(y_val, val_proba, 0.50),
            "at_tuned": val_at_tuned,
        },
        "test": {
            "auc": test_auc,
            "at_default": at_default,
            "at_tuned": at_tuned,
        },
        "history": jsonable(history.history),
        "shap": {
            "computed": not args.no_shap,
            "seconds": None if shap_seconds is None else round(shap_seconds, 1),
            "top": [entry["feature"] for entry in
                    importance_payload["features"][:3]],
        },
        "artifacts": {
            "model": "dnn_model.keras",
            "scaler": "scaler.pkl",
            "preprocess": "dnn_preprocess.json",
            "importance": "feature_importance.json",
            "background": "dnn_background.csv",
        },
        "runtime_seconds": round(time.time() - started, 1),
    }
    (outdir / "dnn_metrics.json").write_text(
        json.dumps(jsonable(metrics_payload), indent=2), encoding="utf-8")

    save_training_plot(history, outdir / "dnn_training_curve.png")
    save_roc_plot(y_test, test_proba, tuned_threshold,
                  outdir / "dnn_roc_curve.png")

    # ── console report ──────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("STAGE 1 — DNN RESULTS")
    print("=" * 66)
    print(f"Rows {len(frame):,} | features {len(feature_names)} | "
          f"params {model.count_params():,} | epochs {len(history.history['loss'])}")
    print(f"Test AUC            : {test_auc:.4f}")
    print(f"At threshold 0.50   : accuracy {at_default['accuracy']:.4f} | "
          f"precision {at_default['precision']:.4f} | "
          f"recall {at_default['recall']:.4f} | F1 {at_default['f1']:.4f}")
    print(f"At tuned threshold {tuned_threshold:.3f}: accuracy "
          f"{at_tuned['accuracy']:.4f} | precision {at_tuned['precision']:.4f} | "
          f"recall {at_tuned['recall']:.4f} | F1 {at_tuned['f1']:.4f}")
    print("\nClassification report @ tuned threshold")
    print(classification_report(y_test, (test_proba >= tuned_threshold).astype(int),
                                target_names=["No diabetes", "Diabetes"],
                                digits=3, zero_division=0))
    print(f"Artifacts written to {outdir}")
    print(f"Total runtime: {metrics_payload['runtime_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
