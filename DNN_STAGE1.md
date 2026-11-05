# Stage 1 — Model & Backend (DNN)

Diabetes-risk classifier trained on `diabetes_prediction_dataset.csv`
(Kaggle, 100,000 rows, 8 input columns). This stage produces the model and the
data files the Streamlit app will load in Stage 2.

Run everything from this folder.

---

## Results (full 100,000-row run, seed 42)

| Metric | Value |
|---|---|
| Test AUC | **0.9765** |
| Accuracy @ 0.478 | 0.8839 |
| Precision @ 0.478 | 0.4183 |
| Recall @ 0.478 | 0.9357 |
| F1 @ 0.478 | 0.5781 |
| Accuracy @ 0.50 | 0.8901 |
| Precision @ 0.50 | 0.4316 |
| Recall @ 0.50 | 0.9255 |
| F1 @ 0.50 | 0.5887 |

* Split: 70,000 train / 15,000 validation / 15,000 test (stratified).
* Model: 64 → batch-norm → dropout 0.3 → 32 → dropout 0.2 → 1 sigmoid,
  3,265 parameters, Adam 1e-3, batch 256, early stopping on `val_auc`
  (35 epochs run, best at epoch 25).
* Decision threshold **0.478**, chosen by Youden's J on the validation split.
* Prevalence in the dataset is 8.5%, so accuracy alone is misleading — the
  model is deliberately recall-oriented (class weights) and catches ~94% of
  true cases at the cost of precision.

### Global SHAP importance (KernelSHAP, 100 background / 200 explained rows)

| Feature | mean \|SHAP\| | Share |
|---|---|---|
| HbA1c level | 0.1222 | 35.7% |
| Blood glucose | 0.0826 | 24.1% |
| Age | 0.0532 | 15.5% |
| BMI | 0.0263 | 7.7% |
| Smoking: never | 0.0134 | 3.9% |
| Hypertension | 0.0121 | 3.5% |
| Gender: male | 0.0086 | 2.5% |
| Smoking: current | 0.0077 | 2.3% |
| Smoking: not current | 0.0056 | 1.6% |
| Smoking: former | 0.0050 | 1.5% |
| Heart disease | 0.0031 | 0.9% |
| Smoking: ever | 0.0028 | 0.8% |
| Gender: other | 0.0000 | 0.0% |

HbA1c + blood glucose account for **59.8%** of the model's decisions, as
expected for a diabetes diagnosis target.

---

## Files

| File | Purpose |
|---|---|
| `dnn_data.py` | Feature list, cleaning, one-hot encoding, scaler fit/save/load. Shared by training and the app. |
| `train_dnn.py` | Trains, evaluates, computes SHAP, writes every artifact below. |
| `check_dnn.py` | Loads the artifacts and asserts they are consistent. Prints `RESULT: PASS`. |
| `requirements-dnn.txt` | Python dependencies for this stage. |
| `dnn_model.keras` | Best Keras 3 model (architecture + weights). |
| `scaler.pkl` | `StandardScaler` fitted on the training split only. |
| `dnn_preprocess.json` | Feature order, display names, medians, category levels. |
| `dnn_metrics.json` | AUC, threshold, metrics, reports, training history, split sizes. |
| `feature_importance.json` | Global mean \|SHAP\| per feature, sorted. |
| `dnn_background.csv` | 300 raw training rows (features + target) for local explanations. |
| `dnn_training_curve.png`, `dnn_roc_curve.png` | Figures for the report. |
| `diabetes_prediction_dataset.csv` | Local copy of the dataset (reproducible runs). |

### Why `.keras` and not `.h5`

The model is saved in Keras 3's native format, which is what
`tf.keras.models.load_model("dnn_model.keras")` expects. Legacy `.h5` is still
supported (`model.save("dnn_model.h5")`) but loses nothing here and is the
older format. If your submission requires `.h5` explicitly, add:

```python
import tensorflow as tf
tf.keras.models.load_model("dnn_model.keras").save("dnn_model.h5")
```

---

## Commands

```powershell
# Full training run (~6 min on CPU; SHAP is ~5 min of that)
python train_dnn.py

# Fast pipeline smoke test (4k rows, 3 epochs, no SHAP)
python train_dnn.py --quick

# Verify the artifacts before deploying
python check_dnn.py
```

Useful flags: `--epochs`, `--limit`, `--no-shap`, `--shap-samples`,
`--shap-background`, `--shap-nsamples`, `--data`, `--outdir`.

---

## Feature contract

13 model inputs, in this exact order (from `dnn_preprocess.json`):

```
age, bmi, HbA1c_level, blood_glucose_level,
hypertension, heart_disease,
gender_Male, gender_Other,
smoking_history_current, smoking_history_ever, smoking_history_former,
smoking_history_never, smoking_history_not current
```

* Numeric columns are imputed with training medians, then standardised.
* Reference levels are `gender=Female` and `smoking_history=No Info`; unknown
  values fall back to the reference instead of an all-zero row.
* The app must call `dnn_data.transform_frame(frame, preprocessor)` rather than
  building features by hand, so serving matches training exactly.

---

## Deployment notes (Streamlit Community Cloud, free tier)

* `tensorflow` is ~600 MB installed and counts against the 1 GB memory limit.
  `tensorflow-cpu` is a smaller option: replace the first line of
  `requirements-dnn.txt` with `tensorflow-cpu>=2.16`.
* Ship these files with the app: `dnn_data.py`, `dnn_model.keras`,
  `scaler.pkl`, `dnn_preprocess.json`, `dnn_metrics.json`,
  `feature_importance.json`, `dnn_background.csv`.
  Do **not** ship the CSV of 100k rows or `train_dnn.py` unless you want
  in-app retraining.
* Load the model inside `@st.cache_resource` so it is created once per session
  group.

---

## Known limitation

Class weighting inflates the raw probabilities: the background sample averages
19.6% predicted risk against an 8.5% true prevalence. Ranking (AUC) is
unaffected and the tuned threshold handles the decision, but the number shown
to users reads higher than the true probability. Options for Stage 2:

1. Keep raw probabilities + tuned threshold (current behaviour, simplest).
2. Isotonic calibration on the validation split, saved as `calibrator.pkl`
   (~20 lines, makes the percentage honest without touching AUC).
