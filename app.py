"""DiabPredict Pro - Streamlit clinical decision support for diabetes risk.

Five pages:
    Dashboard          aggregate statistics from this session's predictions
    Single Prediction  full input form, risk gauge, SHAP waterfall, PDF export
    What-If Simulator  live sliders showing how risk responds to changes
    Batch Prediction   CSV upload, scored table, download
    Model Insights     global SHAP, test metrics, confusion matrix, training curves

All artefacts are loaded once at start-up (``@st.cache_resource``); nothing is
retrained. Preprocessing goes through ``dnn_data`` so serving matches training.
"""

from __future__ import annotations

import hmac
import io
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import dnn_data
from dnn_data import load_preprocessor, transform_frame

# ── page config (must be the first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="DiabPredict Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── paths ───────────────────────────────────────────────────────────────────
MODEL_PATH = BASE_DIR / "dnn_model.keras"
PREPROCESS_PATH = BASE_DIR / "dnn_preprocess.json"
SCALER_PATH = BASE_DIR / "scaler.pkl"
IMPORTANCE_PATH = BASE_DIR / "feature_importance.json"
BACKGROUND_PATH = BASE_DIR / "dnn_background.csv"
METRICS_PATH = BASE_DIR / "dnn_metrics.json"
LOG_PATH = BASE_DIR / "predictions_log.csv"

APP_NAME = "DiabPredict Pro"
APP_VERSION = "1.0.0"
DEFAULT_PASSWORD = "clinician123"

# ── palette ─────────────────────────────────────────────────────────────────
PRIMARY = "#1E88E5"
SUCCESS = "#43A047"
WARNING = "#FB8C00"
DANGER = "#E53935"
CRITICAL = "#B71C1C"
INK = "#0F2A44"
MUTED = "#5C7186"
LINE = "#E3EBF5"

BAND_ORDER = ["Low", "Moderate", "High", "Critical"]
BAND_COLOR = {
    "Low": SUCCESS,
    "Moderate": WARNING,
    "High": DANGER,
    "Critical": CRITICAL,
}
BAND_TINT = {
    "Low": "#E9F7EC",
    "Moderate": "#FFF4E5",
    "High": "#FDECEC",
    "Critical": "#FBE4E4",
}
BAND_TINT_DARK = {
    "Low": "#123322",
    "Moderate": "#3A2A12",
    "High": "#3A1A1A",
    "Critical": "#3A1418",
}
BAND_INK_DARK = {
    "Low": "#6FDD8B",
    "Moderate": "#FFB74D",
    "High": "#FF8A80",
    "Critical": "#FF8A80",
}
BAND_HELP = {
    "Low": "below the 30% action threshold",
    "Moderate": "worth a conversation with a clinician",
    "High": "strongly suggests further testing",
    "Critical": "urgent clinical review recommended",
}

# Column -> (label, min, max, step, unit, help)
NUMERIC_INPUTS = {
    "age": ("Age", 1.0, 80.0, 1.0, "years", "Patient age at assessment."),
    "bmi": ("BMI", 10.0, 70.0, 0.1, "kg/m²", "Body mass index."),
    "HbA1c_level": ("HbA1c level", 3.5, 9.0, 0.1, "%",
                    "Glycated haemoglobin. 5.7-6.4% is prediabetes, "
                    "6.5%+ is the diabetes range."),
    "blood_glucose_level": ("Blood glucose", 80.0, 300.0, 1.0, "mg/dL",
                            "Fasting or random blood glucose."),
}

DEFAULTS = {
    "gender": "Female",
    "age": 45.0,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 27.3,
    "HbA1c_level": 5.8,
    "blood_glucose_level": 140.0,
}

PAGES = [
    "Dashboard",
    "Single Prediction",
    "What-If Simulator",
    "Batch Prediction",
    "Model Insights",
]
PAGE_CAPTION = {
    "Dashboard": "Population overview",
    "Single Prediction": "Patient assessment",
    "What-If Simulator": "Scenario analysis",
    "Batch Prediction": "Cohort scoring",
    "Model Insights": "Model transparency",
}

# ── clinician vs patient presentation ───────────────────────────────────────
VIEW_MODES = ["Clinician", "Patient"]

BAND_PLAIN = {
    "Low": "Lower risk",
    "Moderate": "Some risk",
    "High": "High risk",
    "Critical": "Very high risk",
}
BAND_PATIENT_NOTE = {
    "Low": "Your results look reassuring. Keep up your routine check-ups.",
    "Moderate": "Some of your results are worth discussing with your doctor.",
    "High": "Your results suggest you should see a doctor soon for a blood test.",
    "Critical": "Your results suggest you should seek medical advice promptly.",
}
PLAIN_FEATURE = {
    "age": "Your age",
    "bmi": "Your weight for height (BMI)",
    "HbA1c_level": "Your average blood sugar (HbA1c)",
    "blood_glucose_level": "Your blood sugar reading",
    "hypertension": "High blood pressure",
    "heart_disease": "Heart disease",
    "gender_Male": "Your sex",
    "gender_Other": "Your sex",
    "smoking_history_current": "Smoking",
    "smoking_history_ever": "Smoking",
    "smoking_history_former": "Smoking",
    "smoking_history_never": "Smoking",
    "smoking_history_not current": "Smoking",
}


# ════════════════════════════════════════════════════════════════════════════
# Styling
# ════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="st-"], .stApp, button, input, textarea, select {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}
.stApp { background: #F7FAFD; }
[data-testid="stHeader"] { background: #F7FAFD; }
.block-container { padding-top: 3.1rem; padding-bottom: 3rem; max-width: 1280px; }
#MainMenu, footer { visibility: hidden; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E3EBF5;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
[data-testid="stSidebar"] hr { margin: 0.9rem 0; border-color: #EDF3FA; }

.dp-brand { display: flex; align-items: center; gap: 12px; padding: 0 2px 4px; }
.dp-brand-mark {
    width: 46px; height: 46px; border-radius: 13px;
    background: linear-gradient(140deg, #1E88E5 0%, #42A5F5 100%);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 6px 16px rgba(30, 136, 229, 0.28);
}
.dp-brand-title { font-size: 1.06rem; font-weight: 700; color: #0F2A44; line-height: 1.15; }
.dp-brand-sub { font-size: 0.68rem; color: #7B8FA6; font-weight: 500; letter-spacing: .04em; text-transform: uppercase; white-space: nowrap; }

.dp-nav-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; color: #93A5BA; margin: 0.2rem 0 0.1rem 2px;
}

[data-testid="stSidebar"] div[role="radiogroup"] { gap: 3px; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 9px 12px; border-radius: 10px; width: 100%;
    transition: background .15s ease, color .15s ease;
    cursor: pointer;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: #F0F6FE; }
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: #E8F2FE; box-shadow: inset 3px 0 0 #1E88E5;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #1467B4 !important; font-weight: 600;
}
[data-testid="stSidebar"] div[role="radiogroup"] p { font-size: 0.9rem; color: #40566E; margin: 0; }

/* ---------- top-right theme switch ---------- */
[data-testid="stHorizontalBlock"]:has(.dp-title) > [data-testid="stColumn"]:first-child {
    flex: 1 1 auto !important;
}
[data-testid="stHorizontalBlock"]:has(.dp-title) > [data-testid="stColumn"]:last-child {
    flex: 0 0 160px !important; max-width: 160px; min-width: 0;
}
.st-key-theme_switch div[role="radiogroup"] {
    justify-content: flex-end; gap: 4px; flex-wrap: nowrap;
}
.st-key-theme_switch div[role="radiogroup"] > label {
    width: auto; flex: 0 0 auto; min-width: 0; padding: 4px 10px;
    border-radius: 999px; border: 1px solid #E3EBF5; background: #FFFFFF;
}
.st-key-theme_switch div[role="radiogroup"] > label > div:first-child { display: none; }
.st-key-theme_switch div[role="radiogroup"] > label:has(input:checked) {
    background: #E8F2FE; border-color: #BBDCF8; box-shadow: none;
}
.st-key-theme_switch div[role="radiogroup"] > label:has(input:checked) p {
    color: #1467B4 !important;
}
.st-key-theme_switch div[role="radiogroup"] p {
    font-size: 0.76rem; color: #5C7186; white-space: nowrap;
}
.st-key-theme_switch { margin-top: 2px; }

/* ---------- clinician / patient switch ---------- */
.st-key-view_switch div[role="radiogroup"] { flex-wrap: nowrap; gap: 6px; }
.st-key-view_switch div[role="radiogroup"] > label {
    flex: 1 1 0; width: auto; min-width: 0; padding: 7px 4px;
    border: 1px solid #E3EEFB; border-radius: 9px; background: #FFFFFF;
    justify-content: center; box-shadow: none;
}
.st-key-view_switch div[role="radiogroup"] > label:hover { background: #F4F9FF; }
.st-key-view_switch div[role="radiogroup"] > label:has(input:checked) {
    background: #E8F2FE; border-color: #BBDCF8; box-shadow: none;
}
.st-key-view_switch div[role="radiogroup"] > label:has(input:checked) p {
    color: #1467B4 !important;
}
.st-key-view_switch div[role="radiogroup"] p { font-size: 0.82rem; }
.dp-view-hint { font-size: 0.72rem; color: #93A5BA; margin: 4px 0 0 2px; }

.dp-side-card {
    background: #F6FAFF; border: 1px solid #E3EEFB; border-radius: 12px;
    padding: 12px 14px; font-size: 0.76rem; color: #4A6478; line-height: 1.65;
}
.dp-side-card b { color: #0F2A44; }
.dp-side-row { display: flex; justify-content: space-between; gap: 8px; }
.dp-side-row span:last-child { font-weight: 600; color: #0F2A44; }

/* ---------- typography ---------- */
.dp-eyebrow {
    font-size: 0.7rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: #1E88E5; margin-bottom: 2px;
}
.dp-title { font-size: 1.72rem; font-weight: 700; color: #0F2A44; letter-spacing: -0.02em; margin: 0; }
.dp-sub { color: #5C7186; font-size: 0.93rem; margin-top: 4px; }
.dp-section {
    font-size: 0.95rem; font-weight: 700; color: #0F2A44;
    margin: 4px 0 2px; display: flex; align-items: center; gap: 8px;
}
.dp-section:before {
    content: ""; width: 4px; height: 15px; border-radius: 3px;
    background: #1E88E5; display: inline-block;
}

/* ---------- cards ---------- */
.dp-card {
    background: #FFFFFF; border: 1px solid #E6EEF8; border-radius: 14px;
    padding: 16px 18px; box-shadow: 0 1px 2px rgba(16, 42, 74, .04), 0 8px 22px rgba(16, 42, 74, .05);
}
.dp-stat { position: relative; overflow: hidden; }
.dp-stat:after {
    content: ""; position: absolute; inset: 0 auto 0 0; width: 3px; background: #1E88E5;
}
.dp-stat-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: .07em;
    text-transform: uppercase; color: #8296AC;
}
.dp-stat-value { font-size: 1.75rem; font-weight: 700; color: #0F2A44; line-height: 1.25; margin-top: 2px; }
.dp-stat-sub { font-size: 0.78rem; color: #7B8FA6; margin-top: 2px; }

/* ---------- theme-aware tables ---------- */
.dp-table-wrap {
    overflow: auto; border: 1px solid #E6EEF8; border-radius: 12px;
    background: #FFFFFF; max-width: 100%; scrollbar-width: thin;
}
.dp-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.dp-table th {
    position: sticky; top: 0; z-index: 1; background: #F3F7FB;
    color: #5C7186; text-align: left; font-weight: 600;
    padding: 9px 12px; white-space: nowrap;
    border-bottom: 1px solid #E6EEF8;
}
.dp-table td {
    padding: 8px 12px; color: #0F2A44; white-space: nowrap;
    border-bottom: 1px solid #EFF4FA;
}
.dp-table tbody tr:last-child td { border-bottom: none; }
.dp-table tbody tr:hover td { background: #F6FAFF; }

.dp-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: .01em;
}
.dp-badge .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

.dp-kv {
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 16px; padding: 6px 0; border-bottom: 1px dashed #EDF3FA;
    font-size: 0.85rem;
}
.dp-kv:last-child { border-bottom: none; }
.dp-kv span:first-child { color: #6B8098; flex: 0 0 auto; }
.dp-kv span:last-child {
    color: #0F2A44; font-weight: 600; text-align: right;
    min-width: 0; overflow-wrap: anywhere;
}

.dp-note {
    background: #F3F8FF; border: 1px solid #DCECFF; border-left: 4px solid #1E88E5;
    border-radius: 10px; padding: 11px 14px; font-size: 0.84rem; color: #31506C; line-height: 1.55;
}
.dp-warn {
    background: #FFF8EC; border: 1px solid #FBE2BC; border-left: 4px solid #FB8C00;
    border-radius: 10px; padding: 11px 14px; font-size: 0.84rem; color: #7A4E00; line-height: 1.55;
}

/* ---------- gauge ---------- */
.dp-gauge-wrap { display: flex; flex-direction: column; align-items: center; padding: 4px 0 0; }
.dp-gauge-arc {
    animation: dp-gauge-fill 900ms cubic-bezier(.22,.61,.36,1) forwards;
}
@keyframes dp-gauge-fill {
    from { stroke-dashoffset: var(--dp-len); }
}
.dp-gauge-band { margin-top: -6px; }
.dp-gauge-caption { font-size: 0.8rem; color: #7B8FA6; text-align: center; margin-top: 6px; }

.dp-delta {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    font-size: 0.88rem; font-weight: 600; color: #40566E;
    background: #F6FAFF; border: 1px solid #E3EEFB; border-radius: 10px;
    padding: 9px 12px; margin-top: 10px;
}

/* ---------- widgets ---------- */
.stButton > button, .stFormSubmitButton > button {
    border-radius: 10px; border: none; background: #1E88E5; color: #FFFFFF;
    font-weight: 600; padding: 0.55rem 1.15rem; width: 100%;
    box-shadow: 0 4px 12px rgba(30, 136, 229, .22);
    transition: transform .12s ease, background .15s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: #1668B8; color: #FFFFFF; transform: translateY(-1px);
}
.stButton > button:focus, .stFormSubmitButton > button:focus { color: #FFFFFF !important; }
.stDownloadButton > button {
    border-radius: 10px; border: 1px solid #CFE3F7; background: #FFFFFF; color: #1467B4;
    font-weight: 600; width: 100%;
}
.stDownloadButton > button:hover { border-color: #1E88E5; background: #F3F8FF; color: #1467B4; }

[data-testid="stForm"] {
    background: #FFFFFF; border: 1px solid #E6EEF8; border-radius: 14px;
    padding: 18px 20px 8px; box-shadow: 0 8px 22px rgba(16, 42, 74, .05);
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
}
[data-testid="stColumn"]:has(.dp-login-title) {
    background: #FFFFFF; border: 1px solid #E6EEF8; border-radius: 20px;
    box-shadow: 0 18px 48px rgba(16, 42, 74, .10);
    padding: 22px 24px 16px;
}
[data-testid="stColumn"]:has(.dp-wi-mark) {
    background: #FFFFFF; border: 1px solid #E6EEF8; border-radius: 16px;
    box-shadow: 0 10px 30px rgba(16, 42, 74, .06);
    padding: 12px 16px 4px;
}
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
[data-testid="stFileUploader"] section {
    border: 1.5px dashed #BCD8F2; border-radius: 12px; background: #FAFCFF;
}
[data-testid="stFileUploader"] section:hover { border-color: #1E88E5; }

div[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #E6EEF8; border-radius: 14px;
    padding: 14px 16px; box-shadow: 0 8px 22px rgba(16, 42, 74, .05);
}

.stSlider [data-baseweb="slider"] div[role="slider"] { box-shadow: 0 0 0 4px rgba(30, 136, 229, .15); }

div[data-testid="stDataFrame"] {
    border: 1px solid #E6EEF8; border-radius: 12px; overflow: hidden; background: #FFFFFF;
}
[data-testid="stExpander"] {
    background: #FFFFFF; border: 1px solid #E6EEF8; border-radius: 12px;
}
[data-testid="stExpander"] summary { font-weight: 600; color: #0F2A44; }
hr { border-color: #E9F0F8; }

/* ---------- login ---------- */
.dp-login-card {
    background: #FFFFFF; border: 1px solid #E6EEF8; border-radius: 20px;
    padding: 34px 34px 26px; box-shadow: 0 18px 48px rgba(16, 42, 74, .10);
}
.dp-login-title { font-size: 1.5rem; font-weight: 700; color: #0F2A44; margin: 14px 0 2px; }
.dp-login-sub { color: #6B8098; font-size: 0.88rem; }
.dp-login-foot { color: #93A5BA; font-size: 0.74rem; text-align: center; margin-top: 14px; }
.dp-login-or { color: #93A5BA; font-size: 0.76rem; text-align: center; margin: 12px 0 8px; }
/* ---------- responsive ---------- */
@media (max-width: 1100px) {
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; row-gap: 12px; }
    [data-testid="stHorizontalBlock"]:not(:has(.dp-title)) > [data-testid="stColumn"] {
        flex: 1 1 280px !important; max-width: 100%;
    }
    [data-testid="stHorizontalBlock"]:has(.dp-stat) > [data-testid="stColumn"] {
        flex: 1 1 150px !important;
    }
    [data-testid="stHorizontalBlock"]:has(.dp-table-wrap) > [data-testid="stColumn"] {
        flex: 1 1 100% !important; max-width: 100% !important;
    }
    .dp-stat-value { font-size: 1.5rem; }
}

/* ---------- phones ---------- */
@media (max-width: 700px) {
    [data-testid="stHorizontalBlock"]:has(.dp-title) { flex-wrap: wrap !important; }
    [data-testid="stHorizontalBlock"]:has(.dp-title) > [data-testid="stColumn"]:last-child {
        flex: 1 1 100% !important; max-width: 100% !important;
    }
    .st-key-theme_switch div[role="radiogroup"] { justify-content: flex-start; }
    .st-key-theme_switch { margin-top: 0; }
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .dp-title { font-size: 1.5rem; line-height: 1.2; }
    .dp-sub { font-size: 0.85rem; }
    .dp-card { padding: 13px 14px; border-radius: 12px; }
    .dp-note { padding: 10px 12px; font-size: 0.8rem; }
    .dp-table th, .dp-table td { padding: 7px 9px; font-size: 0.78rem; }
    .dp-login-title { font-size: 1.35rem; }
    .dp-login-card { padding: 22px 18px 18px; }
    .stButton > button, .stDownloadButton > button,
    .stFormSubmitButton > button { min-height: 42px; }
}

@media (max-width: 420px) {
    [data-testid="stHorizontalBlock"]:has(.dp-stat) > [data-testid="stColumn"] {
        flex: 1 1 100% !important;
    }
    .dp-title { font-size: 1.32rem; }
    .dp-eyebrow { font-size: 0.64rem; }
    .dp-stat-value { font-size: 1.45rem; }
    .dp-kv { gap: 10px; font-size: 0.8rem; }
}

/* ---------- motion ---------- */
@media (prefers-reduced-motion: no-preference) {
    [data-testid="stMain"] .block-container { animation: dpFade .38s ease-out both; }
    .stApp, [data-testid="stSidebar"], [data-testid="stHeader"] {
        transition: background-color .3s ease;
    }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        transition: background-color .25s ease, border-color .25s ease;
    }
    .dp-card, .dp-note, .dp-badge, .dp-table-wrap {
        transition: background-color .25s ease, border-color .25s ease,
                    color .2s ease, box-shadow .25s ease;
    }
    .dp-card:hover { border-color: #CFE3F7; }
    [data-testid="stSidebar"] div[role="radiogroup"] > label,
    [data-testid="stSidebar"] .dp-side-card,
    .stButton > button, .stDownloadButton > button,
    .stFormSubmitButton > button, .st-key-theme_switch label {
        transition: background-color .18s ease, color .18s ease,
                    border-color .18s ease, box-shadow .18s ease,
                    transform .12s ease;
    }
    .stButton > button:active, .stDownloadButton > button:active,
    .stFormSubmitButton > button:active { transform: translateY(1px); }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .001ms !important;
    }
}
@keyframes dpFade {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
"""


# ════════════════════════════════════════════════════════════════════════════
# Small helpers
# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# Theme: clinical light (default) and midnight dark
# ════════════════════════════════════════════════════════════════════════════
THEMES = ["Light", "Dark"]

# Neutral colours in the stylesheet that need to flip in dark mode. Accents
# (blue / green / amber / red) are deliberately left alone.
DARK_TEXT = {
    "#F7FAFD": "#0B1220",
    "#0F2A44": "#E8F0FB",
    "#40566E": "#C3D2E4",
    "#31506C": "#C3D2E4",
    "#5C7186": "#A7B9CE",
    "#6B8098": "#9FB3C8",
    "#7B8FA6": "#8FA3B8",
    "#8296AC": "#8FA3B8",
    "#93A5BA": "#7E93A9",
    "#9AAEC2": "#7E93A9",
    "#A3B4C6": "#6F8499",
    "#1467B4": "#7EC0FF",
    "#E3EBF5": "#22314A",
    "#E6EEF8": "#22314A",
    "#E9F0F8": "#1E2C42",
    "#EDF3FA": "#1E2C42",
    "#EFF4FA": "#1E2C42",
    "#E8EFF8": "#22314A",
    "#DCE7F3": "#2A3A52",
    "#CFE3F7": "#2F4763",
    "#E3EEFB": "#22314A",
    "#DCECFF": "#243A56",
    "#BBDCF8": "#2F5C8A",
    "#F6FAFF": "#16233A",
    "#F3F8FF": "#16233A",
    "#FAFCFF": "#131F33",
    "#F0F6FE": "#1B2A42",
    "#E8F2FE": "#17304F",
    "#F3F7FB": "#16233A",
    "#F4F9FF": "#16233A",
    "#4A6478": "#A7B9CE",
    "#FFF8EC": "#2A2113",
    "#FBE2BC": "#4A3A17",
    "#7A4E00": "#F5C46B",
    "#BCD8F2": "#2F5C8A",
}

DARK_EXTRA = """
<style>
html, body, .stApp { background: #0B1220; }
[data-testid="stHeader"] { background: #0B1220; }
[data-testid="stSidebar"] { background: #0E1728; border-right-color: #22314A; }
[data-testid="stSidebar"] .block-container { background: transparent; }

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 { color: #DCE8F6 !important; }
[data-testid="stCaptionContainer"] p { color: #93A9C0 !important; }
[data-testid="stWidgetLabel"] p { color: #C3D2E4 !important; }

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #131F33 !important; color: #E8F0FB !important;
    border-color: #22314A !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] svg { fill: #A7B9CE; }
[data-testid="stForm"] { background: #131C2E !important; border-color: #22314A !important; }
[data-testid="stExpander"] { background: #131C2E !important; border-color: #22314A !important; }
[data-testid="stExpander"] summary { color: #E8F0FB !important; }
[data-testid="stExpander"] summary:hover { color: #7EC0FF !important; }
[data-testid="stMetric"] { background: #131C2E !important; border-color: #22314A !important; }
div[data-testid="stDataFrame"] { background: #131C2E !important; border-color: #22314A !important; }
div[data-testid="stDataFrame"] [role="columnheader"] {
    background: #182338 !important; color: #C3D2E4 !important;
}
[data-testid="stFileUploader"] section {
    background: #131F33 !important; border-color: #2F4763 !important;
}
[data-testid="stFileUploader"] section * { color: #C3D2E4 !important; }
[data-testid="stFileUploader"] section button {
    background: #16233A !important; border: 1px solid #2F4763 !important;
    color: #7EC0FF !important;
}
[data-testid="stFileUploaderFile"] { color: #C3D2E4 !important; }
[data-testid="stFileUploaderFile"] * { color: #93A9C0 !important; }
[data-testid="stFileUploaderFile"] svg { fill: #7E93A9; }
[data-testid="stTooltipIcon"] svg { fill: #7E93A9; }
[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"] { color: #7E93A9 !important; }
[data-testid="stSliderThumbValue"] { color: #7EC0FF !important; }
[data-testid="stToolbar"] { background: transparent; }
[data-testid="stMetricLabel"] p { color: #93A9C0 !important; }

[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: #16233A; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: #17304F;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #7EC0FF !important;
}
[data-testid="stSidebar"] .dp-side-card {
    background: #16233A !important; border-color: #22314A !important;
}
[data-testid="stSidebar"] .dp-side-card b { color: #E8F0FB; }

.stDownloadButton > button {
    background: #16233A !important; border-color: #2F4763 !important;
    color: #7EC0FF !important;
}
.stDownloadButton > button:hover {
    background: #1B2A42 !important; border-color: #1E88E5 !important;
    color: #7EC0FF !important;
}
</style>
"""

# Chart colours, refreshed by apply_theme() on every rerun.
CHART = {
    "ink": INK,
    "font": "#40566E",
    "soft": "#7B8FA6",
    "grid": "#EFF4FA",
    "surface": "#FFFFFF",
    "axis": "#DCE7F3",
}


def theme_dark() -> bool:
    return st.session_state.get("theme") == "Dark"


def band_tint(band: str) -> str:
    palette = BAND_TINT_DARK if theme_dark() else BAND_TINT
    return palette.get(band, "#2A3A52" if theme_dark() else "#EEF1F4")


def band_ink(band: str) -> str:
    if theme_dark():
        return BAND_INK_DARK.get(band, "#E8F0FB")
    return BAND_COLOR.get(band, INK)


def theme_text(text: str) -> str:
    """Swap the neutral palette for dark mode. Accents are preserved."""
    if not theme_dark():
        return text
    out = (text.replace("background: #FFFFFF", "background: #131C2E")
                .replace("background:#FFFFFF", "background:#131C2E"))
    for light, dark in DARK_TEXT.items():
        out = out.replace(light, dark)
    return out


def render_css() -> str:
    return theme_text(CSS) + (DARK_EXTRA if theme_dark() else "")


def md(body: str, **kwargs) -> None:
    """st.markdown with the active theme's palette applied."""
    st.markdown(theme_text(body), **kwargs)


def apply_theme() -> None:
    dark = theme_dark()
    CHART.update({
        "ink": "#E8F0FB" if dark else INK,
        "font": "#C3D2E4" if dark else "#40566E",
        "soft": "#8FA3B8" if dark else "#7B8FA6",
        "grid": "#1E2C42" if dark else "#EFF4FA",
        "surface": "#131C2E" if dark else "#FFFFFF",
        "axis": "#2A3A52" if dark else "#DCE7F3",
    })


def band_for(risk: float) -> str:
    if risk < 0.30:
        return "Low"
    if risk < 0.60:
        return "Moderate"
    if risk < 0.80:
        return "High"
    return "Critical"


def patient_view() -> bool:
    """True when the sidebar switch is set to plain-language Patient mode."""
    return st.session_state.get("view_mode") == "Patient"


def plain_name(item: dict) -> str:
    feature = item.get("feature", "")
    if feature in PLAIN_FEATURE:
        return PLAIN_FEATURE[feature]
    if feature.startswith("smoking_history_"):
        return "Smoking"
    if feature.startswith("gender_"):
        return "Your sex"
    return item.get("display", feature)


def effect_words(value: float) -> str:
    """Qualitative strength label so patients never see SHAP points."""
    magnitude = abs(value) * 100
    if magnitude >= 20:
        adverb = "strongly"
    elif magnitude >= 8:
        adverb = "clearly"
    elif magnitude >= 3:
        adverb = "slightly"
    else:
        adverb = "a little"
    return f"{adverb} {'raises' if value >= 0 else 'lowers'}"


def patient_tips(values: dict) -> list[str]:
    """Practical, non-prescriptive next steps for the plain-language view."""
    tips: list[str] = []
    if values["HbA1c_level"] >= 6.5:
        tips.append("Your HbA1c is in the diabetes range. Ask your doctor about a "
                    "confirmatory blood test.")
    elif values["HbA1c_level"] >= 5.7:
        tips.append("Your HbA1c is in the prediabetes range. Small changes to food "
                    "and daily activity can bring it down.")
    if values["blood_glucose_level"] >= 200:
        tips.append("Your blood sugar reading is high and should be checked by a "
                    "clinician soon.")
    elif values["blood_glucose_level"] >= 140:
        tips.append("Your blood sugar reading is above the usual fasting range. "
                    "A repeat test can confirm it.")
    if values["bmi"] >= 30:
        tips.append("Losing even a modest amount of weight can lower the chance of "
                    "type 2 diabetes.")
    elif values["bmi"] >= 25:
        tips.append("Staying active and keeping a healthy weight helps lower the "
                    "chance.")
    if values["smoking_history"] in ("current", "ever", "former"):
        tips.append("Stopping smoking, or staying smoke-free, lowers diabetes and "
                    "heart risk.")
    if values["hypertension"] or values["heart_disease"]:
        tips.append("Keep blood pressure and heart health reviews up to date with "
                    "your doctor.")
    if not tips:
        tips.append("Keep up routine check-ups and a healthy lifestyle.")
    return tips


def badge_html(band: str, text: str | None = None) -> str:
    colour = band_ink(band)
    tint = band_tint(band)
    return (f'<span class="dp-badge" style="color:{colour};background:{tint}">'
            f'<span class="dot"></span>{text or band}</span>')


def stat_card(label: str, value: str, sub: str = "", accent: str = PRIMARY) -> str:
    return (
        f'<div class="dp-card dp-stat" style="--accent:{accent}">'
        f'<div class="dp-stat-label">{label}</div>'
        f'<div class="dp-stat-value">{value}</div>'
        f'<div class="dp-stat-sub">{sub}</div></div>'
    )


def page_header(title: str, eyebrow: str, subtitle: str) -> None:
    heading, switch = st.columns([1, 0.34], gap="small")
    with heading:
        md(
            f'<div class="dp-eyebrow">{eyebrow}</div>'
            f'<div class="dp-title">{title}</div>'
            f'<div class="dp-sub">{subtitle}</div>'
            f'<div style="height:14px"></div>',
            unsafe_allow_html=True,
        )
    with switch:
        with st.container(key="theme_switch"):
            st.radio(
                "Theme", THEMES, horizontal=True, key="theme",
                format_func=lambda value: ("☀  Light" if value == "Light"
                                           else "☾  Dark"),
                label_visibility="collapsed",
            )


def section(title: str) -> None:
    md(f'<div class="dp-section">{title}</div>', unsafe_allow_html=True)


def kv_rows(pairs: list[tuple[str, str]]) -> str:
    body = "".join(
        f'<div class="dp-kv"><span>{key}</span><span>{value}</span></div>'
        for key, value in pairs
    )
    return f'<div class="dp-card">{body}</div>'


def data_table(frame, tint_column: str | None = None,
               tints: list[str] | None = None,
               max_height: int = 430) -> None:
    """Render a theme-aware HTML table (the native grid ignores our palette)."""
    head = "".join(f"<th>{xml_escape(str(col))}</th>" for col in frame.columns)
    body = []
    for index, (_, row) in enumerate(frame.iterrows()):
        cells = []
        for col in frame.columns:
            text = xml_escape(str(row[col]))
            if tint_column and col == tint_column:
                band = tints[index] if tints else str(row[col])
                cells.append(
                    f'<td style="background:{band_tint(band)};'
                    f'color:{band_ink(band)};font-weight:700">{text}</td>')
            else:
                cells.append(f"<td>{text}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    md(
        f'<div class="dp-table-wrap" style="max-height:{max_height}px">'
        f'<table class="dp-table"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


# ── circular gauge (pure SVG + CSS so it animates on every rerun) ───────────
def gauge_svg(risk: float, caption: str = "", label: str | None = None) -> str:
    risk = float(np.clip(risk, 0.0, 1.0))
    band = band_for(risk)
    colour = BAND_COLOR[band]

    radius = 84.0
    cx, cy = 110.0, 108.0
    sweep = 240.0
    arc_len = 2 * math.pi * radius * (sweep / 360.0)
    offset = arc_len * (1.0 - risk)

    start = math.radians(150.0)          # bottom-left
    end = math.radians(150.0 + sweep)    # bottom-right
    x1, y1 = cx + radius * math.cos(start), cy + radius * math.sin(start)
    x2, y2 = cx + radius * math.cos(end), cy + radius * math.sin(end)
    path = f"M {x1:.2f} {y1:.2f} A {radius} {radius} 0 1 1 {x2:.2f} {y2:.2f}"

    return f"""
<div class="dp-gauge-wrap">
  <svg viewBox="0 0 220 196" style="width:100%;max-width:330px;display:block;">
    <path d="{path}" fill="none" stroke="#E8EFF8" stroke-width="15" stroke-linecap="round"/>
    <path class="dp-gauge-arc" d="{path}" fill="none" stroke="{colour}"
          stroke-width="15" stroke-linecap="round"
          stroke-dasharray="{arc_len:.2f} {arc_len:.2f}"
          stroke-dashoffset="{offset:.2f}"
          style="--dp-len:{arc_len:.2f};"/>
    <text x="{cx}" y="{cy - 2}" text-anchor="middle"
          style="font-family:Inter,sans-serif;font-size:38px;font-weight:700;fill:{INK}">
      {risk * 100:.0f}<tspan style="font-size:19px;font-weight:600;fill:{MUTED}">%</tspan>
    </text>
    <text x="{cx}" y="{cy + 22}" text-anchor="middle"
          style="font-family:Inter,sans-serif;font-size:12.5px;font-weight:700;
                 letter-spacing:.10em;fill:{colour}">{label or f"{band.upper()} RISK"}</text>
    <text x="{x1 + 4:.0f}" y="{y1 + 26:.0f}" text-anchor="middle"
          style="font-family:Inter,sans-serif;font-size:10.5px;fill:#9AAEC2">0%</text>
    <text x="{x2 - 4:.0f}" y="{y2 + 26:.0f}" text-anchor="middle"
          style="font-family:Inter,sans-serif;font-size:10.5px;fill:#9AAEC2">100%</text>
  </svg>
  <div class="dp-gauge-caption">{caption}</div>
</div>
"""


# ════════════════════════════════════════════════════════════════════════════
# Artefacts
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading DiabPredict model…")
def load_artifacts() -> dict:
    """Load the Stage 1 artefacts exactly once per server process."""
    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = load_preprocessor(PREPROCESS_PATH, SCALER_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    importance = json.loads(IMPORTANCE_PATH.read_text(encoding="utf-8"))
    background = pd.read_csv(BACKGROUND_PATH)
    return {
        "model": model,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "importance": importance,
        "background": background,
        "threshold": float(metrics["threshold"]),
    }


ARTIFACTS: dict = {}


def predict_one(values: dict) -> float:
    frame = pd.DataFrame([values])[dnn_data.RAW_FEATURES]
    matrix = transform_frame(frame, ARTIFACTS["preprocessor"])
    return float(ARTIFACTS["model"].predict(matrix, verbose=0).ravel()[0])


@st.cache_resource(show_spinner=False)
def get_explainer() -> "shap.KernelExplainer":
    background = ARTIFACTS["background"].sample(
        min(60, len(ARTIFACTS["background"])), random_state=7)
    matrix = transform_frame(background, ARTIFACTS["preprocessor"])
    model = ARTIFACTS["model"]

    def predict_fn(data):
        return model.predict(np.asarray(data, dtype=np.float32), verbose=0).ravel()

    return shap.KernelExplainer(predict_fn, matrix)


@st.cache_data(show_spinner=False, max_entries=48)
def explain_row(row_key: tuple[float, ...], nsamples: int = 200) -> dict:
    """KernelSHAP for a single patient. Cached on the raw input tuple."""
    values = dict(zip(dnn_data.RAW_FEATURES, row_key))
    frame = pd.DataFrame([values])[dnn_data.RAW_FEATURES]
    matrix = transform_frame(frame, ARTIFACTS["preprocessor"])

    explainer = get_explainer()
    raw = explainer.shap_values(matrix, nsamples=nsamples, l1_reg=0.0, silent=True)
    raw = np.asarray(raw, dtype=float).reshape(-1)
    base = float(np.ravel(explainer.expected_value)[0])
    prediction = float(ARTIFACTS["model"].predict(matrix, verbose=0).ravel()[0])

    names = ARTIFACTS["preprocessor"]["columns"]
    display = ARTIFACTS["preprocessor"]["display_names"]

    def raw_text(name: str):
        """Value for the report: the measurement, or Yes/No for an indicator."""
        if name in values:
            return values[name]
        if name.startswith("gender_"):
            return "Yes" if values["gender"] == name[len("gender_"):] else "No"
        if name.startswith("smoking_history_"):
            level = name[len("smoking_history_"):]
            return "Yes" if values["smoking_history"] == level else "No"
        return "—"

    items = [
        {
            "feature": name,
            "display": display.get(name, name),
            "value": float(raw[i]),
            "raw_value": raw_text(name),
        }
        for i, name in enumerate(names)
    ]
    items.sort(key=lambda item: abs(item["value"]), reverse=True)
    return {"base": base, "prediction": prediction, "items": items}


def risk_row_key(values: dict) -> tuple:
    return tuple(values[name] for name in dnn_data.RAW_FEATURES)


# ════════════════════════════════════════════════════════════════════════════
# Prediction log (session + best-effort CSV persistence)
# ════════════════════════════════════════════════════════════════════════════
LOG_COLUMNS = ["timestamp", "patient", "age", "gender", "HbA1c", "glucose", "bmi",
               "risk", "band", "source"]


def load_log() -> pd.DataFrame:
    if "log" in st.session_state:
        return st.session_state["log"]
    frame = pd.DataFrame(columns=LOG_COLUMNS)
    try:
        if LOG_PATH.exists():
            frame = pd.read_csv(LOG_PATH)
            for column in LOG_COLUMNS:
                if column not in frame.columns:
                    frame[column] = np.nan
            frame = frame[LOG_COLUMNS]
            frame["risk"] = pd.to_numeric(frame["risk"], errors="coerce")
            frame = frame.dropna(subset=["risk"]).reset_index(drop=True)
    except Exception:
        frame = pd.DataFrame(columns=LOG_COLUMNS)
    st.session_state["log"] = frame
    return frame


def log_prediction(values: dict, risk: float, patient: str, source: str = "Single") -> None:
    frame = load_log()
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "patient": patient or "Anonymous",
        "age": float(values["age"]),
        "gender": values["gender"],
        "HbA1c": float(values["HbA1c_level"]),
        "glucose": float(values["blood_glucose_level"]),
        "bmi": float(values["bmi"]),
        "risk": round(float(risk), 4),
        "band": band_for(risk),
        "source": source,
    }
    updated = pd.concat([frame, pd.DataFrame([record])], ignore_index=True)
    st.session_state["log"] = updated
    try:
        updated.to_csv(LOG_PATH, index=False)
    except Exception:
        pass  # read-only filesystem on some hosts: session memory still works


def clear_log() -> None:
    st.session_state["log"] = pd.DataFrame(columns=LOG_COLUMNS)
    try:
        LOG_PATH.unlink(missing_ok=True)
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
# Input widgets
# ════════════════════════════════════════════════════════════════════════════
INPUT_KEYS = ("age", "bmi", "HbA1c_level", "blood_glucose_level",
              "gender", "smoking_history", "hypertension", "heart_disease")


def clear_inputs(prefix: str) -> None:
    """Drop widget state *before* the widgets are created in this run."""
    for name in INPUT_KEYS:
        st.session_state.pop(f"{prefix}_{name}", None)


def seed_inputs(prefix: str, values: dict) -> None:
    """Give widget keys an initial value without tripping Streamlit warnings."""
    defaults = {
        "age": int(round(values["age"])),
        "bmi": float(values["bmi"]),
        "HbA1c_level": float(values["HbA1c_level"]),
        "blood_glucose_level": int(round(values["blood_glucose_level"])),
        "gender": values["gender"],
        "smoking_history": values["smoking_history"],
        "hypertension": int(values["hypertension"]),
        "heart_disease": int(values["heart_disease"]),
    }
    for name, value in defaults.items():
        st.session_state.setdefault(f"{prefix}_{name}", value)


def render_inputs(prefix: str, values: dict) -> dict:
    """Render the 8 raw inputs; widget keys survive reruns via session state."""
    out: dict = {}
    left, right = st.columns(2, gap="large")

    with left:
        out["age"] = st.slider(
            "Age (years)", min_value=1, max_value=80, step=1,
            key=f"{prefix}_age", help="Patient age at assessment.")
        out["gender"] = st.selectbox(
            "Gender", dnn_data.GENDER_LEVELS, key=f"{prefix}_gender")
        out["smoking_history"] = st.selectbox(
            "Smoking history", dnn_data.SMOKING_LEVELS, key=f"{prefix}_smoking_history",
            help="'No Info' is used when smoking status is not recorded.")

    with right:
        out["HbA1c_level"] = st.slider(
            "HbA1c level (%)", min_value=3.5, max_value=9.0, step=0.1,
            key=f"{prefix}_HbA1c_level",
            help=NUMERIC_INPUTS["HbA1c_level"][5])
        out["blood_glucose_level"] = st.slider(
            "Blood glucose (mg/dL)", min_value=80, max_value=300, step=1,
            key=f"{prefix}_blood_glucose_level",
            help=NUMERIC_INPUTS["blood_glucose_level"][5])
        out["bmi"] = st.slider(
            "BMI (kg/m²)", min_value=10.0, max_value=70.0, step=0.1,
            key=f"{prefix}_bmi", help=NUMERIC_INPUTS["bmi"][5])
        out["hypertension"] = int(st.checkbox(
            "Hypertension", key=f"{prefix}_hypertension"))
        out["heart_disease"] = int(st.checkbox(
            "Heart disease", key=f"{prefix}_heart_disease"))

    out["age"] = float(out["age"])
    out["bmi"] = float(out["bmi"])
    out["HbA1c_level"] = float(out["HbA1c_level"])
    out["blood_glucose_level"] = float(out["blood_glucose_level"])
    return out


# ════════════════════════════════════════════════════════════════════════════
# Charts
# ════════════════════════════════════════════════════════════════════════════
def base_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12,
                  color=CHART["font"]),
        hoverlabel=dict(bgcolor=CHART["surface"], font_size=12,
                        font_family="Inter", font_color=CHART["ink"],
                        bordercolor=CHART["axis"]),
    )
    fig.update_xaxes(gridcolor=CHART["grid"], linecolor=CHART["axis"],
                     tickfont=dict(color=CHART["font"]),
                     title_font=dict(color=CHART["font"]), zeroline=False)
    fig.update_yaxes(gridcolor=CHART["grid"], linecolor=CHART["axis"],
                     tickfont=dict(color=CHART["font"]),
                     title_font=dict(color=CHART["font"]), zeroline=False)
    return fig


def donut(series: pd.Series, centre_label: str) -> go.Figure:
    counts = [int((series == band).sum()) for band in BAND_ORDER]
    fig = go.Figure(go.Pie(
        labels=BAND_ORDER,
        values=counts,
        hole=0.64,
        sort=False,
        direction="clockwise",
        marker=dict(colors=[BAND_COLOR[b] for b in BAND_ORDER],
                    line=dict(color=CHART["surface"], width=2)),
        textinfo="percent",
        textfont=dict(size=12, color="#FFFFFF"),
        hovertemplate="%{label}: %{value} patients<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b style='font-size:22px;color:{CHART['ink']}'>{int(sum(counts))}</b><br>"
             f"<span style='font-size:11px;color:{CHART['soft']}'>{centre_label}</span>",
        x=0.5, y=0.5, showarrow=False,
    )
    base_layout(fig, 300)
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"))
    return fig


def waterfall_figure(items: list[dict], base: float, prediction: float,
                     threshold: float, top_n: int = 8) -> plt.Figure:
    """Cumulative SHAP waterfall drawn by hand for consistent styling."""
    shown = items[:top_n]
    shown = list(reversed(shown))  # largest at the top of the axes

    cumulative = base
    segments = []
    for item in shown:
        start = cumulative
        end = cumulative + item["value"]
        segments.append((start, end, item))
        cumulative = end

    fig, ax = plt.subplots(figsize=(6.6, 0.5 * len(shown) + 1.45), dpi=130)
    fig.patch.set_facecolor(CHART["surface"])
    ax.set_facecolor(CHART["surface"])

    for position, (start, end, item) in enumerate(segments):
        colour = DANGER if item["value"] >= 0 else SUCCESS
        ax.barh(position, end - start, left=start, height=0.58,
                color=colour, alpha=0.92, edgecolor="none", zorder=3)
        label = f"{item['value'] * 100:+.1f} pt"
        ax.text(max(start, end) + 0.004, position, label, va="center", ha="left",
                fontsize=9.2, color=CHART["font"], fontweight="bold", zorder=4)

    ax.axvline(base, color=CHART["soft"], ls="--", lw=1.1, zorder=2)
    ax.axvline(threshold, color=PRIMARY, ls=":", lw=1.4, zorder=2)
    ax.text(base, len(shown) - 0.35, f" base {base:.1%}", fontsize=8.6,
            color=CHART["soft"], va="bottom")
    ax.text(threshold, len(shown) - 0.35, f" threshold {threshold:.1%} ", fontsize=8.6,
            color=PRIMARY, va="bottom", ha="right")

    labels = [f"{item['display']}" for _, _, item in segments]
    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels(labels, fontsize=9.4, color=CHART["ink"])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9, colors=CHART["font"])
    ax.set_xlabel("Cumulative predicted risk", fontsize=9.4, color=CHART["font"])
    ax.set_title(f"Final prediction {prediction:.1%}", fontsize=11.5,
                 color=CHART["ink"], loc="left", fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CHART["axis"])
    ax.grid(axis="x", color=CHART["grid"], lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.65, len(shown) + 0.15)
    ax.margins(x=0.16)
    fig.tight_layout()
    return fig


def importance_bar(importance: dict) -> go.Figure:
    entries = list(reversed(importance["features"]))
    labels = [entry["display"] for entry in entries]
    shares = [entry["share"] * 100 for entry in entries]

    fig = go.Figure(go.Bar(
        x=shares, y=labels, orientation="h",
        marker=dict(color=shares, colorscale=[[0, "#BBDEFB"], [1, PRIMARY]],
                    line=dict(width=0)),
        text=[f"{value:.1f}%" for value in shares],
        textposition="outside",
        textfont=dict(size=11, color=CHART["font"]),
        hovertemplate="%{y}: %{x:.1f}% of model importance<extra></extra>",
    ))
    base_layout(fig, 430)
    fig.update_layout(
        xaxis=dict(title="Share of mean |SHAP|", showgrid=True,
                   gridcolor=CHART["grid"], zeroline=False, range=[0, max(shares) * 1.18]),
        yaxis=dict(title="", showgrid=False),
        bargap=0.35,
    )
    return fig


def confusion_heatmap(matrix: list[list[int]]) -> go.Figure:
    matrix = np.asarray(matrix, dtype=int)
    labels = ["No diabetes", "Diabetes"]
    text = [[f"{value:,}" for value in row] for row in matrix]

    fig = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels,
        colorscale=[[0, "#16233A" if theme_dark() else "#F4F9FF"],
                    [0.55, "#2A4C74" if theme_dark() else "#BBDCF8"],
                    [1, "#3D77B8" if theme_dark() else "#7EB8EC"]],
        text=text, texttemplate="%{text}",
        textfont=dict(size=17, color=CHART["ink"]),
        showscale=False, hoverongaps=False, xgap=3, ygap=3,
        hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z:,} patients<extra></extra>",
    ))
    base_layout(fig, 330)
    fig.update_layout(
        xaxis=dict(title="Predicted", side="bottom", showgrid=False),
        yaxis=dict(title="Actual", autorange="reversed", showgrid=False),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


def history_figure(history: dict) -> go.Figure:
    from plotly.subplots import make_subplots

    epochs = list(range(1, len(history.get("loss", [])) + 1))
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.16,
        subplot_titles=("Loss (binary cross-entropy)", "AUC"),
    )
    traces = [
        ("loss", "Train loss", "#9CCBF5", "dot", 1),
        ("val_loss", "Validation loss", PRIMARY, "solid", 1),
        ("auc", "Train AUC", "#A5D6A7", "dot", 2),
        ("val_auc", "Validation AUC", SUCCESS, "solid", 2),
    ]
    for key, name, colour, dash, row in traces:
        if key not in history:
            continue
        fig.add_trace(go.Scatter(
            x=epochs, y=history[key], name=name, mode="lines",
            line=dict(color=colour, width=2.2, dash=dash),
            hovertemplate=f"{name}: %{{y:.4f}}<extra></extra>",
        ), row=row, col=1)

    base_layout(fig, 420)
    fig.update_layout(
        legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center",
                    font=dict(size=11)),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=28, b=10),
    )
    fig.update_xaxes(title_text="Epoch", showgrid=True, gridcolor=CHART["grid"],
                     row=2, col=1, title_font=dict(size=11))
    fig.update_yaxes(showgrid=True, gridcolor=CHART["grid"])
    for annotation in fig.layout.annotations:
        annotation.font.size = 11.5
        annotation.font.color = CHART["ink"]
    return fig


# ════════════════════════════════════════════════════════════════════════════
# PDF report
# ════════════════════════════════════════════════════════════════════════════
def build_pdf(patient: str, values: dict, risk: float,
              items: list[dict], meta: dict, plain: bool = False) -> bytes:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    band = band_for(risk)
    colour = rl_colors.HexColor(BAND_COLOR[band])
    tint = rl_colors.HexColor(BAND_TINT[band])
    ink = rl_colors.HexColor(INK)
    muted = rl_colors.HexColor(MUTED)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"{APP_NAME} report", author=APP_NAME,
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=19, leading=23,
                        textColor=ink, alignment=0, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9, leading=12,
                         textColor=muted, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Normal"], fontSize=10.5, leading=13,
                        textColor=rl_colors.HexColor(PRIMARY), spaceBefore=10,
                        spaceAfter=4, fontName="Helvetica-Bold")
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=12.5,
                          textColor=ink)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=7.6, leading=10,
                           textColor=muted)
    score = ParagraphStyle("score", parent=styles["Normal"], fontSize=30, leading=34,
                           textColor=colour, fontName="Helvetica-Bold", alignment=0)
    band_style = ParagraphStyle("band", parent=styles["Normal"], fontSize=10,
                                leading=13, textColor=colour,
                                fontName="Helvetica-Bold")

    flow = [
        Paragraph(f"{APP_NAME} - Diabetes Risk "
                  f"{'Summary' if plain else 'Report'}", h1),
        Paragraph(f"Generated {datetime.now():%d %b %Y, %H:%M} &nbsp;|&nbsp; "
                  f"Report ID {uuid.uuid4().hex[:8].upper()}", sub),
        HRFlowable(width="100%", thickness=1, color=rl_colors.HexColor(LINE)),
        Spacer(1, 10),
    ]

    patient_cell = [
        Paragraph(f"<b>Patient</b><br/>{xml_escape(patient or 'Anonymous')}", body),
        Paragraph(f"<b>Age / gender</b><br/>{values['age']:.0f} years, "
                  f"{xml_escape(values['gender'])}", body),
        Paragraph(f"<b>Smoking</b><br/>{xml_escape(values['smoking_history'])}", body),
    ]
    if plain:
        score_cell = [Paragraph(f"{risk * 100:.1f}%", score),
                      Paragraph(BAND_PLAIN[band].upper(), band_style),
                      Paragraph(BAND_PATIENT_NOTE[band], small)]
    else:
        score_cell = [Paragraph(f"{risk * 100:.1f}%", score),
                      Paragraph(f"{band.upper()} RISK", band_style),
                      Paragraph(f"Decision threshold {meta['threshold']:.3f}", small)]
    layout = Table([[patient_cell, score_cell]], colWidths=[105 * mm, 62 * mm])
    layout.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (1, 0), (1, 0), tint),
        ("BOX", (1, 0), (1, 0), 0.8, colour),
        ("LEFTPADDING", (1, 0), (1, 0), 10),
        ("RIGHTPADDING", (1, 0), (1, 0), 10),
        ("TOPPADDING", (1, 0), (1, 0), 10),
        ("BOTTOMPADDING", (1, 0), (1, 0), 10),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
    ]))
    flow += [layout, Spacer(1, 12),
             Paragraph("Your measurements" if plain else "Clinical inputs", h2)]

    input_rows = [["Measurement", "Value", "Measurement", "Value"]]
    pairs = [
        ("HbA1c level", f"{values['HbA1c_level']:.1f} %"),
        ("Blood glucose", f"{values['blood_glucose_level']:.0f} mg/dL"),
        ("BMI", f"{values['bmi']:.1f} kg/m2"),
        ("Age", f"{values['age']:.0f} years"),
        ("Hypertension", "Yes" if values["hypertension"] else "No"),
        ("Heart disease", "Yes" if values["heart_disease"] else "No"),
    ]
    for index in range(0, len(pairs), 2):
        first = pairs[index]
        second = pairs[index + 1] if index + 1 < len(pairs) else ("", "")
        input_rows.append([first[0], first[1], second[0], second[1]])

    inputs_table = Table(input_rows, colWidths=[40 * mm, 42 * mm, 40 * mm, 42 * mm])
    inputs_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#F2F8FF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.HexColor("#1467B4")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("TEXTCOLOR", (0, 1), (-1, -1), ink),
        ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.HexColor(LINE)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    flow += [inputs_table,
             Paragraph("What affects your chance" if plain
                       else "What drove this score", h2)]

    if plain:
        factor_rows = [["What", "Your value", "Effect"]]
        for item in items[:6]:
            raw = item.get("raw_value", "")
            if isinstance(raw, float):
                raw = f"{raw:g}"
            factor_rows.append([
                Paragraph(xml_escape(plain_name(item)), body),
                Paragraph(xml_escape(str(raw)), body),
                effect_words(item["value"]).capitalize(),
            ])
        factors_table = Table(factor_rows,
                              colWidths=[67 * mm, 40 * mm, 60 * mm])
    else:
        factor_rows = [["Factor", "Patient value", "Effect", "Direction"]]
        for item in items[:6]:
            effect = item["value"] * 100
            raw = item.get("raw_value", "")
            if isinstance(raw, float):
                raw = f"{raw:g}"
            factor_rows.append([
                Paragraph(xml_escape(item["display"]), body),
                Paragraph(xml_escape(str(raw)), body),
                f"{effect:+.1f} pt",
                "Increases risk" if item["value"] >= 0 else "Reduces risk",
            ])
        factors_table = Table(factor_rows,
                              colWidths=[62 * mm, 38 * mm, 26 * mm, 41 * mm])
    factors_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#F2F8FF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.HexColor("#1467B4")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("TEXTCOLOR", (0, 1), (-1, -1), ink),
        ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.HexColor(LINE)),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    if plain:
        footnote = ("Each row shows whether that measurement raises or lowers the "
                    f"overall chance. The starting chance for an average person is "
                    f"{meta['base']:.0%}.")
        model_heading = "About this screening tool"
        model_rows = [
            ["What it does", "Estimates the chance of diabetes from 8 routine "
                             "measurements"],
            ["Ranking quality", f"{meta['auc']:.3f} out of 1.000 (higher is better)"],
            ["Detection rate", f"Finds about {meta['recall'] * 100:.0f} in 100 "
                               "people who have diabetes"],
            ["False alarms", f"About {100 - meta['precision'] * 100:.0f} of every "
                             "100 positive flags are false alarms"],
            ["Important", "This is not a diagnosis. Only a blood test can confirm "
                          "diabetes."],
        ]
    else:
        footnote = ("Effect is the SHAP contribution to the predicted probability, "
                    "in percentage points. Contributions add up from the background "
                    f"risk of {meta['base']:.1%} to the final score.")
        model_heading = "Model information"
        model_rows = [
            ["Model", "Dense neural network (64 -> 32 -> 1), 13 inputs"],
            ["Test AUC", f"{meta['auc']:.4f}"],
            ["Recall / precision @ threshold",
             f"{meta['recall']:.1%} / {meta['precision']:.1%}"],
            ["Decision threshold", f"{meta['threshold']:.3f} (Youden's J)"],
            ["Risk bands", "Low <30% | Moderate 30-60% | High 60-80% | Critical >80%"],
        ]

    flow += [
        factors_table,
        Spacer(1, 6),
        Paragraph(footnote, small),
        Spacer(1, 12),
        Paragraph(model_heading, h2),
    ]
    model_table = Table(model_rows, colWidths=[52 * mm, 115 * mm])
    model_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("TEXTCOLOR", (0, 0), (0, -1), muted),
        ("TEXTCOLOR", (1, 0), (1, -1), ink),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, rl_colors.HexColor(LINE)),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    if plain:
        disclaimer = ("This summary comes from a statistical screening tool. It is "
                      "not a diagnosis. Discuss the result with your doctor, who may "
                      "arrange a blood test to confirm it.")
    else:
        disclaimer = ("This report is a screening aid produced by a statistical "
                      "model. It is not a diagnosis and must be interpreted alongside "
                      "clinical history and confirmatory laboratory testing. "
                      "Probabilities are not calibrated to population prevalence.")
    flow += [
        model_table, Spacer(1, 14),
        HRFlowable(width="100%", thickness=0.6, color=rl_colors.HexColor(LINE)),
        Spacer(1, 6),
        Paragraph(disclaimer, small),
    ]

    doc.build(flow)
    return buffer.getvalue()


# ════════════════════════════════════════════════════════════════════════════
# Page: Dashboard
# ════════════════════════════════════════════════════════════════════════════
def page_dashboard() -> None:
    page_header("Clinical dashboard", "Overview",
                "Aggregate view of the assessments run in this workspace.")

    log = load_log()
    total = len(log)
    if total:
        risks = pd.to_numeric(log["risk"], errors="coerce").dropna()
        high = int((risks >= 0.60).sum())
        high_pct = high / total if total else 0.0
        avg_hba1c = pd.to_numeric(log["HbA1c"], errors="coerce").mean()
        avg_risk = risks.mean()
    else:
        high, high_pct, avg_hba1c, avg_risk = 0, 0.0, float("nan"), float("nan")

    columns = st.columns(4, gap="medium")
    cards = [
        ("Total predictions", f"{total:,}", "single-patient assessments", PRIMARY),
        ("High risk (≥60%)", f"{high:,}",
         f"{high_pct:.0%} of assessments" if total else "no data yet",
         DANGER if high else SUCCESS),
        ("Average HbA1c", "—" if not total else f"{avg_hba1c:.2f}%",
         "across assessed patients", PRIMARY),
        ("Average risk", "—" if not total else f"{avg_risk:.1%}",
         f"decision threshold {ARTIFACTS['threshold']:.2f}", WARNING),
    ]
    for column, (label, value, sub, accent) in zip(columns, cards):
        with column:
            md(stat_card(label, value, sub, accent), unsafe_allow_html=True)

    md('<div style="height:18px"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        section("Recent predictions")
        if not total:
            md(
                '<div class="dp-card" style="text-align:center;padding:38px 20px;">'
                '<div style="font-size:2rem">🗂️</div>'
                '<div style="font-weight:700;color:#0F2A44;margin-top:6px">'
                'No predictions yet</div>'
                '<div style="color:#6B8098;font-size:.86rem;margin-top:4px">'
                'Run an assessment from <b>Single Prediction</b> and it will appear here.'
                '</div></div>', unsafe_allow_html=True)
        else:
            recent = log.tail(10).iloc[::-1].copy()
            recent["Risk"] = recent["risk"].map(lambda value: f"{value:.1%}")
            recent["Band"] = recent["risk"].map(band_for)
            display = recent[["timestamp", "patient", "HbA1c", "glucose",
                              "Risk"]].copy()
            display["timestamp"] = pd.to_datetime(display["timestamp"],
                                                  errors="coerce")
            display["When"] = display["timestamp"].dt.strftime("%d %b %H:%M")
            display["Patient"] = display["patient"].fillna("Anonymous")
            display["HbA1c %"] = display["HbA1c"].map(lambda v: f"{v:.1f}")
            display["Glucose"] = display["glucose"].map(lambda v: f"{v:.0f}")
            display = display[["When", "Patient", "HbA1c %", "Glucose", "Risk"]]
            data_table(display, tint_column="Risk",
                       tints=list(recent["Band"]),
                       max_height=min(410, 42 + 38 * len(display)))

            with st.expander("History actions"):
                st.caption(f"{total:,} assessments stored in this session "
                           f"(and `predictions_log.csv` when writable).")
                if st.button("Clear prediction history"):
                    clear_log()
                    st.rerun()

    with right:
        section("Risk distribution")
        if not total:
            md(
                '<div class="dp-card" style="padding:60px 20px;text-align:center;'
                'color:#93A5BA;font-size:.86rem">The donut chart appears once '
                'predictions exist.</div>', unsafe_allow_html=True)
        else:
            counts = pd.Series([band_for(value) for value in log["risk"]])
            st.plotly_chart(donut(counts, "patients"), use_container_width=True,
                            config={"displayModeBar": False})

    md('<div style="height:6px"></div>', unsafe_allow_html=True)
    section("Risk bands used in this app")
    legend = st.columns(4, gap="medium")
    for column, band in zip(legend, BAND_ORDER):
        with column:
            md(
                f'<div class="dp-card" style="padding:12px 14px">'
                f'{badge_html(band)}'
                f'<div style="font-size:.78rem;color:#6B8098;margin-top:7px;line-height:1.4">'
                f'{BAND_PATIENT_NOTE[band] if patient_view() else BAND_HELP[band]}'
                f'</div></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Page: Single Prediction
# ════════════════════════════════════════════════════════════════════════════
def render_result(result: dict) -> None:
    values = result["values"]
    risk = float(result["risk"])
    band = band_for(risk)
    items = result["items"]
    plain = patient_view()

    left, right = st.columns([1, 1.12], gap="large")

    with left:
        caption = ("Chance of diabetes" if plain
                   else "Predicted probability of diabetes")
        footer = ""
        if not plain:
            footer = (
                f'<div class="dp-gauge-caption" style="margin-top:-2px">'
                f'Decision threshold {ARTIFACTS["threshold"]:.2f} · '
                f'test AUC {ARTIFACTS["metrics"]["test"]["auc"]:.3f}</div>'
            )
        md(
            f'<div class="dp-card" style="padding:18px 18px 10px">'
            f'{gauge_svg(risk, caption)}'
            f'<div style="text-align:center;margin:2px 0 12px">'
            f'{badge_html(band, BAND_PLAIN[band] if plain else None)}</div>'
            f'{footer}</div>', unsafe_allow_html=True)

        if plain:
            note = (f'<b>{BAND_PLAIN[band]}.</b> {BAND_PATIENT_NOTE[band]} '
                    f'This is a screening estimate, not a diagnosis.')
        else:
            note = (f'<b>{band} risk.</b> The score is {BAND_HELP[band]}. '
                    f'This is a screening estimate, not a diagnosis.')
        md(
            f'<div class="dp-note" style="margin-top:12px">{note}</div>',
            unsafe_allow_html=True)

        try:
            pdf_key = (result.get("patient", ""), round(risk, 6), plain,
                       tuple(sorted((str(k), str(v)) for k, v in values.items())))
            if st.session_state.get("pdf_key") != pdf_key:
                with st.spinner("Preparing PDF report…"):
                    st.session_state["pdf_bytes"] = build_pdf(
                        result.get("patient", ""), values, risk, items,
                        {
                            "threshold": ARTIFACTS["threshold"],
                            "auc": ARTIFACTS["metrics"]["test"]["auc"],
                            "recall":
                                ARTIFACTS["metrics"]["test"]["at_tuned"]["recall"],
                            "precision":
                                ARTIFACTS["metrics"]["test"]["at_tuned"]["precision"],
                            "base": result["base"],
                        },
                        plain=plain,
                    )
                    st.session_state["pdf_key"] = pdf_key
            pdf_bytes = st.session_state["pdf_bytes"]
            st.download_button(
                "⬇  Download patient summary (PDF)" if plain
                else "⬇  Download PDF report",
                data=pdf_bytes,
                file_name=f"diabpredict_{datetime.now():%Y%m%d_%H%M}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:  # noqa: BLE001 - export must never block the score
            st.caption(f"PDF export unavailable: {exc}")

    with right:
        if plain:
            section("What this means for you")
            md(
                f'<div class="dp-note">The tool estimates about a '
                f'<b>{risk * 100:.0f} in 100</b> chance that this patient has '
                f'diabetes. {BAND_PATIENT_NOTE[band]}</div>',
                unsafe_allow_html=True)
            md('<div style="height:12px"></div>', unsafe_allow_html=True)

            rising = [item for item in items if item["value"] > 0][:3]
            falling = [item for item in items if item["value"] < 0][:2]

            if rising:
                section("Things that raise the chance")
                md("\n".join(
                    f"- **{plain_name(item)}** — {effect_words(item['value'])} "
                    f"the chance"
                    for item in rising))
            if falling:
                section("Things that lower the chance")
                md("\n".join(
                    f"- **{plain_name(item)}** — {effect_words(item['value'])} "
                    f"the chance"
                    for item in falling))

            md('<div style="height:6px"></div>', unsafe_allow_html=True)
            section("What could help")
            md("\n".join(f"- {tip}" for tip in patient_tips(values)))
            st.caption("Only a blood test can confirm diabetes. This tool is a "
                       "screening aid, not a diagnosis.")
        else:
            section("Why this score")
            st.caption("SHAP contributions for this patient, largest first. "
                       "Red pushes risk up, green pulls it down.")
            figure = waterfall_figure(items, result["base"], risk,
                                      ARTIFACTS["threshold"])
            st.pyplot(figure, use_container_width=True)
            plt.close(figure)

            with st.expander("Plain-language summary", expanded=True):
                rising = [item for item in items if item["value"] > 0][:3]
                falling = [item for item in items if item["value"] < 0][:2]
                if rising:
                    md("**Pushing risk up**")
                    for item in rising:
                        md(f"- {item['display']}: "
                                    f"**+{item['value'] * 100:.1f} pts**")
                if falling:
                    md("**Pulling risk down**")
                    for item in falling:
                        md(f"- {item['display']}: "
                                    f"**{item['value'] * 100:.1f} pts**")
                md(
                    f"The model started from a background risk of "
                    f"**{result['base']:.1%}** and arrived at **{risk:.1%}**.")


def page_single() -> None:
    plain = patient_view()
    page_header(
        "Check the chance of diabetes" if plain else "Single patient prediction",
        "Screening" if plain else "Assessment",
        "Enter eight simple measurements to get an estimated chance and a plain "
        "explanation." if plain else
        "Enter the eight routine measurements to generate a risk score "
        "and a patient-level explanation.")

    default_values = st.session_state.get("patient", DEFAULTS)
    seed_inputs("sp", default_values)

    with st.form("single_form", border=False):
        section("Patient details")
        patient = st.text_input(
            "Patient name or ID", value=st.session_state.get("patient_name", ""),
            placeholder="e.g. PT-1042",
            help="Used to label the report and dashboard row. Not stored remotely.")
        section("Measurements")
        values = render_inputs("sp", default_values)
        submitted = st.form_submit_button("Assess risk", type="primary")

    if submitted:
        risk = predict_one(values)
        with st.spinner("Computing patient-level explanation…"):
            try:
                explained = explain_row(risk_row_key(values))
                items, base = explained["items"], explained["base"]
            except Exception:
                items = [{"feature": key, "display": key, "value": 0.0,
                          "raw_value": values.get(key)} for key in values]
                base = 0.0
        st.session_state["patient"] = values
        st.session_state["patient_name"] = patient
        st.session_state["result"] = {
            "values": values, "risk": risk, "items": items, "base": base,
            "patient": patient, "timestamp": datetime.now(),
        }
        log_prediction(values, risk, patient, "Single")
        clear_inputs("wi")
        st.session_state["wi_seeded"] = False

    result = st.session_state.get("result")
    if result is None:
        md(
            '<div class="dp-card" style="text-align:center;padding:34px 20px;margin-top:14px">'
            '<div style="font-size:2rem">🩺</div>'
            '<div style="font-weight:700;color:#0F2A44;margin-top:6px">'
            'Ready when you are</div>'
            '<div style="color:#6B8098;font-size:.86rem;margin-top:4px">'
            'Fill in the measurements above and press <b>Assess risk</b>.'
            '</div></div>', unsafe_allow_html=True)
        return

    md('<div style="height:10px"></div>', unsafe_allow_html=True)
    md(
        f'<div class="dp-sub" style="margin-bottom:10px">Result for '
        f'<b>{xml_escape(result.get("patient") or "Anonymous")}</b> · '
        f'{result["timestamp"]:%d %b %Y, %H:%M}</div>', unsafe_allow_html=True)
    render_result(result)


# ════════════════════════════════════════════════════════════════════════════
# Page: What-If Simulator
# ════════════════════════════════════════════════════════════════════════════
def page_whatif() -> None:
    plain = patient_view()
    page_header(
        "What if I change something?" if plain else "What-if simulator",
        "Your choices" if plain else "Scenario analysis",
        "Move the sliders to see how the estimated chance changes." if plain
        else "Adjust modifiable measurements and watch the risk update instantly.")

    base_values = st.session_state.get("patient")
    result = st.session_state.get("result")

    if base_values is None or result is None:
        md(
            '<div class="dp-card" style="text-align:center;padding:38px 20px">'
            '<div style="font-size:2rem">🎛️</div>'
            '<div style="font-weight:700;color:#0F2A44;margin-top:6px">'
            'No patient loaded</div>'
            '<div style="color:#6B8098;font-size:.86rem;margin-top:4px">'
            'Run an assessment in <b>Single Prediction</b> first, or load a '
            'default profile below.</div></div>', unsafe_allow_html=True)
        md('<div style="height:10px"></div>', unsafe_allow_html=True)
        if st.button("Load a default patient"):
            st.session_state["patient"] = dict(DEFAULTS)
            st.session_state["patient_name"] = "Demo patient"
            risk = predict_one(DEFAULTS)
            explained = explain_row(risk_row_key(DEFAULTS))
            st.session_state["result"] = {
                "values": dict(DEFAULTS), "risk": risk,
                "items": explained["items"], "base": explained["base"],
                "patient": "Demo patient", "timestamp": datetime.now(),
            }
            clear_inputs("wi")
            st.session_state["wi_seeded"] = False
            st.rerun()
        return

    original_risk = float(result["risk"])
    original = dict(base_values)

    if st.session_state.pop("wi_reset", False):
        clear_inputs("wi")
        st.session_state["wi_seeded"] = False

    if not st.session_state.get("wi_seeded"):
        seed_inputs("wi", original)
        st.session_state["wi_seeded"] = True

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        md('<div class="dp-wi-mark"></div>', unsafe_allow_html=True)
        section("Change the numbers" if plain else "Adjust measurements")
        current = render_inputs("wi", original)

    new_risk = predict_one(current)
    delta = new_risk - original_risk

    with right:
        md(
            f'<div class="dp-card" style="padding:18px 18px 14px">'
            f'<div style="font-size:.72rem;font-weight:700;letter-spacing:.08em;'
            f'text-transform:uppercase;color:#8296AC;text-align:center">'
            f'{"Estimated chance" if plain else "Simulated risk"}</div>'
            f'{gauge_svg(new_risk, "")}'
            f'<div style="text-align:center;margin-top:2px">'
            f'{badge_html(band_for(new_risk), BAND_PLAIN[band_for(new_risk)] if plain else None)}'
            f'</div>'
            f'</div>', unsafe_allow_html=True)

        if abs(delta) < 0.0005:
            arrow, colour, text = "→", MUTED, (
                "No change from the original estimate" if patient_view()
                else "Unchanged from the original assessment")
        elif delta > 0:
            arrow, colour = "▲", DANGER
            text = (f"The chance goes up {delta * 100:.1f} in 100 "
                    f"({new_risk:.0%} vs {original_risk:.0%})"
                    if patient_view() else
                    f"Risk up {delta * 100:.1f} points "
                    f"({new_risk:.1%} vs {original_risk:.1%})")
        else:
            arrow, colour = "▼", SUCCESS
            text = (f"The chance goes down {abs(delta) * 100:.1f} in 100 "
                    f"({new_risk:.0%} vs {original_risk:.0%})"
                    if patient_view() else
                    f"Risk down {abs(delta) * 100:.1f} points "
                    f"({new_risk:.1%} vs {original_risk:.1%})")

        md(
            f'<div class="dp-delta" style="color:{colour}">'
            f'<span style="font-size:1rem">{arrow}</span><span>{text}</span></div>',
            unsafe_allow_html=True)

        changed = []
        for key in dnn_data.RAW_FEATURES:
            if original[key] != current[key]:
                changed.append((key, original[key], current[key]))
        if changed:
            md('<div style="height:8px"></div>', unsafe_allow_html=True)
            rows = []
            for key, before, after in changed:
                label = NUMERIC_INPUTS.get(key, (key,))[0]
                unit = NUMERIC_INPUTS[key][4] if key in NUMERIC_INPUTS else ""
                rows.append((label, f"{before:g} {unit} → {after:g} {unit}".strip()))
            md(kv_rows(rows), unsafe_allow_html=True)

        if st.button("Reset to original values"):
            st.session_state["wi_reset"] = True
            st.rerun()

    md('<div style="height:14px"></div>', unsafe_allow_html=True)
    if patient_view():
        md(
            '<div class="dp-warn">Only one measurement is changed at a time, so this '
            'is a guide rather than a promise. Real bodies are more complicated than '
            'the model, and other things you change may matter more.</div>',
            unsafe_allow_html=True)
    else:
        md(
            f'<div class="dp-warn">Changes are applied one measurement at a time while '
            f'the other inputs stay fixed. Real risk factors interact, so treat this as '
            f'directional guidance rather than a guaranteed outcome.</div>',
            unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Page: Batch Prediction
# ════════════════════════════════════════════════════════════════════════════
def template_csv() -> bytes:
    rows = [
        {"gender": "Female", "age": 45, "hypertension": 0, "heart_disease": 0,
         "smoking_history": "never", "bmi": 27.3, "HbA1c_level": 5.8,
         "blood_glucose_level": 140},
        {"gender": "Male", "age": 62, "hypertension": 1, "heart_disease": 0,
         "smoking_history": "former", "bmi": 32.1, "HbA1c_level": 7.4,
         "blood_glucose_level": 190},
        {"gender": "Female", "age": 31, "hypertension": 0, "heart_disease": 0,
         "smoking_history": "No Info", "bmi": 22.8, "HbA1c_level": 5.1,
         "blood_glucose_level": 92},
    ]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def page_batch() -> None:
    page_header("Batch prediction", "Cohort scoring",
                "Upload a CSV of patients, score them all, and download the results.")

    top = st.columns([1, 1], gap="medium")
    with top[0]:
        md(
            '<div class="dp-card" style="padding:14px 16px">'
            '<div style="font-weight:700;color:#0F2A44;font-size:.9rem">'
            'CSV format</div>'
            '<div style="color:#6B8098;font-size:.82rem;margin-top:4px;line-height:1.5">'
            'One row per patient with columns: gender, age, hypertension, '
            'heart_disease, smoking_history, bmi, HbA1c_level, '
            'blood_glucose_level. A <code>diabetes</code> column is optional and '
            'ignored.</div></div>', unsafe_allow_html=True)
    with top[1]:
        md(
            '<div class="dp-card" style="padding:14px 16px">'
            '<div style="font-weight:700;color:#0F2A44;font-size:.9rem">'
            'Before you upload</div>'
            '<div style="color:#6B8098;font-size:.82rem;margin-top:4px;line-height:1.5">'
            'Rows with missing or invalid values are imputed from training medians, '
            'so every row receives a score. Review the flagged columns below the '
            'results.</div></div>', unsafe_allow_html=True)

    md('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.download_button("⬇  Download CSV template", data=template_csv(),
                       file_name="diabpredict_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload patient CSV", type=["csv"],
                                label_visibility="collapsed")
    if uploaded is None:
        return

    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read that CSV: {exc}")
        return

    missing = [column for column in dnn_data.RAW_FEATURES if column not in frame.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(f"`{m}`" for m in missing))
        return

    if len(frame) == 0:
        st.warning("The uploaded file has no rows.")
        return

    with st.spinner(f"Scoring {len(frame):,} rows…"):
        matrix = transform_frame(frame, ARTIFACTS["preprocessor"])
        risks = ARTIFACTS["model"].predict(matrix, batch_size=512, verbose=0).ravel()

    results = frame.copy()
    results["risk"] = risks
    results["risk_pct"] = (risks * 100).round(2)
    results["band"] = [band_for(value) for value in risks]

    md('<div style="height:6px"></div>', unsafe_allow_html=True)
    bands = pd.Series(results["band"])
    columns = st.columns(4, gap="medium")
    summary = [
        ("Rows scored", f"{len(results):,}", "in this upload", PRIMARY),
        ("High or critical", f"{int((risks >= 0.60).sum()):,}",
         f"{(risks >= 0.60).mean():.0%} of the cohort", DANGER),
        ("Mean risk", f"{risks.mean():.1%}", "average probability", WARNING),
        ("Above threshold", f"{int((risks >= ARTIFACTS['threshold']).sum()):,}",
         f"threshold {ARTIFACTS['threshold']:.2f}", PRIMARY),
    ]
    for column, (label, value, sub, accent) in zip(columns, summary):
        with column:
            md(stat_card(label, value, sub, accent), unsafe_allow_html=True)

    md('<div style="height:16px"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.5, 1], gap="large")

    with left:
        section("Scored cohort")
        show = results.head(500).copy()
        show["risk_pct"] = show["risk_pct"].map(lambda v: f"{v:.1f}%")
        show["age"] = show["age"].map(lambda v: f"{v:g}")
        show["bmi"] = show["bmi"].map(lambda v: f"{v:.1f}")
        show["HbA1c_level"] = show["HbA1c_level"].map(lambda v: f"{v:.1f}")
        show["blood_glucose_level"] = show["blood_glucose_level"].map(
            lambda v: f"{v:.0f}")
        preview = show.drop(columns=["risk"]).rename(columns={
            "gender": "Gender", "age": "Age", "hypertension": "HTN",
            "heart_disease": "Heart", "smoking_history": "Smoking",
            "bmi": "BMI", "HbA1c_level": "HbA1c %",
            "blood_glucose_level": "Glucose", "risk_pct": "Risk",
            "band": "Band",
        })
        data_table(preview, tint_column="Band", max_height=420)
        if len(results) > len(show):
            st.caption(f"Showing the first {len(show):,} of {len(results):,} rows. "
                       "The download contains every row.")

    with right:
        section("Risk distribution")
        counts = bands.value_counts().reindex(BAND_ORDER, fill_value=0)
        figure = go.Figure(go.Bar(
            x=BAND_ORDER, y=counts.values,
            marker_color=[BAND_COLOR[b] for b in BAND_ORDER],
            text=counts.values, textposition="outside",
            hovertemplate="%{x}: %{y} patients<extra></extra>",
        ))
        base_layout(figure, 300)
        figure.update_layout(
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(title="Patients", showgrid=True, gridcolor=CHART["grid"]),
            bargap=0.45, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(figure, use_container_width=True,
                        config={"displayModeBar": False})

    export = results.drop(columns=["risk_pct"])
    st.download_button(
        "⬇  Download scored results (CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name=f"diabpredict_batch_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Data quality notes"):
        numeric = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
        issues = []
        for column in numeric:
            values = pd.to_numeric(frame[column], errors="coerce")
            gaps = int(values.isna().sum())
            if gaps:
                issues.append(f"`{column}`: {gaps:,} missing or non-numeric "
                              f"(imputed with {ARTIFACTS['preprocessor']['median'][column]:g})")
        for column, levels in (("gender", dnn_data.GENDER_LEVELS),
                               ("smoking_history", dnn_data.SMOKING_LEVELS)):
            unknown = int((~frame[column].astype(str).isin(levels)).sum())
            if unknown:
                issues.append(f"`{column}`: {unknown:,} unrecognised values "
                              "(mapped to the reference category)")
        md("\n".join(f"- {item}" for item in issues) if issues
                    else "No data-quality issues detected.")


# ════════════════════════════════════════════════════════════════════════════
# Page: Model Insights
# ════════════════════════════════════════════════════════════════════════════
def render_patient_insights() -> None:
    metrics = ARTIFACTS["metrics"]
    test = metrics["test"]["at_tuned"]
    importance = ARTIFACTS["importance"]

    page_header("How the score is worked out", "Plain language",
                "What this screening tool looks at, and how far to trust it.")

    columns = st.columns(3, gap="medium")
    cards = [
        ("Screening results correct", f"{test['accuracy']:.0%}",
         "overall accuracy", PRIMARY),
        ("Diabetes found", f"{test['recall'] * 100:.0f} in 100",
         "of people who actually have it", SUCCESS),
        ("Flags needing a test", f"{100 - test['precision'] * 100:.0f} in 100",
         "positive flags are false alarms", WARNING),
    ]
    for column, (label, value, sub, accent) in zip(columns, cards):
        with column:
            md(stat_card(label, value, sub, accent), unsafe_allow_html=True)

    st.caption(f"Measured on {metrics['split']['test']:,} patients the tool had "
               f"never seen before.")

    md('<div style="height:16px"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.4, 1], gap="large")

    with left:
        section("What the tool pays most attention to")
        st.caption("Longer bars mean the measurement matters more to the estimate.")
        st.plotly_chart(importance_bar(importance), use_container_width=True,
                        config={"displayModeBar": False})

    with right:
        section("In everyday terms")
        top = sorted(importance["features"], key=lambda entry: entry["share"],
                     reverse=True)[:4]
        rows = "".join(
            f'<div class="dp-kv"><span>{plain_name(entry)}</span>'
            f'<span>{entry["share"] * 100:.0f}%</span></div>'
            for entry in top
        )
        md(f'<div class="dp-card">{rows}</div>', unsafe_allow_html=True)
        md('<div style="height:10px"></div>', unsafe_allow_html=True)
        md(
            '<div class="dp-note">Average blood sugar (HbA1c) and the current '
            'blood sugar reading matter most. Age, weight and smoking history add '
            'smaller amounts. Changes to the first two show up fastest in the '
            'What-If Simulator.</div>', unsafe_allow_html=True)

    md('<div style="height:14px"></div>', unsafe_allow_html=True)
    section("What this tool cannot do")
    md(
        "- It cannot diagnose diabetes — only a blood test can do that.\n"
        "- It was trained on public data, so it may be less accurate in a real "
        "clinic.\n"
        "- Its percentages are relative, not exact odds.\n"
        "- It does not replace advice from your doctor or nurse.")


def page_insights() -> None:
    if patient_view():
        render_patient_insights()
        return

    page_header("Model insights", "Transparency",
                "Global explainability and validated performance for the deployed "
                "neural network.")

    metrics = ARTIFACTS["metrics"]
    test = metrics["test"]["at_tuned"]
    importance = ARTIFACTS["importance"]

    columns = st.columns(5, gap="small")
    cards = [
        ("Test AUC", f"{metrics['test']['auc']:.3f}", PRIMARY),
        ("Accuracy", f"{test['accuracy']:.1%}", PRIMARY),
        ("Recall", f"{test['recall']:.1%}", SUCCESS),
        ("Precision", f"{test['precision']:.1%}", WARNING),
        ("F1 score", f"{test['f1']:.3f}", PRIMARY),
    ]
    for column, (label, value, accent) in zip(columns, cards):
        with column:
            md(stat_card(label, value, "", accent), unsafe_allow_html=True)

    st.caption(f"Evaluated on the held-out test split "
               f"({metrics['split']['test']:,} patients) at the tuned threshold "
               f"of {metrics['threshold']:.3f}.")

    md('<div style="height:16px"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        section("Global feature importance")
        st.caption(f"KernelSHAP mean |SHAP| across {importance.get('n_explained', '—')} "
                   f"patients, {importance.get('n_background', '—')} background rows.")
        st.plotly_chart(importance_bar(importance), use_container_width=True,
                        config={"displayModeBar": False})

    with right:
        section("Confusion matrix")
        st.caption("Rows are actual diagnoses, columns are model predictions.")
        st.plotly_chart(confusion_heatmap(test["confusion_matrix"]),
                        use_container_width=True, config={"displayModeBar": False})

        tn, fp = test["confusion_matrix"][0]
        fn, tp = test["confusion_matrix"][1]
        md(
            kv_rows([
                ("True negatives", f"{tn:,}"),
                ("False positives", f"{fp:,}"),
                ("False negatives", f"{fn:,}"),
                ("True positives", f"{tp:,}"),
            ]), unsafe_allow_html=True)

    md('<div style="height:16px"></div>', unsafe_allow_html=True)
    section("Training history")
    st.caption(f"{metrics['model']['epochs_run']} epochs run, best validation AUC "
               f"at epoch {metrics['model']['best_epoch'] + 1}. Early stopping "
               f"restored the best weights.")
    st.plotly_chart(history_figure(metrics["history"]), use_container_width=True,
                    config={"displayModeBar": False})

    md('<div style="height:16px"></div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")

    with left:
        section("Model card")
        md(kv_rows([
            ("Architecture", metrics["model"]["architecture"]),
            ("Parameters", f"{metrics['model']['params']:,}"),
            ("Inputs", f"{metrics['features']['count']} features"),
            ("Training rows", f"{metrics['split']['train']:,}"),
            ("Batch size", f"{metrics['model']['batch_size']}"),
            ("Decision rule", f"Youden's J → {metrics['threshold']:.3f}"),
        ]), unsafe_allow_html=True)

    with right:
        section("Dataset")
        md(kv_rows([
            ("Rows", f"{metrics['dataset']['rows']:,}"),
            ("Positive cases", f"{metrics['dataset']['positives']:,}"),
            ("Prevalence", f"{metrics['dataset']['prevalence']:.1%}"),
            ("Validation", f"{metrics['split']['validation']:,}"),
            ("Test", f"{metrics['split']['test']:,}"),
            ("Source", "Kaggle diabetes prediction dataset"),
        ]), unsafe_allow_html=True)

    with st.expander("Full classification report (test split)"):
        report = test["classification_report"]
        rows = []
        for label in ("0", "1"):
            entry = report.get(label, {})
            rows.append({
                "Class": "No diabetes" if label == "0" else "Diabetes",
                "Precision": f"{entry.get('precision', 0):.3f}",
                "Recall": f"{entry.get('recall', 0):.3f}",
                "F1": f"{entry.get('f1-score', 0):.3f}",
                "Support": f"{int(entry.get('support', 0)):,}",
            })
        data_table(pd.DataFrame(rows), max_height=380)

    with st.expander("Important limitations"):
        md(
            "- Trained on a synthetic-leaning public dataset; performance on a real "
            "clinic population will be lower.\n"
            "- Probabilities are inflated by class weighting (background mean risk "
            "≈ 19.6% vs 8.5% prevalence). Use the tuned threshold, not the raw "
            "percentage, for decisions.\n"
            "- The model is a screening aid and must not replace laboratory "
            "diagnosis or clinical judgement.")


# ════════════════════════════════════════════════════════════════════════════
# Login + shell
# ════════════════════════════════════════════════════════════════════════════
def brand_html(mark_size: int = 46) -> str:
    return f"""
<div class="dp-brand">
  <div class="dp-brand-mark" style="width:{mark_size}px;height:{mark_size}px">
    <svg width="{int(mark_size * 0.56)}" height="{int(mark_size * 0.56)}"
         viewBox="0 0 24 24" fill="none">
      <path d="M12 3v18M3 12h18" stroke="white" stroke-width="2.6"
            stroke-linecap="round"/>
      <circle cx="12" cy="12" r="9.2" stroke="white" stroke-width="1.4"
              stroke-opacity=".55"/>
    </svg>
  </div>
  <div>
    <div class="dp-brand-title">DiabPredict <span style="color:#1E88E5">Pro</span></div>
    <div class="dp-brand-sub">Clinical risk intelligence</div>
  </div>
</div>
"""


def oauth_ready() -> bool:
    """True when Google/OIDC secrets are configured for st.login()."""
    try:
        auth = st.secrets.get("auth")
    except Exception:
        return False
    return bool(auth and auth.get("client_id") and auth.get("cookie_secret"))


def sso_signed_in() -> bool:
    if not oauth_ready():
        return False
    try:
        return bool(st.user.is_logged_in)
    except Exception:
        return False


def password_ok(candidate: str) -> bool:
    expected = DEFAULT_PASSWORD
    try:
        expected = st.secrets.get("APP_PASSWORD", expected)
    except Exception:
        pass
    expected = os.environ.get("APP_PASSWORD", expected)
    return hmac.compare_digest(str(candidate), str(expected))


MAX_ATTEMPTS = 5
LOCK_SECONDS = 60


def login_locked_for() -> int:
    """Seconds left on the brute-force cool-down, 0 when unlocked."""
    remaining = float(st.session_state.get("lock_until", 0)) - time.time()
    return max(0, int(round(remaining)))


def _submit_login() -> None:
    """Button callback: runs before the script body, so a correct password
    renders the dashboard on the same pass (no st.rerun needed)."""
    if login_locked_for():
        return
    if password_ok(st.session_state.get("password_input", "")):
        st.session_state["auth"] = True
        st.session_state["login_error"] = ""
        st.session_state["attempts"] = 0
        return
    attempts = int(st.session_state.get("attempts", 0)) + 1
    st.session_state["attempts"] = attempts
    if attempts >= MAX_ATTEMPTS:
        st.session_state["lock_until"] = time.time() + LOCK_SECONDS
        st.session_state["login_error"] = (
            f"Too many attempts. For your security, sign-in is paused for "
            f"{LOCK_SECONDS} seconds.")
    else:
        left = MAX_ATTEMPTS - attempts
        st.session_state["login_error"] = (
            f"Incorrect password. {left} attempt{'s' if left != 1 else ''} left.")


def render_login() -> None:
    md('<div style="height:5vh"></div>', unsafe_allow_html=True)
    _, middle, _ = st.columns([1, 1.05, 1])
    with middle:
        md(brand_html(52), unsafe_allow_html=True)
        md(
            '<div class="dp-login-title">Clinician sign in</div>'
            '<div class="dp-login-sub">Secure access to patient risk assessment '
            'and cohort scoring.</div>', unsafe_allow_html=True)
        md('<div style="height:10px"></div>', unsafe_allow_html=True)
        st.text_input("Access password", type="password", key="password_input",
                      placeholder="Enter your access password")
        st.button("Sign in", type="primary", on_click=_submit_login,
                  disabled=login_locked_for() > 0, use_container_width=True)
        if login_locked_for():
            st.warning(f"Sign-in paused for {login_locked_for()}s. "
                       "Please wait and try again.")
        elif st.session_state.get("login_error"):
            st.error(st.session_state["login_error"])

        if oauth_ready():
            md('<div class="dp-login-or">or</div>', unsafe_allow_html=True)
            st.button("Continue with Google", on_click=st.login,
                      use_container_width=True)

        md(
            '<div class="dp-login-foot">Authorised clinical users only · '
            f'DiabPredict Pro v{APP_VERSION}</div>',
            unsafe_allow_html=True)
    md(
        '<div style="text-align:center;color:#93A5BA;font-size:.74rem;'
        'margin-top:16px">Demonstration build · password provided by your '
        'administrator</div>', unsafe_allow_html=True)


def render_sidebar() -> str:
    with st.sidebar:
        md(brand_html(), unsafe_allow_html=True)
        md('<div style="height:14px"></div>', unsafe_allow_html=True)
        md('<div class="dp-nav-label">Workspace</div>', unsafe_allow_html=True)

        labels = [f"{name}" for name in PAGES]
        choice = st.radio("Navigation", labels, label_visibility="collapsed",
                          key="nav")
        md('<div style="height:10px"></div>', unsafe_allow_html=True)
        md(f'<div class="dp-login-foot" style="text-align:left">'
                    f'{PAGE_CAPTION[choice]}</div>', unsafe_allow_html=True)

        md('<div style="height:12px"></div>', unsafe_allow_html=True)
        md('<div class="dp-nav-label">View mode</div>',
                    unsafe_allow_html=True)
        with st.container(key="view_switch"):
            st.radio("View mode", VIEW_MODES, horizontal=True, key="view_mode",
                     label_visibility="collapsed")
        md(
            '<div class="dp-view-hint">'
            + ("Plain language for patients." if patient_view()
               else "Full detail for clinical staff.")
            + '</div>', unsafe_allow_html=True)

        md("<hr/>", unsafe_allow_html=True)
        metrics = ARTIFACTS["metrics"]
        if patient_view():
            md(
                '<div class="dp-side-card">'
                '<div style="font-weight:700;color:#0F2A44;margin-bottom:6px">'
                'About this tool</div>'
                '<div style="margin-bottom:6px">Estimates the chance of diabetes '
                'from 8 routine measurements.</div>'
                '<div style="margin-bottom:6px">A screening aid only. Only a blood '
                'test can diagnose diabetes.</div>'
                f'<div style="color:#93A5BA">Version v{APP_VERSION}</div>'
                '</div>', unsafe_allow_html=True)
        else:
            md(
                '<div class="dp-side-card">'
                '<div style="font-weight:700;color:#0F2A44;margin-bottom:6px">'
                'Model information</div>'
                f'<div class="dp-side-row"><span>Network</span><span>64 → 32 → 1</span></div>'
                f'<div class="dp-side-row"><span>Test AUC</span>'
                f'<span>{metrics["test"]["auc"]:.3f}</span></div>'
                f'<div class="dp-side-row"><span>Recall</span>'
                f'<span>{metrics["test"]["at_tuned"]["recall"]:.1%}</span></div>'
                f'<div class="dp-side-row"><span>Threshold</span>'
                f'<span>{ARTIFACTS["threshold"]:.3f}</span></div>'
                f'<div class="dp-side-row"><span>Inputs</span>'
                f'<span>{metrics["features"]["count"]} features</span></div>'
                f'<div class="dp-side-row"><span>Version</span><span>v{APP_VERSION}</span></div>'
                '</div>', unsafe_allow_html=True)

        md('<div style="height:10px"></div>', unsafe_allow_html=True)
        st.caption("Screening aid only — not a diagnosis.")
        if st.button("Sign out"):
            if oauth_ready():
                try:
                    st.logout()
                except Exception:
                    pass
            st.session_state.clear()
            st.cache_resource.clear()
            st.rerun()
    return choice


def main() -> None:
    apply_theme()
    st.markdown(render_css(), unsafe_allow_html=True)

    if sso_signed_in():
        st.session_state["auth"] = True

    if not st.session_state.get("auth"):
        render_login()
        return

    missing = [path.name for path in
               (MODEL_PATH, PREPROCESS_PATH, SCALER_PATH, IMPORTANCE_PATH,
                BACKGROUND_PATH, METRICS_PATH) if not path.exists()]
    if missing:
        st.error("Required model artefacts are missing: " + ", ".join(missing))
        st.info("Copy them next to `app.py`, then reload the page.")
        return

    try:
        ARTIFACTS.update(load_artifacts())
    except Exception as exc:
        st.error("The model artefacts could not be loaded.")
        st.exception(exc)
        return

    choice = render_sidebar()
    routes = {
        "Dashboard": page_dashboard,
        "Single Prediction": page_single,
        "What-If Simulator": page_whatif,
        "Batch Prediction": page_batch,
        "Model Insights": page_insights,
    }
    routes[choice]()

    md(
        f'<div style="text-align:center;color:#A3B4C6;font-size:.72rem;'
        f'margin-top:28px">{APP_NAME} v{APP_VERSION} · Dense neural network · '
        f'Test AUC {ARTIFACTS["metrics"]["test"]["auc"]:.3f} · '
        f'Screening aid, not a diagnosis</div>', unsafe_allow_html=True)


main()
