"""Headless smoke test for the DiabPredict Pro Streamlit app.

Runs every page through Streamlit's AppTest harness (no browser needed) and
asserts that nothing raises. Also exercises the PDF builder and the batch
scoring path directly.

    python smoke_test_app.py
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd
from streamlit.testing.v1 import AppTest

BASE = Path(__file__).resolve().parent
APP = BASE / "app.py"

CHECKS: list[tuple[str, bool]] = []


def check(name: str, passed: bool) -> None:
    CHECKS.append((name, bool(passed)))
    print(f"{'PASS' if passed else 'FAIL'}  {name}")


def login(at: AppTest) -> AppTest:
    at.session_state["auth"] = True
    at.run()
    return at


def state(at: AppTest, key: str, default=None):
    try:
        return at.session_state[key]
    except KeyError:
        return default


def radio_by_label(at: AppTest, label: str):
    for radio in at.radio:
        if radio.label == label:
            return radio
    raise AssertionError(f"radio control {label!r} not found")


def markdown_text(at: AppTest) -> str:
    return " ".join(str(m.value) for m in at.markdown)


def main() -> int:
    # ── login gate ──────────────────────────────────────────────────────────
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    check("app starts without exception", not at.exception)
    check("login screen shown", any("Clinician sign in" in str(m.value)
                                    for m in at.markdown))

    at.text_input[0].set_value("wrong-password")
    at.button[0].click().run()
    check("wrong password rejected", not state(at, "auth"))
    check("error message shown", bool(at.error))

    # brute-force cool-down after five failures
    for _ in range(4):
        at.text_input[0].set_value("wrong-password")
        at.button[0].click().run()
    check("lockout after five failures", bool(at.warning))
    check("sign-in disabled while locked", bool(at.button[0].proto.disabled))

    at.session_state["lock_until"] = 0
    at.session_state["attempts"] = 0
    at.run()
    check("lockout clears", not bool(at.button[0].proto.disabled))

    at.text_input[0].set_value("clinician123")
    at.button[0].click().run()
    check("correct password accepted", bool(state(at, "auth")))

    # ── every page renders ──────────────────────────────────────────────────
    pages = ["Dashboard", "Single Prediction", "What-If Simulator",
             "Batch Prediction", "Model Insights"]
    for page in pages:
        radio_by_label(at, "Navigation").set_value(page).run()
        check(f"page renders: {page}", not at.exception)

    # ── theme switch ────────────────────────────────────────────────────────
    theme = radio_by_label(at, "Theme")
    check("theme switch present", theme.value == "Light")
    check("theme switch offers both modes",
          [option.split()[-1] for option in theme.options] == ["Light", "Dark"])

    radio_by_label(at, "Theme").set_value("Dark").run()
    check("dark theme selected", state(at, "theme") == "Dark")
    check("dark theme styles emitted", "#0B1220" in markdown_text(at))
    check("dark theme renders", not at.exception)

    radio_by_label(at, "Theme").set_value("Light").run()
    check("light theme restored", "#F7FAFD" in markdown_text(at))

    # ── single prediction flow ──────────────────────────────────────────────
    radio_by_label(at, "Navigation").set_value("Single Prediction").run()
    at.text_input[-1].set_value("PT-0001")
    at.button[0].click().run()
    check("prediction runs", not at.exception and state(at, "result") is not None)
    result = state(at, "result") or {}
    risk = float(result.get("risk", -1))
    check("risk is a probability", 0.0 <= risk <= 1.0)
    check("SHAP items returned", len(result.get("items", [])) == 13)
    check("PDF download offered", len(at.get("download_button")) >= 1)
    check("clinician view shows SHAP waterfall",
          "Why this score" in markdown_text(at))

    # ── what-if flow ────────────────────────────────────────────────────────
    radio_by_label(at, "Navigation").set_value("What-If Simulator").run()
    check("what-if renders with patient", not at.exception)
    sliders = at.slider
    check("what-if exposes sliders", len(sliders) >= 4)

    # ── clinician / patient toggle ──────────────────────────────────────────
    view = radio_by_label(at, "View mode")
    check("view toggle defaults to clinician", view.value == "Clinician")
    check("view toggle is a two-way switch", list(view.options) == ["Clinician", "Patient"])

    radio_by_label(at, "View mode").set_value("Patient").run()
    check("patient view selected", state(at, "view_mode") == "Patient")
    check("patient what-if renders", not at.exception)

    radio_by_label(at, "Navigation").set_value("Single Prediction").run()
    plain_text = markdown_text(at)
    check("patient view renders", not at.exception)
    check("patient view uses plain language", "What this means for you" in plain_text)
    check("patient view hides SHAP waterfall", "Why this score" not in plain_text)
    check("patient view hides the decision threshold", "Decision threshold" not in plain_text)
    check("patient PDF summary offered",
          any("patient summary" in str(getattr(b, "label", "")).lower()
              for b in at.get("download_button")))

    radio_by_label(at, "Navigation").set_value("Model Insights").run()
    check("patient insights render", not at.exception)
    check("patient insights are plain",
          "How the score is worked out" in markdown_text(at))

    radio_by_label(at, "View mode").set_value("Clinician").run()
    check("clinician view restored",
          "Model insights" in markdown_text(at) or "Global feature importance" in markdown_text(at))

    # ── dashboard after a prediction ────────────────────────────────────────
    radio_by_label(at, "Navigation").set_value("Dashboard").run()
    dashboard_markup = markdown_text(at)
    check("dashboard table renders", "dp-table" in dashboard_markup)
    check("dashboard shows the recorded score", "PT-0001" in dashboard_markup)
    check("dashboard theme switch still present",
          radio_by_label(at, "Theme").value == "Light")

    # ── direct helpers ──────────────────────────────────────────────────────
    sys.path.insert(0, str(BASE))
    import app as model_app  # noqa: E402  (executes the Streamlit script)

    model_app.ARTIFACTS.update(model_app.load_artifacts())

    check("demo password accepted", model_app.password_ok("clinician123"))
    check("password gate active without OAuth secrets",
          not model_app.oauth_ready())
    os.environ["APP_PASSWORD"] = "hospital-secret"
    check("environment password overrides the demo default",
          model_app.password_ok("hospital-secret")
          and not model_app.password_ok("clinician123"))
    os.environ.pop("APP_PASSWORD", None)

    values = dict(model_app.DEFAULTS)
    risk = model_app.predict_one(values)
    check("direct predict works", 0.0 <= risk <= 1.0)

    explained = model_app.explain_row(model_app.risk_row_key(values))
    check("direct SHAP works", abs(sum(i["value"] for i in explained["items"])
                                   + explained["base"] - explained["prediction"]) < 1e-3)

    meta = {
        "threshold": model_app.ARTIFACTS["threshold"],
        "auc": model_app.ARTIFACTS["metrics"]["test"]["auc"],
        "recall": model_app.ARTIFACTS["metrics"]["test"]["at_tuned"]["recall"],
        "precision": model_app.ARTIFACTS["metrics"]["test"]["at_tuned"]["precision"],
        "base": explained["base"],
    }
    pdf = model_app.build_pdf("PT-0001", values, risk, explained["items"], meta)
    check("PDF has a PDF header", pdf[:5] == b"%PDF-" and len(pdf) > 2000)

    pdf_plain = model_app.build_pdf("PT-0001", values, risk, explained["items"],
                                    meta, plain=True)
    check("patient PDF builds", pdf_plain[:5] == b"%PDF-" and len(pdf_plain) > 1500)

    batch = pd.read_csv(BASE / "dnn_background.csv").head(50)
    matrix = model_app.transform_frame(batch, model_app.ARTIFACTS["preprocessor"])
    scores = model_app.ARTIFACTS["model"].predict(matrix, verbose=0).ravel()
    check("batch scoring works", len(scores) == 50 and scores.min() >= 0
          and scores.max() <= 1)

    template = pd.read_csv(io.BytesIO(model_app.template_csv()))
    check("batch template has all columns",
          set(model_app.dnn_data.RAW_FEATURES) <= set(template.columns))

    failed = [name for name, ok in CHECKS if not ok]
    print()
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    print("RESULT:", "PASS" if not failed else "FAIL -> " + ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
