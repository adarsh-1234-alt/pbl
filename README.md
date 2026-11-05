# DiabPredict Pro

Clinical diabetes-risk screening app built on the Stage 1 DNN
(13 inputs, 64 → 32 → 1, test AUC **0.977**).

White/blue clinical interface, five pages, password-gated, with a clinician /
patient view switch and a light / dark theme switch. All model artefacts load
once at start-up — nothing is trained at runtime.

```
Dashboard          summary stats, recent predictions, risk donut
Single Prediction  input form → gauge → SHAP waterfall → PDF report
What-If Simulator  live sliders, delta vs the original score
Batch Prediction   CSV upload → scored table → CSV download
Model Insights     global SHAP, confusion matrix, training curves
```

## Run locally

```powershell
cd diabpredict_pro
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 and sign in with **`clinician123`**.

Set your own password in either place (secrets wins over the environment
variable, both win over the demo default):

```toml
# .streamlit/secrets.toml  (git-ignored)
APP_PASSWORD = "your-password"
```

```powershell
$env:APP_PASSWORD = "your-password"
```

After five wrong attempts the sign-in form pauses for 60 seconds.

### Optional Google sign-in

Add an `[auth]` block to `.streamlit/secrets.toml` (or the host's secrets) and
the login page gains a **Continue with Google** button:

```toml
[auth]
redirect_uri = "https://your-app.example.com/oauth2callback"
cookie_secret = "a-long-random-string"
client_id = "xxxx.apps.googleusercontent.com"
client_secret = "xxxx"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

Register the same redirect URI in the Google Cloud console. The shared password
stays available as a fallback unless you remove it from the code.

## Verification

```powershell
python smoke_test_app.py
```

Runs every page through Streamlit's `AppTest` harness, both view modes, both
themes, the PDF builders, the SHAP path and batch scoring. Expect
`50/50 checks passed` and `RESULT: PASS`.

## Interface

* **View mode** — *Clinician* shows the full SHAP waterfall, threshold and
  metrics; *Patient* swaps in plain language, everyday risk wording and a
  patient-friendly PDF summary.
* **Theme** — the ☀ Light / ☾ Dark pills at the top-right of every page.
  Dark mode remaps the whole palette (cards, tables, charts, Plotly,
  Matplotlib) and remembers the choice for the session.
* **Mobile** — below 700 px the header stacks, the theme pills move under the
  title, stat cards reflow, tables scroll horizontally and buttons meet the
  42 px touch target. Motion is disabled automatically when the operating
  system asks for reduced motion.
* **Loading feedback** — spinners cover model start-up, SHAP explanations,
  batch scoring and PDF generation.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository (it is self-contained).
2. On <https://share.streamlit.io> choose **New app** → repo → main file
   `app.py`.
3. Under *Settings → Secrets* add:

   ```toml
   APP_PASSWORD = "your-password"
   ```

4. Optionally set the Python version to **3.11** in *Advanced settings*.

`packages.txt` installs `libgomp1`, which TensorFlow needs on the slim image.
The free tier gives 1 GB RAM; `tensorflow-cpu` fits, but the first cold start
takes ~15–30 s while TensorFlow imports and the model loads.

## Deploy on Render

The repository ships a `render.yaml` blueprint.

1. Push the folder to GitHub.
2. In Render choose **New → Blueprint** and select the repo.
3. Render creates the web service from `render.yaml`; when prompted, enter
   `APP_PASSWORD` (marked `sync: false`, so it is never committed).
4. Deploy. Health checks hit `/_stcore/health`.

The start command is:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 \
  --server.headless true --browser.gatherUsageStats false
```

The free 512 MB instance can run out of memory while TensorFlow loads. If that
happens, upgrade to the 2 GB instance or pin `tensorflow-cpu==2.16.1` in
`requirements.txt`. `Procfile` covers platforms that prefer one (Railway,
Heroku-style hosts).

## Files

```
app.py                    Streamlit UI, CSS, charts, PDF, auth
dnn_data.py               preprocessing contract shared with training
smoke_test_app.py         headless verification suite
requirements.txt          runtime dependencies
render.yaml / Procfile    deployment definitions
runtime.txt               Python 3.11 for hosted builders
packages.txt              apt packages for managed Python images
.streamlit/config.toml    Streamlit theme + upload limit
.streamlit/secrets.toml.example
dnn_model.keras  scaler.pkl  dnn_preprocess.json
dnn_metrics.json  feature_importance.json  dnn_background.csv
```

Everything above except `secrets.toml` must be committed. `predictions_log.csv`
is generated at runtime and git-ignored.

## How the model is used

Every prediction path — form, what-if, batch — calls the same function:

```python
matrix = transform_frame(frame, preprocessor)   # dnn_data.py
risk = model.predict(matrix, verbose=0)
```

`transform_frame` handles medians, `StandardScaler`, one-hot encoding and the
fixed feature order, so serving can never drift from training. `dnn_data.py`
must stay identical to the module used by `train_dnn.py`.

* **Threshold**: 0.478 (Youden's J on validation), stored in `dnn_metrics.json`.
* **Bands**: Low < 30%, Moderate 30–60%, High 60–80%, Critical > 80%.
* **SHAP**: `shap.KernelExplainer` over 60 background rows, 200 samples per
  patient, cached per input vector. Falls back to "explanation unavailable"
  without blocking the score.

## Limits and privacy

* Trained on a public, synthetic-leaning dataset — expect lower performance in a
  real clinic.
* Class weighting inflates the raw probabilities (background mean 19.6% vs 8.5%
  prevalence). Use the tuned threshold for decisions; the percentage is relative,
  not calibrated.
* The demo password gate is a single shared secret, not real authentication.
  For production, put the app behind SSO / an identity-aware proxy and keep
  patient identifiers out of URLs and logs.
* `predictions_log.csv` is written next to `app.py` when the filesystem is
  writable. Streamlit Cloud and Render use ephemeral disks, so the dashboard
  history resets when the container restarts; session state keeps it for the
  life of a session.
* Screening aid only, not a diagnosis.
