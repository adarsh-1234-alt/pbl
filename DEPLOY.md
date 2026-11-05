# Deploying DiabPredict Pro

The app is a single Streamlit folder and runs unchanged on Streamlit Community
Cloud, Render, Railway or any Docker/VM host. Model artefacts are committed —
nothing is trained during deploy.

## 0. Pre-flight (local)

```powershell
cd diabpredict_pro
python -m py_compile app.py
python smoke_test_app.py          # expect 50/50, RESULT: PASS
```

Only these files are needed at runtime:

```
app.py  dnn_data.py  requirements.txt  runtime.txt  packages.txt
render.yaml  Procfile
.streamlit/config.toml
dnn_model.keras  scaler.pkl  dnn_preprocess.json
dnn_metrics.json  feature_importance.json  dnn_background.csv
```

`.streamlit/secrets.toml`, `predictions_log.csv`, `__pycache__`, PDFs and logs
are ignored by `.gitignore`.

## 1. Push to GitHub

```powershell
cd diabpredict_pro
git init
git add .
git commit -m "DiabPredict Pro: DNN risk app with clinician/patient views and themes"
git branch -M main
git remote add origin https://github.com/<you>/diabpredict-pro.git
git push -u origin main
```

Use a personal access token or GitHub Desktop if the push asks for a password.
The folder is self-contained, so it can be the repository root.

## 2. Streamlit Community Cloud

1. <https://share.streamlit.io> → **New app**.
2. Repository `<you>/diabpredict-pro`, branch `main`, main file `app.py`.
3. **Advanced settings** → Python **3.11**.
4. **Secrets**:

   ```toml
   APP_PASSWORD = "choose-a-strong-password"
   ```

5. Deploy. First boot takes 15–30 s (TensorFlow import + model load); later
   reruns are cached.

If the build fails, check:

* **Build log** for `libgomp.so.1` — `packages.txt` already installs `libgomp1`.
* **Memory** — the free tier has 1 GB; `tensorflow-cpu` fits, but pin
  `tensorflow-cpu==2.16.1` if a build dies while importing TensorFlow.
* **Model file** — `dnn_model.keras` must be committed (about 77 KB).

## 3. Render

1. Push the repo (above), then Render → **New → Blueprint** → select it.
2. Render reads `render.yaml` and creates the service.
3. When prompted, set `APP_PASSWORD` (`sync: false`, never stored in git).
4. Open the service URL. Health checks use `/_stcore/health`.

Manual setup instead of the blueprint:

```
Environment: Python 3.11
Build:       pip install -r requirements.txt
Start:       streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

The free instance (512 MB) may run out of memory loading TensorFlow. Use the
2 GB instance if the service restarts during boot.

## 4. Any other host

```bash
pip install -r requirements.txt
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

`Procfile` is included for hosts that use one.

## 5. Post-deploy checklist

- [ ] Login page appears; wrong passwords are rejected and the 5-attempt
      cool-down engages.
- [ ] `clinician123` (or your secret) signs in.
- [ ] Every page renders: Dashboard, Single Prediction, What-If, Batch,
      Model Insights.
- [ ] Theme pills switch Light ↔ Dark and survive page navigation.
- [ ] Clinician ↔ Patient view switch changes the wording and the PDF.
- [ ] Single prediction returns a score, SHAP bars/waterfall and a PDF.
- [ ] Batch upload scores a CSV and the download contains the rows.
- [ ] On a phone (or DevTools device mode) the header stacks, cards reflow and
      tables scroll horizontally.

## 6. Security notes

The password gate is a single shared secret suitable for a demo. For real
clinical use, either configure the built-in Google OIDC login or put the app
behind an identity-aware proxy (Cloudflare Access, Google IAP, Azure Easy Auth,
oauth2-proxy), and avoid entering identifiable patient data.

Google sign-in is off by default. Add this block to the host's secrets (or
`.streamlit/secrets.toml`) and a **Continue with Google** button appears under
the password form:

```toml
[auth]
redirect_uri = "https://your-app.example.com/oauth2callback"
cookie_secret = "a-long-random-string"
client_id = "xxxx.apps.googleusercontent.com"
client_secret = "xxxx"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

Register the same redirect URI in the Google Cloud console. Sign-out calls
`st.logout()` when OAuth is configured. Without `[auth]`, the app runs the
password gate only.

`APP_PASSWORD` resolves in this order: `.streamlit/secrets.toml` → environment
variable → built-in demo default. Five wrong attempts pause sign-in for 60
seconds, per browser session.
