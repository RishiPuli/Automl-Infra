# DEPLOYMENT.md — CloudAutoML Deployment Guide

## Overview

This file covers deploying every component of CloudAutoML:

| Component | Platform | URL after deploy |
|---|---|---|
| Streamlit dashboard | Streamlit Community Cloud | `https://<your-app>.streamlit.app` |
| FastAPI inference endpoint | Render (free tier) | `https://cloudautoml-api.onrender.com` |

---

## STEP 1 — Streamlit Cloud Deployment

### Prerequisites

- A GitHub account
- This repository pushed to GitHub (see Git setup below)
- A Streamlit Community Cloud account at [share.streamlit.io](https://share.streamlit.io)

### Git Setup (first time)

```bash
git init
git add .
git commit -m "Initial commit: CloudAutoML"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

### Deploy steps

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **"New app"**.
3. Under **Repository**, select your GitHub repository.
4. Under **Branch**, select `main`.
5. Under **Main file path**, enter:
   ```
   app/streamlit_app.py
   ```
6. Click **"Deploy!"**
7. Streamlit will install `requirements.txt`, build, and launch.
8. Your app will be live at a URL like:
   ```
   https://<your-username>-cloudautoml-<hash>.streamlit.app
   ```

### Cloud-safety notes

All of the following are already handled in `app/streamlit_app.py`:

| Risk | How it is handled |
|---|---|
| `psutil` restricted | Wrapped in `try/except`; sidebar shows "unavailable" gracefully |
| `mlruns/` missing | Wrapped in `try/except`; shows info message instead of crashing |
| Absolute file paths | All writes use `os.path.join` with relative paths |
| Heavy model files | Models are excluded from git via `.gitignore` |

### Verify locally before deploying

```bash
streamlit run app/streamlit_app.py
```

Expected: browser opens at `http://localhost:8501`, CSV upload works,
training completes, results display — zero import errors or path errors.

---

## STEP 2 — GitHub Actions CI/CD

The workflow file is already at `.github/workflows/ci.yml`.
It triggers on every push to `main` and on every pull request.

### What the pipeline does

1. Checks out the code on `ubuntu-latest`
2. Sets up Python 3.10
3. Caches pip to speed up subsequent runs
4. Runs `pip install -r requirements.txt pytest httpx`
5. Runs `pytest tests/ -v --tb=short`
6. Prints **"All tests passed. Safe to deploy."** on success

### Verify locally before pushing

```bash
pytest tests/ -v
```

All tests must be green. Then push:

```bash
git add .
git commit -m "Add CI tests"
git push origin main
```

Go to **GitHub → your repo → Actions tab**.
The "CI — CloudAutoML Tests" workflow must show a ✅ green checkmark.

---

## STEP 3 — FastAPI Inference Endpoint on Render

### Prerequisites

- A [render.com](https://render.com) account (free tier is sufficient)
- The repository pushed to GitHub (same repo used for Streamlit)

### Deploy steps

1. Go to [render.com](https://render.com) and sign in / create a free account.
2. Click **"New +"** → **"Web Service"**.
3. Click **"Connect GitHub"** and authorise Render to access your repositories.
4. Select this repository from the list.
5. Fill in the service settings:

   | Field | Value |
   |---|---|
   | **Name** | `cloudautoml-api` |
   | **Region** | Any (e.g. Oregon) |
   | **Branch** | `main` |
   | **Runtime** | `Python 3` |
   | **Build Command** | `pip install -r requirements_api.txt` |
   | **Start Command** | `uvicorn inference_api:app --host 0.0.0.0 --port $PORT` |

6. Under **Instance Type**, select **"Free"**.
7. Click **"Create Web Service"**.
8. Render pulls the repo, runs the build command, and starts uvicorn.
9. The API will be live at:
   ```
   https://cloudautoml-api.onrender.com
   ```

> **Note:** The `render.yaml` file in the project root pre-configures all
> of the above. Render will detect it automatically.

### Test the live API

```bash
# Liveness check
curl https://cloudautoml-api.onrender.com/health

# Expected response:
# {"status":"ok","model_loaded":false}
```

```bash
# Swagger / interactive docs
open https://cloudautoml-api.onrender.com/docs
```

### Send a prediction request (once a model is loaded)

```bash
curl -X POST https://cloudautoml-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [{"feature_0": 1.2, "feature_1": 0.5, "feature_2": -0.3, "feature_3": 0.8}]}'
```

If no model has been trained and uploaded yet, you will receive:
```json
{"detail":"No model loaded. Train a model first."}
```
with HTTP status **503** — this is the expected and correct response.

### Verify locally before deploying

```bash
uvicorn inference_api:app --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs).
The Swagger UI must show all four endpoints:
- `GET /`
- `GET /health`
- `POST /predict`
- `GET /model-info`

Run the inference API tests:

```bash
pytest tests/test_inference_api.py -v
```

All tests must pass before pushing to GitHub.

---

## Final Verification Checklist

### Step 1 — Streamlit

- [ ] `streamlit run app/streamlit_app.py` launches without errors
- [ ] CSV upload works in the browser
- [ ] Training completes and results display correctly
- [ ] `requirements.txt` exists with pinned versions
- [ ] `.streamlit/config.toml` exists
- [ ] `.gitignore` excludes `venv/`, `mlruns/`, `models/*.pkl`

### Step 2 — CI/CD

- [ ] `pytest tests/ -v` passes locally (all green)
- [ ] `.github/workflows/ci.yml` exists
- [ ] Pushed to GitHub → Actions tab shows ✅ green checkmark
- [ ] `pytest.ini` exists in project root

### Step 3 — FastAPI / Render

- [ ] `uvicorn inference_api:app --reload` starts without errors
- [ ] `GET /` returns HTTP 200
- [ ] `GET /health` returns `{"status": "ok", ...}`
- [ ] `POST /predict` returns HTTP 503 when no model is loaded
- [ ] `POST /predict` returns predictions when a model is loaded
- [ ] `pytest tests/test_inference_api.py -v` passes
- [ ] `render.yaml` exists in project root
- [ ] `requirements_api.txt` exists with pinned versions
