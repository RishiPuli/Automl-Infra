# ☁️ CloudAutoML — Production AutoML Platform

[![CI/CD](https://github.com/YOUR_USERNAME/cppe_project/actions/workflows/main.yml/badge.svg)](https://github.com/YOUR_USERNAME/cppe_project/actions)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED)

> A simplified cloud-native AutoML platform that takes any CSV → auto-analyzes → allocates resources → trains multiple models → tracks everything → displays a full dashboard.

---

## 🎯 Features at a Glance

| Feature | Description |
|---|---|
| 📂 **Dataset Upload** | Upload any CSV from the UI; no code changes needed |
| 🔍 **Auto Analysis** | Detects rows, features, dtypes, missing values, complexity |
| 🤖 **Auto Model Selection** | Picks classification or regression models based on target |
| ⚙️ **Resource-Aware Engine** | Allocates CPUs & RAM based on dataset size |
| 📊 **Full Metrics** | Accuracy / F1 / RMSE / R² + confusion matrix + feature importance |
| 📈 **MLflow Integration** | Logs every run: params, metrics, artifacts, models |
| 🌐 **Streamlit UI** | One-page dashboard with live progress + log window |
| 📥 **Download Best Model** | One-click `.pkl` download |
| 🐳 **Docker Ready** | `docker build` → `docker run` |
| ⚙️ **CI/CD** | GitHub Actions: lint → test → docker build → smoke test |

---

## 📁 Project Structure

```
CloudAutoML/
│
├── app/
│   └── streamlit_app.py        ← Full Streamlit web interface
│
├── core/
│   ├── __init__.py
│   ├── orchestrator.py         ← Training pipeline coordinator
│   ├── resource_manager.py     ← CPU/RAM allocation + peak-memory monitoring
│   └── evaluator.py            ← Metrics computation + plot generation
│
├── utils/
│   ├── __init__.py
│   ├── dataset_analyzer.py     ← CSV analysis + complexity detection
│   └── logger.py               ← Structured rotating logger
│
├── tests/
│   ├── test_dataset_analyzer.py
│   └── test_evaluator.py
│
├── src/                        ← Legacy CLI modules (retained)
│   ├── data_processing.py
│   ├── train_models.py
│   ├── orchestrator.py
│   └── experiment_tracker.py
│
├── models/                     ← Saved .pkl model artifacts
├── data/                       ← Input datasets
├── reports/                    ← Auto-generated plot PNGs
├── logs/                       ← Rotating log files
├── mlruns/                     ← MLflow tracking store
│
├── Dockerfile
├── .github/workflows/main.yml  ← CI/CD pipeline
├── .gitignore
├── requirements.txt
├── main.py                     ← CLI entry point
└── README.md
```

---

## 🚀 Quick Start

### 1. Create & activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3a. Launch the Web App (Recommended)

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) — upload a CSV and click **Start Training**.

### 3b. Run CLI (headless)

```bash
# Auto-generates a demo dataset if no CSV is given
python main.py

# With your own dataset
python main.py --csv data/my_dataset.csv --target label_column
```

### 4. View MLflow Dashboard

```bash
# In a separate terminal
mlflow ui
```

Open [http://localhost:5000](http://localhost:5000)

---

## 🐳 Docker

```bash
# Build
docker build -t cloudautoml .

# Run (Streamlit on :8501)
docker run -p 8501:8501 cloudautoml
```

---

## 🧪 Tests

```bash
pytest tests/ -v --tb=short
```

---

## ⚙️ Resource Allocation Strategy

| Dataset Complexity | CPU Budget | Memory Budget |
|---|---|---|
| **Small** (≤5K rows, ≤15 features) | 1–2 cores | 256 MB |
| **Medium** (≤100K rows, ≤50 features) | 2–4 cores | 512 MB |
| **Large** (anything above) | 4–8 cores | 1 GB+ |

All allocations are **capped** by actual host hardware at runtime.

---

## 🤖 Auto Model Selection

| Task | Models |
|---|---|
| **Classification** | Logistic Regression, Random Forest, XGBoost, SVM |
| **Regression** | Linear Regression, Random Forest Regressor, XGBoost Regressor |

Task type is **auto-detected** from the target column's dtype and cardinality.

---

## 📊 Metrics

| Task | Metrics |
|---|---|
| Classification | Accuracy, Precision, Recall, F1 Score, Confusion Matrix |
| Regression | RMSE, MAE, R² Score |

**Plots generated:** model comparison bar chart · confusion matrix heatmap · feature importance · regression scatter

---

## 🌐 Deployment (Streamlit Cloud)

1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set **Main file path** → `app/streamlit_app.py`
4. Click **Deploy**

---

## 📈 CI/CD Pipeline

`.github/workflows/main.yml` runs on every push to `main`:

1. **Lint** — `flake8` code quality check
2. **Unit Tests** — `pytest` with coverage
3. **Docker Build** — verifies the image builds successfully
4. **Integration Smoke Test** — runs `python main.py` (demo mode) end-to-end

---

## 📄 License

MIT — free to use, modify, and distribute.
