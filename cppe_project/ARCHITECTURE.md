# CloudML — Cloud-Based Intelligent Simulation & Control Platform
## Architecture Reference

---

## System Overview

CloudML is a **cloud-native AutoML platform** that ingests raw tabular datasets and
autonomously executes the full machine learning lifecycle: profiling → resource
allocation → hyperparameter optimisation → parallel training → evaluation → model
export → REST API inference. Every stage is logged, versioned with MLflow, and
containerised for reproducible deployment.

---

## Component Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER LAYER                                     │
│  Browser / CLI                                                          │
│    │                                                                    │
│    ├── Streamlit Web UI  (app/streamlit_app.py)  :8501                 │
│    └── CLI Entry Point   (main.py)                                      │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────────┐
│                     PIPELINE CORE (cloud/backend)                       │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  utils/dataset_analyzer.py                                       │  │
│  │  · Detects target column, task type, complexity                  │  │
│  │  · Returns structured profile dict used by every other module    │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │ analysis dict                             │
│  ┌──────────────────────────▼───────────────────────────────────────┐  │
│  │  core/stability_predictor.py  (Meta-ML)                          │  │
│  │  · RF classifier trained on synthetic meta-dataset               │  │
│  │  · Predicts best model *before* training via dataset profile     │  │
│  │  · Outputs ranked confidence scores logged to orchestrator       │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │ ranking                                   │
│  ┌──────────────────────────▼───────────────────────────────────────┐  │
│  │  core/performance_optimizer.py  (Strategy Planner)               │  │
│  │  · Estimates per-model latency & memory before training          │  │
│  │  · Recommends: thread_pool vs sequential, HPO on/off             │  │
│  │  · Flags memory-budget overflow risks                            │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │ plan                                      │
│  ┌──────────────────────────▼───────────────────────────────────────┐  │
│  │  core/resource_manager.py                                        │  │
│  │  · Reads host psutil stats (CPUs, RAM)                           │  │
│  │  · Maps complexity → CPU/RAM budget                              │  │
│  │  · PeakMemoryMonitor polls RSS every 300ms during training       │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │ allocation                                │
│  ┌──────────────────────────▼───────────────────────────────────────┐  │
│  │  core/orchestrator.py  (Pipeline Coordinator)                    │  │
│  │  · Preprocessing: ColumnTransformer (StandardScaler + OHE)       │  │
│  │  · Parallel training: ThreadPoolExecutor (n=cpu_allocated)       │  │
│  │  · Per-model: HPO (Optuna TPE) → fit → CV → metrics             │  │
│  │  · Evaluator plots → SHAP → Stacking Ensemble → MLflow         │  │
│  │  · Saves: best_model.pkl + preprocessor.pkl + deployment_meta   │  │
│  └──────────────────────────┬───────────────────────────────────────┘  │
│                             │                                           │
│  ┌──────────────────────────▼───────────────────────────────────────┐  │
│  │  Supporting Modules                                              │  │
│  │  core/hyperparameter_tuner.py  — Optuna TPE per model           │  │
│  │  core/evaluator.py             — Metrics + matplotlib plots     │  │
│  │  core/explainer.py             — SHAP bar plots                 │  │
│  │  core/ensemble_builder.py      — StackingClassifier/Regressor   │  │
│  │  core/report_generator.py      — Multi-page PDF via PdfPages    │  │
│  │  utils/metrics_logger.py       — Structured JSONL event log     │  │
│  │  utils/logger.py               — Rotating file logger           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────────────┘
                         │  artifacts
                         │  (best_model.pkl, preprocessor.pkl,
                         │   deployment_meta.json, plots, PDF)
┌────────────────────────▼────────────────────────────────────────────────┐
│                     INFERENCE / API LAYER                               │
│                                                                         │
│  api/prediction_server.py  (FastAPI)                           :8000   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Middleware stack (outermost → innermost)                        │   │
│  │  1. CORS (CORSMiddleware)                                        │   │
│  │  2. X-Request-ID injection  (uuid4 per request)                 │   │
│  │  3. X-Response-Time-MS measurement                               │   │
│  │  4. JSONL audit log (metrics_logger.log_api_request)            │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  Endpoints                                                       │   │
│  │  GET  /health      — liveness probe (Docker, Render, CI)        │   │
│  │  GET  /model-info  — deployment metadata JSON                    │   │
│  │  GET  /metrics     — p50/p95/p99 latency + error rate + events  │   │
│  │  POST /predict     — batch inference with schema validation      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────────┐
│                     EXPERIMENT TRACKING                                 │
│  MLflow Tracking Server (mlruns/)                              :5000   │
│  · Params: model, task, complexity, hpo_enabled                        │
│  · Metrics: accuracy/r2, f1, rmse, cv_mean, cv_std, train_time, ram   │
│  · Artifacts: .pkl, plots, SHAP images                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
CSV Upload
    │
    ▼
analyze_dataset() → profile dict (rows, features, dtypes, missing, complexity)
    │
    ▼
StabilityPredictor.fit().predict_ranking()  →  ranked model list
    │
    ▼
PerformanceOptimiser.recommend()  →  strategy plan (latency/memory estimates)
    │
    ▼
allocate_resources()  →  cpu_allocated, memory_budget_mb
    │
    ▼
preprocess()  →  X_train, X_test, y_train, y_test, ColumnTransformer, LabelEncoder
    │
    ├── ThreadPoolExecutor (n workers) ─────┐
    │   ├── Worker: Optuna HPO → fit      │ ← parallel
    │   ├── Worker: Optuna HPO → fit      │
    │   └── Worker: Optuna HPO → fit      │
    │                                      │
    ◄──────────────────────────────────────┘ results list
    │
    ├── Evaluator plots (main thread — matplotlib-safe)
    ├── SHAP explanations (main thread)
    ├── Stacking Ensemble builder
    ├── MLflow logging
    ├── Best model selection + artifact export
    └── PDF Report generation
```

---

## Horizontal Scalability Design

| Component          | Scaling Approach |
|--------------------|-----------------|
| **Training**       | ThreadPoolExecutor — adds workers as CPU budget increases |
| **Inference API**  | Stateless FastAPI → run N replicas behind a load balancer |
| **Data tier**      | CSV → swap to S3 / GCS presigned URL with minimal code change |
| **MLflow**         | Local → point `MLFLOW_TRACKING_URI` to a remote PostgreSQL store |
| **Containers**     | Docker image → Kubernetes HPA on CPU utilisation |

---

## Security Design

| Layer              | Mechanism |
|--------------------|-----------|
| **API Headers**    | `X-Request-ID` (UUID4) for distributed tracing on every response |
| **Input Validation** | Feature schema validated against training-time `feature_names` list |
| **CORS**           | Configurable origin whitelist (currently open for development) |
| **Secrets**        | Never committed; loaded from environment variables |
| **Audit Log**      | Every API request written to `logs/metrics.jsonl` (JSONL, append-only) |
| **Dependencies**   | Pinned versions in `requirements.txt`; `pip audit` in CI |

---

## Technology Stack

| Concern            | Library / Tool | Version |
|--------------------|----------------|---------|
| Core ML            | scikit-learn   | 1.4.1   |
| Gradient boosting  | XGBoost        | 2.0.3   |
| HPO                | Optuna (TPE)   | 3.6.1   |
| Explainability     | SHAP           | 0.45.0  |
| Experiment tracking| MLflow         | 2.11.1  |
| System monitoring  | psutil         | 5.9.8   |
| Visualisation      | matplotlib, seaborn | 3.8.3 / 0.13.2 |
| Web UI             | Streamlit      | 1.32.0  |
| Inference API      | FastAPI + uvicorn | 0.110.0 / 0.27.1 |
| Containerisation   | Docker (python:3.11-slim) | — |
| CI/CD              | GitHub Actions | — |
| Testing            | pytest + httpx | — |
