# CloudML — Project Evaluation Document
## Technical Reference for CPPE Reviews 1 – Final

---

## 1. Problem Definition

Traditional machine learning workflows require:
- **Manual model selection** — engineers must know which algorithm suits which data type.
- **Manual hyperparameter tuning** — Grid/Random search is slow and expert-driven.
- **Manual resource management** — No awareness of CPU/RAM constraints during training.
- **No integrated monitoring** — Training outcomes and inference metrics live in separate tools.

**CloudML solves all four problems** through a cloud-native, resource-aware AutoML
pipeline that takes any CSV file and produces a trained, evaluated, tracked, and
deployable model — with zero manual intervention required.

**Real-world justification:** Similar to AWS AutoML / Google AutoML
but implemented at student scale using open-source tools, making it reproducible,
debuggable, and academically defensible.

---

## 2. Architecture Summary (Review 1)

> See `ARCHITECTURE.md` for the full component diagram and data flow.

### Layer breakdown

| Layer | Component | What it does |
|-------|-----------|-------------|
| Frontend | `app/streamlit_app.py` | Upload CSV, toggle pipeline options, display results |
| Backend / Orchestrator | `core/orchestrator.py` | Coordinates 11 pipeline stages |
| Meta-ML | `core/stability_predictor.py` | Predicts best model before training starts |
| Strategy Planner | `core/performance_optimizer.py` | Estimates latency, recommends HPO on/off |
| Resource Manager | `core/resource_manager.py` | Maps dataset complexity → CPU/RAM budget |
| HPO | `core/hyperparameter_tuner.py` | Optuna TPE per model (3-fold CV objective) |
| Evaluation | `core/evaluator.py` | Metrics + confusion matrix + feature importance |
| Explainability | `core/explainer.py` | SHAP bar plots per model |
| Ensemble | `core/ensemble_builder.py` | Stacking on top-N base learners |
| Tracking | MLflow | Params, metrics, artifacts per run |
| Inference | `api/prediction_server.py` | FastAPI REST, /predict, /metrics, /health |
| Monitoring | `utils/metrics_logger.py` | JSONL append-only structured event log |
| Reporting | `core/report_generator.py` | Multi-page dark-theme PDF export |

### Cloud-native principles applied

- **Stateless inference API** — any number of replicas can serve the same model artifact.
- **Resource-aware training** — CPU/RAM budgets computed from live host stats (psutil).
- **Decoupled training and serving** — model saved to disk; API reads it independently.
- **Environment parity** — Docker image matches local and production environments exactly.

---

## 3. Agile Backlog (Review 1)

| Sprint | User Story | Priority |
|--------|-----------|----------|
| 1 | As a developer I can run `python main.py` and see a full training cycle | Must |
| 1 | As a user I can upload a CSV and click Start Training in the Streamlit UI | Must |
| 2 | As a user the system auto-detects my target column and task type | Must |
| 2 | As a user models train in parallel so I don't wait sequentially | Must |
| 2 | Resources (CPUs, RAM) are allocated proportional to dataset complexity | Must |
| 3 | HPO uses Bayesian (Optuna TPE) search — not random grid search | Should |
| 3 | SHAP explains *why* each model made each prediction | Should |
| 3 | A stacking ensemble is auto-built from the best base models | Should |
| 3 | Every training run is versioned in MLflow with full artifact logging | Must |
| 4 | A FastAPI server serves the best model as a REST API | Must |
| 4 | The API exposes /metrics with latency percentiles and error rates | Should |
| 4 | X-Request-ID is injected on every API response for tracing | Should |
| 4 | CI/CD pipeline runs on every git push: lint → test → docker → smoke | Must |
| 5 | A meta-ML model (StabilityPredictor) ranks algorithms before training | Could |
| 5 | A PerformanceOptimiser estimates latency and recommends compute strategy | Could |
| 5 | A multi-page PDF report is auto-generated after each training run | Could |

---

## 4. Implementation Detail (Review 2)

### 4.1 End-to-End Workflow

```
User uploads CSV
  → analyze_dataset()         : profile (rows, dtypes, missing, complexity)
  → StabilityPredictor        : predict model ranking from dataset meta-features
  → PerformanceOptimiser      : estimate latency, recommend HPO setting
  → allocate_resources()      : compute CPU/RAM budget from psutil host stats
  → preprocess()              : StandardScaler + OneHotEncoder via ColumnTransformer
  → ThreadPoolExecutor        : train N models in parallel
     └─ per-worker:
        ├── Optuna TPE (n_trials proportional to complexity)
        ├── model.fit()  with PeakMemoryMonitor polling RSS every 300ms
        ├── model.predict() → compute_metrics()
        └── cross_val_score() k=3 folds → cv_mean, cv_std
  → (main thread) evaluator plots: confusion matrix, feature importance, bars
  → (main thread) SHAP explanations
  → (main thread) stacking ensemble on top base learners
  → (main thread) MLflow: log params + metrics + artifacts
  → Select best (highest accuracy / R²)
  → Save: best_model.pkl, preprocessor.pkl, label_encoder.pkl, deployment_meta.json
  → generate_report()         : multi-page PDF
```

### 4.2 ML Integration (defensible at viva)

| Component | Library | Justification |
|-----------|---------|---------------|
| Classification models | LogisticReg, RandomForest, XGBoost, SVM | Cover linear, tree, boosted, kernel families |
| Regression models | LinearReg (Ridge), RFRegressor, XGBoostReg | Same family coverage for continuous targets |
| HPO | Optuna TPE | Bayesian beats random search: exploits past trial results |
| Cross-validation | StratifiedKFold (clf), KFold (reg) | Prevents data leakage; stratified maintains class balance |
| Ensemble | StackingClassifier/Regressor (sklearn) | Reduces variance by combining diverse learners |
| Explainability | SHAP TreeExplainer | Model-agnostic; produces Shapley values from game theory |
| Meta-learning | RandomForestClassifier on synthetic meta-features | Directs search before HPO; saves time on large datasets |

### 4.3 Preprocessing Pipeline

```python
ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])
```
- **Why StandardScaler?** SVM and LogisticRegression are sensitive to feature scale.
- **Why OHE over LabelEncoder?** Avoids ordinal assumptions in tree models.
- **Why handle_unknown="ignore"?** Prevents inference-time crashes on unseen categories.

### 4.4 Parallelism Strategy

```
ThreadPoolExecutor(max_workers=cpu_allocated)
  ↳ each model trains in its own thread
  ↳ thread-safe contract: no matplotlib, no MLflow in threads
  ↳ results collected via a threading.Lock()-guarded list
  ↳ MLflow/plots run on main thread (both require main-thread-only resources)
```

Parallel speedup ratio = `sum(individual_train_times) / wall_clock_time`
— displayed in the CLI output and logged to MLflow.

---

## 5. CI/CD Pipeline (Review 2)

```
Git push to main/develop
    │
    ├── Job 1: lint-and-test
    │   ├── flake8 (max-line-length=120)
    │   ├── pytest tests/ (unit + integration)
    │   │     --cov=core --cov=utils --cov-report=xml
    │   └── Upload coverage.xml artifact
    │
    ├── Job 2: docker-build  (needs: lint-and-test)
    │   ├── docker/build-push-action (build only, push=false)
    │   └── Verify: docker run python -c "import sklearn, xgboost, mlflow..."
    │
    └── Job 3: integration-test  (needs: lint-and-test)
        ├── python main.py --no-hpo --no-shap --no-ensemble --no-report
        ├── Assert: models/best_model.pkl exists
        ├── Assert: models/deployment_meta.json exists
        └── uvicorn inference_api:app & → curl /health → assert status=ok
```

**Branching strategy:**
- `main` — production-stable; protected; requires CI pass + PR review
- `develop` — integration branch; all feature branches merge here first
- `feature/*` — short-lived feature branches

---

## 6. Testing Strategy (Review 3)

### 6.1 Test Suite Overview

| Module | Test File | Tests | Type |
|--------|-----------|-------|------|
| `dataset_analyzer` | test_dataset_analyzer.py | ~20 | Unit |
| `stability_predictor` | test_stability_predictor.py | 20 | Unit |
| `performance_optimizer` | test_performance_optimizer.py | 35 | Unit |
| `resource_manager` | test_resource_manager.py | ~18 | Unit |
| `evaluator` | test_evaluator.py | ~12 | Unit |
| `orchestrator` | test_orchestrator.py | 25 | Integration |
| `api/prediction_server` | test_api_server.py | 30 | Integration |
| `train_models` (src legacy) | test_train_models.py | ~15 | Unit |

**Run all tests:**
```bash
pytest tests/ -v --tb=short --cov=core --cov=utils --cov-report=term-missing
```

### 6.2 Test Case Tables

#### Dataset Analyzer — Selected Test Cases

| TC-ID | Input | Expected Output | Type |
|-------|-------|-----------------|------|
| DA-01 | DataFrame with `target` column (integer) | `task_type = "classification"` | Unit |
| DA-02 | DataFrame where target has 100 unique float values | `task_type = "regression"` | Unit |
| DA-03 | 500-row DataFrame | `complexity = "small"` | Unit |
| DA-04 | 5000-row DataFrame | `complexity = "medium"` | Unit |
| DA-05 | DataFrame with NaN cells | `missing_values = True, total_missing > 0` | Unit |
| DA-06 | Empty DataFrame | `ValueError` raised | Edge |
| DA-07 | `target_hint="price"` with matching column | `target_column = "price"` | Unit |

#### API Server — Selected Test Cases

| TC-ID | Endpoint | Input | Expected | Type |
|-------|----------|-------|----------|------|
| API-01 | GET /health | — | `status=ok, HTTP 200` | Integration |
| API-02 | GET /model-info | No model loaded | `HTTP 503` | Integration |
| API-03 | POST /predict | `{"wrong_key": []}` | `HTTP 422` | Validation |
| API-04 | POST /predict | No model loaded | `HTTP 503` | Integration |
| API-05 | GET /health | — | `X-Request-ID` header present | Security |
| API-06 | GET /health (×5) | — | All 5 X-Request-IDs unique | Security |
| API-07 | GET /metrics | — | `uptime_seconds ≥ 0, error_rate ∈ [0,1]` | Integration |

---

## 7. Performance & Monitoring (Review 3)

### 7.1 Key Performance Metrics

| Metric | Where captured | Tool |
|--------|---------------|------|
| Model accuracy / R² | `compute_metrics()` | Internal + MLflow |
| Cross-validation mean ± std | `cross_val_score()` | Internal + MLflow |
| Training wall-clock time | `time.perf_counter()` | Per-model + MLflow |
| Peak RAM per model (MB) | `PeakMemoryMonitor` (psutil RSS) | Per-model + MLflow |
| Parallel speedup ratio | `seq_estimate / wall_total` | CLI + Streamlit |
| Inference latency (ms) | `time.perf_counter()` in `/predict` | JSONL log |
| API error rate | `predict_errors / predict_calls` | `/metrics` endpoint |
| API latency percentiles (p50/p95/p99) | Rolling list → sorted | `/metrics` endpoint |

### 7.2 Monitoring Architecture

```
Training events  → utils/metrics_logger.py → logs/metrics.jsonl  (JSONL append)
API requests     → middleware               → logs/metrics.jsonl  (JSONL append)
All ML runs      → core/orchestrator.py    → mlruns/             (MLflow artifact store)
Log rotation     → utils/logger.py         → logs/cloudautoml.log (RotatingFileHandler)
Live API stats   → api/prediction_server.py → GET /metrics (JSON)
```

**JSONL entry example:**
```json
{"event": "training_complete", "timestamp": "2026-04-16T16:30:00Z",
 "model_name": "XGBoost", "task_type": "classification",
 "metrics": {"accuracy": 0.9312, "f1_score": 0.9287},
 "train_time_s": 3.14, "peak_ram_mb": 187.4,
 "complexity": "medium", "hpo_enabled": true}
```

---

## 8. Risk Analysis (Review 3)

| # | Risk | Category | Probability | Impact | Mitigation |
|---|------|----------|-------------|--------|------------|
| R1 | SVM training time O(n²) — unusable on large datasets | Technical | High | High | SVM excluded for medium/large; PerformanceOptimiser warns |
| R2 | XGBoost memory spike on very wide datasets | Technical | Medium | Medium | Memory budget cap; PerformanceOptimiser flags overruns |
| R3 | StabilityPredictor ranking incorrect for novel data distributions | ML | Medium | Low | Non-binding; pipeline trains all models regardless of ranking |
| R4 | Optuna HPO timeout in CI (limited runner CPU) | ML / CI | Medium | Medium | CI integration test runs `--no-hpo`; HPO disabled for large datasets |
| R5 | Threading race condition in result collection | Technical | Low | High | `threading.Lock()` guards shared `_raw` list; tested in orchestrator tests |
| R6 | MLflow artifact logging fails on restricted environments | Deployment | Low | Low | Wrapped in try/except; pipeline continues without MLflow |
| R7 | Docker image size exceeds Render free tier RAM | Deployment | Medium | Medium | `python:3.11-slim` base + no-cache pip install; currently ~650 MB |
| R8 | Streamlit Cloud psutil restrictions crash resource manager | Deployment | High | Medium | All psutil calls wrapped in try/except with sensible fallback allocations |
| R9 | Feature schema mismatch at inference time | ML | Medium | High | `/predict` validates incoming columns against `deployment_meta.json` |
| R10 | No authentication on API endpoints | Security | High | High | Noted limitation; mitigation: add API key middleware before public deploy |

---

## 9. Deployment Strategy (Review 3)

### Local (Development)
```bash
# 1. Activate venv
venv\Scripts\activate

# 2. Install
pip install -r requirements.txt

# 3a. Streamlit UI
streamlit run app/streamlit_app.py           # http://localhost:8501

# 3b. CLI (headless)
python main.py                               # auto-generates demo dataset
python main.py --csv data/my.csv --target label

# 4. FastAPI inference server
uvicorn api.prediction_server:app --reload   # http://localhost:8000

# 5. MLflow dashboard
mlflow ui                                    # http://localhost:5000

# 6. Tests
pytest tests/ -v --tb=short
```

### Docker (Containerised)
```bash
docker build -t cloudautoml .
docker run -p 8501:8501 cloudautoml         # Streamlit
docker run -p 8000:8000 cloudautoml \
  uvicorn api.prediction_server:app --host 0.0.0.0 --port 8000
```

### Cloud (Streamlit Cloud + Render)
| Service | Component | Config |
|---------|-----------|--------|
| Streamlit Cloud | `app/streamlit_app.py` | Automatic from `.streamlit/config.toml` |
| Render (FastAPI) | `api/prediction_server.py` | `render.yaml` — start command: `uvicorn api.prediction_server:app --host 0.0.0.0 --port $PORT` |

---

## 10. Viva Preparation — Key Questions & Answers

**Q: Why ThreadPoolExecutor instead of ProcessPoolExecutor?**
> scikit-learn and XGBoost both release the GIL during their C-extension computations,
> making threads genuinely parallel for the CPU-bound training step while avoiding the
> pickling overhead and memory duplication of multi-processing.

**Q: Why is matplotlib called only on the main thread?**
> matplotlib is not thread-safe. Calling `plt.savefig()` from worker threads causes
> non-deterministic segfaults. All plot generation is deferred until
> `as_completed()` finishes so it always runs on the main thread.

**Q: What is the StabilityPredictor actually doing?**
> It's a meta-learning layer. A RandomForestClassifier is trained on 900 synthetic
> observations where each row represents a (dataset profile → best algorithm) mapping.
> At inference time, 7 meta-features are extracted from the incoming dataset
> (log₁₀(rows), feature count, numeric ratio, missing %, class count, complexity
> encoding, task flag) and the meta-RF returns per-model probability scores.
> This is directionally correct rather than precisely accurate — its value is in
> informing the orchestrator's log output and the user, not in hard-filtering models.

**Q: How does the PerformanceOptimiser work?**
> It uses empirically-grounded heuristics (not a trained ML model) to estimate per-model
> wall-clock training time as a function of base latency × complexity scale × dataset
> size factors. HPO is flagged as inadvisable if enabling it would multiply the
> parallel wall-time by more than 5×. This prevents runaway training on large datasets.

**Q: What happens if MLflow is unavailable?**
> Every MLflow call in the orchestrator is wrapped in a try/except block. If it
> fails, a warning is logged and the pipeline continues. No training result is lost.

**Q: How is the API secured?**
> Three layers: (1) CORS middleware controls origin access, (2) the middleware layer
> injects a UUID X-Request-ID for distributed tracing, (3) the /predict endpoint
> validates the incoming feature schema against the training-time schema stored in
> `deployment_meta.json` — rejecting mismatched columns with HTTP 400.

**Q: Why JSONL for monitoring instead of a database?**
> JSONL is append-only, human-readable, immediately queryable with standard tools
> (jq, pandas), and has zero operational overhead. For a production system at scale,
> the next step would be to ship these events to a time-series store (Prometheus,
> InfluxDB) using a sidecar exporter.

**Q: What is the parallelism speedup ratio?**
> `speedup = sum(individual model train times) / wall_clock_pipeline_time`.
> A value of 2.8× means that if you had run models sequentially, it would have taken
> 2.8× longer. This is printed in the CLI and logged to MLflow as a metric.

**Q: How was the stacking ensemble implemented?**
> `sklearn.ensemble.StackingClassifier` (or Regressor) is built from the top-N base
> learners ranked by CV mean score. A LogisticRegression or Ridge meta-learner is
> fitted on the out-of-fold predictions of the base models. This reduces variance
> by leveraging the diversity of different algorithm families.
