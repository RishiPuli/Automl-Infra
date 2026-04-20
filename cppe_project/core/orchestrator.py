"""
core/orchestrator.py  (v3)
--------------------------
CloudAutoML — Unified Training Pipeline

Pipeline stages
---------------
1.  Pre-select      StabilityPredictor meta-ML model ranking (non-critical)
2.  Preprocess      ColumnTransformer: StandardScaler + OHE
3.  HPO  (opt.)     Optuna TPE per model, runs inside worker thread
4.  Train           Parallel via ThreadPoolExecutor; PeakMemoryMonitor per run
5.  Cross-validate  k-fold CV on training split, per trained model
6.  Eval plots      Confusion matrix / regression scatter / feature importance
7.  SHAP  (opt.)    SHAP bar plots per model (main thread, matplotlib-safe)
8.  Ensemble (opt.) Stacking on top-N base learners
9.  MLflow          Parameters, metrics, artifacts (main thread)
10. Report  (opt.)  Multi-page PDF via matplotlib PdfPages
11. Artifact save   best_model.pkl + deployment_meta.json for inference API
"""

import os
import json
import time
import joblib
import threading
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

from core.resource_manager import PeakMemoryMonitor
from utils.logger import get_logger

log = get_logger("Orchestrator")

# Use absolute paths so cloud deployments write to correct dir
_HERE           = os.path.dirname(os.path.abspath(__file__))
_CPPE_ROOT      = os.path.abspath(os.path.join(_HERE, ".."))
MODELS_DIR      = os.path.join(_CPPE_ROOT, "models")
REPORTS_DIR     = os.path.join(_CPPE_ROOT, "reports")
EXPERIMENT_NAME = "CloudAutoML"
SAMPLE_LIMIT    = 5000

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── Preprocessing ──────────────────────────────────────────────────────────────

def build_preprocessor(num_cols, cat_cols):
    """Build a ColumnTransformer for numeric scaling and categorical encoding."""
    steps = []
    if num_cols:
        steps.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                            sparse_output=False), cat_cols))
    if not steps:
        steps.append(("num", StandardScaler(), slice(None)))
    return ColumnTransformer(transformers=steps, remainder="drop")


def preprocess(df, analysis, test_size=0.2, random_state=42):
    """
    Prepare data for training.

    Returns
    -------
    X_train, X_test, y_train, y_test,
    preprocessor, label_encoder, feature_cols
    """
    target   = analysis["target_column"]
    num_cols = analysis["numerical_cols"]
    cat_cols = analysis["categorical_cols"]
    task     = analysis["task_type"]

    if len(df) > SAMPLE_LIMIT:
        log.info("Sampling dataset from %d to %d rows", len(df), SAMPLE_LIMIT)
        df = df.sample(n=SAMPLE_LIMIT, random_state=random_state).reset_index(drop=True)

    feat_cols = [c for c in df.columns if c != target]
    X         = df[feat_cols].copy()
    y_raw     = df[target].copy()

    for c in num_cols:
        if c in X.columns:
            X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        if c in X.columns:
            mode = X[c].mode()
            X[c] = X[c].fillna(mode.iloc[0] if len(mode) else "unknown")

    le = None
    if task == "classification":
        le = LabelEncoder()
        y  = le.fit_transform(y_raw.astype(str))
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0).values.astype(float)

    prep = build_preprocessor(num_cols, cat_cols)

    stratify = None
    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        if np.all(counts >= 2):
            stratify = y

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=stratify,
    )
    X_train = prep.fit_transform(X_tr)
    X_test  = prep.transform(X_te)
    return X_train, X_test, y_tr, y_te, prep, le, feat_cols


def get_feature_names(prep, feat_cols):
    try:
        return list(prep.get_feature_names_out())
    except Exception:
        return feat_cols


# ── Model factory ──────────────────────────────────────────────────────────────

def build_default_model(name, n_jobs):
    """Instantiate a named model with sensible defaults."""
    registry = {
        "LogisticRegression":    lambda: LogisticRegression(
            C=1.0, solver="saga", max_iter=500, n_jobs=n_jobs, random_state=42),
        "RandomForest":          lambda: RandomForestClassifier(
            n_estimators=100, max_depth=8, n_jobs=n_jobs, random_state=42),
        "XGBoost":               lambda: XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            eval_metric="mlogloss", verbosity=0, n_jobs=n_jobs, random_state=42),
        "SVM":                   lambda: SVC(
            kernel="rbf", C=1.0, gamma="scale", random_state=42),
        "LinearRegression":      lambda: LinearRegression(n_jobs=n_jobs),
        "RandomForestRegressor": lambda: RandomForestRegressor(
            n_estimators=100, max_depth=8, n_jobs=n_jobs, random_state=42),
        "XGBoostRegressor":      lambda: XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            verbosity=0, n_jobs=n_jobs, random_state=42),
    }
    fn = registry.get(name)
    return fn() if fn else None


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, task):
    """Return task-appropriate evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score, f1_score,
        mean_squared_error, r2_score, mean_absolute_error,
    )
    if task == "classification":
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred,
                               average="weighted", zero_division=0)), 4),
        }
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": round(rmse, 4),
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
    }


def compute_cv(model, X_train, y_train, task, n_splits=3):
    """k-fold cross-validation on the training split. Returns (mean, std)."""
    scoring = "accuracy" if task == "classification" else "r2"
    cv = (StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
          if task == "classification"
          else KFold(n_splits=n_splits, shuffle=True, random_state=42))
    try:
        scores = cross_val_score(model, X_train, y_train,
                                  cv=cv, scoring=scoring, n_jobs=1)
        return round(float(scores.mean()), 4), round(float(scores.std()), 4)
    except Exception:
        return 0.0, 0.0


# ── Worker (thread-safe) ───────────────────────────────────────────────────────

def _train_worker(name, X_train, X_test, y_train, y_test,
                  task, complexity, n_jobs, emit, enable_hpo):
    """
    Trains one model.  Optionally runs Optuna HPO before fitting.

    Thread-safe contract:
      - No matplotlib calls (not thread-safe)
      - No MLflow calls (not thread-safe)
      - Appends to result dict; uses only local variables
    """
    emit(f"[START] {name}")
    best_params = {}

    try:
        # ── Optional HPO ──────────────────────────────────────────────────────
        if enable_hpo:
            try:
                from core.hyperparameter_tuner import (
                    tune_hyperparameters, build_tuned_model,
                )
                n_trials = {"small": 30, "medium": 20, "large": 10}.get(complexity, 20)
                emit(f"  [HPO]  {name} — {n_trials} Optuna TPE trials")
                best_params = tune_hyperparameters(
                    name, task, X_train, y_train, complexity, emit=emit,
                )
                model = build_tuned_model(name, task, best_params, n_jobs)
                if model is None:
                    model       = build_default_model(name, n_jobs)
                    best_params = {}
            except Exception as hpo_err:
                emit(f"  [HPO]  {name} fallback to defaults — {hpo_err}")
                model       = build_default_model(name, n_jobs)
                best_params = {}
        else:
            model = build_default_model(name, n_jobs)

        if model is None:
            emit(f"[SKIP]  {name} — not in model registry")
            return None

        # ── Train + RAM tracking ───────────────────────────────────────────────
        monitor = PeakMemoryMonitor()
        monitor.start()
        t0         = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = round(time.perf_counter() - t0, 3)
        monitor.stop()
        peak_ram   = monitor.peak_mb

        # ── Evaluation ────────────────────────────────────────────────────────
        y_pred           = model.predict(X_test)
        metrics          = compute_metrics(y_test, y_pred, task)
        cv_mean, cv_std  = compute_cv(model, X_train, y_train, task)

        # ── Persist model ─────────────────────────────────────────────────────
        model_path = os.path.join(MODELS_DIR,
                                   f"{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)

        primary = "accuracy" if task == "classification" else "r2"
        emit(
            f"[DONE]  {name} | {primary}={metrics.get(primary, 0):.4f} "
            f"| cv={cv_mean:.4f}±{cv_std:.4f} | {train_time}s | {peak_ram:.0f}MB"
        )

        return {
            "name":        name,
            "model":       model,
            "metrics":     metrics,
            "train_time":  train_time,
            "peak_ram_mb": peak_ram,
            "n_jobs":      n_jobs,
            "model_path":  model_path,
            "hpo_params":  best_params,
            "cv_mean":     cv_mean,
            "cv_std":      cv_std,
            "plot_paths":  [],
        }

    except Exception as exc:
        emit(f"[FAIL]  {name} — {exc}")
        log.exception("Error training %s", name)
        return None


# ── MLflow logging (main thread only) ─────────────────────────────────────────

def _log_to_mlflow(results, task, complexity):
    """Log all training results to MLflow. Must run on the main thread.
    Uses lazy import so mlflow failures never block the orchestrator import."""
    try:
        import mlflow          # lazy import — safe if mlflow not installed
        import mlflow.sklearn
    except ImportError:
        log.warning("mlflow not installed — skipping experiment tracking.")
        return

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mlruns_path  = os.path.join(project_root, "mlruns")
        os.makedirs(mlruns_path, exist_ok=True)
        mlflow.set_tracking_uri(f"file:///{mlruns_path}")
        mlflow.set_experiment(EXPERIMENT_NAME)

        for r in results:
            with mlflow.start_run(run_name=r["name"]):
                mlflow.log_param("model",       r["name"])
                mlflow.log_param("task",        task)
                mlflow.log_param("complexity",  complexity)
                mlflow.log_param("n_jobs",      r.get("n_jobs", 1))
                mlflow.log_param("hpo_enabled", bool(r.get("hpo_params")))
                mlflow.log_metric("train_time", r["train_time"])
                mlflow.log_metric("peak_ram_mb", r["peak_ram_mb"])
                mlflow.log_metric("cv_mean",    r.get("cv_mean", 0.0))
                mlflow.log_metric("cv_std",     r.get("cv_std",  0.0))
                for k, v in r["metrics"].items():
                    mlflow.log_metric(k, v)
                for p in r.get("plot_paths", []):
                    if os.path.isfile(p):
                        mlflow.log_artifact(p)
                if os.path.isfile(r.get("model_path", "")):
                    mlflow.log_artifact(r["model_path"])
                try:
                    mlflow.sklearn.log_model(r["model"], r["name"])
                except Exception:
                    pass

        log.info("MLflow: logged %d runs -> %s", len(results), mlruns_path)

    except Exception as exc:
        log.exception("MLflow logging failed: %s", exc)


# ── Deployment artifact export ─────────────────────────────────────────────────

def _save_deployment_meta(best, analysis, prep, le, feat_names):
    """
    Persist:
      - models/best_model.pkl      (already saved above)
      - models/preprocessor.pkl
      - models/label_encoder.pkl   (classification only)
      - models/deployment_meta.json

    deployment_meta.json is read by api/prediction_server.py at startup.
    """
    if not best:
        return
    try:
        prep_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
        joblib.dump(prep, prep_path)

        le_path = ""
        if le is not None:
            le_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
            joblib.dump(le, le_path)

        meta = {
            "model_name":         best["name"],
            "model_path":         os.path.join(MODELS_DIR, "best_model.pkl"),
            "preprocessor_path":  prep_path,
            "label_encoder_path": le_path,
            "task_type":          analysis.get("task_type", ""),
            "feature_names":      feat_names,
            "n_features":         analysis.get("features", 0),
            "metrics":            best.get("metrics", {}),
            "cv_mean":            best.get("cv_mean", 0.0),
            "hpo_params":         best.get("hpo_params", {}),
        }
        meta_path = os.path.join(MODELS_DIR, "deployment_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        log.info("Deployment meta saved -> %s", meta_path)
    except Exception as exc:
        log.warning("Could not save deployment meta: %s", exc)


# ── Main pipeline entry point ──────────────────────────────────────────────────

def run_pipeline(
    df, analysis, model_names, allocation,
    test_size=0.2, log_fn=None, progress_fn=None,
    enable_hpo=True, enable_shap=True,
    enable_ensemble=True, enable_report=True,
):
    """
    End-to-end AutoML pipeline.

    Parameters
    ----------
    df              : raw DataFrame
    analysis        : output of analyze_dataset()
    model_names     : list of model name strings
    allocation      : output of allocate_resources()
    test_size       : test split fraction
    log_fn          : thread-safe callable(str) for log messages
    progress_fn     : callable(float, str) for progress updates (main thread)
    enable_hpo      : run Optuna hyperparameter optimisation
    enable_shap     : run SHAP model explanations
    enable_ensemble : build stacking ensemble from top-N models
    enable_report   : generate multi-page PDF report

    Returns
    -------
    dict with keys:
        results          list of per-model result dicts
        best             best result dict (alias: best_model)
        best_model       same as 'best' (main.py compatibility)
        summary_df       DataFrame summary of all results
        label_enc        fitted LabelEncoder (None for regression)
        prep             fitted ColumnTransformer
        speedup          float  parallel speedup ratio
        speedup_info     dict   wall + sequential estimate
        pdf_report_path  str    path to generated PDF (empty if disabled)
    """
    def emit(msg):
        log.info(msg)
        if log_fn:
            log_fn(msg)

    task       = analysis["task_type"]
    complexity = analysis["complexity"]
    n_jobs     = allocation["cpu_allocated"]
    n_models   = len(model_names)

    # ── Stage 1: Meta-learning pre-selection ──────────────────────────────────
    try:
        from core.stability_predictor import StabilityPredictor
        ranking = StabilityPredictor().fit().predict_ranking(analysis)
        emit("StabilityPredictor ranking: " +
             " > ".join(f"{n}({s:.2f})" for n, s in ranking[:4]))
    except Exception:
        pass  # non-critical; pipeline continues regardless

    # ── Stage 1.5: Performance optimiser — strategy recommendation ────────────
    perf_plan = {}
    try:
        from core.performance_optimizer import PerformanceOptimiser
        perf_plan = PerformanceOptimiser().fit().recommend(
            analysis, allocation, model_names,
            hpo_requested=enable_hpo,
        )
        emit(
            f"PerformanceOptimiser: strategy={perf_plan['strategy']} "
            f"workers={perf_plan['n_workers']} "
            f"est_parallel={perf_plan['total_est_par']:.0f}s "
            f"HPO={'recommended' if perf_plan['hpo_recommended'] else 'auto-disabled'}"
        )
        if perf_plan.get("memory_warnings"):
            for w in perf_plan["memory_warnings"]:
                emit(f"  [MEM WARN] {w}")
        # Auto-disable HPO if optimiser says it's inadvisable
        if enable_hpo and not perf_plan["hpo_recommended"]:
            emit("  [PERF] HPO auto-disabled: estimated overhead too high for dataset size")
            enable_hpo = False
    except Exception as perf_err:
        emit(f"[WARN] PerformanceOptimiser skipped: {perf_err}")

    # ── Stage 2: Preprocess ───────────────────────────────────────────────────
    emit("Preprocessing data ...")
    X_tr, X_te, y_tr, y_te, prep, le, feat_cols = preprocess(
        df, analysis, test_size=test_size,
    )
    feat_names = get_feature_names(prep, feat_cols)
    emit(
        f"Train: {X_tr.shape[0]} samples | "
        f"Test: {X_te.shape[0]} samples | "
        f"Features: {X_tr.shape[1]}"
    )

    # ── Stage 3+4: HPO + Parallel Training ────────────────────────────────────
    _raw       = []
    _lock      = threading.Lock()
    completed  = 0
    wall_start = time.perf_counter()

    hpo_label = "ON" if enable_hpo else "OFF"
    emit(f"Parallel training: {n_models} models x {n_jobs} workers (HPO={hpo_label})")

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {
            pool.submit(
                _train_worker,
                name, X_tr, X_te, y_tr, y_te,
                task, complexity, n_jobs, emit, enable_hpo,
            ): name
            for name in model_names
        }
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
                if result is not None:
                    with _lock:
                        _raw.append(result)
            except Exception as exc:
                emit(f"Thread error: {exc}")
            if progress_fn:
                progress_fn(
                    completed / n_models * 0.65,
                    f"{completed}/{n_models} models complete",
                )

    wall_train = time.perf_counter() - wall_start
    emit(f"Training done: {len(_raw)}/{n_models} succeeded in {wall_train:.1f}s")

    # ── Stage 5: Evaluator plots (main thread — matplotlib-safe) ──────────────
    if progress_fn:
        progress_fn(0.70, "Generating evaluation plots ...")
    try:
        from core.evaluator import (
            plot_model_comparison, plot_training_time,
            plot_confusion_matrix, plot_feature_importance,
            plot_regression_scatter,
        )
        plot_model_comparison(_raw, task)
        plot_training_time(_raw)
        for r in _raw:
            paths = []
            if task == "classification":
                p = plot_confusion_matrix(
                    r["model"], X_te, y_te, r["name"],
                    class_labels=(list(le.classes_) if le else None),
                )
                if p:
                    paths.append(p)
            else:
                p = plot_regression_scatter(r["model"], X_te, y_te, r["name"])
                if p:
                    paths.append(p)
            p2 = plot_feature_importance(r["model"], feat_names, r["name"])
            if p2:
                paths.append(p2)
            r["plot_paths"] = paths
    except Exception as exc:
        emit(f"[WARN] Eval plots: {exc}")

    # ── Stage 6: SHAP explanations (main thread — matplotlib-safe) ────────────
    if enable_shap:
        if progress_fn:
            progress_fn(0.76, "Running SHAP explanations ...")
        try:
            from core.explainer import explain_model
            emit("SHAP feature importance ...")
            for r in _raw:
                shap_paths = explain_model(
                    r["model"], X_tr, X_te, feat_names, r["name"], task,
                )
                r["plot_paths"].extend(shap_paths)
        except Exception as exc:
            emit(f"[WARN] SHAP skipped: {exc}")

    # ── Stage 7: Stacking Ensemble ────────────────────────────────────────────
    if enable_ensemble and len(_raw) >= 2:
        if progress_fn:
            progress_fn(0.82, "Building stacking ensemble ...")
        try:
            from core.ensemble_builder import build_stacking_ensemble
            emit("Building stacking ensemble ...")
            ens = build_stacking_ensemble(
                _raw, X_tr, y_tr, X_te, y_te, task, emit=emit,
            )
            if ens is not None:
                _raw.append(ens)
                emit(f"Ensemble metrics: {ens['metrics']}")
        except Exception as exc:
            emit(f"[WARN] Ensemble: {exc}")

    # ── Stage 8: MLflow ───────────────────────────────────────────────────────
    if progress_fn:
        progress_fn(0.88, "Logging to MLflow ...")
    if _raw:
        emit("Logging results to MLflow ...")
        _log_to_mlflow(_raw, task, complexity)

    # ── Stage 9: Select best ──────────────────────────────────────────────────
    primary = "accuracy" if task == "classification" else "r2"
    valid   = [r for r in _raw if primary in r.get("metrics", {})]
    best    = max(valid, key=lambda r: r["metrics"][primary]) if valid else None

    if best:
        emit(
            f"Best model: {best['name']} "
            f"({primary}={best['metrics'][primary]:.4f} | "
            f"cv={best.get('cv_mean', 0):.4f}+/-{best.get('cv_std', 0):.4f})"
        )
        best_path = os.path.join(MODELS_DIR, "best_model.pkl")
        joblib.dump(best["model"], best_path)
        best["model_path"] = best_path
        _save_deployment_meta(best, analysis, prep, le, feat_names)

    # ── Stage 10: Speedup calculation ─────────────────────────────────────────
    wall_total   = round(time.perf_counter() - wall_start, 2)
    seq_estimate = round(sum(r["train_time"] for r in _raw), 2)
    speedup_val  = round(seq_estimate / max(wall_total, 0.001), 2)
    speedup_info = {
        "wall":                wall_total,
        "sequential_estimate": seq_estimate,
        "speedup":             speedup_val,
    }

    # ── Stage 11: PDF Report ──────────────────────────────────────────────────
    pdf_path = ""
    if enable_report and _raw:
        if progress_fn:
            progress_fn(0.94, "Generating PDF report ...")
        try:
            from core.report_generator import generate_report
            pdf_path = generate_report(
                analysis=analysis,
                allocation=allocation,
                results=_raw,
                best=best or {},
                task_type=task,
                speedup_info=speedup_info,
            )
            if pdf_path:
                emit(f"PDF report -> {pdf_path}")
        except Exception as exc:
            emit(f"[WARN] Report failed: {exc}")

    if progress_fn:
        progress_fn(1.0, "Pipeline complete")

    # ── Metrics logging ───────────────────────────────────────────────────────
    try:
        from utils.metrics_logger import log_training_result
        for r in _raw:
            log_training_result(
                model_name=r["name"],
                task_type=task,
                metrics=r["metrics"],
                train_time=r["train_time"],
                peak_ram_mb=r["peak_ram_mb"],
                complexity=complexity,
                hpo_enabled=bool(r.get("hpo_params")),
            )
    except Exception:
        pass

    # ── Summary DataFrame ──────────────────────────────────────────────────────
    rows = []
    for r in _raw:
        row = {
            "Model":          r["name"],
            "Train Time (s)": r["train_time"],
            "Peak RAM (MB)":  round(r["peak_ram_mb"], 1),
            "CV Mean":        r.get("cv_mean", 0),
            "CV Std":         r.get("cv_std",  0),
            "Workers":        r.get("n_jobs",  1),
        }
        row.update(r["metrics"])
        rows.append(row)

    return {
        "results":         _raw,
        "best":            best,        # streamlit_app.py uses "best"
        "best_model":      best,        # main.py uses "best_model"
        "summary_df":      pd.DataFrame(rows),
        "label_enc":       le,
        "prep":            prep,
        "speedup":         speedup_val,  # float  — for main.py print
        "speedup_info":    speedup_info, # dict   — for report_generator
        "pdf_report_path": pdf_path,
    }
