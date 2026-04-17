"""
core/ensemble_builder.py
-------------------------
Auto-Stacking Ensemble Builder for CloudAutoML.

Takes the top-N trained models as base learners, generates
out-of-fold (OOF) predictions via cross-validation, then
trains a meta-learner on those OOF features.

Result dict is compatible with the main pipeline so the
ensemble appears seamlessly in all UI tables and charts.
"""

import os
import joblib
import numpy as np
from typing import Optional
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from utils.logger import get_logger
from core.resource_manager import measure_training

log      = get_logger("Ensemble")
MODELS_DIR = "models"


def build_stacking_ensemble(
    results: list,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    task_type: str,
    top_n: int = 3,
    emit=None,
) -> Optional[dict]:
    """
    Build a stacked ensemble from the top-N models.

    Returns a result dict or None if not enough models.
    """
    def _emit(msg):
        log.info(msg)
        if emit:
            emit(msg)

    primary = "accuracy" if task_type == "classification" else "r2"
    ranked  = sorted(
        [r for r in results if primary in r.get("metrics", {})],
        key=lambda r: r["metrics"][primary], reverse=True,
    )
    base = ranked[:min(top_n, len(ranked))]

    if len(base) < 2:
        _emit("⚠️ Stacking requires ≥ 2 models — skipped.")
        return None

    base_models = [r["model"] for r in base]
    base_names  = [r["name"]  for r in base]
    _emit(f"\n🔗 Building Stacking Ensemble from: {', '.join(base_names)}")

    cv = (StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
          if task_type == "classification"
          else KFold(n_splits=5, shuffle=True, random_state=42))

    # ── Out-of-fold predictions ────────────────────────────────────────────────
    oof_train = np.zeros((X_train.shape[0], len(base_models)))
    oof_test  = np.zeros((X_test.shape[0],  len(base_models)))

    for i, (model, name) in enumerate(zip(base_models, base_names)):
        try:
            method = "predict_proba" if (task_type == "classification"
                                         and hasattr(model, "predict_proba")) else "predict"
            if method == "predict_proba":
                oof_p = cross_val_predict(model, X_train, y_train, cv=cv, method=method)
                # Use the argmax as a single column
                oof_train[:, i] = np.argmax(oof_p, axis=1)
            else:
                oof_train[:, i] = cross_val_predict(model, X_train, y_train, cv=cv)
            oof_test[:, i] = model.predict(X_test)
        except Exception as e:
            _emit(f"   ⚠️ OOF failed for {name}: {e} — using direct predict")
            oof_train[:, i] = model.predict(X_train)
            oof_test[:, i]  = model.predict(X_test)

    # ── Meta-learner ──────────────────────────────────────────────────────────
    meta = (LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
            if task_type == "classification"
            else Ridge(alpha=1.0))

    train_time, peak_ram = measure_training(meta, oof_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    from core.evaluator import evaluate_model
    metrics = evaluate_model(meta, oof_test, y_test, "StackingEnsemble", task_type)
    score   = metrics.get(primary, 0)
    _emit(f"   ✅ StackingEnsemble: {primary}={score:.4f} | time={train_time:.2f}s")

    # ── Save ──────────────────────────────────────────────────────────────────
    path = os.path.join(MODELS_DIR, "stacking_ensemble.pkl")
    joblib.dump({"meta": meta, "base_models": base_models,
                 "base_names": base_names}, path)

    return {
        "name":        "StackingEnsemble",
        "model":       meta,
        "metrics":     metrics,
        "train_time":  round(train_time, 3),
        "wall_start":  0.0,
        "wall_end":    round(train_time, 3),
        "peak_ram_mb": peak_ram,
        "cpu_cores":   1,
        "model_path":  path,
        "plot_paths":  [],
        "run_id":      "ensemble",
        "hpo_params":  {},
        "cv_mean":     score,
        "cv_std":      0.0,
        "is_ensemble": True,
        "base_models": base_names,
    }
