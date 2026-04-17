"""
core/evaluator.py
------------------
Metrics + Visualisation Engine for CloudAutoML.

Classification metrics: Accuracy, Precision, Recall, F1
Regression metrics:     RMSE, MAE, R²

Plots:
  - Model comparison bar chart
  - Confusion matrix heatmap (only if classes < 40)
  - Feature importance plot
  - Training time comparison
  - Regression actual vs predicted scatter
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)
from utils.logger import get_logger

log = get_logger("Evaluator")

REPORTS_DIR    = "reports"
MAX_CM_CLASSES = 40  # confusion matrix capped to this many classes

os.makedirs(REPORTS_DIR, exist_ok=True)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B2", "#937860", "#DA8BC3", "#64B5CD"]

plt.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D27",
    "axes.edgecolor":   "#444",
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "grid.color":       "#2d3748",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _savefig(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(REPORTS_DIR, filename)
    try:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    except Exception as e:
        log.warning("Could not save plot %s: %s", filename, e)
        path = ""
    plt.close(fig)
    return path


# ─── Metric computation ───────────────────────────────────────────────────────

def evaluate_classification(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    avg = "weighted"
    m = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred,    average=avg, zero_division=0)), 4),
        "f1_score":  round(float(f1_score(y_test, y_pred,        average=avg, zero_division=0)), 4),
    }
    log.info("[%s] Accuracy=%.4f  F1=%.4f", model_name, m["accuracy"], m["f1_score"])
    return m


def evaluate_regression(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    m = {
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "mae":  round(float(mean_absolute_error(y_test, y_pred)),          4),
        "r2":   round(float(r2_score(y_test, y_pred)),                     4),
    }
    log.info("[%s] RMSE=%.4f  MAE=%.4f  R²=%.4f", model_name, m["rmse"], m["mae"], m["r2"])
    return m


def evaluate_model(model, X_test, y_test, model_name: str, task_type: str) -> dict:
    if task_type == "classification":
        return evaluate_classification(model, X_test, y_test, model_name)
    return evaluate_regression(model, X_test, y_test, model_name)


# ─── Plot generators ──────────────────────────────────────────────────────────

def plot_model_comparison(results: list, task_type: str) -> str:
    """Bar chart of primary metric across all trained models."""
    primary = "accuracy" if task_type == "classification" else "r2"
    label   = "Accuracy"  if task_type == "classification" else "R² Score"

    valid  = [r for r in results if primary in r.get("metrics", {})]
    if not valid:
        return ""

    names  = [r["name"]             for r in valid]
    values = [r["metrics"][primary] for r in valid]

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.8), 5))
    bars = ax.bar(names, values, color=PALETTE[:len(names)], edgecolor="#222", width=0.55)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="white")

    y_max = max(values) * 1.25 if max(values) > 0 else 1.0
    ax.set_ylim(0, min(1.2, y_max))
    ax.set_title(f"Model Comparison — {label}", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Model"); ax.set_ylabel(label)
    ax.yaxis.grid(True); ax.set_axisbelow(True)

    return _savefig(fig, "metrics_comparison.png")


def plot_training_time(results: list) -> str:
    """Bar chart: training wall-clock time per model."""
    valid = [r for r in results if "train_time" in r]
    if not valid:
        return ""

    names = [r["name"]       for r in valid]
    times = [r["train_time"] for r in valid]

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.8), 4))
    ax.bar(names, times, color=PALETTE[1], edgecolor="#222", width=0.55)
    ax.set_title("Training Time Comparison (seconds)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Model"); ax.set_ylabel("Seconds")
    ax.yaxis.grid(True); ax.set_axisbelow(True)

    return _savefig(fig, "training_time.png")


def plot_confusion_matrix(model, X_test, y_test,
                          model_name: str, class_labels=None) -> str:
    """
    Confusion matrix heatmap.
    Automatically skipped if n_classes > MAX_CM_CLASSES
    to prevent matplotlib 2^16 image-size crash.
    """
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    n_cls  = cm.shape[0]

    if n_cls > MAX_CM_CLASSES:
        log.warning("[%s] CM skipped: %d classes > limit %d.", model_name, n_cls, MAX_CM_CLASSES)
        return ""

    annotate = n_cls <= 15
    sz       = max(5, n_cls * 0.5)
    fig, ax  = plt.subplots(figsize=(sz, sz * 0.85))

    sns.heatmap(
        cm, annot=annotate, fmt="d", cmap="Blues",
        xticklabels=class_labels if (class_labels and len(class_labels) == n_cls) else "auto",
        yticklabels=class_labels if (class_labels and len(class_labels) == n_cls) else "auto",
        ax=ax, linewidths=0.3 if n_cls <= 20 else 0,
    )
    ax.set_title(f"Confusion Matrix ({n_cls} classes) — {model_name}",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    return _savefig(fig, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")


def plot_feature_importance(model, feature_names: list,
                            model_name: str, top_n: int = 20) -> str:
    """Horizontal bar chart of feature importances (tree-based + linear models)."""
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        c = model.coef_
        importances = np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c)

    if importances is None or len(importances) == 0:
        return ""

    names = (feature_names if len(feature_names) == len(importances)
             else [f"f{i}" for i in range(len(importances))])

    df_imp = (pd.DataFrame({"feature": names, "importance": importances})
              .sort_values("importance", ascending=False)
              .head(top_n)
              .reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(9, max(4, len(df_imp) * 0.45)))
    ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1],
            color=PALETTE[0], edgecolor="#222")
    ax.set_title(f"Feature Importance (Top {len(df_imp)}) — {model_name}",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Importance"); ax.xaxis.grid(True); ax.set_axisbelow(True)

    return _savefig(fig, f"feature_importance_{model_name.lower().replace(' ','_')}.png")


def plot_regression_scatter(model, X_test, y_test, model_name: str) -> str:
    """Actual vs Predicted scatter plot for regression models."""
    y_pred = model.predict(X_test)
    _min   = min(float(y_test.min()), float(y_pred.min()))
    _max   = max(float(y_test.max()), float(y_pred.max()))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.35, s=18, color=PALETTE[0])
    ax.plot([_min, _max], [_min, _max], "r--", lw=1.5, label="Perfect fit")
    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.legend(fontsize=9)

    return _savefig(fig, f"regression_scatter_{model_name.lower().replace(' ','_')}.png")


# ─── Best model ───────────────────────────────────────────────────────────────

def select_best_model(results: list, task_type: str) -> dict:
    primary = "accuracy" if task_type == "classification" else "r2"
    valid   = [r for r in results if primary in r.get("metrics", {})]
    if not valid:
        return {}
    best = max(valid, key=lambda r: r["metrics"][primary])
    log.info("Best model: [%s] %s=%.4f", best["name"], primary, best["metrics"][primary])
    return best
