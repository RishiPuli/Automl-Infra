"""
core/explainer.py
------------------
SHAP-based Model Explainability for CloudAutoML.

- TreeExplainer  → Random Forest, XGBoost (fast)
- LinearExplainer → Logistic/Linear Regression (fast)
- KernelExplainer → SVM and any other model (slow fallback)

Multi-class safe: averages per-class SHAP values.
Outputs dark-themed bar summary plots to reports/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.logger import get_logger

log = get_logger("Explainer")
REPORTS_DIR = "reports"

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    log.warning("SHAP not installed — explainability disabled. pip install shap")


def explain_model(
    model, X_train: np.ndarray, X_test: np.ndarray,
    feature_names: list, model_name: str,
    task_type: str, max_samples: int = 150,
) -> list:
    """
    Generate SHAP explanation plots for a model.
    Returns list of saved plot paths (may be empty on failure).
    """
    if not SHAP_AVAILABLE:
        return []

    os.makedirs(REPORTS_DIR, exist_ok=True)
    paths = []
    slug  = model_name.lower().replace(" ", "_")

    n       = min(max_samples, X_test.shape[0])
    X_samp  = X_test[:n]
    fn      = feature_names if len(feature_names) == X_samp.shape[1] else None

    try:
        # ── Choose explainer ──────────────────────────────────────────────────
        if hasattr(model, "feature_importances_"):          # tree-based
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_samp)
        elif hasattr(model, "coef_"):                       # linear
            explainer   = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_samp)
        else:                                               # kernel (SVM etc.)
            bg          = shap.sample(X_train, min(50, X_train.shape[0]))
            explainer   = shap.KernelExplainer(model.predict, bg)
            shap_values = explainer.shap_values(X_samp, nsamples=50)

        # ── Handle multi-class (list of arrays) ───────────────────────────────
        sv = shap_values
        if isinstance(shap_values, list):
            # Average absolute SHAP values across all classes
            sv = np.array([np.abs(s) for s in shap_values]).mean(axis=0)

        # ── Mean absolute importance per feature ──────────────────────────────
        mean_abs = np.abs(sv).mean(axis=0)
        top_n    = min(20, len(mean_abs))
        indices  = np.argsort(mean_abs)[::-1][:top_n]

        labels = ([fn[i] for i in indices] if fn
                  else [f"f{i}" for i in indices])
        values = mean_abs[indices]

        # ── Dark-themed horizontal bar chart ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.42)),
                               facecolor="#0F1117")
        ax.set_facecolor("#1A1D27")
        ax.barh(labels[::-1], values[::-1], color="#00d2ff", edgecolor="#0F1117")
        ax.set_xlabel("Mean |SHAP value|", color="white")
        ax.set_title(f"SHAP Feature Importance — {model_name}",
                     color="white", fontweight="bold", fontsize=12, pad=10)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.xaxis.grid(True, color="#2d3748", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        path = os.path.join(REPORTS_DIR, f"shap_{slug}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="#0F1117")
        plt.close(fig)
        paths.append(path)
        log.info("SHAP plot saved → %s", path)

    except Exception as e:
        log.warning("SHAP failed for %s: %s", model_name, e)

    return paths
