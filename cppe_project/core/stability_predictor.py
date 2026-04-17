"""
core/stability_predictor.py
-----------------------------
Meta-learning module that predicts the most suitable model configuration
given an incoming dataset profile, acting as an intelligent pre-selection
layer before Optuna HPO runs.

Approach
--------
A RandomForestClassifier is trained on a synthetic meta-dataset whose rows
represent (dataset_profile_vector -> best_model_label) pairs.  The synthetic
observations are constructed using empirically grounded rules about the
relationship between dataset characteristics and algorithm performance:

  LogisticRegression  performs best on small, low-dimensional, linearly
                      separable datasets with few classes.
  RandomForest        favours medium-sized datasets with moderate feature
                      counts and mixed numerical/categorical inputs.
  XGBoost             excels on medium-to-large datasets with many features
                      and some categorical content.
  SVM                 is well-suited to tiny, all-numeric, low-dimensional
                      datasets where kernel tricks provide a clear advantage.

  LinearRegression      for low-noise, low-dimension regression.
  RandomForestRegressor for medium regression tasks with nonlinearity.
  XGBoostRegressor      for larger regression tasks.

At inference time the meta-features of the incoming dataset are extracted
and passed through the classifier to produce per-model probability scores.
These scores are returned as a ranked list and logged by the orchestrator.

Usage
-----
    from core.stability_predictor import StabilityPredictor

    sp      = StabilityPredictor().fit()
    ranking = sp.predict_ranking(analysis)   # analysis from analyze_dataset()
    # [("XGBoost", 0.42), ("RandomForest", 0.35), ...]

    # Or get a filtered name list directly:
    top3 = sp.recommend_models(analysis, top_n=3)
"""

import numpy as np
from utils.logger import get_logger

log = get_logger("StabilityPredictor")

# ── Label maps ────────────────────────────────────────────────────────────────

_CLF_LABELS = {0: "LogisticRegression", 1: "RandomForest",
               2: "XGBoost",            3: "SVM"}
_REG_LABELS = {0: "LinearRegression",   1: "RandomForestRegressor",
               2: "XGBoostRegressor"}


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_meta_features(analysis: dict) -> np.ndarray:
    """
    Convert a dataset analysis dict into a fixed-length feature vector.

    Dimensions (7)
    --------------
    0  log10(n_rows)            — dataset scale
    1  n_features (clipped 200) — dimensionality
    2  numerical_ratio (0..1)   — proportion of numeric features
    3  missing_pct (0..100)     — data quality
    4  n_classes (reg → 1)      — target cardinality
    5  complexity_encoded        — small=0 / medium=1 / large=2
    6  is_classification (0/1)  — task flag
    """
    n_rows    = max(analysis.get("rows", 1), 1)
    nf        = analysis.get("features", 1)
    num_cols  = len(analysis.get("numerical_cols", []))
    cat_cols  = len(analysis.get("categorical_cols", []))
    total_f   = max(num_cols + cat_cols, 1)
    num_ratio = num_cols / total_f
    miss_pct  = analysis.get("missing_pct", 0.0)
    n_cls     = analysis.get("n_classes", 1) or 1
    cplx_map  = {"small": 0, "medium": 1, "large": 2}
    cplx      = cplx_map.get(analysis.get("complexity", "medium"), 1)
    is_clf    = 1 if analysis.get("task_type") == "classification" else 0

    return np.array([
        np.log10(n_rows),
        min(nf, 200),
        num_ratio,
        miss_pct,
        min(n_cls, 50),
        cplx,
        is_clf,
    ], dtype=float)


# ── Synthetic meta-dataset ────────────────────────────────────────────────────

def _make_clf_meta_dataset(n_per_class: int = 150):
    """
    Generate (X_meta, y_meta) for classification model selection.
    Each row is a jittered prototype representing a typical dataset profile
    where the labelled model historically performs best.
    """
    rng  = np.random.default_rng(42)
    rows, labels = [], []

    def _add(prototype, label, n, jitter=0.12):
        base  = np.array(prototype, dtype=float)
        noise = rng.normal(0, jitter, size=(n, len(base)))
        for r in np.clip(base + noise, 0, None):
            rows.append(r)
            labels.append(label)

    n = n_per_class

    # 0 = LogisticRegression — small / few features / mostly numeric / few classes
    _add([2.5,  8,  0.95, 0.5, 2, 0, 1], 0, n,      jitter=0.12)
    _add([3.0, 12,  0.90, 1.0, 3, 0, 1], 0, n // 2, jitter=0.10)

    # 1 = RandomForest — medium / moderate features / mixed types
    _add([3.5, 20,  0.70, 3.0, 4, 1, 1], 1, n,      jitter=0.18)
    _add([3.8, 30,  0.60, 5.0, 5, 1, 1], 1, n // 2, jitter=0.14)

    # 2 = XGBoost — medium-large / many features / some categoricals
    _add([4.0, 40,  0.60, 5.0, 6, 2, 1], 2, n,      jitter=0.20)
    _add([4.2, 60,  0.50, 8.0, 8, 2, 1], 2, n // 2, jitter=0.16)

    # 3 = SVM — tiny / all numeric / low dimensionality
    _add([2.0,  6,  1.00, 0.0, 2, 0, 1], 3, n,      jitter=0.08)
    _add([2.3,  8,  0.95, 0.5, 2, 0, 1], 3, n // 2, jitter=0.07)

    return np.vstack(rows), np.array(labels, dtype=int)


def _make_reg_meta_dataset(n_per_class: int = 150):
    """Generate (X_meta, y_meta) for regression model selection."""
    rng  = np.random.default_rng(7)
    rows, labels = [], []

    def _add(prototype, label, n, jitter=0.12):
        base  = np.array(prototype, dtype=float)
        noise = rng.normal(0, jitter, size=(n, len(base)))
        for r in np.clip(base + noise, 0, None):
            rows.append(r)
            labels.append(label)

    n = n_per_class

    # 0 = LinearRegression — small dataset, many numeric features, low noise
    _add([2.5,  8, 0.95, 0.0, 1, 0, 0], 0, n,      jitter=0.10)
    _add([3.0, 12, 0.90, 1.0, 1, 0, 0], 0, n // 2, jitter=0.08)

    # 1 = RandomForestRegressor — medium, mixed types, moderate noise
    _add([3.5, 20, 0.70, 3.0, 1, 1, 0], 1, n,      jitter=0.16)
    _add([3.8, 30, 0.60, 5.0, 1, 1, 0], 1, n // 2, jitter=0.14)

    # 2 = XGBoostRegressor — large, many features, some missing
    _add([4.0, 40, 0.60, 6.0, 1, 2, 0], 2, n,      jitter=0.18)
    _add([4.2, 60, 0.50, 9.0, 1, 2, 0], 2, n // 2, jitter=0.15)

    return np.vstack(rows), np.array(labels, dtype=int)


# ── Public class ──────────────────────────────────────────────────────────────

class StabilityPredictor:
    """
    Meta-learner that predicts per-model suitability probabilities for a
    given dataset profile.

    Wraps two RandomForestClassifiers — one for classification tasks and one
    for regression tasks — each trained on synthetic meta-features.

    The predictor is lightweight: fitting takes < 0.3 s and does not require
    any external training data.  It is used by the orchestrator as a
    pre-selection step to inform model ranking before HPO begins.
    """

    def __init__(self):
        self._clf_meta = None   # meta-learner for classification datasets
        self._reg_meta = None   # meta-learner for regression datasets

    def fit(self) -> "StabilityPredictor":
        """
        Train both meta-learners on synthetic data.

        Returns self so it can be chained:
            ranking = StabilityPredictor().fit().predict_ranking(analysis)
        """
        from sklearn.ensemble import RandomForestClassifier

        def _train(X, y):
            m = RandomForestClassifier(
                n_estimators=120, max_depth=6,
                random_state=42, n_jobs=1,
            )
            m.fit(X, y)
            return m

        X_clf, y_clf     = _make_clf_meta_dataset()
        self._clf_meta   = _train(X_clf, y_clf)

        X_reg, y_reg     = _make_reg_meta_dataset()
        self._reg_meta   = _train(X_reg, y_reg)

        log.info(
            "StabilityPredictor fitted: clf=%d samples, reg=%d samples",
            len(y_clf), len(y_reg),
        )
        return self

    def predict_ranking(self, analysis: dict) -> list:
        """
        Return a ranked list of (model_name, confidence_score) tuples,
        sorted descending by predicted suitability for the given dataset.

        Parameters
        ----------
        analysis : dict output of utils.dataset_analyzer.analyze_dataset()

        Returns
        -------
        list of (model_name: str, confidence: float) tuples
        """
        if self._clf_meta is None or self._reg_meta is None:
            self.fit()

        is_clf = analysis.get("task_type") == "classification"
        meta   = self._clf_meta if is_clf else self._reg_meta
        labels = _CLF_LABELS if is_clf else _REG_LABELS

        x     = _extract_meta_features(analysis).reshape(1, -1)
        proba = meta.predict_proba(x)[0]

        ranking = []
        for cls_idx, prob in zip(meta.classes_, proba):
            model_name = labels.get(int(cls_idx))
            if model_name:
                ranking.append((model_name, round(float(prob), 4)))

        ranking.sort(key=lambda t: t[1], reverse=True)
        log.info("StabilityPredictor ranking: %s", ranking)
        return ranking

    def recommend_models(self, analysis: dict, top_n: int = 3) -> list:
        """Return the top-N model names ordered by predicted suitability."""
        ranking = self.predict_ranking(analysis)
        return [name for name, _ in ranking[:top_n]]
