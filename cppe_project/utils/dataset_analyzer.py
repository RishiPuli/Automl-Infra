"""
utils/dataset_analyzer.py
--------------------------
Analyzes a CSV dataset and returns properties used by the orchestrator.
"""

import pandas as pd
import numpy as np
from typing import Optional

# Thresholds
UNIQUE_REGRESSION_THRESHOLD = 50   # target unique values > this → regression
SAMPLE_THRESHOLD            = 5000  # rows above this → auto-sample
COMPLEXITY_SMALL_MAX        = 1000
COMPLEXITY_MEDIUM_MAX       = 10000


def detect_target_column(df: pd.DataFrame, hint: Optional[str] = None) -> str:
    """
    Find the target column.
    Priority: user hint → common names → last column.
    """
    if hint:
        hint = hint.strip()
        if hint in df.columns:
            return hint
        # Case-insensitive check
        lower = {c.lower(): c for c in df.columns}
        if hint.lower() in lower:
            return lower[hint.lower()]

    for name in ["target", "label", "class", "y", "output",
                 "Target", "Label", "Class"]:
        if name in df.columns:
            return name

    return df.columns[-1]


def detect_task_type(series: pd.Series) -> str:
    """
    Determine if the task is classification or regression.

    Rules:
      - Non-numeric column        → classification
      - > 50 unique values        → regression
      - Integer-like values ≤ 50  → classification
      - Otherwise                 → regression
    """
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"

    n_unique = series.nunique(dropna=True)

    if n_unique > UNIQUE_REGRESSION_THRESHOLD:
        return "regression"

    # Accept integer dtype
    if pd.api.types.is_integer_dtype(series):
        return "classification"

    # Float columns that store whole numbers (e.g. 1.0, 2.0)
    non_null = series.dropna()
    if len(non_null) > 0 and np.allclose(non_null, non_null.round()):
        return "classification"

    return "regression"


def detect_complexity(n_rows: int) -> str:
    if n_rows < COMPLEXITY_SMALL_MAX:
        return "small"
    elif n_rows <= COMPLEXITY_MEDIUM_MAX:
        return "medium"
    return "large"


def analyze_dataset(df: pd.DataFrame,
                    target_hint: Optional[str] = None) -> dict:
    """
    Analyze a DataFrame and return a structured profile.

    Returns
    -------
    dict with keys used by orchestrator and UI.
    """
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    target = detect_target_column(df, target_hint)
    feat_cols = [c for c in df.columns if c != target]
    feat_df   = df[feat_cols]
    target_s  = df[target]

    num_cols = feat_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = feat_df.select_dtypes(exclude=[np.number]).columns.tolist()

    total_missing    = int(df.isnull().sum().sum())
    missing_by_col   = df.isnull().sum()
    cols_with_missing = {c: int(v)
                         for c, v in missing_by_col[missing_by_col > 0].items()}
    missing_pct      = round(total_missing / (df.shape[0] * df.shape[1]) * 100, 2) \
                       if df.shape[0] * df.shape[1] > 0 else 0.0

    n_rows    = len(df)
    task_type = detect_task_type(target_s)
    n_unique  = int(target_s.nunique(dropna=True))
    complexity= detect_complexity(n_rows)
    mem_mb    = round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3)

    class_dist = {}
    if task_type == "classification":
        class_dist = {str(k): int(v)
                      for k, v in target_s.value_counts().head(20).items()}

    return {
        "target_column":      target,
        "task_type":          task_type,
        "complexity":         complexity,
        "rows":               n_rows,
        "features":           len(feat_cols),
        "numerical_cols":     num_cols,
        "categorical_cols":   cat_cols,
        "numerical":          len(num_cols),
        "categorical":        len(cat_cols),
        "missing_values":     total_missing > 0,
        "total_missing":      total_missing,
        "missing_pct":        missing_pct,
        "cols_with_missing":  cols_with_missing,
        "n_unique_target":    n_unique,
        "n_classes":          n_unique if task_type == "classification" else None,
        "memory_mb":          mem_mb,
        "should_sample":      n_rows > SAMPLE_THRESHOLD,
        "sample_size":        SAMPLE_THRESHOLD,
        "class_distribution": class_dist,
    }


def get_recommended_models(task_type: str, complexity: str) -> list:
    """
    Select models based on task type and dataset complexity.

    Rules:
      - Skip SVM for medium and large datasets (slow)
      - Skip XGBoost for large datasets (memory intensive)
    """
    if task_type == "classification":
        models = ["LogisticRegression", "RandomForest"]
        if complexity != "large":
            models.append("XGBoost")
        if complexity == "small":
            models.append("SVM")
        return models
    else:
        return ["LinearRegression", "RandomForestRegressor", "XGBoostRegressor"]
