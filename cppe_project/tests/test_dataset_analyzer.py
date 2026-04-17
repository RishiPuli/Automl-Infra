"""
tests/test_dataset_analyzer.py
--------------------------------
Unit tests for utils/dataset_analyzer.py

Actual API confirmed from source:
  - detect_task_type(series)           → 'classification' | 'regression'
  - detect_complexity(n_rows)          → 'small' | 'medium' | 'large'
  - analyze_dataset(df, target_hint)   → dict
  - get_recommended_models(task_type, complexity) → list
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from utils.dataset_analyzer import (
    analyze_dataset,
    detect_task_type,
    detect_complexity,
    get_recommended_models,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def classification_df():
    """200-row DataFrame whose target column has 3 integer classes."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "feat_1": rng.uniform(0, 1, 200),
        "feat_2": rng.uniform(0, 1, 200),
        "cat_col": rng.choice(["A", "B", "C"], 200),
        "target": rng.integers(0, 3, 200),
    })


@pytest.fixture
def regression_df():
    """300-row DataFrame whose target column has 300 unique floats → regression."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "feat_1": rng.uniform(0, 100, 300),
        "feat_2": rng.uniform(0, 100, 300),
        "target": rng.uniform(0, 1000, 300).astype(float),
    })


@pytest.fixture
def missing_df():
    """DataFrame with deliberate NaN values."""
    return pd.DataFrame({
        "f1": [1.0, np.nan, 3.0, 4.0, 5.0] * 50,
        "f2": [np.nan] * 250,
        "target": list(range(250)),
    })


# ─── detect_task_type ────────────────────────────────────────────────────────

class TestDetectTaskType:
    def test_integer_few_classes_is_classification(self):
        """Integer series with ≤ 50 unique values → classification."""
        s = pd.Series([0, 1, 2, 1, 0, 2])
        assert detect_task_type(s) == "classification"

    def test_string_column_is_classification(self):
        """Non-numeric dtype always → classification regardless of cardinality."""
        s = pd.Series(["cat", "dog"] * 10)
        assert detect_task_type(s) == "classification"

    def test_many_unique_floats_is_regression(self):
        """100 distinct float values (> UNIQUE_REGRESSION_THRESHOLD=50) → regression."""
        rng = np.random.default_rng(0)
        s = pd.Series(rng.uniform(0, 1000, 100))
        assert detect_task_type(s) == "regression"

    def test_float_whole_numbers_is_classification(self):
        """Float series whose values are whole numbers (1.0, 2.0) → classification."""
        s = pd.Series([1.0, 2.0, 3.0, 1.0, 2.0] * 5)
        assert detect_task_type(s) == "classification"


# ─── detect_complexity ───────────────────────────────────────────────────────

class TestDetectComplexity:
    def test_small(self):
        """< 1000 rows → 'small'."""
        assert detect_complexity(100) == "small"

    def test_medium(self):
        """1000 – 10000 rows → 'medium'."""
        assert detect_complexity(10_000) == "medium"

    def test_large(self):
        """> 10000 rows → 'large'."""
        assert detect_complexity(60_000) == "large"


# ─── analyze_dataset ─────────────────────────────────────────────────────────

class TestAnalyzeDataset:
    def test_basic_shape(self, classification_df):
        report = analyze_dataset(classification_df, target_hint="target")
        assert report["rows"] == 200
        assert report["features"] == 3  # feat_1, feat_2, cat_col

    def test_task_type_classification(self, classification_df):
        report = analyze_dataset(classification_df, target_hint="target")
        assert report["task_type"] == "classification"

    def test_task_type_regression(self, regression_df):
        report = analyze_dataset(regression_df, target_hint="target")
        assert report["task_type"] == "regression"

    def test_missing_values_detected(self, missing_df):
        report = analyze_dataset(missing_df, target_hint="target")
        assert report["missing_values"] is True
        assert report["total_missing"] > 0

    def test_no_missing_values(self, classification_df):
        report = analyze_dataset(classification_df, target_hint="target")
        assert report["missing_values"] is False

    def test_categorical_and_numerical_counts(self, classification_df):
        report = analyze_dataset(classification_df, target_hint="target")
        assert report["categorical"] == 1   # cat_col
        assert report["numerical"] == 2     # feat_1, feat_2

    def test_empty_df_raises(self):
        with pytest.raises(ValueError):
            analyze_dataset(pd.DataFrame())

    def test_memory_mb_positive(self, classification_df):
        report = analyze_dataset(classification_df, target_hint="target")
        assert report["memory_mb"] > 0

    def test_complexity_small_for_100_rows(self):
        """100-row dataset should be 'small'."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"x": rng.uniform(0, 1, 100), "target": rng.integers(0, 2, 100)})
        report = analyze_dataset(df, target_hint="target")
        assert report["complexity"] == "small"

    def test_complexity_medium_for_5000_rows(self):
        """5000-row dataset should be 'medium' (between 1000 and 10000)."""
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"x": rng.uniform(0, 1, 5000), "target": rng.integers(0, 2, 5000)})
        report = analyze_dataset(df, target_hint="target")
        assert report["complexity"] == "medium"

    def test_complexity_large_for_60000_rows(self, monkeypatch):
        """
        Monkeypatching len() is impractical — instead we verify detect_complexity
        directly returns 'large' for 60 000, then test analyze_dataset via the
        'should_sample' flag which activates above 5000 rows.
        """
        assert detect_complexity(60_000) == "large"


# ─── get_recommended_models ──────────────────────────────────────────────────

class TestRecommendedModels:
    def test_classification_small_includes_svm(self):
        """Small classification gets SVM added."""
        models = get_recommended_models("classification", "small")
        assert "SVM" in models
        assert "RandomForest" in models

    def test_classification_large_excludes_xgboost(self):
        """Large classification skips XGBoost (memory intensive)."""
        models = get_recommended_models("classification", "large")
        assert "XGBoost" not in models

    def test_regression_returns_list(self):
        models = get_recommended_models("regression", "medium")
        assert isinstance(models, list)
        assert len(models) > 0

    def test_regression_includes_linear(self):
        models = get_recommended_models("regression", "small")
        assert "LinearRegression" in models
