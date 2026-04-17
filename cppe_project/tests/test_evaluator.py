"""
tests/test_evaluator.py
------------------------
Unit tests for the core evaluator (metrics + plotting).
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from core.evaluator import (
    evaluate_classification, evaluate_regression,
    select_best_model, plot_model_comparison,
)


@pytest.fixture(scope="module")
def clf_data():
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X[:240], y[:240])
    return model, X[240:], y[240:]


@pytest.fixture(scope="module")
def reg_data():
    X, y = make_regression(n_samples=300, n_features=10, random_state=42)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X[:240], y[:240])
    return model, X[240:], y[240:]


class TestClassificationMetrics:
    def test_keys_present(self, clf_data):
        model, X_test, y_test = clf_data
        metrics = evaluate_classification(model, X_test, y_test, "RF")
        for key in ("accuracy", "precision", "recall", "f1_score"):
            assert key in metrics

    def test_accuracy_in_range(self, clf_data):
        model, X_test, y_test = clf_data
        metrics = evaluate_classification(model, X_test, y_test, "RF")
        assert 0.0 <= metrics["accuracy"] <= 1.0


class TestRegressionMetrics:
    def test_keys_present(self, reg_data):
        model, X_test, y_test = reg_data
        metrics = evaluate_regression(model, X_test, y_test, "RF_R")
        for key in ("rmse", "mae", "r2"):
            assert key in metrics

    def test_rmse_non_negative(self, reg_data):
        model, X_test, y_test = reg_data
        metrics = evaluate_regression(model, X_test, y_test, "RF_R")
        assert metrics["rmse"] >= 0


class TestBestModelSelection:
    def test_select_best_classification(self):
        results = [
            {"name": "A", "metrics": {"accuracy": 0.85}},
            {"name": "B", "metrics": {"accuracy": 0.92}},
            {"name": "C", "metrics": {"accuracy": 0.78}},
        ]
        best = select_best_model(results, "classification")
        assert best["name"] == "B"

    def test_select_best_regression(self):
        results = [
            {"name": "X", "metrics": {"r2": 0.72}},
            {"name": "Y", "metrics": {"r2": 0.89}},
        ]
        best = select_best_model(results, "regression")
        assert best["name"] == "Y"

    def test_empty_results(self):
        best = select_best_model([], "classification")
        assert best == {}
