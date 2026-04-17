"""
tests/test_train_models.py
---------------------------
Unit tests for src/train_models.py

Tested functions:
  train_random_forest(X_train, y_train, save_path, cpu_cores)
  train_logistic_regression(X_train, y_train, save_path, cpu_cores)
  train_svm(X_train, y_train, save_path, cpu_cores)
  train_xgboost(X_train, y_train, save_path, cpu_cores)

Each returns: (model, duration_seconds, ram_mb, cpu_cores_used)
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.train_models import (
    train_random_forest,
    train_logistic_regression,
    train_svm,
    train_xgboost,
)


# ─── Shared dataset ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clf_splits():
    """
    Synthetic 100-sample, 5-feature binary classification dataset.
    Returns (X_train, X_test, y_train, y_test) as numpy arrays.
    """
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.20, random_state=42)


@pytest.fixture(scope="module")
def tmp_model_dir():
    """Temporary directory for .pkl outputs — auto-deleted after the session."""
    with tempfile.TemporaryDirectory() as d:
        yield d


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _assert_return_contract(result, X_test, y_test, model_name: str):
    """
    All training functions must return: (model, duration, ram_mb, cpu_cores).
    Asserts the full return-value contract plus prediction shape.
    """
    assert len(result) == 4, f"{model_name}: expected 4-tuple"
    model, duration, ram_mb, cpu_cores = result

    # model must have a predict method
    assert hasattr(model, "predict"), f"{model_name}: model has no predict()"

    # duration must be a non-negative float
    assert isinstance(duration, float), f"{model_name}: duration is not float"
    assert duration >= 0.0, f"{model_name}: negative duration={duration}"

    # ram_mb must be positive (minimum 0.01 enforced in source)
    assert isinstance(ram_mb, float), f"{model_name}: ram_mb is not float"
    assert ram_mb > 0.0, f"{model_name}: ram_mb={ram_mb} not positive"

    # cpu_cores must be a positive integer
    assert isinstance(cpu_cores, int), f"{model_name}: cpu_cores is not int"
    assert cpu_cores >= 1, f"{model_name}: cpu_cores={cpu_cores} < 1"

    # predict output must have same length as test set
    preds = model.predict(X_test)
    assert len(preds) == len(y_test), (
        f"{model_name}: predict() returned {len(preds)} values, "
        f"expected {len(y_test)}"
    )


# ─── RandomForest ────────────────────────────────────────────────────────────

class TestTrainRandomForest:
    def test_returns_four_tuple(self, clf_splits, tmp_model_dir):
        X_train, X_test, y_train, y_test = clf_splits
        result = train_random_forest(X_train, y_train, save_path=tmp_model_dir, cpu_cores=1)
        _assert_return_contract(result, X_test, y_test, "RandomForest")

    def test_model_has_predict(self, clf_splits, tmp_model_dir):
        X_train, X_test, y_train, y_test = clf_splits
        model, *_ = train_random_forest(X_train, y_train, save_path=tmp_model_dir, cpu_cores=1)
        assert hasattr(model, "predict")

    def test_predict_output_length(self, clf_splits, tmp_model_dir):
        X_train, X_test, y_train, y_test = clf_splits
        model, *_ = train_random_forest(X_train, y_train, save_path=tmp_model_dir, cpu_cores=1)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_pkl_file_created(self, clf_splits, tmp_model_dir):
        X_train, X_test, y_train, y_test = clf_splits
        train_random_forest(X_train, y_train, save_path=tmp_model_dir, cpu_cores=1)
        assert os.path.isfile(os.path.join(tmp_model_dir, "random_forest.pkl"))


# ─── LogisticRegression ──────────────────────────────────────────────────────

class TestTrainLogisticRegression:
    def test_returns_four_tuple(self, clf_splits, tmp_model_dir):
        X_train, X_test, y_train, y_test = clf_splits
        result = train_logistic_regression(X_train, y_train, save_path=tmp_model_dir, cpu_cores=1)
        _assert_return_contract(result, X_test, y_test, "LogisticRegression")

    def test_model_has_predict(self, clf_splits, tmp_model_dir):
        X_train, X_test, y_train, y_test = clf_splits
        model, *_ = train_logistic_regression(X_train, y_train, save_path=tmp_model_dir, cpu_cores=1)
        assert hasattr(model, "predict")

    def test_predict_output_length(self, clf_splits, tmp_model_dir):
        X_train, X_test, y_train, y_test = clf_splits
        model, *_ = train_logistic_regression(X_train, y_train, save_path=tmp_model_dir, cpu_cores=1)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_pkl_file_created(self, clf_splits, tmp_model_dir):
        X_train, X_test, y_train, y_test = clf_splits
        train_logistic_regression(X_train, y_train, save_path=tmp_model_dir, cpu_cores=1)
        assert os.path.isfile(os.path.join(tmp_model_dir, "logistic_regression.pkl"))


# ─── SVM ─────────────────────────────────────────────────────────────────────

class TestTrainSVM:
    def test_returns_four_tuple(self, clf_splits, tmp_model_dir):
        X_train, X_test, y_train, y_test = clf_splits
        result = train_svm(X_train, y_train, save_path=tmp_model_dir, cpu_cores=1)
        _assert_return_contract(result, X_test, y_test, "SVM")

    def test_model_has_predict(self, clf_splits, tmp_model_dir):
        X_train, X_test, y_train, y_test = clf_splits
        model, *_ = train_svm(X_train, y_train, save_path=tmp_model_dir, cpu_cores=1)
        assert hasattr(model, "predict")

    def test_predict_output_length(self, clf_splits, tmp_model_dir):
        X_train, X_test, y_train, y_test = clf_splits
        model, *_ = train_svm(X_train, y_train, save_path=tmp_model_dir, cpu_cores=1)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_svm_cpu_cores_always_1(self, clf_splits, tmp_model_dir):
        """SVM source hard-codes cpu_cores return value to 1 regardless of input."""
        X_train, _, y_train, _ = clf_splits
        _, _, _, returned_cores = train_svm(X_train, y_train, save_path=tmp_model_dir, cpu_cores=4)
        assert returned_cores == 1


# ─── XGBoost ─────────────────────────────────────────────────────────────────

class TestTrainXGBoost:
    def test_returns_four_tuple(self, clf_splits, tmp_model_dir):
        X_train, X_test, y_train, y_test = clf_splits
        result = train_xgboost(X_train, y_train, save_path=tmp_model_dir, cpu_cores=1)
        _assert_return_contract(result, X_test, y_test, "XGBoost")

    def test_model_has_predict(self, clf_splits, tmp_model_dir):
        X_train, X_test, y_train, y_test = clf_splits
        model, *_ = train_xgboost(X_train, y_train, save_path=tmp_model_dir, cpu_cores=1)
        assert hasattr(model, "predict")

    def test_predict_output_length(self, clf_splits, tmp_model_dir):
        X_train, X_test, y_train, y_test = clf_splits
        model, *_ = train_xgboost(X_train, y_train, save_path=tmp_model_dir, cpu_cores=1)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_pkl_file_created(self, clf_splits, tmp_model_dir):
        X_train, X_test, y_train, y_test = clf_splits
        train_xgboost(X_train, y_train, save_path=tmp_model_dir, cpu_cores=1)
        assert os.path.isfile(os.path.join(tmp_model_dir, "xgboost.pkl"))
