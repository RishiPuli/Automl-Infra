"""
tests/test_stability_predictor.py
-----------------------------------
Unit tests for core/stability_predictor.py

Tests cover:
  _extract_meta_features()     — correct vector shape and value ranges
  StabilityPredictor.fit()     — returns self, trains both meta-learners
  StabilityPredictor.predict_ranking()  — output format and ordering
  StabilityPredictor.recommend_models() — correct length, names are strings
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from core.stability_predictor import (
    StabilityPredictor,
    _extract_meta_features,
    _CLF_LABELS,
    _REG_LABELS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clf_analysis():
    """Typical medium classification dataset profile."""
    return {
        "task_type":       "classification",
        "rows":            3000,
        "features":        18,
        "numerical_cols":  [f"n{i}" for i in range(12)],
        "categorical_cols": [f"c{i}" for i in range(6)],
        "missing_pct":     3.5,
        "n_classes":       4,
        "complexity":      "medium",
    }


@pytest.fixture(scope="module")
def reg_analysis():
    """Typical large regression dataset profile."""
    return {
        "task_type":       "regression",
        "rows":            50000,
        "features":        40,
        "numerical_cols":  [f"n{i}" for i in range(35)],
        "categorical_cols": [f"c{i}" for i in range(5)],
        "missing_pct":     7.2,
        "n_classes":       None,
        "complexity":      "large",
    }


@pytest.fixture(scope="module")
def fitted_predictor():
    return StabilityPredictor().fit()


# ── _extract_meta_features ────────────────────────────────────────────────────

class TestExtractMetaFeatures:
    def test_returns_numpy_array(self, clf_analysis):
        vec = _extract_meta_features(clf_analysis)
        assert isinstance(vec, np.ndarray)

    def test_output_shape_is_7(self, clf_analysis):
        vec = _extract_meta_features(clf_analysis)
        assert vec.shape == (7,), f"Expected shape (7,), got {vec.shape}"

    def test_log10_rows_positive(self, clf_analysis):
        vec = _extract_meta_features(clf_analysis)
        assert vec[0] > 0.0, "log10(rows) must be positive for rows > 1"

    def test_n_features_clipped_at_200(self):
        analysis = {
            "task_type": "classification", "rows": 100, "features": 500,
            "numerical_cols": [f"n{i}" for i in range(500)],
            "categorical_cols": [], "missing_pct": 0.0,
            "n_classes": 2, "complexity": "small",
        }
        vec = _extract_meta_features(analysis)
        assert vec[1] <= 200.0

    def test_numerical_ratio_in_0_1(self, clf_analysis):
        vec = _extract_meta_features(clf_analysis)
        assert 0.0 <= vec[2] <= 1.0

    def test_is_classification_flag_set(self, clf_analysis):
        vec = _extract_meta_features(clf_analysis)
        assert vec[6] == 1.0

    def test_is_classification_flag_clear_for_regression(self, reg_analysis):
        vec = _extract_meta_features(reg_analysis)
        assert vec[6] == 0.0

    def test_complexity_small_encodes_to_0(self):
        analysis = {
            "task_type": "classification", "rows": 500, "features": 5,
            "numerical_cols": ["n0"], "categorical_cols": [],
            "missing_pct": 0.0, "n_classes": 2, "complexity": "small",
        }
        vec = _extract_meta_features(analysis)
        assert vec[5] == 0.0

    def test_complexity_large_encodes_to_2(self, reg_analysis):
        vec = _extract_meta_features(reg_analysis)
        assert vec[5] == 2.0


# ── StabilityPredictor.fit() ──────────────────────────────────────────────────

class TestStabilityPredictorFit:
    def test_fit_returns_self(self):
        sp = StabilityPredictor()
        result = sp.fit()
        assert result is sp

    def test_clf_meta_is_set_after_fit(self):
        sp = StabilityPredictor().fit()
        assert sp._clf_meta is not None

    def test_reg_meta_is_set_after_fit(self):
        sp = StabilityPredictor().fit()
        assert sp._reg_meta is not None

    def test_clf_meta_has_predict_proba(self):
        sp = StabilityPredictor().fit()
        assert hasattr(sp._clf_meta, "predict_proba")

    def test_chaining_syntax_works(self, clf_analysis):
        """StabilityPredictor().fit().predict_ranking(analysis) must not raise."""
        ranking = StabilityPredictor().fit().predict_ranking(clf_analysis)
        assert isinstance(ranking, list)


# ── StabilityPredictor.predict_ranking() ──────────────────────────────────────

class TestPredictRanking:
    def test_returns_list(self, fitted_predictor, clf_analysis):
        ranking = fitted_predictor.predict_ranking(clf_analysis)
        assert isinstance(ranking, list)

    def test_list_is_not_empty(self, fitted_predictor, clf_analysis):
        ranking = fitted_predictor.predict_ranking(clf_analysis)
        assert len(ranking) > 0

    def test_each_item_is_tuple_of_two(self, fitted_predictor, clf_analysis):
        for item in fitted_predictor.predict_ranking(clf_analysis):
            assert len(item) == 2, f"Expected (name, score), got {item}"

    def test_model_names_are_strings(self, fitted_predictor, clf_analysis):
        for name, _ in fitted_predictor.predict_ranking(clf_analysis):
            assert isinstance(name, str)

    def test_scores_are_floats_in_0_1(self, fitted_predictor, clf_analysis):
        for _, score in fitted_predictor.predict_ranking(clf_analysis):
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_ranking_sorted_descending(self, fitted_predictor, clf_analysis):
        ranking = fitted_predictor.predict_ranking(clf_analysis)
        scores  = [s for _, s in ranking]
        assert scores == sorted(scores, reverse=True), \
            "Ranking must be sorted descending by confidence score"

    def test_clf_model_names_from_clf_labels(self, fitted_predictor, clf_analysis):
        valid_names = set(_CLF_LABELS.values())
        for name, _ in fitted_predictor.predict_ranking(clf_analysis):
            assert name in valid_names, f"Unexpected model name: {name}"

    def test_reg_model_names_from_reg_labels(self, fitted_predictor, reg_analysis):
        valid_names = set(_REG_LABELS.values())
        for name, _ in fitted_predictor.predict_ranking(reg_analysis):
            assert name in valid_names, f"Unexpected model name: {name}"

    def test_scores_sum_to_approx_1(self, fitted_predictor, clf_analysis):
        """Probability scores from predict_proba must sum to ~1."""
        scores = [s for _, s in fitted_predictor.predict_ranking(clf_analysis)]
        assert abs(sum(scores) - 1.0) < 0.01, \
            f"Scores sum to {sum(scores):.4f}, expected ~1.0"


# ── StabilityPredictor.recommend_models() ─────────────────────────────────────

class TestRecommendModels:
    def test_returns_list(self, fitted_predictor, clf_analysis):
        result = fitted_predictor.recommend_models(clf_analysis, top_n=3)
        assert isinstance(result, list)

    def test_correct_length(self, fitted_predictor, clf_analysis):
        result = fitted_predictor.recommend_models(clf_analysis, top_n=2)
        assert len(result) == 2

    def test_all_items_are_strings(self, fitted_predictor, clf_analysis):
        result = fitted_predictor.recommend_models(clf_analysis, top_n=3)
        for name in result:
            assert isinstance(name, str)

    def test_top1_matches_ranking_first(self, fitted_predictor, clf_analysis):
        ranking = fitted_predictor.predict_ranking(clf_analysis)
        top1    = fitted_predictor.recommend_models(clf_analysis, top_n=1)
        assert top1[0] == ranking[0][0], \
            "recommend_models top-1 must equal first item of predict_ranking"

    def test_top_n_clamped_to_available(self, fitted_predictor, clf_analysis):
        """top_n > available models should not raise; returns all available."""
        result = fitted_predictor.recommend_models(clf_analysis, top_n=100)
        assert len(result) <= len(_CLF_LABELS)
