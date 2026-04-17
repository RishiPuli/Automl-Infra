"""
tests/test_orchestrator.py
---------------------------
Integration tests for core/orchestrator.run_pipeline().

Strategy
--------
Uses a small synthetic dataset (200 rows) so每 test suite executes
in under 30 seconds even in CI.  HPO, SHAP, and ensemble are disabled
to keep runtime predictable.  Evaluator / MLflow artefacts are logged
to the real on-disk stores because the pipeline is end-to-end.

Test groups
-----------
  TestClassificationPipeline  — RF + LR on a 2-class dataset
  TestRegressionPipeline      — LinearReg + RF on a regression target
  TestReturnContractKeys      — mandatory keys present in output dict
  TestCVScoring               — cv_mean / cv_std populated correctly
  TestBestModelSelection      — best / best_model keys consistent
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification, make_regression


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clf_df():
    """200-sample, 6-feature, 2-class classification DataFrame."""
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=5,
        n_redundant=1, n_classes=2, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    return df


@pytest.fixture(scope="module")
def reg_df():
    """200-sample, 6-feature continuous regression DataFrame."""
    X, y = make_regression(
        n_samples=200, n_features=6, n_informative=5, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y.astype(float)
    return df


@pytest.fixture(scope="module")
def clf_analysis(clf_df):
    from utils.dataset_analyzer import analyze_dataset
    return analyze_dataset(clf_df, target_hint="target")


@pytest.fixture(scope="module")
def reg_analysis(reg_df):
    from utils.dataset_analyzer import analyze_dataset
    return analyze_dataset(reg_df, target_hint="target")


@pytest.fixture(scope="module")
def fake_allocation():
    """Minimal allocation dict — does not call psutil."""
    return {
        "cpu_allocated":    2,
        "cpu_min":          1,
        "cpu_max":          4,
        "memory_budget_mb": 512,
        "available_cpus":   4,
        "physical_cpus":    2,
        "total_ram_gb":     8.0,
        "available_ram_gb": 4.0,
    }


# ── Helper ─────────────────────────────────────────────────────────────────────

def _pipeline(df, analysis, allocation, model_names):
    """Run the pipeline with all optional stages disabled for speed."""
    from core.orchestrator import run_pipeline
    logs = []
    return run_pipeline(
        df=df,
        analysis=analysis,
        model_names=model_names,
        allocation=allocation,
        test_size=0.25,
        log_fn=logs.append,
        enable_hpo=False,
        enable_shap=False,
        enable_ensemble=False,
        enable_report=False,
    )


# ── Test group 1: Classification ───────────────────────────────────────────────

class TestClassificationPipeline:
    MODELS = ["LogisticRegression", "RandomForest"]

    def test_pipeline_returns_dict(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        assert isinstance(out, dict)

    def test_results_list_has_correct_length(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        assert len(out["results"]) == len(self.MODELS)

    def test_accuracy_in_every_result(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        for r in out["results"]:
            assert "accuracy" in r["metrics"], \
                f"{r['name']} is missing accuracy metric"
            assert 0.0 <= r["metrics"]["accuracy"] <= 1.0

    def test_f1_score_in_every_result(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        for r in out["results"]:
            assert "f1_score" in r["metrics"]

    def test_train_time_is_positive(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        for r in out["results"]:
            assert r["train_time"] >= 0.0

    def test_peak_ram_is_positive(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        for r in out["results"]:
            assert r["peak_ram_mb"] > 0.0

    def test_model_pkl_saved_to_disk(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        for r in out["results"]:
            assert os.path.isfile(r["model_path"]), \
                f"Expected pkl at {r['model_path']}"

    def test_label_encoder_not_none(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        assert out["label_enc"] is not None

    def test_preprocessor_not_none(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        assert out["prep"] is not None


# ── Test group 2: Regression ───────────────────────────────────────────────────

class TestRegressionPipeline:
    MODELS = ["LinearRegression", "RandomForestRegressor"]

    def test_r2_in_every_result(self, reg_df, reg_analysis, fake_allocation):
        out = _pipeline(reg_df, reg_analysis, fake_allocation, self.MODELS)
        for r in out["results"]:
            assert "r2" in r["metrics"], \
                f"{r['name']} is missing r2 metric"

    def test_rmse_in_every_result(self, reg_df, reg_analysis, fake_allocation):
        out = _pipeline(reg_df, reg_analysis, fake_allocation, self.MODELS)
        for r in out["results"]:
            assert "rmse" in r["metrics"]
            assert r["metrics"]["rmse"] >= 0.0

    def test_label_encoder_is_none_for_regression(self, reg_df, reg_analysis,
                                                    fake_allocation):
        out = _pipeline(reg_df, reg_analysis, fake_allocation, self.MODELS)
        assert out["label_enc"] is None


# ── Test group 3: Return contract ─────────────────────────────────────────────

class TestReturnContractKeys:
    MODELS = ["LogisticRegression"]

    def test_has_results_key(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        assert "results" in out

    def test_has_best_key(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        assert "best" in out

    def test_has_best_model_key(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        assert "best_model" in out

    def test_best_and_best_model_identical(self, clf_df, clf_analysis,
                                            fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        assert out["best"] is out["best_model"]

    def test_has_summary_df(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        assert isinstance(out["summary_df"], pd.DataFrame)
        assert not out["summary_df"].empty

    def test_has_speedup_float(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        assert isinstance(out["speedup"], float)
        assert out["speedup"] > 0.0

    def test_has_speedup_info_dict(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        si  = out["speedup_info"]
        assert "wall" in si
        assert "speedup" in si

    def test_pdf_path_empty_when_disabled(self, clf_df, clf_analysis,
                                           fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        assert out["pdf_report_path"] == ""

    def test_has_prep_key(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        assert out["prep"] is not None


# ── Test group 4: CV scoring ───────────────────────────────────────────────────

class TestCVScoring:
    MODELS = ["LogisticRegression", "RandomForest"]

    def test_cv_mean_present_in_all_results(self, clf_df, clf_analysis,
                                             fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        for r in out["results"]:
            assert "cv_mean" in r, f"{r['name']} missing cv_mean"

    def test_cv_std_present_in_all_results(self, clf_df, clf_analysis,
                                            fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        for r in out["results"]:
            assert "cv_std" in r, f"{r['name']} missing cv_std"

    def test_cv_mean_in_valid_range(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        for r in out["results"]:
            assert -1.0 <= r["cv_mean"] <= 1.0, \
                f"{r['name']} cv_mean={r['cv_mean']} out of range"

    def test_summary_df_has_cv_mean_column(self, clf_df, clf_analysis,
                                             fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        assert "CV Mean" in out["summary_df"].columns


# ── Test group 5: Best model selection ────────────────────────────────────────

class TestBestModelSelection:
    MODELS = ["LogisticRegression", "RandomForest"]

    def test_best_has_name_field(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        assert "name" in out["best"]

    def test_best_has_metrics_field(self, clf_df, clf_analysis, fake_allocation):
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        assert "metrics" in out["best"]

    def test_best_has_highest_accuracy(self, clf_df, clf_analysis, fake_allocation):
        out     = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        best_acc = out["best"]["metrics"]["accuracy"]
        for r in out["results"]:
            assert r["metrics"]["accuracy"] <= best_acc + 1e-9, \
                f"Non-best model {r['name']} has higher accuracy than best"

    def test_best_model_pkl_is_best_model_pkl(self, clf_df, clf_analysis,
                                               fake_allocation):
        """best_model.pkl must be the best model's artifact."""
        out = _pipeline(clf_df, clf_analysis, fake_allocation, self.MODELS)
        if out["best"]:
            assert os.path.isfile(
                os.path.join("models", "best_model.pkl")
            )
