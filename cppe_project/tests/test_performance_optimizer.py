"""
tests/test_performance_optimizer.py
------------------------------------
Unit tests for core/performance_optimizer.py

This module verifies that the PerformanceOptimiser:
  - Returns a correctly structured plan dict
  - Produces positive time estimates for every model
  - Flags memory warnings when models exceed budget
  - Correctly toggles HPO recommendation based on dataset size
  - Selects the right parallelism strategy
  - Generates a non-empty human-readable reason string
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from core.performance_optimizer import PerformanceOptimiser


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def optimiser():
    return PerformanceOptimiser().fit()


@pytest.fixture(scope="module")
def small_analysis():
    return {
        "task_type":       "classification",
        "rows":            500,
        "features":        6,
        "numerical_cols":  [f"n{i}" for i in range(6)],
        "categorical_cols": [],
        "missing_pct":     0.0,
        "n_classes":       2,
        "complexity":      "small",
    }


@pytest.fixture(scope="module")
def large_analysis():
    return {
        "task_type":       "classification",
        "rows":            80_000,
        "features":        80,
        "numerical_cols":  [f"n{i}" for i in range(70)],
        "categorical_cols": [f"c{i}" for i in range(10)],
        "missing_pct":     5.0,
        "n_classes":       5,
        "complexity":      "large",
    }


@pytest.fixture(scope="module")
def small_allocation():
    return {
        "cpu_allocated":    1,
        "cpu_min":          1,
        "cpu_max":          2,
        "memory_budget_mb": 256,
        "available_cpus":   2,
        "physical_cpus":    1,
        "total_ram_gb":     4.0,
        "available_ram_gb": 2.0,
    }


@pytest.fixture(scope="module")
def medium_allocation():
    return {
        "cpu_allocated":    4,
        "cpu_min":          2,
        "cpu_max":          4,
        "memory_budget_mb": 512,
        "available_cpus":   8,
        "physical_cpus":    4,
        "total_ram_gb":     16.0,
        "available_ram_gb": 8.0,
    }


CLF_MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "SVM"]
REG_MODELS = ["LinearRegression", "RandomForestRegressor", "XGBoostRegressor"]


# ── TestFit ────────────────────────────────────────────────────────────────────

class TestFit:
    def test_fit_returns_self(self):
        opt = PerformanceOptimiser()
        result = opt.fit()
        assert result is opt

    def test_chaining_works(self, small_analysis, small_allocation):
        plan = PerformanceOptimiser().fit().recommend(
            small_analysis, small_allocation, CLF_MODELS
        )
        assert isinstance(plan, dict)


# ── TestPlanStructure ──────────────────────────────────────────────────────────

class TestPlanStructure:
    REQUIRED_KEYS = [
        "hpo_recommended", "strategy", "n_workers",
        "estimated_times", "memory_warnings",
        "total_est_seq", "total_est_par", "reason",
    ]

    def test_all_keys_present(self, optimiser, small_analysis, small_allocation):
        plan = optimiser.recommend(small_analysis, small_allocation, CLF_MODELS)
        for key in self.REQUIRED_KEYS:
            assert key in plan, f"Missing key: {key}"

    def test_hpo_recommended_is_bool(self, optimiser, small_analysis, small_allocation):
        plan = optimiser.recommend(small_analysis, small_allocation, CLF_MODELS)
        assert isinstance(plan["hpo_recommended"], bool)

    def test_strategy_is_valid_string(self, optimiser, small_analysis, small_allocation):
        plan = optimiser.recommend(small_analysis, small_allocation, CLF_MODELS)
        assert plan["strategy"] in ("thread_pool", "sequential")

    def test_n_workers_is_positive_int(self, optimiser, small_analysis, small_allocation):
        plan = optimiser.recommend(small_analysis, small_allocation, CLF_MODELS)
        assert isinstance(plan["n_workers"], int)
        assert plan["n_workers"] >= 1

    def test_estimated_times_dict_keys_match_models(
            self, optimiser, small_analysis, small_allocation):
        plan = optimiser.recommend(small_analysis, small_allocation, CLF_MODELS)
        assert set(plan["estimated_times"].keys()) == set(CLF_MODELS)

    def test_memory_warnings_is_list(self, optimiser, small_analysis, small_allocation):
        plan = optimiser.recommend(small_analysis, small_allocation, CLF_MODELS)
        assert isinstance(plan["memory_warnings"], list)

    def test_total_est_seq_is_positive(self, optimiser, small_analysis, small_allocation):
        plan = optimiser.recommend(small_analysis, small_allocation, CLF_MODELS)
        assert plan["total_est_seq"] > 0.0

    def test_total_est_par_is_positive(self, optimiser, small_analysis, small_allocation):
        plan = optimiser.recommend(small_analysis, small_allocation, CLF_MODELS)
        assert plan["total_est_par"] > 0.0

    def test_reason_is_non_empty_string(self, optimiser, small_analysis, small_allocation):
        plan = optimiser.recommend(small_analysis, small_allocation, CLF_MODELS)
        assert isinstance(plan["reason"], str)
        assert len(plan["reason"]) > 10


# ── TestTimeEstimates ──────────────────────────────────────────────────────────

class TestTimeEstimates:
    def test_all_models_have_positive_estimates(
            self, optimiser, small_analysis, small_allocation):
        plan = optimiser.recommend(small_analysis, small_allocation, CLF_MODELS)
        for model, t in plan["estimated_times"].items():
            assert t > 0.0, f"Model {model} has non-positive time estimate {t}"

    def test_large_dataset_produces_higher_times_than_small(
            self, optimiser, small_analysis, large_analysis,
            small_allocation, medium_allocation):
        plan_small = optimiser.recommend(small_analysis, small_allocation,
                                         ["RandomForest"], hpo_requested=False)
        plan_large = optimiser.recommend(large_analysis, medium_allocation,
                                         ["RandomForest"], hpo_requested=False)
        assert (plan_large["estimated_times"]["RandomForest"] >
                plan_small["estimated_times"]["RandomForest"]), (
            "Large dataset should have higher estimated training time than small"
        )

    def test_hpo_enabled_increases_estimated_time(
            self, optimiser, small_analysis, small_allocation):
        plan_no_hpo = optimiser.recommend(
            small_analysis, small_allocation, ["XGBoost"], hpo_requested=False)
        plan_hpo = optimiser.recommend(
            small_analysis, small_allocation, ["XGBoost"], hpo_requested=True)
        assert (plan_hpo["estimated_times"]["XGBoost"] >=
                plan_no_hpo["estimated_times"]["XGBoost"]), (
            "HPO should not decrease estimated training time"
        )

    def test_regression_models_have_positive_estimates(
            self, optimiser, medium_allocation):
        reg_analysis = {
            "task_type": "regression", "rows": 3000, "features": 20,
            "numerical_cols": [f"n{i}" for i in range(20)],
            "categorical_cols": [], "missing_pct": 0.0,
            "n_classes": None, "complexity": "medium",
        }
        plan = optimiser.recommend(reg_analysis, medium_allocation, REG_MODELS)
        for model, t in plan["estimated_times"].items():
            assert t > 0.0


# ── TestStrategy ───────────────────────────────────────────────────────────────

class TestStrategy:
    def test_single_model_single_worker_is_sequential(
            self, optimiser, small_analysis):
        alloc_1 = {
            "cpu_allocated": 1, "cpu_min": 1, "cpu_max": 1,
            "memory_budget_mb": 256, "available_cpus": 1,
            "physical_cpus": 1, "total_ram_gb": 4.0, "available_ram_gb": 2.0,
        }
        plan = optimiser.recommend(small_analysis, alloc_1, ["LogisticRegression"])
        assert plan["strategy"] == "sequential"

    def test_multiple_models_multi_worker_is_thread_pool(
            self, optimiser, small_analysis, medium_allocation):
        plan = optimiser.recommend(small_analysis, medium_allocation, CLF_MODELS)
        assert plan["strategy"] == "thread_pool"


# ── TestMemoryWarnings ─────────────────────────────────────────────────────────

class TestMemoryWarnings:
    def test_no_warnings_with_huge_budget(
            self, optimiser, small_analysis):
        alloc = {
            "cpu_allocated": 2, "cpu_min": 1, "cpu_max": 4,
            "memory_budget_mb": 99999,
            "available_cpus": 4, "physical_cpus": 2,
            "total_ram_gb": 32.0, "available_ram_gb": 16.0,
        }
        plan = optimiser.recommend(small_analysis, alloc, CLF_MODELS)
        # With a 99 GB budget no model should trigger a warning
        assert len(plan["memory_warnings"]) == 0

    def test_warnings_present_with_tiny_budget(
            self, optimiser, large_analysis):
        alloc_tiny = {
            "cpu_allocated": 2, "cpu_min": 1, "cpu_max": 4,
            "memory_budget_mb": 10,   # absurdly small
            "available_cpus": 4, "physical_cpus": 2,
            "total_ram_gb": 4.0, "available_ram_gb": 0.5,
        }
        plan = optimiser.recommend(large_analysis, alloc_tiny, CLF_MODELS)
        assert len(plan["memory_warnings"]) > 0
