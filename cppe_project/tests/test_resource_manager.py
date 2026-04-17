"""
tests/test_resource_manager.py
---------------------------------
Unit tests for core/resource_manager.py

Tested API:
  allocate_resources(complexity)  → dict
    keys: cpu_allocated, cpu_min, cpu_max, memory_budget_mb,
          available_cpus, physical_cpus, total_ram_gb, available_ram_gb

  get_host_stats()  → HostStats
    attrs: total_cpus, physical_cpus, total_ram_gb, available_ram_gb
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from core.resource_manager import allocate_resources, get_host_stats

REQUIRED_ALLOC_KEYS = {
    "cpu_allocated",
    "cpu_min",
    "cpu_max",
    "memory_budget_mb",
    "available_cpus",
    "physical_cpus",
    "total_ram_gb",
    "available_ram_gb",
}


# ─── get_host_stats ──────────────────────────────────────────────────────────

class TestGetHostStats:
    def test_returns_host_stats_object(self):
        host = get_host_stats()
        assert host is not None

    def test_total_cpus_positive_integer(self):
        host = get_host_stats()
        assert isinstance(host.total_cpus, int)
        assert host.total_cpus >= 1

    def test_physical_cpus_positive_integer(self):
        host = get_host_stats()
        assert isinstance(host.physical_cpus, int)
        assert host.physical_cpus >= 1

    def test_total_ram_gb_positive(self):
        host = get_host_stats()
        assert isinstance(host.total_ram_gb, float)
        assert host.total_ram_gb > 0.0

    def test_available_ram_gb_non_negative(self):
        host = get_host_stats()
        assert host.available_ram_gb >= 0.0

    def test_available_ram_not_exceed_total(self):
        host = get_host_stats()
        assert host.available_ram_gb <= host.total_ram_gb


# ─── allocate_resources ──────────────────────────────────────────────────────

class TestAllocateResources:
    @pytest.mark.parametrize("complexity", ["small", "medium", "large"])
    def test_returns_dict(self, complexity):
        result = allocate_resources(complexity)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("complexity", ["small", "medium", "large"])
    def test_all_required_keys_present(self, complexity):
        result = allocate_resources(complexity)
        for key in REQUIRED_ALLOC_KEYS:
            assert key in result, f"Missing key '{key}' for complexity='{complexity}'"

    @pytest.mark.parametrize("complexity", ["small", "medium", "large"])
    def test_cpu_allocated_is_positive_integer(self, complexity):
        result = allocate_resources(complexity)
        assert isinstance(result["cpu_allocated"], int)
        assert result["cpu_allocated"] >= 1

    @pytest.mark.parametrize("complexity", ["small", "medium", "large"])
    def test_memory_budget_mb_is_positive_number(self, complexity):
        result = allocate_resources(complexity)
        assert isinstance(result["memory_budget_mb"], (int, float))
        assert result["memory_budget_mb"] > 0

    def test_small_cpu_max_is_2(self):
        """Small complexity must cap CPUs at 2."""
        result = allocate_resources("small")
        assert result["cpu_max"] == 2

    def test_medium_cpu_max_is_4(self):
        result = allocate_resources("medium")
        assert result["cpu_max"] == 4

    def test_large_cpu_max_is_8(self):
        result = allocate_resources("large")
        assert result["cpu_max"] == 8

    def test_cpu_allocated_within_min_max(self):
        """Allocated CPUs must sit between the plan's min and max."""
        for complexity in ("small", "medium", "large"):
            result = allocate_resources(complexity)
            assert result["cpu_min"] <= result["cpu_allocated"] <= result["cpu_max"], (
                f"cpu_allocated={result['cpu_allocated']} out of range "
                f"[{result['cpu_min']}, {result['cpu_max']}] for '{complexity}'"
            )

    def test_memory_does_not_exceed_plan_ceiling(self):
        """Budget must never exceed the hard ceiling defined in ALLOCATION_TABLE."""
        ceilings = {"small": 256, "medium": 512, "large": 1024}
        for complexity, ceiling in ceilings.items():
            result = allocate_resources(complexity)
            assert result["memory_budget_mb"] <= ceiling, (
                f"memory_budget_mb={result['memory_budget_mb']} exceeds "
                f"ceiling={ceiling} for '{complexity}'"
            )

    def test_available_cpus_positive(self):
        result = allocate_resources("small")
        assert result["available_cpus"] >= 1

    def test_total_ram_gb_positive(self):
        result = allocate_resources("small")
        assert result["total_ram_gb"] > 0.0


# ─── measure_training ─────────────────────────────────────────────────────────

class TestMeasureTraining:
    """Tests for the measure_training() helper used by ensemble_builder.py."""

    @pytest.fixture(scope="class")
    def tiny_data(self):
        """50-sample, 4-feature classification data as numpy arrays."""
        import numpy as np
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=50, n_features=4, n_informative=3,
            n_redundant=1, random_state=0,
        )
        return X, y

    def test_returns_two_tuple(self, tiny_data):
        from sklearn.linear_model import LogisticRegression
        from core.resource_manager import measure_training
        X, y = tiny_data
        result = measure_training(LogisticRegression(max_iter=200), X, y)
        assert len(result) == 2

    def test_duration_is_non_negative_float(self, tiny_data):
        from sklearn.linear_model import LogisticRegression
        from core.resource_manager import measure_training
        X, y = tiny_data
        duration, _ = measure_training(LogisticRegression(max_iter=200), X, y)
        assert isinstance(duration, float)
        assert duration >= 0.0

    def test_peak_ram_is_positive_float(self, tiny_data):
        from sklearn.linear_model import LogisticRegression
        from core.resource_manager import measure_training
        X, y = tiny_data
        _, peak_ram = measure_training(LogisticRegression(max_iter=200), X, y)
        assert isinstance(peak_ram, float)
        assert peak_ram > 0.0

    def test_model_is_fitted_after_call(self, tiny_data):
        """The model must be callable for predict() after measure_training."""
        from sklearn.linear_model import LogisticRegression
        from core.resource_manager import measure_training
        X, y = tiny_data
        model = LogisticRegression(max_iter=200)
        measure_training(model, X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_works_with_random_forest(self, tiny_data):
        from sklearn.ensemble import RandomForestClassifier
        from core.resource_manager import measure_training
        X, y = tiny_data
        duration, peak_ram = measure_training(
            RandomForestClassifier(n_estimators=10, random_state=0), X, y
        )
        assert duration >= 0.0
        assert peak_ram > 0.0
