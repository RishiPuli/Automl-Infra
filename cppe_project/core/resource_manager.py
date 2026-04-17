"""
core/resource_manager.py
-------------------------
Resource allocation and monitoring.

Maps dataset complexity to CPU/memory budgets.
Tracks peak RAM usage during training.
"""

import os
import time
import threading
import psutil
from utils.logger import get_logger

log = get_logger("ResourceManager")

# CPU and memory budgets by complexity tier
ALLOCATION_TABLE = {
    "small":  {"cpu_min": 1, "cpu_max": 2,  "memory_mb": 256},
    "medium": {"cpu_min": 2, "cpu_max": 4,  "memory_mb": 512},
    "large":  {"cpu_min": 4, "cpu_max": 8,  "memory_mb": 1024},
}


class HostStats:
    """Snapshot of host machine resources."""
    def __init__(self):
        vm = psutil.virtual_memory()
        self.total_ram_gb     = vm.total     / (1024 ** 3)
        self.available_ram_gb = vm.available / (1024 ** 3)
        self.total_cpus       = psutil.cpu_count(logical=True)  or 1
        self.physical_cpus    = psutil.cpu_count(logical=False) or 1


def get_host_stats() -> HostStats:
    return HostStats()


def allocate_resources(complexity: str) -> dict:
    """
    Return CPU and memory budget for the given complexity tier.
    Caps allocation to what the host machine actually has.
    """
    host  = get_host_stats()
    plan  = ALLOCATION_TABLE.get(complexity, ALLOCATION_TABLE["medium"])
    cpu   = min(plan["cpu_max"], max(plan["cpu_min"], host.total_cpus // 2))
    # Cap memory to 60% of available RAM
    mem   = min(plan["memory_mb"], int(host.available_ram_gb * 1024 * 0.6))

    alloc = {
        "cpu_allocated":    cpu,
        "cpu_min":          plan["cpu_min"],
        "cpu_max":          plan["cpu_max"],
        "memory_budget_mb": mem,
        "available_cpus":   host.total_cpus,
        "physical_cpus":    host.physical_cpus,
        "total_ram_gb":     round(host.total_ram_gb, 2),
        "available_ram_gb": round(host.available_ram_gb, 2),
    }
    log.info("Allocated for '%s': %d CPUs, %d MB", complexity, cpu, mem)
    return alloc


class PeakMemoryMonitor:
    """
    Background thread that polls process RSS memory every 0.3 seconds
    and records the peak value seen during a model training run.

    Usage:
        mon = PeakMemoryMonitor()
        mon.start()
        model.fit(X, y)
        mon.stop()
        print(mon.peak_mb)
    """

    def __init__(self, interval: float = 0.3):
        self._interval = interval
        self._peak_mb  = 0.0
        self._running  = False
        self._thread   = None
        self._proc     = psutil.Process(os.getpid())

    def _poll(self):
        while self._running:
            try:
                rss = self._proc.memory_info().rss / (1024 ** 2)
                if rss > self._peak_mb:
                    self._peak_mb = rss
            except psutil.NoSuchProcess:
                break
            time.sleep(self._interval)

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def peak_mb(self) -> float:
        return round(self._peak_mb, 2)


def measure_training(model, X_train, y_train) -> tuple:
    """
    Fit *model* on (X_train, y_train) and return (duration_seconds, peak_ram_mb).

    Used by ensemble_builder.py to time and profile the meta-learner fit.
    The return value is a plain 2-tuple so it unpacks cleanly::

        train_time, peak_ram = measure_training(meta, X_oof, y_train)
    """
    monitor = PeakMemoryMonitor()
    monitor.start()
    t0       = time.perf_counter()
    model.fit(X_train, y_train)
    duration = round(time.perf_counter() - t0, 3)
    monitor.stop()
    return duration, monitor.peak_mb
