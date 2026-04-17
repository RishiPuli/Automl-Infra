"""
utils/metrics_logger.py
------------------------
Lightweight structured metrics logger for CloudAutoML.

Appends JSON Lines (JSONL) entries to logs/metrics.jsonl.
Each entry is a self-contained JSON object capturing one event.

Event types
-----------
  training_complete   — logged after each model trains in the pipeline
  api_request         — logged by the inference API on every HTTP request
  prediction          — logged after each /predict call

File format (JSONL — one JSON object per line)
----------------------------------------------
  {"event": "training_complete", "timestamp": "...", "model_name": "XGBoost", ...}
  {"event": "api_request",       "timestamp": "...", "endpoint": "/predict", ...}

Usage
-----
    from utils.metrics_logger import log_training_result, log_prediction

    log_training_result(
        model_name="XGBoost",
        task_type="classification",
        metrics={"accuracy": 0.93, "f1_score": 0.92},
        train_time=2.14,
        peak_ram_mb=184.0,
        complexity="medium",
        hpo_enabled=True,
    )

    log_prediction(
        model_name="XGBoost",
        n_samples=10,
        latency_ms=8.3,
    )

    recent = read_recent(50)   # returns last 50 entries as list of dicts
"""

import os
import json
import threading
from datetime import datetime, timezone

from utils.logger import get_logger

log       = get_logger("MetricsLogger")
_LOCK     = threading.Lock()

LOGS_DIR     = "logs"
METRICS_FILE = os.path.join(LOGS_DIR, "metrics.jsonl")

os.makedirs(LOGS_DIR, exist_ok=True)


# ── Internal writer ───────────────────────────────────────────────────────────

def _write(entry: dict) -> None:
    """Thread-safe append of one JSONL entry."""
    entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    line = json.dumps(entry, default=str) + "\n"
    with _LOCK:
        try:
            with open(METRICS_FILE, "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception as exc:
            log.warning("metrics_logger write failed: %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────

def log_training_result(
    model_name: str,
    task_type: str,
    metrics: dict,
    train_time: float,
    peak_ram_mb: float,
    complexity: str = "",
    hpo_enabled: bool = False,
) -> None:
    """
    Log a completed training run.

    Parameters
    ----------
    model_name   : name of the trained model (e.g. "XGBoost")
    task_type    : "classification" or "regression"
    metrics      : dict of metric_name -> float
    train_time   : wall-clock training duration in seconds
    peak_ram_mb  : peak RSS memory during training in megabytes
    complexity   : dataset complexity tier ("small", "medium", "large")
    hpo_enabled  : whether Optuna HPO was used for this run
    """
    _write({
        "event":        "training_complete",
        "model_name":   model_name,
        "task_type":    task_type,
        "metrics":      metrics,
        "train_time_s": round(train_time, 3),
        "peak_ram_mb":  round(peak_ram_mb, 1),
        "complexity":   complexity,
        "hpo_enabled":  hpo_enabled,
    })


def log_prediction(
    model_name: str,
    n_samples: int,
    latency_ms: float,
    status: str = "ok",
    error: str = "",
) -> None:
    """
    Log one /predict API call.

    Parameters
    ----------
    model_name  : currently loaded model name
    n_samples   : number of rows in the prediction request
    latency_ms  : end-to-end inference latency in milliseconds
    status      : "ok" or "error"
    error       : error message if status == "error"
    """
    _write({
        "event":      "prediction",
        "model_name": model_name,
        "n_samples":  n_samples,
        "latency_ms": round(latency_ms, 2),
        "status":     status,
        "error":      error,
    })


def log_api_request(
    endpoint: str,
    method: str,
    status_code: int,
    latency_ms: float,
) -> None:
    """
    Log one HTTP request received by the inference API.

    Parameters
    ----------
    endpoint    : request path (e.g. "/predict")
    method      : HTTP verb (e.g. "POST")
    status_code : HTTP response status code
    latency_ms  : time from request receipt to response sent
    """
    _write({
        "event":       "api_request",
        "endpoint":    endpoint,
        "method":      method,
        "status_code": status_code,
        "latency_ms":  round(latency_ms, 2),
    })


def read_recent(n: int = 100) -> list:
    """
    Return the last *n* entries from the metrics log as a list of dicts.
    Returns an empty list if the file does not exist or cannot be parsed.
    """
    if not os.path.isfile(METRICS_FILE):
        return []
    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        entries = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries
    except Exception as exc:
        log.warning("read_recent failed: %s", exc)
        return []


def clear_log() -> None:
    """
    Truncate the metrics log file.
    Useful for test isolation — not intended for production use.
    """
    with _LOCK:
        try:
            open(METRICS_FILE, "w").close()
        except Exception:
            pass
