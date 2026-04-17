"""
api/prediction_server.py
-------------------------
FastAPI Production Prediction Server for CloudAutoML.

Loads trained artifacts saved by the orchestrator after training.

Endpoints
---------
  GET  /health        — liveness check (model name, task type, loaded flag)
  GET  /model-info    — full deployment metadata dict
  POST /predict       — single or batch prediction (JSON)
  GET  /metrics       — operational metrics (latency percentiles, error rate)

Security features
-----------------
  - X-Request-ID header injected on every response (distributed tracing)
  - X-Response-Time-MS latency header for SLA monitoring
  - /predict validates incoming feature names against training-time schema
  - CORS enabled (configurable via environment variable in production)

Run
---
    uvicorn api.prediction_server:app --host 0.0.0.0 --port 8000 --reload

Or from project root:
    python api/prediction_server.py
"""

import os
import sys
import json
import time
import uuid
import threading
from contextlib import asynccontextmanager

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import joblib
from utils.logger import get_logger

log = get_logger("PredictionServer")

# ── Try importing FastAPI ──────────────────────────────────────────────────────
FASTAPI_AVAILABLE = False
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    log.error("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")


# ── Runtime counters for /metrics endpoint ────────────────────────────────────
_COUNTERS = {
    "requests_total":      0,
    "predict_calls":       0,
    "predict_errors":      0,
    "predict_latency_ms":  [],   # rolling list (capped at 500 entries)
    "server_start_time":   time.time(),
}
_COUNTER_LOCK = threading.Lock()


def _inc(key: str, value: float = 1):
    """Thread-safe counter increment / latency append."""
    with _COUNTER_LOCK:
        if isinstance(_COUNTERS[key], list):
            _COUNTERS[key].append(value)
            if len(_COUNTERS[key]) > 500:
                _COUNTERS[key] = _COUNTERS[key][-500:]
        else:
            _COUNTERS[key] += value


# ── Artifact loading ──────────────────────────────────────────────────────────
_model        = None
_preprocessor = None
_label_enc    = None
_meta: dict   = {}

META_PATH = os.path.join("models", "deployment_meta.json")


def _load():
    global _model, _preprocessor, _label_enc, _meta
    if not os.path.exists(META_PATH):
        log.warning("No deployment_meta.json — run CloudAutoML training first.")
        return

    with open(META_PATH) as f:
        _meta = json.load(f)

    mp = _meta.get("model_path", "")
    if mp and os.path.exists(mp):
        _model = joblib.load(mp)
        log.info("Model loaded: %s", mp)

    pp = _meta.get("preprocessor_path", "")
    if pp and os.path.exists(pp):
        _preprocessor = joblib.load(pp)
        log.info("Preprocessor loaded: %s", pp)

    lp = _meta.get("label_encoder_path", "")
    if lp and os.path.exists(lp):
        _label_enc = joblib.load(lp)
        log.info("LabelEncoder loaded: %s", lp)


# ── FastAPI app ───────────────────────────────────────────────────────────────
if FASTAPI_AVAILABLE:
    @asynccontextmanager
    async def _lifespan(app):
        """Modern FastAPI lifespan handler — loads model artifacts at startup."""
        _load()
        yield

    app = FastAPI(
        title="CloudAutoML Prediction API",
        description="Production REST API — AutoML trained model inference",
        version="2.0.0",
        lifespan=_lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Security middleware: X-Request-ID + response time header ──────────────
    @app.middleware("http")
    async def request_instrumentation(request: Request, call_next):
        """
        Injects two security/observability headers into every response:
          X-Request-ID      — UUID for distributed tracing and log correlation
          X-Response-Time-MS — actual server-side latency in milliseconds

        Also increments the global request counter and appends a JSONL entry
        to logs/metrics.jsonl for every request, regardless of outcome.
        """
        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-MS"] = str(latency_ms)
        _inc("requests_total")
        try:
            from utils.metrics_logger import log_api_request
            log_api_request(
                endpoint=str(request.url.path),
                method=request.method,
                status_code=response.status_code,
                latency_ms=latency_ms,
            )
        except Exception:
            pass
        return response

    @asynccontextmanager
    async def _lifespan(app):
        """Modern FastAPI lifespan handler (replaces deprecated on_event)."""
        _load()          # Load model artifacts at startup
        yield
        # (shutdown cleanup can go here if needed)

    # ── Request schema ────────────────────────────────────────────────────────
    class PredictRequest(BaseModel):
        data: list          # list of dicts (records format) OR list of lists
        feature_names: list = []

    # ── Endpoints ─────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        """Liveness probe used by Render, Docker HEALTHCHECK, and CI."""
        return {
            "status":       "ok",
            "model_loaded": _model is not None,
            "model_name":   _meta.get("model_name", "none"),
            "task_type":    _meta.get("task_type",  "unknown"),
        }

    @app.get("/model-info")
    async def model_info():
        """Return full deployment metadata saved by the orchestrator."""
        if not _meta:
            raise HTTPException(503, "No model loaded. Run CloudAutoML training first.")
        return _meta

    @app.post("/predict")
    async def predict(req: PredictRequest):
        """
        Accept a JSON payload and return model predictions.

        Accepts two payload formats:
          - List of dicts: [{\"col_a\": 1.2, \"col_b\": 3.4}, ...]
          - List of lists: [[1.2, 3.4], ...] with feature_names supplied

        Validates incoming feature names against the training-time schema.
        Returns predictions, probabilities (if available), and inference latency.
        """
        if _model is None:
            raise HTTPException(503, "Model not loaded. Run training and restart server.")

        _inc("predict_calls")
        t_start = time.perf_counter()

        try:
            # Accept list-of-dicts or list-of-lists
            if req.data and isinstance(req.data[0], dict):
                df = pd.DataFrame(req.data)
            else:
                names = (req.feature_names
                         or _meta.get("feature_names", [])
                         or [f"f{i}" for i in range(len(req.data[0]))])
                df = pd.DataFrame(req.data, columns=names)

            # Security: validate feature schema against training-time schema
            expected_features = _meta.get("feature_names", [])
            if expected_features:
                incoming = set(df.columns.tolist())
                expected = set(expected_features)
                missing = expected - incoming
                extra   = incoming - expected
                if missing or extra:
                    raise ValueError(
                        f"Feature schema mismatch. "
                        f"Missing: {sorted(missing)}. Extra: {sorted(extra)}."
                    )

            X = _preprocessor.transform(df) if _preprocessor is not None else df.values
            preds = _model.predict(X)

            # Probabilities (if available)
            proba = None
            if hasattr(_model, "predict_proba"):
                proba = _model.predict_proba(X).tolist()

            # Decode labels
            if _label_enc is not None:
                preds_out = _label_enc.inverse_transform(preds.astype(int)).tolist()
            else:
                preds_out = preds.tolist()

            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            _inc("predict_latency_ms", latency_ms)

            try:
                from utils.metrics_logger import log_prediction
                log_prediction(
                    model_name=_meta.get("model_name", type(_model).__name__),
                    n_samples=len(preds_out),
                    latency_ms=latency_ms,
                    status="ok",
                )
            except Exception:
                pass

            return {
                "predictions":   preds_out,
                "probabilities": proba,
                "model":         _meta.get("model_name", "unknown"),
                "task_type":     _meta.get("task_type",  "unknown"),
                "n_samples":     len(preds_out),
                "latency_ms":    latency_ms,
            }

        except Exception as e:
            _inc("predict_errors")
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
            try:
                from utils.metrics_logger import log_prediction
                log_prediction(
                    model_name=_meta.get("model_name", "unknown"),
                    n_samples=len(req.data),
                    latency_ms=latency_ms,
                    status="error",
                    error=str(e),
                )
            except Exception:
                pass
            raise HTTPException(400, str(e))

    @app.get("/metrics")
    async def metrics():
        """
        Operational monitoring endpoint.

        Returns runtime counters, latency percentiles (p50/p95/p99), current
        error rate, and the last 20 structured events from logs/metrics.jsonl.

        Designed to be scraped by a Prometheus exporter, a monitoring dashboard,
        or a CI health-check script.  Does not require authentication in this
        prototype — in production, protect with an API key middleware.
        """
        with _COUNTER_LOCK:
            latencies         = list(_COUNTERS["predict_latency_ms"])
            counters_snapshot = {
                k: v for k, v in _COUNTERS.items()
                if not isinstance(v, list)
            }

        latency_stats = {}
        if latencies:
            arr = sorted(latencies)
            n   = len(arr)
            latency_stats = {
                "p50_ms":  round(arr[n // 2], 2),
                "p95_ms":  round(arr[min(int(n * 0.95), n - 1)], 2),
                "p99_ms":  round(arr[min(int(n * 0.99), n - 1)], 2),
                "avg_ms":  round(sum(arr) / n, 2),
                "count":   n,
            }

        recent_logs = []
        try:
            from utils.metrics_logger import read_recent
            recent_logs = read_recent(20)
        except Exception:
            pass

        uptime_s   = round(time.time() - _COUNTERS["server_start_time"], 1)
        calls      = counters_snapshot.get("predict_calls", 0)
        errors     = counters_snapshot.get("predict_errors", 0)
        error_rate = round(errors / calls, 4) if calls > 0 else 0.0

        return {
            "uptime_seconds": uptime_s,
            "model_loaded":   _model is not None,
            "model_name":     _meta.get("model_name", "none"),
            "counters":       counters_snapshot,
            "latency":        latency_stats,
            "error_rate":     error_rate,
            "recent_events":  recent_logs,
        }

    # ── Dev runner ────────────────────────────────────────────────────────────
    if __name__ == "__main__":
        _load()  # pre-load before server starts
        uvicorn.run("api.prediction_server:app",
                    host="0.0.0.0", port=8000, reload=False)
else:
    # Stub so imports don't crash when FastAPI isn't installed
    app = None
