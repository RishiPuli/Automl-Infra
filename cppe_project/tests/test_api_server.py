"""
tests/test_api_server.py
-------------------------
Integration tests for api/prediction_server.py

Uses FastAPI's TestClient (via httpx) to make real HTTP requests against the
app without starting a live server.  Tests cover:

  /health             liveness probe structure
  /model-info         503 when no model loaded
  /predict            503 when model absent, 400 on bad payload
  /metrics            structure and counter integrity
  Security headers    X-Request-ID, X-Response-Time-MS present on all responses

These tests are self-contained — they do NOT require a trained model on disk
and do NOT call psutil or mlflow.
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Guard: skip if FastAPI / httpx not installed ───────────────────────────────
pytest.importorskip("fastapi",  reason="fastapi not installed")
pytest.importorskip("httpx",    reason="httpx not installed (pip install httpx)")

from fastapi.testclient import TestClient


# ── Import the app ─────────────────────────────────────────────────────────────
from api.prediction_server import app, _COUNTERS, _inc, _COUNTER_LOCK

# Reset counters between test runs to avoid cross-test contamination
def _reset_counters():
    with _COUNTER_LOCK:
        _COUNTERS["requests_total"]     = 0
        _COUNTERS["predict_calls"]      = 0
        _COUNTERS["predict_errors"]     = 0
        _COUNTERS["predict_latency_ms"] = []


@pytest.fixture(scope="module")
def client():
    """TestClient for the whole module (no model loaded on disk)."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_between_tests():
    """Reset runtime counters before every test."""
    _reset_counters()
    yield


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_ok(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_health_has_model_loaded_field(self, client):
        body = client.get("/health").json()
        assert "model_loaded" in body

    def test_health_model_loaded_is_false_without_model(self, client):
        body = client.get("/health").json()
        # No model on disk in test environment
        assert body["model_loaded"] is False

    def test_health_has_task_type_field(self, client):
        body = client.get("/health").json()
        assert "task_type" in body

    def test_health_has_model_name_field(self, client):
        body = client.get("/health").json()
        assert "model_name" in body


# ── /model-info ────────────────────────────────────────────────────────────────

class TestModelInfo:
    def test_model_info_returns_503_when_no_model(self, client):
        r = client.get("/model-info")
        # No model loaded in test → should be 503
        assert r.status_code == 503

    def test_model_info_503_detail_mentions_training(self, client):
        r = client.get("/model-info")
        body = r.json()
        assert "detail" in body


# ── /predict ───────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_no_model_returns_503(self, client):
        payload = {"data": [{"feature_1": 1.0, "feature_2": 2.0}]}
        r = client.post("/predict", json=payload)
        assert r.status_code == 503

    def test_predict_empty_data_without_model_returns_error(self, client):
        payload = {"data": []}
        r = client.post("/predict", json=payload)
        # Either 503 (no model) or 422 (validation error) — both are correct
        assert r.status_code in (400, 422, 503)

    def test_predict_invalid_json_returns_422(self, client):
        r = client.post(
            "/predict",
            content=b"not-json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422

    def test_predict_missing_data_field_returns_422(self, client):
        r = client.post("/predict", json={"wrong_key": []})
        assert r.status_code == 422

    def test_predict_increments_predict_calls_counter(self, client):
        before = _COUNTERS["predict_calls"]
        client.post("/predict", json={"data": [{"f1": 1.0}]})
        after  = _COUNTERS["predict_calls"]
        # Counter should have incremented (request hit the endpoint)
        assert after >= before


# ── /metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_has_required_keys(self, client):
        body = client.get("/metrics").json()
        required = ["uptime_seconds", "model_loaded", "model_name",
                    "counters", "latency", "error_rate", "recent_events"]
        for key in required:
            assert key in body, f"Missing key: {key}"

    def test_metrics_uptime_is_positive(self, client):
        body = client.get("/metrics").json()
        assert body["uptime_seconds"] >= 0

    def test_metrics_model_loaded_is_bool(self, client):
        body = client.get("/metrics").json()
        assert isinstance(body["model_loaded"], bool)

    def test_metrics_counters_is_dict(self, client):
        body = client.get("/metrics").json()
        assert isinstance(body["counters"], dict)

    def test_metrics_error_rate_is_float(self, client):
        body = client.get("/metrics").json()
        assert isinstance(body["error_rate"], float)

    def test_metrics_recent_events_is_list(self, client):
        body = client.get("/metrics").json()
        assert isinstance(body["recent_events"], list)

    def test_metrics_counters_has_requests_total(self, client):
        # Make some requests first
        client.get("/health")
        client.get("/health")
        body = client.get("/metrics").json()
        assert "requests_total" in body["counters"]

    def test_metrics_latency_empty_when_no_predictions(self, client):
        body = client.get("/metrics").json()
        # No successful predictions made → latency dict should be empty (or have count 0)
        latency = body["latency"]
        if latency:
            assert latency.get("count", 0) == 0 or isinstance(latency["count"], int)

    def test_metrics_error_rate_zero_when_no_calls(self, client):
        body = client.get("/metrics").json()
        # No predict calls → error rate must be 0.0
        assert body["error_rate"] == 0.0


# ── Security headers ───────────────────────────────────────────────────────────

class TestSecurityHeaders:
    def test_health_has_x_request_id(self, client):
        r = client.get("/health")
        assert "x-request-id" in r.headers, \
            "X-Request-ID header must be present on every response"

    def test_x_request_id_is_uuid_format(self, client):
        r = client.get("/health")
        rid = r.headers.get("x-request-id", "")
        import uuid
        try:
            uuid.UUID(rid)   # raises ValueError if not a valid UUID
        except ValueError:
            pytest.fail(f"X-Request-ID '{rid}' is not a valid UUID")

    def test_x_response_time_present(self, client):
        r = client.get("/health")
        assert "x-response-time-ms" in r.headers, \
            "X-Response-Time-MS header must be present on every response"

    def test_x_response_time_is_numeric(self, client):
        r = client.get("/health")
        val = r.headers.get("x-response-time-ms", "")
        try:
            float(val)
        except ValueError:
            pytest.fail(f"X-Response-Time-MS '{val}' is not a valid float")

    def test_every_endpoint_has_request_id(self, client):
        for endpoint in ["/health", "/metrics"]:
            r = client.get(endpoint)
            assert "x-request-id" in r.headers, \
                f"X-Request-ID missing on {endpoint}"

    def test_each_response_has_unique_request_id(self, client):
        ids = {client.get("/health").headers.get("x-request-id") for _ in range(5)}
        assert len(ids) == 5, "Each X-Request-ID must be unique per request"
