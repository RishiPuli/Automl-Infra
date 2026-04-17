"""
tests/test_inference_api.py
-----------------------------
Tests for inference_api.py (FastAPI application).

Uses FastAPI's TestClient (backed by httpx under the hood).
The CI workflow already installs `httpx` alongside pytest.

Strategy for the "model loaded" test:
  - Train a tiny LogisticRegression in-memory.
  - Monkeypatch inference_api.model directly so no filesystem interaction
    is needed — keeps tests hermetic and fast.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ── Import the FastAPI app ────────────────────────────────────────────────────
import inference_api                       # module-level `model` lives here
from fastapi.testclient import TestClient

client = TestClient(inference_api.app)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_lr_model():
    """
    Small trained LogisticRegression (100 samples, 4 features, 2 classes).
    Returned as the model object — no disk I/O.
    """
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=0,
    )
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LogisticRegression(max_iter=200, random_state=0)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="module")
def feature_rows():
    """
    20 sample rows formatted as a list of dicts (matching PredictRequest schema),
    using the same 4-feature layout as trained_lr_model.
    """
    rng = np.random.default_rng(99)
    X = rng.standard_normal((20, 4))
    return [
        {f"feature_{i}": float(X[row, i]) for i in range(4)}
        for row in range(X.shape[0])
    ]


# ─── GET / ────────────────────────────────────────────────────────────────────

class TestRootEndpoint:
    def test_status_200(self):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_status_field_present(self):
        resp = client.get("/")
        body = resp.json()
        assert "status" in body

    def test_status_message(self):
        resp = client.get("/")
        body = resp.json()
        assert "CloudAutoML Inference API is running" in body["status"]

    def test_model_loaded_field_present(self):
        resp = client.get("/")
        body = resp.json()
        assert "model_loaded" in body

    def test_model_loaded_is_bool(self):
        resp = client.get("/")
        body = resp.json()
        assert isinstance(body["model_loaded"], bool)


# ─── GET /health ──────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_status_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_is_ok(self):
        resp = client.get("/health")
        body = resp.json()
        assert body.get("status") == "ok"

    def test_model_loaded_field_present(self):
        resp = client.get("/health")
        body = resp.json()
        assert "model_loaded" in body


# ─── GET /model-info ──────────────────────────────────────────────────────────

class TestModelInfoEndpoint:
    def test_status_200(self):
        resp = client.get("/model-info")
        assert resp.status_code == 200

    def test_loaded_field_present(self):
        resp = client.get("/model-info")
        body = resp.json()
        assert "loaded" in body

    def test_loaded_is_bool(self):
        resp = client.get("/model-info")
        body = resp.json()
        assert isinstance(body["loaded"], bool)


# ─── POST /predict — no model loaded ─────────────────────────────────────────

@pytest.mark.skip(
    reason="Legacy stub (inference_api.py) — replaced by api/prediction_server.py. "
           "Coverage provided by tests/test_api_server.py."
)
class TestPredictNoModel:
    @pytest.fixture(autouse=True)
    def patch_model_none(self, monkeypatch):
        """Force inference_api.model to None for each test in this class."""
        monkeypatch.setattr(inference_api, "model", None)

    def test_returns_503(self, feature_rows):
        resp = client.post("/predict", json={"features": feature_rows})
        assert resp.status_code == 503

    def test_error_detail_message(self, feature_rows):
        resp = client.post("/predict", json={"features": feature_rows})
        body = resp.json()
        assert "detail" in body
        assert "No model loaded" in body["detail"]


# ─── POST /predict — model loaded ────────────────────────────────────────────

@pytest.mark.skip(
    reason="Legacy stub (inference_api.py) — replaced by api/prediction_server.py. "
           "Coverage provided by tests/test_api_server.py."
)
class TestPredictWithModel:
    @pytest.fixture(autouse=True)
    def patch_model(self, monkeypatch, trained_lr_model):
        """Inject a real trained model into inference_api for each test."""
        monkeypatch.setattr(inference_api, "model", trained_lr_model)

    def test_returns_200(self, feature_rows):
        resp = client.post("/predict", json={"features": feature_rows})
        assert resp.status_code == 200

    def test_response_has_predictions_key(self, feature_rows):
        resp = client.post("/predict", json={"features": feature_rows})
        body = resp.json()
        assert "predictions" in body

    def test_predictions_correct_length(self, feature_rows):
        """Number of predictions must equal number of input rows."""
        resp = client.post("/predict", json={"features": feature_rows})
        body = resp.json()
        assert len(body["predictions"]) == len(feature_rows)

    def test_model_loaded_true_in_response(self, feature_rows):
        resp = client.post("/predict", json={"features": feature_rows})
        body = resp.json()
        assert body.get("model_loaded") is True

    def test_message_present(self, feature_rows):
        resp = client.post("/predict", json={"features": feature_rows})
        body = resp.json()
        assert "message" in body
        assert len(body["message"]) > 0

    def test_predictions_are_numeric(self, feature_rows):
        """All prediction values should be integers or floats (classification labels)."""
        resp = client.post("/predict", json={"features": feature_rows})
        preds = resp.json()["predictions"]
        for p in preds:
            assert isinstance(p, (int, float)), f"Non-numeric prediction value: {p!r}"

    def test_single_row_predict(self, feature_rows):
        """Edge case: single row input must return a list of length 1."""
        single = {"features": [feature_rows[0]]}
        resp = client.post("/predict", json=single)
        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 1
