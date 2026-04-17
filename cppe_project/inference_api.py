"""
inference_api.py  (v2)
-----------------------
CloudAutoML Inference API — FastAPI prediction endpoint.

Improvements over v1
---------------------
- Loads deployment_meta.json produced by the orchestrator at training time
  so the preprocessor and label encoder are also available, making predictions
  on raw (unscaled) data correct.
- Falls back gracefully to best_model.pkl only when meta file is absent
  (e.g. models trained with an older version of the pipeline).
- Structured prediction latency logging via utils/metrics_logger.py.

Run locally
-----------
    uvicorn inference_api:app --reload

Deploy on Render
----------------
    Build:  pip install -r requirements_api.txt
    Start:  uvicorn inference_api:app --host 0.0.0.0 --port $PORT
"""

import os
import sys
import time
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib
import sklearn
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CloudAutoML Inference API",
    description=(
        "Production REST endpoint that serves predictions from the best "
        "model trained by the CloudAutoML pipeline."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Artifact loading ──────────────────────────────────────────────────────────

MODEL_PATH = os.path.join("models", "best_model.pkl")
META_PATH  = os.path.join("models", "deployment_meta.json")

_model       = None
_preprocessor = None
_label_enc   = None
_meta: dict  = {}


def _load_artifacts():
    """
    Load model artifacts at startup.

    Priority order:
      1. deployment_meta.json  — loads model + preprocessor + label encoder
      2. best_model.pkl        — model only (legacy / no meta file)
    """
    global _model, _preprocessor, _label_enc, _meta

    if os.path.isfile(META_PATH):
        try:
            with open(META_PATH) as fh:
                _meta = json.load(fh)

            mp = _meta.get("model_path", "")
            if mp and os.path.isfile(mp):
                _model = joblib.load(mp)
                print(f"[INFO] Model loaded: {mp}")

            pp = _meta.get("preprocessor_path", "")
            if pp and os.path.isfile(pp):
                _preprocessor = joblib.load(pp)
                print(f"[INFO] Preprocessor loaded: {pp}")

            lp = _meta.get("label_encoder_path", "")
            if lp and os.path.isfile(lp):
                _label_enc = joblib.load(lp)
                print(f"[INFO] LabelEncoder loaded: {lp}")

            return
        except Exception as exc:
            print(f"[WARN] Could not load from deployment_meta.json: {exc}")

    # Fallback: model only
    if os.path.isfile(MODEL_PATH):
        try:
            _model = joblib.load(MODEL_PATH)
            print(f"[INFO] Model loaded (no meta): {MODEL_PATH}")
        except Exception as exc:
            print(f"[ERROR] Failed to load model: {exc}")
    else:
        print(f"[WARN] No model at {MODEL_PATH}. Train first, then restart.")


_load_artifacts()


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: List[Dict[str, Any]]
    """List of row dicts, e.g. [{"col_a": 1.2, "col_b": 3.4}, ...]"""


class PredictResponse(BaseModel):
    predictions:  List[Any]
    probabilities: Any          # list of lists for clf, None for reg
    model_loaded: bool
    model_name:   str
    message:      str
    latency_ms:   float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Welcome / liveness check."""
    return {
        "status":       "CloudAutoML Inference API is running",
        "model_loaded": _model is not None,
        "model_name":   _meta.get("model_name", "none"),
        "version":      "2.0.0",
    }


@app.get("/health")
def health():
    """Liveness probe for Render and external monitors."""
    return {
        "status":       "ok",
        "model_loaded": _model is not None,
        "model_name":   _meta.get("model_name", "none"),
    }


@app.get("/model-info")
def model_info():
    """Metadata about the loaded model artifact."""
    if _model is None:
        return {"loaded": False}
    return {
        "loaded":            True,
        "model_class":       type(_model).__name__,
        "model_name":        _meta.get("model_name", type(_model).__name__),
        "task_type":         _meta.get("task_type", "unknown"),
        "sklearn_version":   sklearn.__version__,
        "n_features":        _meta.get("n_features", "unknown"),
        "metrics":           _meta.get("metrics", {}),
        "cv_mean":           _meta.get("cv_mean", 0.0),
        "preprocessor":      _preprocessor is not None,
        "label_encoder":     _label_enc is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    """
    Accept a list of row dicts and return predictions.

    - HTTP 503 when no model is loaded.
    - HTTP 400 when input is malformed or prediction fails.
    - Logs latency and prediction count to logs/metrics.jsonl.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Train a model first, then restart the API.",
        )

    t_start = time.perf_counter()

    try:
        df = pd.DataFrame(req.features)

        # Apply the preprocessor when available (trained on scaled data)
        if _preprocessor is not None:
            X = _preprocessor.transform(df)
        else:
            X = df.values

        raw_preds = _model.predict(X)

        # Decode integer labels back to original class names
        if _label_enc is not None:
            preds_out = _label_enc.inverse_transform(
                raw_preds.astype(int)
            ).tolist()
        else:
            preds_out = raw_preds.tolist()

        # Optional class probabilities
        proba = None
        if hasattr(_model, "predict_proba"):
            try:
                proba = _model.predict_proba(X).tolist()
            except Exception:
                pass

        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        # Structured metrics logging
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

        return PredictResponse(
            predictions=preds_out,
            probabilities=proba,
            model_loaded=True,
            model_name=_meta.get("model_name", type(_model).__name__),
            message=f"Predicted {len(preds_out)} sample(s) successfully.",
            latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        try:
            from utils.metrics_logger import log_prediction
            log_prediction(
                model_name=_meta.get("model_name", "unknown"),
                n_samples=len(req.features),
                latency_ms=latency_ms,
                status="error",
                error=str(exc),
            )
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
