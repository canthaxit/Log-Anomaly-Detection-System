#!/usr/bin/env python3
"""
REST API for Log Anomaly Detection
FastAPI server for integration with any platform
"""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import joblib
import pandas as pd
from io import StringIO

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    from typing import Literal
    import uvicorn
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
except ImportError:
    print("FastAPI not installed. Install with: pip install fastapi uvicorn python-multipart slowapi")
    exit(1)

# Import anomaly detection components
from log_anomaly_detection_lite import (
    LogParser,
    LogFeaturePipeline,
    StatisticalAnomalyDetector,
    AnomalyScorer,
    preprocess_logs
)
from sklearn.ensemble import IsolationForest

# Import security utilities
import sys, os
_parent = os.path.join(os.path.dirname(__file__), '..')
if os.path.isfile(os.path.join(_parent, 'common', 'security.py')):
    sys.path.insert(0, _parent)
from common.security import validate_model_path, verify_model_file, get_verify_api_key, ALLOWED_ORIGINS
from fastapi import Depends

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anomaly-api")

# Initialize FastAPI
app = FastAPI(
    title="Log Anomaly Detection API",
    description="AI-powered security threat detection for system logs",
    version="1.0.0"
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


# Thread safety lock for MODEL_STATE
_model_lock = threading.Lock()

# Enable CORS (restricted origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "Accept"],
)

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
    return response


# Auth dependency
verify_api_key = get_verify_api_key()

# Global state
MODEL_STATE = {
    "feature_pipeline": None,
    "isolation_forest": None,
    "statistical_detector": None,
    "scorer": None,
    "threshold": None,
    "loaded": False,
    "loaded_at": None
}


# Pydantic Models
class LogEvent(BaseModel):
    """Single log event."""
    timestamp: str = Field(..., max_length=64)
    user: str = Field(..., min_length=1, max_length=128)
    source_ip: str = Field(..., max_length=45)
    dest_ip: Optional[str] = Field("unknown", max_length=45)
    event_type: str = Field(..., max_length=64)
    action: str = Field(..., max_length=64)
    message: str = Field(..., max_length=2048)
    severity: Optional[Literal["low", "medium", "high", "critical"]] = "low"


MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_LOG_EVENTS = 10_000


class AnalysisRequest(BaseModel):
    """Request for log analysis."""
    logs: List[LogEvent] = Field(..., description="List of log events to analyze", max_length=MAX_LOG_EVENTS)
    return_all_events: bool = Field(False, description="Return all events with scores, not just anomalies")


class Anomaly(BaseModel):
    """Detected anomaly."""
    timestamp: str
    user: str
    source_ip: str
    dest_ip: str
    event_type: str
    action: str
    message: str
    severity: str
    anomaly_score: float
    threat_type: str


class AnalysisResponse(BaseModel):
    """Analysis results."""
    status: str
    total_events: int
    anomalies_detected: int
    anomaly_rate: float
    threshold: float
    anomalies: List[Anomaly]
    processing_time_ms: float


class ModelInfo(BaseModel):
    """Model information."""
    loaded: bool
    loaded_at: Optional[str]
    threshold: Optional[float]
    n_features: Optional[int]
    feature_names: Optional[List[str]]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    timestamp: str


# Utility Functions
def classify_threat(row: pd.Series) -> str:
    """Classify threat type."""
    if 'failed' in str(row.get('action', '')).lower():
        return 'brute_force'
    elif 'sudo' in str(row.get('message', '')).lower():
        return 'privilege_escalation'
    elif any(word in str(row.get('message', '')).lower() for word in ['shadow', 'passwd', 'secret']):
        return 'data_exfiltration'
    elif row.get('event_type') == 'network':
        return 'lateral_movement'
    else:
        return 'unknown'


def assign_severity(score: float) -> str:
    """Assign severity based on score."""
    if score >= 0.95:
        return 'critical'
    elif score >= 0.85:
        return 'high'
    elif score >= 0.7:
        return 'medium'
    else:
        return 'low'


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=MODEL_STATE["loaded"],
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/models/info", response_model=ModelInfo, dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def get_model_info(request: Request):
    """Get information about loaded models."""
    with _model_lock:
        snapshot = dict(MODEL_STATE)
    if not snapshot["loaded"]:
        return ModelInfo(loaded=False, loaded_at=None, threshold=None, n_features=None, feature_names=None)

    return ModelInfo(
        loaded=True,
        loaded_at=snapshot["loaded_at"],
        threshold=float(snapshot["threshold"]) if snapshot["threshold"] else None,
        n_features=len(snapshot["feature_pipeline"].feature_names_),
        feature_names=snapshot["feature_pipeline"].feature_names_
    )


@app.post("/models/load", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def load_models(request: Request, model_dir: str = "anomaly_outputs"):
    """Load trained models from disk."""
    try:
        model_path = validate_model_path(model_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Model directory is outside allowed paths")

    try:
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model directory not found")

        with _model_lock:
            for pkl in ("feature_pipeline.pkl", "isolation_forest_model.pkl", "statistical_detector.pkl"):
                verify_model_file(model_path / pkl)
            MODEL_STATE["feature_pipeline"] = joblib.load(model_path / "feature_pipeline.pkl")
            MODEL_STATE["isolation_forest"] = joblib.load(model_path / "isolation_forest_model.pkl")
            MODEL_STATE["statistical_detector"] = joblib.load(model_path / "statistical_detector.pkl")

            # Load inference package
            if (model_path / "inference_package.pkl").exists():
                verify_model_file(model_path / "inference_package.pkl")
                package = joblib.load(model_path / "inference_package.pkl")
                MODEL_STATE["scorer"] = package.get("scorer")
                MODEL_STATE["threshold"] = package.get("threshold")
            else:
                MODEL_STATE["scorer"] = AnomalyScorer()
                MODEL_STATE["threshold"] = 0.7

            MODEL_STATE["loaded"] = True
            MODEL_STATE["loaded_at"] = datetime.utcnow().isoformat()

        logger.info(f"Models loaded successfully from {model_dir}")

        return {
            "status": "success",
            "message": f"Models loaded from {model_dir}",
            "loaded_at": MODEL_STATE["loaded_at"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/analyze", response_model=AnalysisResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def analyze_logs(request: Request, analysis: AnalysisRequest):
    """Analyze logs for anomalies."""
    start_time = datetime.utcnow()

    # Atomic snapshot of MODEL_STATE
    with _model_lock:
        model_snap = dict(MODEL_STATE)

    if not model_snap["loaded"]:
        raise HTTPException(
            status_code=400,
            detail="Models not loaded. Call POST /models/load first."
        )

    try:
        # Convert request to DataFrame
        logs_dict = [log.dict() for log in analysis.logs]
        df = pd.DataFrame(logs_dict)

        # Preprocess
        parser = LogParser()
        df = parser._normalize_schema(df)
        df = preprocess_logs(df)

        if len(df) == 0:
            return AnalysisResponse(
                status="success",
                total_events=0,
                anomalies_detected=0,
                anomaly_rate=0.0,
                threshold=float(model_snap["threshold"]),
                anomalies=[],
                processing_time_ms=0.0
            )

        # Extract features
        features = model_snap["feature_pipeline"].transform(df)

        # Detect anomalies
        iso_scores = -model_snap["isolation_forest"].score_samples(features)
        stat_scores = model_snap["statistical_detector"].detect_all(df)

        # Combine scores
        combined_scores = model_snap["scorer"].combine_scores({
            'isolation_forest': iso_scores,
            'statistical': stat_scores
        })

        # Identify anomalies
        threshold = model_snap["threshold"]
        is_anomaly = combined_scores > threshold

        # Build response
        if analysis.return_all_events:
            result_df = df.copy()
            result_df['anomaly_score'] = combined_scores
            result_df['is_anomaly'] = is_anomaly
        else:
            result_df = df[is_anomaly].copy()
            result_df['anomaly_score'] = combined_scores[is_anomaly]

        # Classify threats and assign severity
        result_df['threat_type'] = result_df.apply(classify_threat, axis=1)
        result_df['severity'] = result_df['anomaly_score'].apply(assign_severity)

        # Convert to response format
        anomalies = []
        for _, row in result_df.iterrows():
            if analysis.return_all_events or row.get('is_anomaly', True):
                anomalies.append(Anomaly(
                    timestamp=str(row['timestamp']),
                    user=row['user'],
                    source_ip=row['source_ip'],
                    dest_ip=row.get('dest_ip', 'unknown'),
                    event_type=row['event_type'],
                    action=row['action'],
                    message=row['message'],
                    severity=row['severity'],
                    anomaly_score=float(row['anomaly_score']),
                    threat_type=row['threat_type']
                ))

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return AnalysisResponse(
            status="success",
            total_events=len(df),
            anomalies_detected=int(is_anomaly.sum()),
            anomaly_rate=float(is_anomaly.sum() / len(df)),
            threshold=float(threshold),
            anomalies=anomalies if not analysis.return_all_events else [a for a in anomalies if a.anomaly_score > threshold],
            processing_time_ms=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/analyze/file", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def analyze_file(request: Request, file: UploadFile = File(...)):
    """Analyze logs from uploaded file."""
    if not MODEL_STATE["loaded"]:
        raise HTTPException(
            status_code=400,
            detail="Models not loaded. Call POST /models/load first."
        )

    try:
        # Read file with size check
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)} MB")
        content_str = content.decode('utf-8')

        # Null filename check
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Parse based on file extension
        if file.filename.endswith('.json'):
            logs = json.loads(content_str)
            if isinstance(logs, dict):
                logs = [logs]
            df = pd.DataFrame(logs)
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(StringIO(content_str))
            if len(df.columns) > 50:
                raise HTTPException(status_code=400, detail="CSV has too many columns (max 50)")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use JSON or CSV.")

        if len(df) > MAX_LOG_EVENTS:
            raise HTTPException(status_code=413, detail=f"Too many log events. Maximum is {MAX_LOG_EVENTS}")

        # Convert to request format
        logs_list = df.to_dict(orient='records')
        log_events = [LogEvent(**log) for log in logs_list]

        # Use existing analyze endpoint
        analysis_req = AnalysisRequest(logs=log_events)
        return await analyze_logs(request, analysis_req)

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        logger.error(f"File analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/stats", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def get_statistics(request: Request):
    """Get system statistics."""
    with _model_lock:
        snapshot = dict(MODEL_STATE)
    if not snapshot["loaded"]:
        raise HTTPException(status_code=400, detail="Models not loaded")

    return {
        "models_loaded": True,
        "loaded_at": snapshot["loaded_at"],
        "threshold": float(snapshot["threshold"]),
        "isolation_forest": {
            "n_estimators": snapshot["isolation_forest"].n_estimators,
            "contamination": snapshot["isolation_forest"].contamination
        },
        "features": {
            "count": len(snapshot["feature_pipeline"].feature_names_),
            "names": snapshot["feature_pipeline"].feature_names_
        }
    }


if __name__ == "__main__":
    # Auto-load models if available
    try:
        model_path = validate_model_path("anomaly_outputs")
        if model_path.exists():
            for pkl in ("feature_pipeline.pkl", "isolation_forest_model.pkl", "statistical_detector.pkl"):
                verify_model_file(model_path / pkl)
            with _model_lock:
                MODEL_STATE["feature_pipeline"] = joblib.load(model_path / "feature_pipeline.pkl")
                MODEL_STATE["isolation_forest"] = joblib.load(model_path / "isolation_forest_model.pkl")
                MODEL_STATE["statistical_detector"] = joblib.load(model_path / "statistical_detector.pkl")

                if (model_path / "inference_package.pkl").exists():
                    verify_model_file(model_path / "inference_package.pkl")
                    package = joblib.load(model_path / "inference_package.pkl")
                    MODEL_STATE["scorer"] = package.get("scorer")
                    MODEL_STATE["threshold"] = package.get("threshold")
                else:
                    MODEL_STATE["scorer"] = AnomalyScorer()
                    MODEL_STATE["threshold"] = 0.7

                MODEL_STATE["loaded"] = True
                MODEL_STATE["loaded_at"] = datetime.utcnow().isoformat()
            logger.info("Models auto-loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to auto-load models: {e}")

    # Run server
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
