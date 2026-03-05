#!/usr/bin/env python3
"""
REST API for Log Anomaly Detection with Google Chronicle Integration
FastAPI server with automatic Chronicle SIEM forwarding
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

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal
import uvicorn
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.security import validate_model_path, get_verify_api_key, ALLOWED_ORIGINS
from fastapi import Depends

# Import Chronicle integration
from google_chronicle_integration import ChronicleClient, ChronicleConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anomaly-api-chronicle")

# Initialize FastAPI
app = FastAPI(
    title="Log Anomaly Detection API with Google Chronicle",
    description="AI-powered security threat detection with Google SIEM integration",
    version="1.0.0"
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


# Thread safety locks
_model_lock = threading.Lock()
_chronicle_lock = threading.Lock()

# Enable CORS (restricted origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

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

CHRONICLE_STATE = {
    "client": None,
    "enabled": False,
    "config": None
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
    return_all_events: bool = Field(False, description="Return all events with scores")
    send_to_chronicle: bool = Field(True, description="Automatically send anomalies to Chronicle")


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
    chronicle_sent: Optional[bool] = None


class ChronicleStatus(BaseModel):
    """Chronicle integration status."""
    enabled: bool
    configured: bool
    customer_id: Optional[str] = None
    region: Optional[str] = None
    events_sent_session: int = 0


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


# Chronicle Integration
async def send_to_chronicle_background(anomalies: List[Dict[str, Any]]):
    """Background task to send anomalies to Chronicle."""
    if not CHRONICLE_STATE["enabled"] or not CHRONICLE_STATE["client"]:
        logger.debug("Chronicle integration not enabled")
        return

    try:
        result = CHRONICLE_STATE["client"].send_anomalies(anomalies)
        if result['status'] == 'success':
            logger.info(f"Sent {len(anomalies)} anomalies to Chronicle")
        else:
            logger.error(f"Chronicle ingestion failed: {result.get('message')}")
    except Exception as e:
        logger.error(f"Error sending to Chronicle: {e}")


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": MODEL_STATE["loaded"],
        "chronicle_enabled": CHRONICLE_STATE["enabled"],
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/chronicle/status", response_model=ChronicleStatus, dependencies=[Depends(verify_api_key)])
async def chronicle_status():
    """Get Chronicle integration status."""
    return ChronicleStatus(
        enabled=CHRONICLE_STATE["enabled"],
        configured=CHRONICLE_STATE["client"] is not None,
        customer_id=CHRONICLE_STATE["config"].get("customer_id") if CHRONICLE_STATE["config"] else None,
        region=CHRONICLE_STATE["config"].get("region") if CHRONICLE_STATE["config"] else None
    )


@app.post("/chronicle/enable", dependencies=[Depends(verify_api_key)])
async def enable_chronicle(
    customer_id: str = None,
    region: str = "us"
):
    """Enable Chronicle integration."""
    try:
        # Load config
        config = ChronicleConfig()

        # Credentials file from env var only (never user-supplied)
        creds = os.environ.get("CHRONICLE_CREDENTIALS_FILE", config.get("credentials_file"))
        cust_id = customer_id or config.get("customer_id")
        reg = region or config.get("region", "us")

        # Validate credentials file exists
        if not Path(creds).exists():
            raise HTTPException(
                status_code=400,
                detail="Chronicle credentials file not found"
            )

        # Initialize Chronicle client
        chronicle_client = ChronicleClient(
            credentials_file=creds,
            customer_id=cust_id,
            region=reg
        )

        with _chronicle_lock:
            CHRONICLE_STATE["client"] = chronicle_client
            CHRONICLE_STATE["enabled"] = True
            CHRONICLE_STATE["config"] = {
                "customer_id": cust_id,
                "region": reg,
            }

        logger.info("Chronicle integration enabled")

        return {
            "status": "success",
            "message": "Chronicle integration enabled",
            "customer_id": cust_id,
            "region": reg
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable Chronicle: {e}")
        raise HTTPException(status_code=500, detail="Failed to enable Chronicle integration")


@app.post("/chronicle/disable", dependencies=[Depends(verify_api_key)])
async def disable_chronicle():
    """Disable Chronicle integration."""
    with _chronicle_lock:
        CHRONICLE_STATE["enabled"] = False
        CHRONICLE_STATE["client"] = None
    return {
        "status": "success",
        "message": "Chronicle integration disabled"
    }


@app.post("/chronicle/test", dependencies=[Depends(verify_api_key)])
async def test_chronicle():
    """Test Chronicle connection."""
    if not CHRONICLE_STATE["enabled"]:
        raise HTTPException(
            status_code=400,
            detail="Chronicle integration not enabled. Call POST /chronicle/enable first."
        )

    # Send test event
    test_anomaly = {
        "timestamp": datetime.utcnow().isoformat() + 'Z',
        "user": "test_user",
        "source_ip": "192.168.1.100",
        "dest_ip": "unknown",
        "event_type": "login",
        "action": "failed",
        "message": "Test event from API",
        "severity": "low",
        "anomaly_score": 0.65,
        "threat_type": "brute_force"
    }

    try:
        result = CHRONICLE_STATE["client"].send_anomalies([test_anomaly])
        return result
    except Exception as e:
        logger.error(f"Chronicle test failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chronicle connection test failed")


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
            raise HTTPException(status_code=404, detail=f"Model directory not found: {model_dir}")

        with _model_lock:
            MODEL_STATE["feature_pipeline"] = joblib.load(model_path / "feature_pipeline.pkl")
            MODEL_STATE["isolation_forest"] = joblib.load(model_path / "isolation_forest_model.pkl")
            MODEL_STATE["statistical_detector"] = joblib.load(model_path / "statistical_detector.pkl")

            # Load inference package
            if (model_path / "inference_package.pkl").exists():
                package = joblib.load(model_path / "inference_package.pkl")
                MODEL_STATE["scorer"] = package.get("scorer")
                MODEL_STATE["threshold"] = package.get("threshold")
            else:
                MODEL_STATE["scorer"] = AnomalyScorer()
                MODEL_STATE["threshold"] = 0.7

            MODEL_STATE["loaded"] = True
            MODEL_STATE["loaded_at"] = datetime.utcnow().isoformat()

        logger.info(f"Models loaded successfully from {model_dir}")

        # Try to auto-enable Chronicle if configured
        try:
            config = ChronicleConfig()
            if Path(config.get("credentials_file")).exists():
                await enable_chronicle()
        except Exception:
            logger.info("Chronicle auto-enable skipped (not configured)")

        return {
            "status": "success",
            "message": f"Models loaded from {model_dir}",
            "loaded_at": MODEL_STATE["loaded_at"],
            "chronicle_enabled": CHRONICLE_STATE["enabled"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/analyze", response_model=AnalysisResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
async def analyze_logs(request: Request, analysis: AnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze logs for anomalies and optionally send to Chronicle."""
    start_time = datetime.utcnow()

    if not MODEL_STATE["loaded"]:
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
                threshold=float(MODEL_STATE["threshold"]),
                anomalies=[],
                processing_time_ms=0.0,
                chronicle_sent=False
            )

        # Extract features
        features = MODEL_STATE["feature_pipeline"].transform(df)

        # Detect anomalies
        iso_scores = -MODEL_STATE["isolation_forest"].score_samples(features)
        stat_scores = MODEL_STATE["statistical_detector"].detect_all(df)

        # Combine scores
        combined_scores = MODEL_STATE["scorer"].combine_scores({
            'isolation_forest': iso_scores,
            'statistical': stat_scores
        })

        # Identify anomalies
        threshold = MODEL_STATE["threshold"]
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
        anomalies_for_chronicle = []

        for _, row in result_df.iterrows():
            anomaly_dict = {
                "timestamp": str(row['timestamp']),
                "user": row['user'],
                "source_ip": row['source_ip'],
                "dest_ip": row.get('dest_ip', 'unknown'),
                "event_type": row['event_type'],
                "action": row['action'],
                "message": row['message'],
                "severity": row['severity'],
                "anomaly_score": float(row['anomaly_score']),
                "threat_type": row['threat_type']
            }

            if analysis.return_all_events or row.get('is_anomaly', True):
                anomalies.append(Anomaly(**anomaly_dict))

                # Collect for Chronicle
                if row['anomaly_score'] > threshold:
                    anomalies_for_chronicle.append(anomaly_dict)

        # Send to Chronicle in background if enabled
        chronicle_sent = False
        if analysis.send_to_chronicle and anomalies_for_chronicle and CHRONICLE_STATE["enabled"]:
            background_tasks.add_task(send_to_chronicle_background, anomalies_for_chronicle)
            chronicle_sent = True

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return AnalysisResponse(
            status="success",
            total_events=len(df),
            anomalies_detected=int(is_anomaly.sum()),
            anomaly_rate=float(is_anomaly.sum() / len(df)),
            threshold=float(threshold),
            anomalies=anomalies if not analysis.return_all_events else [a for a in anomalies if a.anomaly_score > threshold],
            processing_time_ms=processing_time,
            chronicle_sent=chronicle_sent
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/analyze/file", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def analyze_file(
    request: Request,
    file: UploadFile = File(...),
    send_to_chronicle: bool = True,
    background_tasks: BackgroundTasks = None
):
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

        # Parse based on file extension
        if file.filename.endswith('.json'):
            logs = json.loads(content_str)
            if isinstance(logs, dict):
                logs = [logs]
            df = pd.DataFrame(logs)
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(StringIO(content_str))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use JSON or CSV.")

        if len(df) > MAX_LOG_EVENTS:
            raise HTTPException(status_code=413, detail=f"Too many log events. Maximum is {MAX_LOG_EVENTS}")

        # Convert to request format
        logs_list = df.to_dict(orient='records')
        log_events = [LogEvent(**log) for log in logs_list]

        # Use existing analyze endpoint
        analysis_req = AnalysisRequest(
            logs=log_events,
            send_to_chronicle=send_to_chronicle
        )
        return await analyze_logs(request, analysis_req, background_tasks)

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        logger.error(f"File analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    # Auto-load models if available
    try:
        model_path = validate_model_path("anomaly_outputs")
        if model_path.exists():
            MODEL_STATE["feature_pipeline"] = joblib.load(model_path / "feature_pipeline.pkl")
            MODEL_STATE["isolation_forest"] = joblib.load(model_path / "isolation_forest_model.pkl")
            MODEL_STATE["statistical_detector"] = joblib.load(model_path / "statistical_detector.pkl")

            if (model_path / "inference_package.pkl").exists():
                package = joblib.load(model_path / "inference_package.pkl")
                MODEL_STATE["scorer"] = package.get("scorer")
                MODEL_STATE["threshold"] = package.get("threshold")
            else:
                MODEL_STATE["scorer"] = AnomalyScorer()
                MODEL_STATE["threshold"] = 0.7

            MODEL_STATE["loaded"] = True
            MODEL_STATE["loaded_at"] = datetime.utcnow().isoformat()
            logger.info("Models auto-loaded successfully")

            # Try to auto-enable Chronicle
            try:
                config = ChronicleConfig()
                if Path(config.get("credentials_file")).exists():
                    chronicle_client = ChronicleClient(
                        credentials_file=config.get("credentials_file"),
                        customer_id=config.get("customer_id"),
                        region=config.get("region", "us")
                    )
                    CHRONICLE_STATE["client"] = chronicle_client
                    CHRONICLE_STATE["enabled"] = True
                    CHRONICLE_STATE["config"] = config.config
                    logger.info("Chronicle integration auto-enabled")
            except Exception:
                logger.info("Chronicle auto-enable skipped (not configured)")

    except Exception as e:
        logger.warning(f"Failed to auto-load models: {e}")

    # Run server
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")
