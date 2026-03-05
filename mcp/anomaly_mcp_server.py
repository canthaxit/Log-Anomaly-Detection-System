#!/usr/bin/env python3
"""
MCP Server for Log Anomaly Detection
Exposes anomaly detection capabilities via Model Context Protocol
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio
import joblib
import pandas as pd

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    import mcp.server.stdio
except ImportError:
    print("MCP SDK not installed. Install with: pip install mcp")
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.security import validate_model_path, validate_log_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anomaly-mcp-server")

# Initialize MCP server
app = Server("log-anomaly-detection")

# Global state for loaded models
MODEL_STATE = {
    "feature_pipeline": None,
    "isolation_forest": None,
    "statistical_detector": None,
    "scorer": None,
    "threshold": None,
    "loaded": False
}


def load_models(model_dir: str = "anomaly_outputs") -> Dict[str, Any]:
    """Load trained models from disk."""
    try:
        model_path = validate_model_path(model_dir)
    except ValueError as exc:
        logger.error(f"Model path validation failed: {exc}")
        return {"status": "error", "message": "Model directory is outside allowed paths"}

    try:
        MODEL_STATE["feature_pipeline"] = joblib.load(model_path / "feature_pipeline.pkl")
        MODEL_STATE["isolation_forest"] = joblib.load(model_path / "isolation_forest_model.pkl")
        MODEL_STATE["statistical_detector"] = joblib.load(model_path / "statistical_detector.pkl")

        # Load inference package if available
        if (model_path / "inference_package.pkl").exists():
            package = joblib.load(model_path / "inference_package.pkl")
            MODEL_STATE["scorer"] = package.get("scorer")
            MODEL_STATE["threshold"] = package.get("threshold")
        else:
            MODEL_STATE["scorer"] = AnomalyScorer()
            MODEL_STATE["threshold"] = 0.7

        MODEL_STATE["loaded"] = True
        logger.info(f"Models loaded successfully from {model_dir}")

        return {"status": "success", "message": f"Models loaded from {model_dir}"}

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return {"status": "error", "message": "Failed to load models"}


def analyze_logs(log_data: str, format: str = "json") -> Dict[str, Any]:
    """Analyze log data and return anomalies."""
    if not MODEL_STATE["loaded"]:
        return {"status": "error", "message": "Models not loaded. Call load_models first."}

    try:
        # Parse log data
        if format == "json":
            logs = json.loads(log_data)
            if isinstance(logs, dict):
                logs = [logs]
            df = pd.DataFrame(logs)
        elif format == "csv":
            from io import StringIO
            df = pd.read_csv(StringIO(log_data))
        else:
            return {"status": "error", "message": f"Unsupported format: {format}"}

        # Preprocess
        parser = LogParser()
        df = parser._normalize_schema(df)
        df = preprocess_logs(df)

        if len(df) == 0:
            return {"status": "success", "anomalies": [], "total_events": 0}

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
        anomalies_df = df[is_anomaly].copy()
        anomalies_df['anomaly_score'] = combined_scores[is_anomaly]

        # Classify threats
        anomalies_df['threat_type'] = anomalies_df.apply(
            lambda row: classify_threat(row, MODEL_STATE["statistical_detector"]),
            axis=1
        )

        # Assign severity
        anomalies_df['severity'] = anomalies_df['anomaly_score'].apply(assign_severity)

        # Convert to JSON-serializable format
        anomalies = anomalies_df.to_dict(orient='records')

        # Convert timestamps to strings
        for anomaly in anomalies:
            if 'timestamp' in anomaly and hasattr(anomaly['timestamp'], 'isoformat'):
                anomaly['timestamp'] = anomaly['timestamp'].isoformat()

        return {
            "status": "success",
            "total_events": len(df),
            "anomalies_detected": len(anomalies),
            "anomaly_rate": len(anomalies) / len(df) if len(df) > 0 else 0,
            "anomalies": anomalies,
            "threshold": float(threshold)
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return {"status": "error", "message": "Analysis failed"}


def classify_threat(row: pd.Series, detector: StatisticalAnomalyDetector) -> str:
    """Classify threat type for a single anomaly."""
    # Simple heuristic based on log attributes
    if 'failed' in str(row.get('action', '')).lower():
        return 'brute_force'
    elif 'sudo' in str(row.get('message', '')).lower():
        return 'privilege_escalation'
    elif 'shadow' in str(row.get('message', '')).lower() or 'passwd' in str(row.get('message', '')).lower():
        return 'data_exfiltration'
    elif row.get('event_type') == 'network':
        return 'lateral_movement'
    else:
        return 'unknown'


def assign_severity(score: float) -> str:
    """Assign severity based on anomaly score."""
    if score >= 0.95:
        return 'critical'
    elif score >= 0.85:
        return 'high'
    elif score >= 0.7:
        return 'medium'
    else:
        return 'low'


def analyze_log_file(filepath: str) -> Dict[str, Any]:
    """Analyze logs from a file."""
    try:
        path = validate_log_path(filepath)
    except ValueError:
        return {"status": "error", "message": "File path is outside allowed directories"}

    try:
        if not path.exists():
            return {"status": "error", "message": f"File not found: {filepath}"}

        with open(path, 'r') as f:
            log_data = f.read()

        # Determine format
        format = "json" if path.suffix == ".json" else "csv"

        return analyze_logs(log_data, format)

    except Exception as e:
        logger.error(f"File analysis failed: {e}")
        return {"status": "error", "message": "File analysis failed"}


def get_stats() -> Dict[str, Any]:
    """Get current model statistics."""
    if not MODEL_STATE["loaded"]:
        return {"status": "error", "message": "Models not loaded"}

    try:
        return {
            "status": "success",
            "models_loaded": True,
            "threshold": float(MODEL_STATE["threshold"]),
            "feature_pipeline": {
                "n_features": len(MODEL_STATE["feature_pipeline"].feature_names_),
                "features": MODEL_STATE["feature_pipeline"].feature_names_
            },
            "isolation_forest": {
                "n_estimators": MODEL_STATE["isolation_forest"].n_estimators,
                "contamination": MODEL_STATE["isolation_forest"].contamination
            }
        }
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}", exc_info=True)
        return {"status": "error", "message": "Failed to retrieve stats"}


# MCP Tool Definitions
@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="load_anomaly_models",
            description="Load trained anomaly detection models from disk",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_dir": {
                        "type": "string",
                        "description": "Directory containing model files (default: anomaly_outputs)"
                    }
                }
            }
        ),
        Tool(
            name="analyze_logs",
            description="Analyze log data for anomalies and security threats",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_data": {
                        "type": "string",
                        "description": "JSON or CSV formatted log data"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "csv"],
                        "description": "Data format (default: json)"
                    }
                },
                "required": ["log_data"]
            }
        ),
        Tool(
            name="analyze_log_file",
            description="Analyze logs from a file path",
            inputSchema={
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to log file (JSON or CSV)"
                    }
                },
                "required": ["filepath"]
            }
        ),
        Tool(
            name="get_detection_stats",
            description="Get statistics about loaded detection models",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle MCP tool calls."""
    try:
        if name == "load_anomaly_models":
            model_dir = arguments.get("model_dir", "anomaly_outputs")
            result = load_models(model_dir)

        elif name == "analyze_logs":
            log_data = arguments.get("log_data")
            format = arguments.get("format", "json")
            result = analyze_logs(log_data, format)

        elif name == "analyze_log_file":
            filepath = arguments.get("filepath")
            result = analyze_log_file(filepath)

        elif name == "get_detection_stats":
            result = get_stats()

        else:
            result = {"status": "error", "message": f"Unknown tool: {name}"}

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]

    except Exception as e:
        logger.error(f"Tool execution failed: {e}", exc_info=True)
        return [TextContent(
            type="text",
            text=json.dumps({"status": "error", "message": "Tool execution failed"}, indent=2)
        )]


async def main():
    """Run MCP server."""
    logger.info("Starting Log Anomaly Detection MCP Server")

    # Auto-load models if available
    try:
        validated = validate_model_path("anomaly_outputs")
        if validated.exists():
            load_models("anomaly_outputs")
    except ValueError:
        logger.warning("Default model dir not in allowed paths, skipping auto-load")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
