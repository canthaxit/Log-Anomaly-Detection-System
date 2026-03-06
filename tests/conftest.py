"""Shared fixtures for the Log Anomaly Detection test suite."""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup — mirror the sys.path hacks the production code uses so that
# ``from log_anomaly_detection_lite import …`` and ``from common.security …``
# resolve cleanly regardless of the working directory.
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
_core_dir = _project_root / "core"

for _p in [str(_project_root), str(_core_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Set env vars BEFORE importing security (it reads them at import time)
os.environ.setdefault("ALLOWED_MODEL_DIRS", "anomaly_outputs,tests/tmp_models")
os.environ.setdefault("ALLOWED_LOG_DIRS", "logs,tests")

# Now safe to import project modules
from log_anomaly_detection_lite import (
    AnomalyScorer,
    LogFeaturePipeline,
    LogParser,
    StatisticalAnomalyDetector,
    create_isolation_forest,
    preprocess_logs,
)

# ---------------------------------------------------------------------------
# Paths to test data shipped in the repo
# ---------------------------------------------------------------------------
_TESTS_DIR = Path(__file__).resolve().parent
_NORMAL_JSON = _TESTS_DIR / "test_logs_normal.json"
_ATTACK_JSON = _TESTS_DIR / "test_logs_attack.json"


# ---------------------------------------------------------------------------
# Session-scoped fixtures (expensive — created once per test run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def normal_df() -> pd.DataFrame:
    """Normal log events parsed into a DataFrame."""
    parser = LogParser()
    with open(_NORMAL_JSON) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = parser._normalize_schema(df)
    return preprocess_logs(df)


@pytest.fixture(scope="session")
def attack_df() -> pd.DataFrame:
    """Attack log events parsed into a DataFrame."""
    parser = LogParser()
    with open(_ATTACK_JSON) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = parser._normalize_schema(df)
    return preprocess_logs(df)


@pytest.fixture(scope="session")
def fitted_pipeline(normal_df) -> LogFeaturePipeline:
    """LogFeaturePipeline fit on normal data."""
    pipeline = LogFeaturePipeline()
    pipeline.fit(normal_df)
    return pipeline


@pytest.fixture(scope="session")
def fitted_iso_forest(fitted_pipeline, normal_df):
    """IsolationForest fit on normal features."""
    features = fitted_pipeline.transform(normal_df)
    iso = create_isolation_forest()
    iso.fit(features)
    return iso


@pytest.fixture(scope="session")
def fitted_stat_detector(normal_df) -> StatisticalAnomalyDetector:
    """StatisticalAnomalyDetector fit on normal data."""
    det = StatisticalAnomalyDetector()
    det.fit(normal_df)
    return det


@pytest.fixture(scope="session")
def scorer() -> AnomalyScorer:
    return AnomalyScorer()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, fitted_pipeline, fitted_iso_forest, fitted_stat_detector, scorer):
    """Temp directory with all .pkl artifacts saved via joblib."""
    d = tmp_path_factory.mktemp("models")
    joblib.dump(fitted_pipeline, d / "feature_pipeline.pkl")
    joblib.dump(fitted_iso_forest, d / "isolation_forest_model.pkl")
    joblib.dump(fitted_stat_detector, d / "statistical_detector.pkl")
    joblib.dump({"scorer": scorer, "threshold": 0.7}, d / "inference_package.pkl")
    return d


@pytest.fixture(scope="session")
def api_app(model_dir):
    """FastAPI ``app`` with models pre-loaded from *model_dir*."""
    # Ensure security module picks up the temp model dir
    os.environ["ALLOWED_MODEL_DIRS"] = f"anomaly_outputs,tests/tmp_models,{model_dir}"

    # Force re-read of env (security module reads at import time, so patch the
    # module-level list directly).
    import common.security as sec
    sec.ALLOWED_MODEL_DIRS.append(str(model_dir))

    from api.anomaly_api import app, MODEL_STATE

    # Pre-load models so endpoint tests don't need to call /models/load
    MODEL_STATE["feature_pipeline"] = joblib.load(model_dir / "feature_pipeline.pkl")
    MODEL_STATE["isolation_forest"] = joblib.load(model_dir / "isolation_forest_model.pkl")
    MODEL_STATE["statistical_detector"] = joblib.load(model_dir / "statistical_detector.pkl")
    pkg = joblib.load(model_dir / "inference_package.pkl")
    MODEL_STATE["scorer"] = pkg["scorer"]
    MODEL_STATE["threshold"] = pkg["threshold"]
    MODEL_STATE["loaded"] = True
    MODEL_STATE["loaded_at"] = "2026-01-01T00:00:00"

    return app


@pytest.fixture(scope="session")
def client(api_app):
    """httpx TestClient for the FastAPI app."""
    from starlette.testclient import TestClient
    return TestClient(api_app)
