"""
Log Anomaly Detection Pipeline
================================
A production-ready ML pipeline for unsupervised threat detection in system logs.

Key Features:
- JSON log parsing with unified schema
- Unsupervised anomaly detection (no labels required)
- Multi-model ensemble (Isolation Forest + Autoencoder + Statistical)
- Threat pattern detection (brute force, privilege escalation, exfiltration, lateral movement)
- Temporal and behavioral feature engineering
- Complete artifact persistence for deployment

Author: Transformed for log anomaly detection
"""

import pandas as pd
import numpy as np
import glob
import os
import joblib
import math
import argparse
import json
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.regularizers import l2

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    """Centralized configuration management."""

    def __init__(self, args=None):
        # Data parameters
        self.data_path = args.data_path if args else './logs/'
        self.log_format = args.log_format if args else 'auto'
        self.output_dir = args.output_dir if args else 'anomaly_outputs'

        # Detection parameters
        self.contamination = args.contamination if args else 0.01  # Expected anomaly rate
        self.baseline_period_days = args.baseline_period_days if args else 7
        self.random_state = args.random_state if args else 42

        # Model parameters
        self.iso_forest_estimators = args.iso_forest_estimators if args else 200
        self.autoencoder_epochs = args.autoencoder_epochs if args else 50
        self.batch_size = args.batch_size if args else 1024

        # Feature engineering parameters
        self.time_windows = args.time_windows if args else [3600, 86400, 604800]  # 1h, 24h, 7d in seconds

        # Alert parameters
        self.severity_thresholds = {
            'low': 0.5,
            'medium': 0.7,
            'high': 0.85,
            'critical': 0.95
        }

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Set random seeds for reproducibility
        self._set_seeds()
    
    def _set_seeds(self):
        """Set all random seeds for reproducibility."""
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        os.environ['PYTHONHASHSEED'] = str(self.random_state)


# ==========================================
# LOG PARSING
# ==========================================
class LogParser:
    """Parse JSON log files into unified DataFrame schema."""

    def __init__(self, log_format='auto'):
        """
        Initialize log parser.

        Args:
            log_format: Log format type ('auto', 'auth', 'syslog', 'security')
        """
        self.log_format = log_format

    def parse_log_file(self, filepath: str) -> pd.DataFrame:
        """
        Parse single JSON log file.

        Supports:
        - Line-delimited JSON (JSONL)
        - JSON array
        - Single JSON object

        Args:
            filepath: Path to JSON log file

        Returns:
            DataFrame with parsed logs
        """
        logs = []

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()

            # Try to parse as JSON array
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    logs = data
                elif isinstance(data, dict):
                    logs = [data]
            except json.JSONDecodeError:
                # Try line-delimited JSON
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            logs.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"  Warning: Skipping invalid JSON line: {e}")
                            continue

        if not logs:
            print(f"  Warning: No valid JSON logs found in {filepath}")
            return pd.DataFrame()

        df = pd.DataFrame(logs)
        return self._normalize_schema(df)

    def _normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize log entries to unified schema.

        Expected schema:
        - timestamp: datetime
        - user: str
        - source_ip: str
        - dest_ip: str (optional)
        - event_type: str
        - action: str
        - message: str
        - severity: str
        """
        # Map common field variations to standard schema
        field_mappings = {
            'timestamp': ['timestamp', 'time', '@timestamp', 'datetime', 'date'],
            'user': ['user', 'username', 'uid', 'account', 'identity'],
            'source_ip': ['source_ip', 'src_ip', 'ip', 'client_ip', 'remote_addr'],
            'dest_ip': ['dest_ip', 'dst_ip', 'destination_ip', 'server_ip'],
            'event_type': ['event_type', 'type', 'event', 'category', 'action_type'],
            'action': ['action', 'result', 'status', 'outcome'],
            'message': ['message', 'msg', 'description', 'text', 'log'],
            'severity': ['severity', 'level', 'priority', 'sev']
        }

        normalized_df = pd.DataFrame()

        # Map fields
        for standard_field, variations in field_mappings.items():
            for var in variations:
                if var in df.columns:
                    normalized_df[standard_field] = df[var]
                    break
            # Add default if missing
            if standard_field not in normalized_df.columns:
                if standard_field == 'severity':
                    normalized_df[standard_field] = 'INFO'
                elif standard_field in ['source_ip', 'dest_ip']:
                    normalized_df[standard_field] = 'unknown'
                elif standard_field == 'user':
                    normalized_df[standard_field] = 'unknown'
                elif standard_field == 'event_type':
                    normalized_df[standard_field] = 'general'
                elif standard_field == 'action':
                    normalized_df[standard_field] = 'success'
                elif standard_field == 'message':
                    normalized_df[standard_field] = ''
                else:
                    normalized_df[standard_field] = None

        # Parse timestamp
        if 'timestamp' in normalized_df.columns:
            try:
                normalized_df['timestamp'] = pd.to_datetime(normalized_df['timestamp'], errors='coerce')
            except Exception as e:
                print(f"  Warning: Could not parse timestamps: {e}")
                normalized_df['timestamp'] = pd.NaT

        # Normalize strings
        for col in ['user', 'event_type', 'action', 'severity']:
            if col in normalized_df.columns:
                normalized_df[col] = normalized_df[col].astype(str).str.strip().str.lower()

        return normalized_df

    def load_logs(self, path: str) -> pd.DataFrame:
        """
        Load and parse all JSON logs from directory.

        Args:
            path: Directory containing JSON log files

        Returns:
            Concatenated DataFrame of all logs
        """
        print(f"\n{'='*60}")
        print("LOG LOADING")
        print('='*60)
        print(f"Searching for JSON log files in {path}...")

        all_files = glob.glob(os.path.join(path, "*.json"))

        if not all_files:
            raise ValueError(f"No JSON files found in {path}")

        dfs = []
        for filename in sorted(all_files):
            try:
                print(f"  Loading {os.path.basename(filename)}...")
                df_temp = self.parse_log_file(filename)
                if not df_temp.empty:
                    print(f"    Shape: {df_temp.shape}")
                    dfs.append(df_temp)
            except Exception as e:
                print(f"  Warning: Could not load {filename}. Error: {e}")

        if not dfs:
            raise ValueError("No log files could be loaded.")

        df = pd.concat(dfs, axis=0, ignore_index=True)

        print(f"\nTotal logs loaded: {len(df):,}")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        return df


# ==========================================
# INITIAL PREPROCESSING
# ==========================================
def preprocess_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess log data before analysis.

    Args:
        df: Raw log DataFrame

    Returns:
        Cleaned log DataFrame
    """
    print(f"\n{'='*60}")
    print("LOG PREPROCESSING")
    print('='*60)

    initial_rows = len(df)

    # 1. Remove exact duplicates
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df):,} duplicate rows")

    # 2. Remove logs with invalid timestamps
    invalid_timestamps = df['timestamp'].isna().sum()
    if invalid_timestamps > 0:
        print(f"Removing {invalid_timestamps:,} logs with invalid timestamps")
        df = df[df['timestamp'].notna()]

    # 3. Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 4. Print log statistics
    print(f"\nLog Statistics:")
    print(f"  Total logs: {len(df):,}")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Duration: {df['timestamp'].max() - df['timestamp'].min()}")

    print(f"\nEvent Type Distribution:")
    for event_type, count in df['event_type'].value_counts().head(10).items():
        print(f"  {event_type}: {count:,}")

    print(f"\nAction Distribution:")
    for action, count in df['action'].value_counts().head(5).items():
        print(f"  {action}: {count:,}")

    print(f"\nTop Users:")
    for user, count in df['user'].value_counts().head(5).items():
        print(f"  {user}: {count:,}")

    return df


# ==========================================
# FEATURE PIPELINE FOR LOG DATA
# ==========================================
class LogFeaturePipeline:
    """
    Extract features from log data for anomaly detection.
    Focuses on temporal, behavioral, and entity-based features.
    """

    def __init__(self, time_windows: List[int] = [3600, 86400, 604800]):
        """
        Initialize log feature pipeline.

        Args:
            time_windows: Time windows in seconds for rolling statistics (default: 1h, 24h, 7d)
        """
        self.time_windows = time_windows
        self.user_baselines_ = None
        self.ip_baselines_ = None
        self.scaler_ = RobustScaler()
        self.feature_names_ = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame) -> 'LogFeaturePipeline':
        """
        Fit the pipeline on baseline data (normal behavior).

        Args:
            X: Baseline log DataFrame

        Returns:
            self
        """
        print(f"\n{'='*60}")
        print("FITTING LOG FEATURE PIPELINE")
        print('='*60)

        X = X.copy()

        # Build user baselines
        self.user_baselines_ = X.groupby('user').agg({
            'event_type': 'count',  # Event frequency
            'source_ip': 'nunique',  # Unique IPs per user
            'action': lambda x: (x == 'failed').sum() / len(x) if len(x) > 0 else 0  # Failed ratio
        }).to_dict('index')
        print(f"Built baselines for {len(self.user_baselines_)} users")

        # Build IP baselines
        self.ip_baselines_ = X.groupby('source_ip').agg({
            'user': 'nunique',  # Unique users per IP
            'event_type': 'count'  # Event frequency
        }).to_dict('index')
        print(f"Built baselines for {len(self.ip_baselines_)} IPs")

        # Extract features and fit scaler
        X_features = self._extract_features(X)
        self.scaler_.fit(X_features)
        self.feature_names_ = X_features.columns.tolist()
        print(f"Extracted {len(self.feature_names_)} features")

        self._is_fitted = True
        return self

    def _extract_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from log data."""
        X = X.copy()

        features = pd.DataFrame(index=X.index)

        # 1. Temporal features
        features['hour'] = X['timestamp'].dt.hour
        features['day_of_week'] = X['timestamp'].dt.dayofweek
        features['is_weekend'] = (X['timestamp'].dt.dayofweek >= 5).astype(int)
        features['is_business_hours'] = ((X['timestamp'].dt.hour >= 9) & (X['timestamp'].dt.hour < 17)).astype(int)
        features['is_night'] = ((X['timestamp'].dt.hour < 6) | (X['timestamp'].dt.hour >= 22)).astype(int)

        # 2. Action-based features
        features['is_failed'] = (X['action'] == 'failed').astype(int)
        features['is_denied'] = (X['action'] == 'denied').astype(int)
        features['is_success'] = (X['action'] == 'success').astype(int)

        # 3. Event type features (one-hot encode top event types)
        top_events = X['event_type'].value_counts().head(10).index.tolist()
        for event in top_events:
            features[f'event_{event}'] = (X['event_type'] == event).astype(int)

        # 4. User-based features
        features['user_is_unknown'] = (X['user'] == 'unknown').astype(int)
        features['user_is_root'] = (X['user'] == 'root').astype(int)
        features['user_is_admin'] = (X['user'].str.contains('admin', case=False, na=False)).astype(int)

        # 5. IP-based features
        features['ip_is_unknown'] = (X['source_ip'] == 'unknown').astype(int)
        features['ip_is_localhost'] = (X['source_ip'].str.startswith('127.', na=False)).astype(int)
        features['ip_is_private'] = (X['source_ip'].str.startswith(('192.168.', '10.', '172.'), na=False)).astype(int)

        # 6. Baseline deviation features (if fitted)
        if self.user_baselines_ is not None:
            features['user_event_count_baseline'] = X['user'].map(
                lambda u: self.user_baselines_.get(u, {}).get('event_type', 0)
            )
            features['user_failed_ratio_baseline'] = X['user'].map(
                lambda u: self.user_baselines_.get(u, {}).get('action', 0)
            )

        if self.ip_baselines_ is not None:
            features['ip_event_count_baseline'] = X['source_ip'].map(
                lambda ip: self.ip_baselines_.get(ip, {}).get('event_type', 0)
            )

        # 7. Rolling window features (events per time window)
        X_sorted = X.sort_values('timestamp').copy()
        for window in self.time_windows:
            window_name = f"{window}s"
            # Count events in rolling window per user
            X_sorted[f'events_per_{window_name}'] = X_sorted.groupby('user')['timestamp'].transform(
                lambda x: x.rolling(f'{window}s').count()
            )
            # Map back to original index
            features[f'events_per_{window_name}'] = X_sorted.set_index(X.index)[f'events_per_{window_name}'].fillna(0)

        # Fill NaN values
        features = features.fillna(0)

        return features

    def transform(self, X: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """
        Transform log data using fitted pipeline.

        Args:
            X: Log DataFrame
            scale: Whether to apply scaling

        Returns:
            Transformed feature matrix
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fit before transform")

        X_features = self._extract_features(X)

        # Align columns with training data
        missing_cols = set(self.feature_names_) - set(X_features.columns)
        extra_cols = set(X_features.columns) - set(self.feature_names_)

        # Add missing columns
        for col in missing_cols:
            X_features[col] = 0

        # Remove extra columns
        X_features = X_features.drop(columns=list(extra_cols), errors='ignore')

        # Reorder to match training
        X_features = X_features[self.feature_names_]

        if scale:
            return self.scaler_.transform(X_features)
        return X_features.values

    def fit_transform(self, X: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X, scale=scale)

    def get_feature_names(self) -> List[str]:
        """Return feature names after transformation."""
        return self.feature_names_


# ==========================================
# ANOMALY DETECTION MODELS
# ==========================================
def create_isolation_forest(contamination: float = 0.01, n_estimators: int = 200,
                           random_state: int = 42) -> IsolationForest:
    """
    Create an Isolation Forest for anomaly detection.

    Args:
        contamination: Expected proportion of anomalies
        n_estimators: Number of trees
        random_state: Random seed

    Returns:
        Isolation Forest model
    """
    return IsolationForest(
        n_estimators=n_estimators,
        max_samples='auto',
        contamination=contamination,
        max_features=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )


def create_autoencoder(input_dim: int, encoding_dim: int = 16,
                      learning_rate: float = 0.001) -> Tuple[Sequential, Sequential, Sequential]:
    """
    Create an autoencoder for reconstruction-based anomaly detection.

    Args:
        input_dim: Number of input features
        encoding_dim: Dimension of encoded representation (bottleneck)
        learning_rate: Learning rate

    Returns:
        Tuple of (encoder, decoder, autoencoder)
    """
    # Encoder
    encoder = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(encoding_dim, activation='relu')
    ], name='encoder')

    # Decoder
    decoder = Sequential([
        Dense(32, activation='relu', input_shape=(encoding_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(input_dim, activation='sigmoid')
    ], name='decoder')

    # Autoencoder
    autoencoder = Sequential([encoder, decoder], name='autoencoder')
    autoencoder.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    return encoder, decoder, autoencoder


def compute_reconstruction_error(X: np.ndarray, autoencoder: Sequential) -> np.ndarray:
    """
    Compute reconstruction error for anomaly scoring.

    Args:
        X: Feature matrix
        autoencoder: Trained autoencoder model

    Returns:
        Reconstruction errors (higher = more anomalous)
    """
    reconstructed = autoencoder.predict(X, verbose=0)
    errors = np.mean(np.square(X - reconstructed), axis=1)
    return errors


# ==========================================
# STATISTICAL THREAT DETECTOR
# ==========================================
class StatisticalAnomalyDetector:
    """Rule-based detection for specific threat patterns."""

    def __init__(self):
        self.brute_force_threshold = 10  # Failed logins per hour
        self.sudo_threshold = 5  # Sudo attempts per hour
        self.user_baselines_ = None

    def fit(self, df: pd.DataFrame):
        """Establish baselines from normal behavior."""
        # Build user activity baselines
        self.user_baselines_ = df.groupby('user').agg({
            'event_type': 'count',
            'action': lambda x: (x == 'failed').sum() / max(len(x), 1)
        }).to_dict('index')
        return self

    def detect_brute_force(self, df: pd.DataFrame) -> np.ndarray:
        """Detect credential brute force attacks."""
        scores = np.zeros(len(df))

        # Failed login velocity (count in 1-hour windows)
        df_sorted = df.sort_values('timestamp').copy()
        df_sorted['failed_count_1h'] = 0

        failed_mask = df_sorted['action'] == 'failed'
        if failed_mask.any():
            df_sorted.loc[failed_mask, 'failed_count_1h'] = df_sorted[failed_mask].groupby('user')['timestamp'].transform(
                lambda x: x.rolling('1H').count()
            )

        # Score based on threshold
        scores = (df_sorted.set_index(df.index)['failed_count_1h'] / self.brute_force_threshold).fillna(0).values
        scores = np.clip(scores, 0, 1)

        return scores

    def detect_privilege_escalation(self, df: pd.DataFrame) -> np.ndarray:
        """Detect privilege escalation attempts."""
        scores = np.zeros(len(df))

        # Check for sudo-related events
        sudo_mask = df['event_type'].str.contains('sudo', case=False, na=False)
        scores[sudo_mask] = 0.5  # Base score for any sudo activity

        # Higher score for failed sudo
        failed_sudo = sudo_mask & (df['action'] == 'failed')
        scores[failed_sudo] = 0.8

        # Higher score for unusual users using sudo
        if self.user_baselines_ is not None:
            for idx, row in df[sudo_mask].iterrows():
                baseline_events = self.user_baselines_.get(row['user'], {}).get('event_type', 0)
                if baseline_events == 0:  # New user doing sudo
                    scores[idx] = 1.0

        return scores

    def detect_data_exfiltration(self, df: pd.DataFrame) -> np.ndarray:
        """Detect data exfiltration patterns."""
        scores = np.zeros(len(df))

        # Off-hours file access
        is_night = (df['timestamp'].dt.hour < 6) | (df['timestamp'].dt.hour >= 22)
        file_access = df['event_type'].str.contains('file', case=False, na=False)
        scores[is_night & file_access] = 0.6

        # Access to sensitive directories (if in message)
        sensitive_patterns = ['passwd', 'shadow', '/etc/', 'credentials', 'keys']
        for pattern in sensitive_patterns:
            sensitive_mask = df['message'].str.contains(pattern, case=False, na=False)
            scores[sensitive_mask] = np.maximum(scores[sensitive_mask], 0.7)

        return scores

    def detect_lateral_movement(self, df: pd.DataFrame) -> np.ndarray:
        """Detect lateral movement patterns."""
        scores = np.zeros(len(df))

        # Network/SSH related events
        network_events = df['event_type'].str.contains('network|ssh|rdp', case=False, na=False)
        scores[network_events] = 0.3

        # Multiple IPs for same user (potential credential reuse)
        df_sorted = df.sort_values('timestamp').copy()
        df_sorted['unique_ips_1h'] = df_sorted.groupby('user')['source_ip'].transform(
            lambda x: x.rolling('1H').apply(lambda y: len(set(y)), raw=False)
        )

        high_ip_diversity = df_sorted['unique_ips_1h'] > 5
        scores = np.maximum(scores, (df_sorted.set_index(df.index)['unique_ips_1h'] / 10).fillna(0).values)
        scores = np.clip(scores, 0, 1)

        return scores

    def detect_all(self, df: pd.DataFrame) -> np.ndarray:
        """Run all detectors and combine scores."""
        bf_scores = self.detect_brute_force(df)
        pe_scores = self.detect_privilege_escalation(df)
        de_scores = self.detect_data_exfiltration(df)
        lm_scores = self.detect_lateral_movement(df)

        # Take maximum score across all threat types
        combined = np.maximum.reduce([bf_scores, pe_scores, de_scores, lm_scores])
        return combined


# ==========================================
# ANOMALY SCORING
# ==========================================
class AnomalyScorer:
    """Combine scores from multiple detectors."""

    def __init__(self):
        self.weights = {
            'isolation_forest': 0.35,
            'autoencoder': 0.35,
            'statistical': 0.30
        }

    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        if scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())

    def combine_scores(self, scores_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted combination of anomaly scores."""
        combined = np.zeros(len(list(scores_dict.values())[0]))

        for detector_name, scores in scores_dict.items():
            if detector_name in self.weights:
                # Normalize scores
                normalized = self.normalize_scores(scores)
                # Add weighted contribution
                combined += self.weights[detector_name] * normalized

        return combined

    def calibrate_threshold(self, scores: np.ndarray, false_positive_rate: float = 0.01) -> float:
        """Determine anomaly threshold based on desired FPR."""
        return np.percentile(scores, (1 - false_positive_rate) * 100)


# ==========================================
# THREAT CLASSIFICATION
# ==========================================
def classify_threat_type(df: pd.DataFrame, stat_detector: StatisticalAnomalyDetector) -> List[str]:
    """Assign threat type to each anomaly."""
    threat_types = []

    # Get individual detector scores
    bf_scores = stat_detector.detect_brute_force(df)
    pe_scores = stat_detector.detect_privilege_escalation(df)
    de_scores = stat_detector.detect_data_exfiltration(df)
    lm_scores = stat_detector.detect_lateral_movement(df)

    for i in range(len(df)):
        scores = {
            'brute_force': bf_scores[i],
            'privilege_escalation': pe_scores[i],
            'data_exfiltration': de_scores[i],
            'lateral_movement': lm_scores[i]
        }

        # Assign threat type with highest score
        if max(scores.values()) > 0:
            threat_types.append(max(scores, key=scores.get))
        else:
            threat_types.append('unknown')

    return threat_types


def assign_severity(scores: np.ndarray, thresholds: Dict[str, float]) -> List[str]:
    """Map anomaly scores to severity levels."""
    severities = []
    for score in scores:
        if score >= thresholds['critical']:
            severities.append('critical')
        elif score >= thresholds['high']:
            severities.append('high')
        elif score >= thresholds['medium']:
            severities.append('medium')
        else:
            severities.append('low')
    return severities


# ==========================================
# OUTPUT AND REPORTING
# ==========================================
def save_anomaly_report(anomalies_df: pd.DataFrame, output_dir: str):
    """Save anomaly detection results."""
    # CSV report
    csv_path = os.path.join(output_dir, 'anomalies_detected.csv')
    anomalies_df.to_csv(csv_path, index=False)
    print(f"Saved CSV report: {csv_path}")

    # JSON report (detailed)
    anomalies_json = anomalies_df.to_dict(orient='records')
    json_path = os.path.join(output_dir, 'anomalies_detailed.json')
    with open(json_path, 'w') as f:
        json.dump(anomalies_json, f, indent=2, default=str)
    print(f"Saved JSON report: {json_path}")


def print_summary(anomalies_df: pd.DataFrame, total_events: int):
    """Print detection summary."""
    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)
    print(f"Total events analyzed: {total_events:,}")
    print(f"Anomalies detected: {len(anomalies_df):,} ({len(anomalies_df)/total_events*100:.2f}%)")

    if len(anomalies_df) > 0:
        print("\nThreat Type Breakdown:")
        for threat, count in anomalies_df['threat_type'].value_counts().items():
            print(f"  {threat}: {count}")

        print("\nSeverity Distribution:")
        for severity, count in anomalies_df['severity'].value_counts().items():
            print(f"  {severity.upper()}: {count}")

        print("\nTop 5 Affected Users:")
        for user, count in anomalies_df['user'].value_counts().head(5).items():
            print(f"  {user}: {count}")

        print("\nTop 5 Source IPs:")
        for ip, count in anomalies_df['source_ip'].value_counts().head(5).items():
            print(f"  {ip}: {count}")


def generate_visualizations(df: pd.DataFrame, scores: np.ndarray, threshold: float,
                           anomalies_df: pd.DataFrame, output_dir: str):
    """Create anomaly detection visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Anomaly Timeline
    is_anomaly = scores > threshold
    axes[0, 0].scatter(df['timestamp'], scores, c=is_anomaly,
                      cmap='RdYlGn_r', alpha=0.6, s=10)
    axes[0, 0].axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})', linewidth=2)
    axes[0, 0].set_title('Anomaly Scores Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Anomaly Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Score Distribution
    axes[0, 1].hist(scores, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0, 1].axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})', linewidth=2)
    axes[0, 1].set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Anomaly Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # 3. Threat Type Breakdown
    if len(anomalies_df) > 0:
        threat_counts = anomalies_df['threat_type'].value_counts()
        colors = plt.cm.Set3(range(len(threat_counts)))
        axes[1, 0].bar(range(len(threat_counts)), threat_counts.values, color=colors)
        axes[1, 0].set_xticks(range(len(threat_counts)))
        axes[1, 0].set_xticklabels(threat_counts.index, rotation=45, ha='right')
        axes[1, 0].set_title('Detected Threat Types', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Threat Type')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 0].text(0.5, 0.5, 'No anomalies detected', ha='center', va='center', fontsize=14)
        axes[1, 0].set_title('Detected Threat Types', fontsize=14, fontweight='bold')

    # 4. Top Anomalous Entities
    if len(anomalies_df) > 0:
        top_users = anomalies_df['user'].value_counts().head(10)
        axes[1, 1].barh(range(len(top_users)), top_users.values, color='coral')
        axes[1, 1].set_yticks(range(len(top_users)))
        axes[1, 1].set_yticklabels(top_users.index)
        axes[1, 1].set_title('Top 10 Anomalous Users', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Anomaly Count')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
    else:
        axes[1, 1].text(0.5, 0.5, 'No anomalies detected', ha='center', va='center', fontsize=14)
        axes[1, 1].set_title('Top 10 Anomalous Users', fontsize=14, fontweight='bold')

    plt.tight_layout()

    fig_path = os.path.join(output_dir, 'anomaly_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visualization: {fig_path}")


def save_artifacts(feature_pipeline, iso_forest, autoencoder, stat_detector,
                  scorer, threshold, config):
    """Save all trained artifacts for deployment."""
    # Save feature pipeline
    pipeline_path = os.path.join(config.output_dir, 'feature_pipeline.pkl')
    joblib.dump(feature_pipeline, pipeline_path)

    # Save Isolation Forest
    iso_path = os.path.join(config.output_dir, 'isolation_forest_model.pkl')
    joblib.dump(iso_forest, iso_path)

    # Save Autoencoder
    ae_path = os.path.join(config.output_dir, 'autoencoder_model.keras')
    autoencoder.save(ae_path)

    # Save Statistical Detector
    stat_path = os.path.join(config.output_dir, 'statistical_detector.pkl')
    joblib.dump(stat_detector, stat_path)

    # Save complete inference package
    inference_package = {
        'feature_pipeline': feature_pipeline,
        'isolation_forest': iso_forest,
        'statistical_detector': stat_detector,
        'anomaly_scorer': scorer,
        'threshold': threshold,
        'config': {
            'contamination': config.contamination,
            'severity_thresholds': config.severity_thresholds
        }
    }

    package_path = os.path.join(config.output_dir, 'inference_package.pkl')
    joblib.dump(inference_package, package_path)

    print(f"\nSaved artifacts to {config.output_dir}/:")
    print(f"  - feature_pipeline.pkl")
    print(f"  - isolation_forest_model.pkl")
    print(f"  - autoencoder_model.keras")
    print(f"  - statistical_detector.pkl")
    print(f"  - inference_package.pkl")


# ==========================================
# LEGACY EVALUATION (REMOVE IN PRODUCTION)
# ==========================================
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
                  model_name: str, output_dir: str) -> dict:
    """
    Comprehensive model evaluation with visualizations.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        model_name: Name for saving files
        output_dir: Directory to save plots
        
    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} EVALUATION")
    print('='*60)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'avg_precision': average_precision_score(y_true, y_proba)
    }
    
    # Print metrics
    print(f"\nMetrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1 Score:          {metrics['f1']:.4f}")
    print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"  Average Precision: {metrics['avg_precision']:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Attack']))
    
    # Find optimal threshold
    opt_threshold, opt_f1 = find_optimal_threshold(y_true, y_proba, metric='f1')
    print(f"Optimal Threshold (F1): {opt_threshold:.2f} (F1={opt_f1:.4f})")
    metrics['optimal_threshold'] = opt_threshold
    metrics['optimal_f1'] = opt_f1
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    axes[0, 0].set_title(f'{model_name} - Confusion Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[0, 1].plot(fpr, tpr, 'b-', label=f'ROC (AUC={metrics["roc_auc"]:.4f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title(f'{model_name} - ROC Curve')
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    axes[1, 0].plot(recall, precision, 'g-', label=f'PR (AP={metrics["avg_precision"]:.4f})')
    axes[1, 0].axhline(y=y_true.mean(), color='k', linestyle='--', label='Baseline')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title(f'{model_name} - Precision-Recall Curve')
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Probability Distribution
    axes[1, 1].hist(y_proba[y_true == 0], bins=50, alpha=0.5, label='Benign', density=True)
    axes[1, 1].hist(y_proba[y_true == 1], bins=50, alpha=0.5, label='Attack', density=True)
    axes[1, 1].axvline(x=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    axes[1, 1].axvline(x=opt_threshold, color='g', linestyle='--', label=f'Optimal ({opt_threshold:.2f})')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title(f'{model_name} - Probability Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    safe_name = model_name.lower().replace(' ', '_')
    fig_path = os.path.join(output_dir, f'{safe_name}_evaluation.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved evaluation plots to: {fig_path}")
    
    return metrics


def plot_feature_importance(model: RandomForestClassifier, feature_names: list,
                           output_dir: str, top_n: int = 30):
    """Plot and save feature importance from Random Forest."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
    
    ax.barh(range(len(top_features)), top_importances[::-1], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Feature Importances (Random Forest)')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved feature importance plot to: {fig_path}")
    
    # Also save as CSV
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    csv_path = os.path.join(output_dir, 'feature_importance.csv')
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved feature importance CSV to: {csv_path}")


def plot_training_history(history, output_dir: str):
    """Plot neural network training history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train')
    axes[1].plot(history.history['val_accuracy'], label='Validation')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # AUC
    axes[2].plot(history.history['auc'], label='Train')
    axes[2].plot(history.history['val_auc'], label='Validation')
    axes[2].set_title('Model AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = os.path.join(output_dir, 'nn_training_history.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training history plot to: {fig_path}")


# ==========================================
# MAIN PIPELINE
# ==========================================
def main(config: Config):
    """
    Main anomaly detection pipeline.

    Args:
        config: Configuration object
    """
    print("\n" + "="*60)
    print("LOG ANOMALY DETECTION PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # ==========================================
    # STEP 1: Load and Parse Logs
    # ==========================================
    try:
        log_parser = LogParser(config.log_format)
        df = log_parser.load_logs(config.data_path)
    except Exception as e:
        print(f"ERROR: Log loading failed - {e}")
        return
    
    # ==========================================
    # STEP 2: Preprocess Logs
    # ==========================================
    df = preprocess_logs(df)

    # ==========================================
    # STEP 3: Temporal Split (Baseline vs Analysis)
    # ==========================================
    print(f"\n{'='*60}")
    print("TEMPORAL SPLIT")
    print('='*60)

    cutoff_date = df['timestamp'].min() + pd.Timedelta(days=config.baseline_period_days)
    baseline_df = df[df['timestamp'] < cutoff_date].copy()
    analysis_df = df[df['timestamp'] >= cutoff_date].copy()

    print(f"Baseline period: {len(baseline_df):,} events ({baseline_df['timestamp'].min()} to {baseline_df['timestamp'].max()})")
    print(f"Analysis period: {len(analysis_df):,} events ({analysis_df['timestamp'].min()} to {analysis_df['timestamp'].max()})")

    if len(baseline_df) == 0:
        print("ERROR: No baseline data available. Increase baseline_period_days or check your data.")
        return

    if len(analysis_df) == 0:
        print("ERROR: No analysis data available. Decrease baseline_period_days or check your data.")
        return

    # ==========================================
    # STEP 4: Feature Engineering
    # ==========================================
    feature_pipeline = LogFeaturePipeline(time_windows=config.time_windows)

    # Fit on baseline (normal behavior)
    baseline_features = feature_pipeline.fit_transform(baseline_df, scale=True)

    # Transform analysis period
    analysis_features = feature_pipeline.transform(analysis_df, scale=True)

    print(f"\nFinal feature count: {baseline_features.shape[1]}")

    # ==========================================
    # STEP 5: Train Anomaly Detectors
    # ==========================================
    print(f"\n{'='*60}")
    print("TRAINING ANOMALY DETECTORS")
    print('='*60)

    # 5a. Isolation Forest
    print("\nTraining Isolation Forest...")
    iso_forest = create_isolation_forest(
        contamination=config.contamination,
        n_estimators=config.iso_forest_estimators,
        random_state=config.random_state
    )
    iso_forest.fit(baseline_features)
    print("Isolation Forest trained successfully")

    # 5b. Autoencoder
    print("\nTraining Autoencoder...")
    encoder, decoder, autoencoder = create_autoencoder(baseline_features.shape[1])
    autoencoder.fit(
        baseline_features, baseline_features,
        epochs=config.autoencoder_epochs,
        batch_size=config.batch_size,
        validation_split=0.15,
        verbose=1
    )
    print("Autoencoder trained successfully")

    # 5c. Statistical Detector
    print("\nFitting Statistical Threat Detector...")
    stat_detector = StatisticalAnomalyDetector()
    stat_detector.fit(baseline_df)
    print("Statistical detector fitted successfully")

    # ==========================================
    # STEP 6: Detect Anomalies
    # ==========================================
    print(f"\n{'='*60}")
    print("DETECTING ANOMALIES")
    print('='*60)

    # Get scores from each detector
    print("Computing Isolation Forest scores...")
    iso_scores = -iso_forest.score_samples(analysis_features)

    print("Computing Autoencoder reconstruction errors...")
    ae_scores = compute_reconstruction_error(analysis_features, autoencoder)

    print("Computing Statistical threat scores...")
    stat_scores = stat_detector.detect_all(analysis_df)

    # ==========================================
    # STEP 7: Combine Scores and Threshold
    # ==========================================
    print("\nCombining detector scores...")
    scorer = AnomalyScorer()
    combined_scores = scorer.combine_scores({
        'isolation_forest': iso_scores,
        'autoencoder': ae_scores,
        'statistical': stat_scores
    })

    threshold = scorer.calibrate_threshold(combined_scores, config.contamination)
    print(f"Anomaly threshold: {threshold:.4f}")

    is_anomaly = combined_scores > threshold
    print(f"Anomalies detected: {is_anomaly.sum():,} / {len(analysis_df):,} ({is_anomaly.sum()/len(analysis_df)*100:.2f}%)")

    # ==========================================
    # STEP 8: Classify Threat Types
    # ==========================================
    print(f"\n{'='*60}")
    print("CLASSIFYING THREAT TYPES")
    print('='*60)

    anomalies_df = analysis_df[is_anomaly].copy()
    anomalies_df['anomaly_score'] = combined_scores[is_anomaly]
    anomalies_df['threat_type'] = classify_threat_type(anomalies_df, stat_detector)
    anomalies_df['severity'] = assign_severity(
        combined_scores[is_anomaly],
        config.severity_thresholds
    )

    print(f"Classified {len(anomalies_df)} anomalies")

    # ==========================================
    # STEP 9: Generate Reports and Visualizations
    # ==========================================
    print(f"\n{'='*60}")
    print("GENERATING REPORTS")
    print('='*60)

    if len(anomalies_df) > 0:
        save_anomaly_report(anomalies_df, config.output_dir)

    generate_visualizations(analysis_df, combined_scores, threshold,
                          anomalies_df, config.output_dir)

    # ==========================================
    # STEP 10: Save Artifacts
    # ==========================================
    print(f"\n{'='*60}")
    print("SAVING ARTIFACTS")
    print('='*60)

    save_artifacts(feature_pipeline, iso_forest, autoencoder,
                  stat_detector, scorer, threshold, config)

    # ==========================================
    # STEP 11: Summary
    # ==========================================
    print_summary(anomalies_df, len(analysis_df))

    print(f"\nAll outputs saved to: {config.output_dir}/")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ==========================================
# INFERENCE UTILITIES
# ==========================================
def load_inference_package(package_path: str) -> dict:
    """Load the saved inference package."""
    return joblib.load(package_path)


def predict(data: pd.DataFrame, package_path: str, model: str = 'rf',
           use_optimal_threshold: bool = True) -> tuple:
    """
    Make predictions on new data.
    
    Args:
        data: DataFrame with same structure as training data
        package_path: Path to inference_package.pkl
        model: 'rf' for Random Forest, 'nn' for Neural Network
        use_optimal_threshold: Whether to use optimized threshold
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    package = load_inference_package(package_path)
    pipeline = package['feature_pipeline']
    
    if model == 'rf':
        clf = package['rf_model']
        X_transformed = pipeline.transform(data, scale=False)
        proba = clf.predict_proba(X_transformed)[:, 1]
        threshold = package['config']['optimal_threshold_rf'] if use_optimal_threshold else 0.5
    else:
        nn_model = tf.keras.models.load_model(
            package_path.replace('inference_package.pkl', 'neural_network_model.keras')
        )
        X_transformed = pipeline.transform(data, scale=True)
        proba = nn_model.predict(X_transformed, verbose=0).flatten()
        threshold = package['config']['optimal_threshold_nn'] if use_optimal_threshold else 0.5
    
    predictions = (proba >= threshold).astype(int)
    
    return predictions, proba


# ==========================================
# CLI ENTRY POINT
# ==========================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Log Anomaly Detection Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data_path', type=str, default='./logs/',
                       help='Path to directory containing JSON log files')
    parser.add_argument('--log_format', type=str, default='auto',
                       help='Log format: auto, auth, syslog, security')
    parser.add_argument('--output_dir', type=str, default='anomaly_outputs',
                       help='Directory to save outputs')
    parser.add_argument('--contamination', type=float, default=0.01,
                       help='Expected anomaly rate (0.01 = 1%%)')
    parser.add_argument('--baseline_period_days', type=int, default=7,
                       help='Days of normal behavior for baseline')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--iso_forest_estimators', type=int, default=200,
                       help='Number of trees in Isolation Forest')
    parser.add_argument('--autoencoder_epochs', type=int, default=50,
                       help='Training epochs for autoencoder')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for autoencoder')
    parser.add_argument('--time_windows', type=int, nargs='+', default=[3600, 86400, 604800],
                       help='Time windows for feature extraction (in seconds)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = Config(args)
    main(config)
