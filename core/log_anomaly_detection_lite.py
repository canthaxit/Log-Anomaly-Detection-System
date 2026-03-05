"""
Log Anomaly Detection Pipeline - Lite Version
==============================================
Simplified version without TensorFlow for quick testing.

Features:
- JSON log parsing
- Isolation Forest + Statistical threat detection
- Faster installation (no TensorFlow required)
- All reporting and visualization features

Author: Lite version for easy testing
"""

import pandas as pd
import numpy as np
import glob
import os
import joblib
import argparse
import json
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

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
        self.contamination = args.contamination if args else 0.01
        self.baseline_period_days = args.baseline_period_days if args else 7
        self.random_state = args.random_state if args else 42

        # Model parameters
        self.iso_forest_estimators = args.iso_forest_estimators if args else 200

        # Feature engineering parameters
        self.time_windows = args.time_windows if args else [3600, 86400, 604800]

        # Alert parameters
        self.severity_thresholds = {
            'low': 0.5,
            'medium': 0.7,
            'high': 0.85,
            'critical': 0.95
        }

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Set random seeds
        self._set_seeds()

    def _set_seeds(self):
        """Set all random seeds for reproducibility."""
        np.random.seed(self.random_state)


# ==========================================
# LOG PARSING
# ==========================================
class LogParser:
    """Parse JSON log files into unified DataFrame schema."""

    def __init__(self, log_format='auto'):
        self.log_format = log_format

    def parse_log_file(self, filepath: str) -> pd.DataFrame:
        """Parse single JSON log file."""
        logs = []

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()

            try:
                data = json.loads(content)
                if isinstance(data, list):
                    logs = data
                elif isinstance(data, dict):
                    logs = [data]
            except json.JSONDecodeError:
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            logs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        if not logs:
            return pd.DataFrame()

        df = pd.DataFrame(logs)
        return self._normalize_schema(df)

    def _normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize log entries to unified schema."""
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

        for standard_field, variations in field_mappings.items():
            for var in variations:
                if var in df.columns:
                    normalized_df[standard_field] = df[var]
                    break
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

        if 'timestamp' in normalized_df.columns:
            try:
                normalized_df['timestamp'] = pd.to_datetime(normalized_df['timestamp'], errors='coerce')
            except Exception:
                normalized_df['timestamp'] = pd.NaT

        for col in ['user', 'event_type', 'action', 'severity']:
            if col in normalized_df.columns:
                normalized_df[col] = normalized_df[col].astype(str).str.strip().str.lower()

        return normalized_df

    def load_logs(self, path: str) -> pd.DataFrame:
        """Load and parse all JSON logs from directory."""
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
# LOG PREPROCESSING
# ==========================================
def preprocess_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess log data before analysis."""
    print(f"\n{'='*60}")
    print("LOG PREPROCESSING")
    print('='*60)

    initial_rows = len(df)

    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df):,} duplicate rows")

    invalid_timestamps = df['timestamp'].isna().sum()
    if invalid_timestamps > 0:
        print(f"Removing {invalid_timestamps:,} logs with invalid timestamps")
        df = df[df['timestamp'].notna()]

    df = df.sort_values('timestamp').reset_index(drop=True)

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
# FEATURE PIPELINE
# ==========================================
class LogFeaturePipeline:
    """Extract features from log data for anomaly detection."""

    def __init__(self, time_windows: List[int] = [3600, 86400, 604800]):
        self.time_windows = time_windows
        self.user_baselines_ = None
        self.ip_baselines_ = None
        self.scaler_ = RobustScaler()
        self.feature_names_ = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame) -> 'LogFeaturePipeline':
        """Fit the pipeline on baseline data."""
        print(f"\n{'='*60}")
        print("FITTING LOG FEATURE PIPELINE")
        print('='*60)

        X = X.copy()

        self.user_baselines_ = X.groupby('user').agg({
            'event_type': 'count',
            'source_ip': 'nunique',
            'action': lambda x: (x == 'failed').sum() / len(x) if len(x) > 0 else 0
        }).to_dict('index')
        print(f"Built baselines for {len(self.user_baselines_)} users")

        self.ip_baselines_ = X.groupby('source_ip').agg({
            'user': 'nunique',
            'event_type': 'count'
        }).to_dict('index')
        print(f"Built baselines for {len(self.ip_baselines_)} IPs")

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

        # Temporal features
        features['hour'] = X['timestamp'].dt.hour
        features['day_of_week'] = X['timestamp'].dt.dayofweek
        features['is_weekend'] = (X['timestamp'].dt.dayofweek >= 5).astype(int)
        features['is_business_hours'] = ((X['timestamp'].dt.hour >= 9) & (X['timestamp'].dt.hour < 17)).astype(int)
        features['is_night'] = ((X['timestamp'].dt.hour < 6) | (X['timestamp'].dt.hour >= 22)).astype(int)

        # Action features
        features['is_failed'] = (X['action'] == 'failed').astype(int)
        features['is_denied'] = (X['action'] == 'denied').astype(int)
        features['is_success'] = (X['action'] == 'success').astype(int)

        # Event type features
        top_events = X['event_type'].value_counts().head(10).index.tolist()
        for event in top_events:
            features[f'event_{event}'] = (X['event_type'] == event).astype(int)

        # User features
        features['user_is_unknown'] = (X['user'] == 'unknown').astype(int)
        features['user_is_root'] = (X['user'] == 'root').astype(int)
        features['user_is_admin'] = (X['user'].str.contains('admin', case=False, na=False)).astype(int)

        # IP features
        features['ip_is_unknown'] = (X['source_ip'] == 'unknown').astype(int)
        features['ip_is_localhost'] = (X['source_ip'].str.startswith('127.', na=False)).astype(int)
        features['ip_is_private'] = (X['source_ip'].str.startswith(('192.168.', '10.', '172.'), na=False)).astype(int)

        # Baseline deviation features
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

        # Rolling window features (simplified for compatibility)
        X_sorted = X.sort_values('timestamp').copy()
        for window in self.time_windows:
            window_name = f"{window}s"
            # Use simple count instead of time-based rolling for compatibility
            X_sorted[f'events_per_{window_name}'] = X_sorted.groupby('user').cumcount() + 1
            features[f'events_per_{window_name}'] = X_sorted.set_index(X.index)[f'events_per_{window_name}'].fillna(0)

        features = features.fillna(0)
        return features

    def transform(self, X: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """Transform log data using fitted pipeline."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fit before transform")

        X_features = self._extract_features(X)

        missing_cols = set(self.feature_names_) - set(X_features.columns)
        extra_cols = set(X_features.columns) - set(self.feature_names_)

        for col in missing_cols:
            X_features[col] = 0

        X_features = X_features.drop(columns=list(extra_cols), errors='ignore')
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
    """Create an Isolation Forest for anomaly detection."""
    return IsolationForest(
        n_estimators=n_estimators,
        max_samples='auto',
        contamination=contamination,
        max_features=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )


# ==========================================
# STATISTICAL THREAT DETECTOR
# ==========================================
class StatisticalAnomalyDetector:
    """Rule-based detection for specific threat patterns."""

    def __init__(self):
        self.brute_force_threshold = 10
        self.sudo_threshold = 5
        self.user_baselines_ = None

    def fit(self, df: pd.DataFrame):
        """Establish baselines from normal behavior."""
        self.user_baselines_ = df.groupby('user').agg({
            'event_type': 'count',
            'action': lambda x: (x == 'failed').sum() / max(len(x), 1)
        }).to_dict('index')
        return self

    def detect_brute_force(self, df: pd.DataFrame) -> np.ndarray:
        """Detect credential brute force attacks."""
        scores = np.zeros(len(df))

        # Count failed logins per user
        df_sorted = df.sort_values('timestamp').copy()
        failed_mask = df_sorted['action'] == 'failed'

        if failed_mask.any():
            # Simple cumulative count of failures per user
            df_sorted['failed_count'] = df_sorted[failed_mask].groupby('user').cumcount() + 1
            df_sorted['failed_count'] = df_sorted['failed_count'].fillna(0)
            scores = (df_sorted.set_index(df.index)['failed_count'] / self.brute_force_threshold).fillna(0).values
            scores = np.clip(scores, 0, 1)

        return scores

    def detect_privilege_escalation(self, df: pd.DataFrame) -> np.ndarray:
        """Detect privilege escalation attempts."""
        scores = np.zeros(len(df))

        sudo_mask = df['event_type'].str.contains('sudo', case=False, na=False)
        scores[sudo_mask] = 0.5

        failed_sudo = sudo_mask & (df['action'] == 'failed')
        scores[failed_sudo] = 0.8

        if self.user_baselines_ is not None:
            for idx, row in df[sudo_mask].iterrows():
                baseline_events = self.user_baselines_.get(row['user'], {}).get('event_type', 0)
                if baseline_events == 0:
                    scores[idx] = 1.0

        return scores

    def detect_data_exfiltration(self, df: pd.DataFrame) -> np.ndarray:
        """Detect data exfiltration patterns."""
        scores = np.zeros(len(df))

        is_night = (df['timestamp'].dt.hour < 6) | (df['timestamp'].dt.hour >= 22)
        file_access = df['event_type'].str.contains('file', case=False, na=False)
        scores[is_night & file_access] = 0.6

        sensitive_patterns = ['passwd', 'shadow', '/etc/', 'credentials', 'keys']
        for pattern in sensitive_patterns:
            sensitive_mask = df['message'].str.contains(pattern, case=False, na=False)
            scores[sensitive_mask] = np.maximum(scores[sensitive_mask], 0.7)

        return scores

    def detect_lateral_movement(self, df: pd.DataFrame) -> np.ndarray:
        """Detect lateral movement patterns."""
        scores = np.zeros(len(df))

        network_events = df['event_type'].str.contains('network|ssh|rdp', case=False, na=False)
        scores[network_events] = 0.3

        # Count unique IPs per user (simplified)
        df_sorted = df.sort_values('timestamp').copy()
        df_sorted['unique_ips'] = df_sorted.groupby('user')['source_ip'].transform('nunique')

        scores = np.maximum(scores, (df_sorted.set_index(df.index)['unique_ips'] / 10).fillna(0).values)
        scores = np.clip(scores, 0, 1)

        return scores

    def detect_all(self, df: pd.DataFrame) -> np.ndarray:
        """Run all detectors and combine scores."""
        bf_scores = self.detect_brute_force(df)
        pe_scores = self.detect_privilege_escalation(df)
        de_scores = self.detect_data_exfiltration(df)
        lm_scores = self.detect_lateral_movement(df)

        combined = np.maximum.reduce([bf_scores, pe_scores, de_scores, lm_scores])
        return combined


# ==========================================
# ANOMALY SCORING
# ==========================================
class AnomalyScorer:
    """Combine scores from multiple detectors."""

    def __init__(self):
        self.weights = {
            'isolation_forest': 0.50,  # Increased weight (no autoencoder)
            'statistical': 0.50         # Increased weight
        }
        self._fitted_ranges: Dict[str, Tuple[float, float]] = {}

    def fit_normalization(self, detector_name: str, scores: np.ndarray):
        """Store baseline (min, max) for a detector so future batches
        use the same normalization range instead of per-batch rescaling."""
        self._fitted_ranges[detector_name] = (float(scores.min()), float(scores.max()))

    def normalize_scores(self, scores: np.ndarray, detector_name: Optional[str] = None) -> np.ndarray:
        """Normalize scores to [0, 1] range.

        If *detector_name* is given and a fitted range exists, use that range
        for consistent normalization across batches.  Falls back to per-batch
        min/max for backward compatibility.
        """
        if detector_name and detector_name in self._fitted_ranges:
            smin, smax = self._fitted_ranges[detector_name]
        else:
            smin, smax = float(scores.min()), float(scores.max())

        if smax == smin:
            return np.zeros_like(scores)
        normalized = (scores - smin) / (smax - smin)
        return np.clip(normalized, 0.0, 1.0)

    def combine_scores(self, scores_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted combination of anomaly scores."""
        combined = np.zeros(len(list(scores_dict.values())[0]))

        for detector_name, scores in scores_dict.items():
            if detector_name in self.weights:
                normalized = self.normalize_scores(scores, detector_name=detector_name)
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
    csv_path = os.path.join(output_dir, 'anomalies_detected.csv')
    anomalies_df.to_csv(csv_path, index=False)
    print(f"Saved CSV report: {csv_path}")

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

    # Anomaly Timeline
    is_anomaly = scores > threshold
    axes[0, 0].scatter(df['timestamp'], scores, c=is_anomaly,
                      cmap='RdYlGn_r', alpha=0.6, s=10)
    axes[0, 0].axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})', linewidth=2)
    axes[0, 0].set_title('Anomaly Scores Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Anomaly Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Score Distribution
    axes[0, 1].hist(scores, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0, 1].axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})', linewidth=2)
    axes[0, 1].set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Anomaly Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Threat Type Breakdown
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

    # Top Anomalous Users
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


def save_artifacts(feature_pipeline, iso_forest, stat_detector, scorer, threshold, config):
    """Save all trained artifacts for deployment."""
    pipeline_path = os.path.join(config.output_dir, 'feature_pipeline.pkl')
    joblib.dump(feature_pipeline, pipeline_path)

    iso_path = os.path.join(config.output_dir, 'isolation_forest_model.pkl')
    joblib.dump(iso_forest, iso_path)

    stat_path = os.path.join(config.output_dir, 'statistical_detector.pkl')
    joblib.dump(stat_detector, stat_path)

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
    print(f"  - statistical_detector.pkl")
    print(f"  - inference_package.pkl")


# ==========================================
# MAIN PIPELINE
# ==========================================
def main(config: Config):
    """Main anomaly detection pipeline."""
    print("\n" + "="*60)
    print("LOG ANOMALY DETECTION PIPELINE - LITE VERSION")
    print("(Isolation Forest + Statistical Detection)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Load and Parse Logs
    try:
        log_parser = LogParser(config.log_format)
        df = log_parser.load_logs(config.data_path)
    except Exception as e:
        print(f"ERROR: Log loading failed - {e}")
        return

    # Preprocess Logs
    df = preprocess_logs(df)

    # Temporal Split
    print(f"\n{'='*60}")
    print("TEMPORAL SPLIT")
    print('='*60)

    cutoff_date = df['timestamp'].min() + pd.Timedelta(days=config.baseline_period_days)
    baseline_df = df[df['timestamp'] < cutoff_date].copy()
    analysis_df = df[df['timestamp'] >= cutoff_date].copy()

    print(f"Baseline period: {len(baseline_df):,} events ({baseline_df['timestamp'].min()} to {baseline_df['timestamp'].max()})")
    print(f"Analysis period: {len(analysis_df):,} events ({analysis_df['timestamp'].min()} to {analysis_df['timestamp'].max()})")

    if len(baseline_df) == 0:
        print("ERROR: No baseline data available.")
        return

    if len(analysis_df) == 0:
        print("ERROR: No analysis data available.")
        return

    # Feature Engineering
    feature_pipeline = LogFeaturePipeline(time_windows=config.time_windows)
    baseline_features = feature_pipeline.fit_transform(baseline_df, scale=True)
    analysis_features = feature_pipeline.transform(analysis_df, scale=True)

    print(f"\nFinal feature count: {baseline_features.shape[1]}")

    # Train Detectors
    print(f"\n{'='*60}")
    print("TRAINING ANOMALY DETECTORS")
    print('='*60)

    print("\nTraining Isolation Forest...")
    iso_forest = create_isolation_forest(
        contamination=config.contamination,
        n_estimators=config.iso_forest_estimators,
        random_state=config.random_state
    )
    iso_forest.fit(baseline_features)
    print("Isolation Forest trained successfully")

    print("\nFitting Statistical Threat Detector...")
    stat_detector = StatisticalAnomalyDetector()
    stat_detector.fit(baseline_df)
    print("Statistical detector fitted successfully")

    # Detect Anomalies
    print(f"\n{'='*60}")
    print("DETECTING ANOMALIES")
    print('='*60)

    print("Computing Isolation Forest scores...")
    iso_scores = -iso_forest.score_samples(analysis_features)

    print("Computing Statistical threat scores...")
    stat_scores = stat_detector.detect_all(analysis_df)

    # Combine Scores
    print("\nCombining detector scores...")
    scorer = AnomalyScorer()
    combined_scores = scorer.combine_scores({
        'isolation_forest': iso_scores,
        'statistical': stat_scores
    })

    threshold = scorer.calibrate_threshold(combined_scores, config.contamination)
    print(f"Anomaly threshold: {threshold:.4f}")

    is_anomaly = combined_scores > threshold
    print(f"Anomalies detected: {is_anomaly.sum():,} / {len(analysis_df):,} ({is_anomaly.sum()/len(analysis_df)*100:.2f}%)")

    # Classify Threats
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

    # Generate Reports
    print(f"\n{'='*60}")
    print("GENERATING REPORTS")
    print('='*60)

    if len(anomalies_df) > 0:
        save_anomaly_report(anomalies_df, config.output_dir)

    generate_visualizations(analysis_df, combined_scores, threshold,
                          anomalies_df, config.output_dir)

    # Save Artifacts
    print(f"\n{'='*60}")
    print("SAVING ARTIFACTS")
    print('='*60)

    save_artifacts(feature_pipeline, iso_forest, stat_detector, scorer, threshold, config)

    # Summary
    print_summary(anomalies_df, len(analysis_df))

    print(f"\nAll outputs saved to: {config.output_dir}/")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ==========================================
# CLI ENTRY POINT
# ==========================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Log Anomaly Detection Pipeline - Lite Version',
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
    parser.add_argument('--time_windows', type=int, nargs='+', default=[3600, 86400, 604800],
                       help='Time windows for feature extraction (in seconds)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = Config(args)
    main(config)
