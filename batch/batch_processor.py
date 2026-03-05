#!/usr/bin/env python3
"""
Batch Log Processor
Processes log files in batches at scheduled intervals
"""

import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import joblib

from log_anomaly_detection_lite import (
    LogParser,
    LogFeaturePipeline,
    StatisticalAnomalyDetector,
    AnomalyScorer,
    preprocess_logs
)
from sklearn.ensemble import IsolationForest

# Import security utilities
import sys
_parent = os.path.join(os.path.dirname(__file__), '..')
if os.path.isfile(os.path.join(_parent, 'common', 'security.py')):
    sys.path.insert(0, _parent)
from common.security import validate_model_path, validate_log_path, verify_model_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("batch-processor")


class BatchProcessor:
    """Process log files in batches."""

    def __init__(
        self,
        model_dir: str = "anomaly_outputs",
        log_dir: str = "logs",
        output_dir: str = "batch_outputs",
        interval: int = 3600
    ):
        self.model_dir = Path(model_dir)
        # Validate log_dir is within allowed directories
        try:
            self.log_dir = validate_log_path(log_dir)
        except ValueError:
            logger.error(f"Log directory is outside allowed paths")
            raise
        self.output_dir = Path(output_dir)
        self.interval = interval

        self.output_dir.mkdir(exist_ok=True)

        self.feature_pipeline = None
        self.isolation_forest = None
        self.statistical_detector = None
        self.scorer = None
        self.threshold = None

        self.processed_files = set()
        self.load_processed_files()

    def load_processed_files(self):
        """Load list of already processed files."""
        processed_file = self.output_dir / "processed_files.txt"
        if processed_file.exists():
            with open(processed_file, 'r') as f:
                self.processed_files = set(line.strip() for line in f)
            logger.info(f"Loaded {len(self.processed_files)} processed files from history")

    def save_processed_file(self, filepath: str):
        """Record a processed file."""
        self.processed_files.add(filepath)
        with open(self.output_dir / "processed_files.txt", 'a') as f:
            f.write(f"{filepath}\n")

    def load_models(self):
        """Load trained models."""
        try:
            self.model_dir = validate_model_path(str(self.model_dir))
            for pkl in ("feature_pipeline.pkl", "isolation_forest_model.pkl", "statistical_detector.pkl"):
                verify_model_file(self.model_dir / pkl)
            self.feature_pipeline = joblib.load(self.model_dir / "feature_pipeline.pkl")
            self.isolation_forest = joblib.load(self.model_dir / "isolation_forest_model.pkl")
            self.statistical_detector = joblib.load(self.model_dir / "statistical_detector.pkl")

            if (self.model_dir / "inference_package.pkl").exists():
                verify_model_file(self.model_dir / "inference_package.pkl")
                package = joblib.load(self.model_dir / "inference_package.pkl")
                self.scorer = package.get("scorer")
                self.threshold = package.get("threshold")
            else:
                self.scorer = AnomalyScorer()
                self.threshold = 0.7

            logger.info(f"Models loaded successfully from {self.model_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def find_new_files(self) -> List[Path]:
        """Find new log files to process."""
        all_files = []

        # Find JSON and CSV files
        for pattern in ['*.json', '*.csv']:
            all_files.extend(self.log_dir.glob(pattern))
            all_files.extend(self.log_dir.glob(f'**/{pattern}'))

        # Filter out already processed
        new_files = [f for f in all_files if str(f) not in self.processed_files]

        logger.info(f"Found {len(new_files)} new files to process")
        return new_files

    def process_file(self, filepath: Path) -> Dict[str, Any]:
        """Process a single log file."""
        try:
            logger.info(f"Processing: {filepath}")

            # Parse log file
            parser = LogParser()
            df = parser.parse_log_file(str(filepath))

            if len(df) == 0:
                logger.warning(f"No valid logs in {filepath}")
                return {"status": "empty", "filepath": str(filepath)}

            # Preprocess
            df = preprocess_logs(df)

            # Extract features
            features = self.feature_pipeline.transform(df)

            # Detect anomalies
            iso_scores = -self.isolation_forest.score_samples(features)
            stat_scores = self.statistical_detector.detect_all(df)

            # Combine scores
            combined_scores = self.scorer.combine_scores({
                'isolation_forest': iso_scores,
                'statistical': stat_scores
            })

            # Identify anomalies
            is_anomaly = combined_scores > self.threshold

            # Build result
            anomalies_df = df[is_anomaly].copy()
            anomalies_df['anomaly_score'] = combined_scores[is_anomaly]

            # Classify threats
            anomalies_df['threat_type'] = anomalies_df.apply(
                self.classify_threat, axis=1
            )

            # Assign severity
            anomalies_df['severity'] = anomalies_df['anomaly_score'].apply(
                self.assign_severity
            )

            # Save results
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"{filepath.stem}_{timestamp}.json"

            result = {
                "source_file": str(filepath),
                "processed_at": datetime.utcnow().isoformat(),
                "total_events": len(df),
                "anomalies_detected": len(anomalies_df),
                "anomaly_rate": len(anomalies_df) / len(df),
                "threshold": float(self.threshold),
                "anomalies": anomalies_df.to_dict(orient='records')
            }

            # Convert timestamps to strings
            for anomaly in result["anomalies"]:
                if 'timestamp' in anomaly and hasattr(anomaly['timestamp'], 'isoformat'):
                    anomaly['timestamp'] = anomaly['timestamp'].isoformat()

            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            logger.info(f"Saved results to {output_file}")
            logger.info(f"Detected {len(anomalies_df)} anomalies in {len(df)} events")

            # Mark as processed
            self.save_processed_file(str(filepath))

            return {
                "status": "success",
                "filepath": str(filepath),
                "output_file": str(output_file),
                "anomalies_detected": len(anomalies_df),
                "total_events": len(df)
            }

        except Exception as e:
            logger.error(f"Failed to process {filepath}: {e}", exc_info=True)
            return {"status": "error", "filepath": str(filepath), "error": "Processing failed"}

    def classify_threat(self, row: pd.Series) -> str:
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

    def assign_severity(self, score: float) -> str:
        """Assign severity based on score."""
        if score >= 0.95:
            return 'critical'
        elif score >= 0.85:
            return 'high'
        elif score >= 0.7:
            return 'medium'
        else:
            return 'low'

    def process_batch(self) -> Dict[str, Any]:
        """Process all new files."""
        start_time = datetime.utcnow()

        new_files = self.find_new_files()

        if not new_files:
            logger.info("No new files to process")
            return {
                "status": "success",
                "processed": 0,
                "errors": 0,
                "duration_seconds": 0
            }

        results = []
        errors = 0

        for filepath in new_files:
            result = self.process_file(filepath)
            results.append(result)
            if result["status"] == "error":
                errors += 1

        duration = (datetime.utcnow() - start_time).total_seconds()

        summary = {
            "status": "success",
            "processed": len(results),
            "errors": errors,
            "duration_seconds": duration,
            "results": results
        }

        # Save summary
        summary_file = self.output_dir / f"batch_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Batch complete: {len(results)} files processed, {errors} errors, {duration:.2f}s")

        return summary

    def run_continuous(self):
        """Run continuous batch processing."""
        logger.info(f"Starting continuous batch processor (interval: {self.interval}s)")

        if not self.load_models():
            logger.error("Failed to load models. Exiting.")
            return

        while True:
            try:
                self.process_batch()
            except Exception as e:
                logger.error(f"Batch processing error: {e}")

            logger.info(f"Sleeping for {self.interval} seconds...")
            time.sleep(self.interval)

    def run_once(self):
        """Run batch processing once."""
        logger.info("Running single batch process")

        if not self.load_models():
            logger.error("Failed to load models. Exiting.")
            return

        return self.process_batch()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch Log Processor")
    parser.add_argument('--model-dir', type=str, default='anomaly_outputs',
                       help='Directory containing trained models')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory containing log files to process')
    parser.add_argument('--output-dir', type=str, default='batch_outputs',
                       help='Directory to save results')
    parser.add_argument('--interval', type=int, default=int(os.getenv('BATCH_INTERVAL', 3600)),
                       help='Processing interval in seconds (default: 3600)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (default: continuous)')

    args = parser.parse_args()

    processor = BatchProcessor(
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        interval=args.interval
    )

    if args.once:
        processor.run_once()
    else:
        processor.run_continuous()


if __name__ == "__main__":
    main()
