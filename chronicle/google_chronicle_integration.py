#!/usr/bin/env python3
"""
Google Chronicle SIEM Integration
Sends anomaly detection results to Google Security Operations (Chronicle)
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import base64

try:
    from google.oauth2 import service_account
    from google.auth.transport.requests import AuthorizedSession
    import requests
except ImportError:
    print("Google Cloud libraries not installed.")
    print("Install with: pip install google-auth google-auth-httplib2 requests")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chronicle-integration")


class ChronicleClient:
    """Client for Google Chronicle SIEM integration."""

    def __init__(
        self,
        credentials_file: str,
        customer_id: str,
        region: str = "us",
        log_type: str = "SECURITY_ANOMALY"
    ):
        """
        Initialize Chronicle client.

        Args:
            credentials_file: Path to Google Cloud service account JSON
            customer_id: Chronicle customer ID
            region: Chronicle region (us, europe, asia)
            log_type: Log type for Chronicle ingestion
        """
        if customer_id == "YOUR_CUSTOMER_ID":
            raise ValueError(
                "Chronicle customer_id is still set to the placeholder 'YOUR_CUSTOMER_ID'. "
                "Please configure a real customer ID."
            )
        self.customer_id = customer_id
        self.log_type = log_type

        # Region endpoints
        self.endpoints = {
            "us": "https://malachiteingestion-pa.googleapis.com",
            "europe": "https://europe-malachiteingestion-pa.googleapis.com",
            "asia": "https://asia-southeast1-malachiteingestion-pa.googleapis.com"
        }
        self.base_url = self.endpoints.get(region, self.endpoints["us"])

        # Load credentials
        self.credentials = service_account.Credentials.from_service_account_file(
            credentials_file,
            scopes=['https://www.googleapis.com/auth/chronicle-backstory']
        )

        self.session = AuthorizedSession(self.credentials)
        logger.info(f"Chronicle client initialized for customer {customer_id} in {region}")

    def convert_to_udm(self, anomaly: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert anomaly to Chronicle UDM (Unified Data Model) format.

        UDM Schema: https://cloud.google.com/chronicle/docs/reference/udm-field-list
        """
        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(anomaly['timestamp'].replace('Z', '+00:00'))
        except Exception:
            timestamp = datetime.utcnow()

        # Map severity to UDM severity
        severity_map = {
            'low': 'LOW',
            'medium': 'MEDIUM',
            'high': 'HIGH',
            'critical': 'CRITICAL'
        }

        # Map threat type to UDM event type
        event_type_map = {
            'brute_force': 'USER_LOGIN',
            'privilege_escalation': 'USER_UNCATEGORIZED',
            'data_exfiltration': 'NETWORK_CONNECTION',
            'lateral_movement': 'NETWORK_CONNECTION',
            'unknown': 'GENERIC_EVENT'
        }

        # Build UDM event
        udm_event = {
            "metadata": {
                "event_timestamp": timestamp.isoformat() + 'Z',
                "event_type": event_type_map.get(anomaly.get('threat_type', 'unknown'), 'GENERIC_EVENT'),
                "vendor_name": "Log Anomaly Detection",
                "product_name": "AI Security Anomaly Detector",
                "product_version": "1.0",
                "log_type": self.log_type,
                "severity": severity_map.get(anomaly.get('severity', 'low'), 'LOW'),
                "description": anomaly.get('message', 'Security anomaly detected')
            },
            "principal": {
                "user": {
                    "userid": anomaly.get('user', 'unknown')
                },
                "ip": [anomaly.get('source_ip', '0.0.0.0')]
            },
            "target": {
                "ip": [anomaly.get('dest_ip', 'unknown')] if anomaly.get('dest_ip') != 'unknown' else []
            },
            "security_result": [{
                "severity": severity_map.get(anomaly.get('severity', 'low'), 'LOW'),
                "category_details": [anomaly.get('threat_type', 'unknown')],
                "summary": f"Anomaly detected: {anomaly.get('threat_type', 'unknown')}",
                "confidence": "HIGH" if anomaly.get('anomaly_score', 0) > 0.8 else "MEDIUM",
                "detection_fields": [{
                    "key": "anomaly_score",
                    "value": str(anomaly.get('anomaly_score', 0))
                }, {
                    "key": "event_type",
                    "value": anomaly.get('event_type', 'unknown')
                }, {
                    "key": "action",
                    "value": anomaly.get('action', 'unknown')
                }]
            }],
            "additional": {
                "fields": [{
                    "key": "anomaly_score",
                    "value": {
                        "string_value": str(anomaly.get('anomaly_score', 0))
                    }
                }, {
                    "key": "threat_type",
                    "value": {
                        "string_value": anomaly.get('threat_type', 'unknown')
                    }
                }, {
                    "key": "raw_message",
                    "value": {
                        "string_value": anomaly.get('message', '')
                    }
                }]
            }
        }

        return udm_event

    def ingest_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ingest events into Chronicle.

        Args:
            events: List of UDM-formatted events

        Returns:
            Response from Chronicle API
        """
        url = f"{self.base_url}/v2/unstructuredlogentries:batchCreate"

        # Build request payload
        payload = {
            "customer_id": self.customer_id,
            "log_type": self.log_type,
            "entries": []
        }

        for event in events:
            # Chronicle expects base64-encoded JSON
            event_json = json.dumps(event)
            event_b64 = base64.b64encode(event_json.encode()).decode()

            payload["entries"].append({
                "log_text": event_b64
            })

        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()

            logger.info(f"Successfully ingested {len(events)} events to Chronicle")
            return {
                "status": "success",
                "events_ingested": len(events),
                "response": response.json()
            }

        except Exception as e:
            logger.error(f"Failed to ingest events: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def send_anomalies(self, anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Send anomaly detection results to Chronicle.

        Args:
            anomalies: List of anomaly dictionaries

        Returns:
            Ingestion result
        """
        if not anomalies:
            logger.info("No anomalies to send")
            return {"status": "success", "events_ingested": 0}

        # Convert to UDM format
        udm_events = [self.convert_to_udm(anomaly) for anomaly in anomalies]

        # Ingest to Chronicle
        return self.ingest_events(udm_events)

    def create_detection_rule(self, rule_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a Chronicle detection rule for anomalies.

        Args:
            rule_config: Rule configuration

        Returns:
            Rule creation result
        """
        url = f"{self.base_url}/v2/detect/rules"

        rule = {
            "rule_name": rule_config.get("name", "Anomaly Detection Alert"),
            "rule_text": self._build_yara_l_rule(rule_config),
            "rule_type": "MULTI_EVENT",
            "enabled": True
        }

        try:
            response = self.session.post(url, json=rule)
            response.raise_for_status()

            logger.info(f"Created Chronicle detection rule: {rule['rule_name']}")
            return {
                "status": "success",
                "rule_id": response.json().get("ruleId"),
                "response": response.json()
            }

        except Exception as e:
            logger.error(f"Failed to create rule: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _build_yara_l_rule(self, config: Dict[str, Any]) -> str:
        """Build YARA-L rule for Chronicle detection."""
        severity = config.get("min_severity", "MEDIUM")
        score = config.get("min_score", 0.7)

        rule = f'''
rule {config.get("name", "anomaly_detection").replace(" ", "_")} {{
  meta:
    author = "Anomaly Detection System"
    description = "{config.get("description", "Detect security anomalies")}"
    severity = "{severity}"

  events:
    $anomaly.metadata.log_type = "SECURITY_ANOMALY"
    $anomaly.security_result.severity = /{severity}|CRITICAL/
    $anomaly.additional.fields.key = "anomaly_score"
    $anomaly.additional.fields.value.string_value >= "{score}"

  condition:
    $anomaly
}}
'''
        return rule.strip()


class ChronicleStreamer:
    """Stream anomaly detection results to Chronicle in real-time."""

    def __init__(self, chronicle_client: ChronicleClient, batch_size: int = 100):
        """
        Initialize streamer.

        Args:
            chronicle_client: Configured Chronicle client
            batch_size: Number of events to batch before sending
        """
        self.client = chronicle_client
        self.batch_size = batch_size
        self.buffer = []

    def add_anomaly(self, anomaly: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Add anomaly to buffer and send when batch is full.

        Args:
            anomaly: Anomaly dictionary

        Returns:
            Ingestion result if batch was sent, None otherwise
        """
        self.buffer.append(anomaly)

        if len(self.buffer) >= self.batch_size:
            return self.flush()

        return None

    def flush(self) -> Optional[Dict[str, Any]]:
        """
        Send all buffered anomalies to Chronicle.

        Returns:
            Ingestion result
        """
        if not self.buffer:
            return None

        result = self.client.send_anomalies(self.buffer)
        self.buffer = []
        return result


class ChronicleConfig:
    """Configuration manager for Chronicle integration."""

    def __init__(self, config_file: str = "chronicle_config.json"):
        """Load configuration from file."""
        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load config from file."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "credentials_file": "chronicle_credentials.json",
            "customer_id": "YOUR_CUSTOMER_ID",
            "region": "us",
            "log_type": "SECURITY_ANOMALY",
            "batch_size": 100,
            "min_severity": "medium",
            "auto_create_rules": True,
            "detection_rules": [
                {
                    "name": "High Severity Anomalies",
                    "description": "Alert on high/critical severity anomalies",
                    "min_severity": "HIGH",
                    "min_score": 0.85
                },
                {
                    "name": "Brute Force Detection",
                    "description": "Alert on brute force attacks",
                    "min_severity": "MEDIUM",
                    "min_score": 0.7,
                    "threat_type": "brute_force"
                }
            ]
        }

    def save(self):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value


# Integration with existing anomaly detection
def integrate_with_api():
    """
    Example integration with anomaly_api.py
    Automatically sends detected anomalies to Chronicle
    """
    from anomaly_api import app, MODEL_STATE
    from fastapi import BackgroundTasks

    # Initialize Chronicle
    config = ChronicleConfig()
    chronicle = ChronicleClient(
        credentials_file=config.get("credentials_file"),
        customer_id=config.get("customer_id"),
        region=config.get("region", "us"),
        log_type=config.get("log_type", "SECURITY_ANOMALY")
    )

    # Background task to send to Chronicle
    def send_to_chronicle(anomalies: List[Dict[str, Any]]):
        """Send anomalies to Chronicle in background."""
        try:
            result = chronicle.send_anomalies(anomalies)
            logger.info(f"Chronicle ingestion: {result}")
        except Exception as e:
            logger.error(f"Chronicle ingestion failed: {e}")

    # Modify analyze endpoint to include Chronicle integration
    original_analyze = app.routes[-1].endpoint  # Get analyze endpoint

    async def analyze_with_chronicle(request, background_tasks: BackgroundTasks):
        """Analyze logs and send to Chronicle."""
        result = await original_analyze(request)

        # Send to Chronicle in background
        if result.get('anomalies'):
            background_tasks.add_task(send_to_chronicle, result['anomalies'])

        return result

    return analyze_with_chronicle


# CLI Tool
def main():
    """CLI for Chronicle integration."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Google Chronicle SIEM Integration"
    )

    parser.add_argument('--setup', action='store_true',
                       help='Set up Chronicle configuration')
    parser.add_argument('--send-anomalies', type=str,
                       help='Send anomalies from JSON file to Chronicle')
    parser.add_argument('--create-rules', action='store_true',
                       help='Create detection rules in Chronicle')
    parser.add_argument('--test', action='store_true',
                       help='Test Chronicle connection')

    args = parser.parse_args()

    if args.setup:
        print("Chronicle SIEM Integration Setup")
        print("=" * 50)

        config = ChronicleConfig()

        # Get configuration
        credentials_file = input(f"Service account JSON path [{config.get('credentials_file')}]: ") or config.get('credentials_file')
        customer_id = input(f"Chronicle customer ID [{config.get('customer_id')}]: ") or config.get('customer_id')
        region = input(f"Region (us/europe/asia) [{config.get('region')}]: ") or config.get('region')

        config.set('credentials_file', credentials_file)
        config.set('customer_id', customer_id)
        config.set('region', region)
        config.save()

        print(f"\n✓ Configuration saved to {config.config_file}")

    elif args.test:
        print("Testing Chronicle connection...")

        config = ChronicleConfig()
        chronicle = ChronicleClient(
            credentials_file=config.get("credentials_file"),
            customer_id=config.get("customer_id"),
            region=config.get("region", "us")
        )

        # Send test event
        test_anomaly = {
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "user": "test_user",
            "source_ip": "192.168.1.100",
            "dest_ip": "unknown",
            "event_type": "login",
            "action": "failed",
            "message": "Test anomaly from integration",
            "severity": "low",
            "anomaly_score": 0.65,
            "threat_type": "brute_force"
        }

        result = chronicle.send_anomalies([test_anomaly])

        if result['status'] == 'success':
            print("✓ Successfully sent test event to Chronicle")
        else:
            print(f"✗ Failed: {result.get('message')}")

    elif args.send_anomalies:
        print(f"Sending anomalies from {args.send_anomalies}...")

        with open(args.send_anomalies, 'r') as f:
            anomalies = json.load(f)

        config = ChronicleConfig()
        chronicle = ChronicleClient(
            credentials_file=config.get("credentials_file"),
            customer_id=config.get("customer_id"),
            region=config.get("region", "us")
        )

        result = chronicle.send_anomalies(anomalies)
        print(f"Result: {result}")

    elif args.create_rules:
        print("Creating detection rules in Chronicle...")

        config = ChronicleConfig()
        chronicle = ChronicleClient(
            credentials_file=config.get("credentials_file"),
            customer_id=config.get("customer_id"),
            region=config.get("region", "us")
        )

        for rule_config in config.get("detection_rules", []):
            result = chronicle.create_detection_rule(rule_config)
            print(f"Rule '{rule_config['name']}': {result['status']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
