# Log Anomaly Detection System: Executive Summary

## What It Is

The Log Anomaly Detection System is an AI-powered security threat detection platform that analyzes system and application logs to identify anomalous behavior. It uses unsupervised machine learning -- no labeled training data required. The system learns what "normal" looks like from a baseline period, then flags deviations as potential threats.

## The Problem

Security teams drown in logs. A mid-size organization generates millions of log events daily across servers, firewalls, applications, and cloud services. Manual review is impossible. Traditional rule-based SIEM alerts produce high false-positive rates and miss novel attack patterns that don't match pre-defined signatures.

Attackers exploit this: brute force attempts hidden in noise, slow privilege escalation over days, off-hours data exfiltration, and lateral movement that looks like normal network traffic.

## How It Solves It

The system uses a multi-model ensemble approach that combines machine learning with domain-specific threat rules:

| Component | Technique | What It Catches |
|-----------|-----------|-----------------|
| Isolation Forest | Unsupervised anomaly detection | General outliers across 25-30 features |
| Statistical Detector | Threat-specific rules | Brute force, privilege escalation, exfiltration, lateral movement |
| Feature Pipeline | Temporal + behavioral extraction | Time patterns, user baselines, IP characteristics |
| Anomaly Scorer | Weighted ensemble (50/50) | Combined score with calibrated thresholds |

The pipeline requires no labeled data. It splits logs into a baseline period (typically 7 days of normal operations) and an analysis period, then identifies events that deviate from learned baselines.

## Threat Detection Capabilities

| Threat Type | Detection Method | Example |
|-------------|-----------------|---------|
| Brute Force | Failed login velocity + cumulative count | 12 failed logins on "admin" from one IP in 60 seconds |
| Privilege Escalation | Sudo patterns + baseline comparison | Unknown user executing sudo commands |
| Data Exfiltration | Off-hours access + sensitive file patterns | Root reading /etc/shadow at 2:00 AM |
| Lateral Movement | Network frequency + unique IP deviation | Single user connecting to 15 unique IPs in an hour |

## Severity Classification

Events are scored 0.0 to 1.0 and mapped to severity:

| Severity | Score Range | Meaning |
|----------|------------|---------|
| Low | 0.50 - 0.70 | Unusual but possibly benign |
| Medium | 0.70 - 0.85 | Suspicious, warrants investigation |
| High | 0.85 - 0.95 | Likely threat, prioritize response |
| Critical | 0.95+ | Active attack, immediate action required |

## Deployment Options

| Mode | Description | Use Case |
|------|-------------|----------|
| CLI Script | `python log_anomaly_detection_lite.py --data_path ./logs/` | Ad-hoc analysis, security audits |
| REST API | FastAPI server on port 8000 with auth + rate limiting | SIEM integration, dashboards, automation |
| Batch Processor | Scheduled scan of log directories | 24/7 monitoring, hourly/daily analysis |
| MCP Server | Claude Desktop / AI tool integration | Analyst-in-the-loop investigation |
| Docker | Multi-service composition (API + MCP + Batch) | Production deployment |

## Architecture

```
Log Sources (JSON/CSV)
        |
        v
  [Log Parser]
  (40+ field aliases, auto-format detection)
        |
        v
  [Feature Pipeline]
  (25-30 features: temporal, behavioral, entity, rolling windows)
        |
        v
  +-----+-----+
  |           |
  v           v
[Isolation  [Statistical
 Forest]     Detector]
  |           |
  +-----+-----+
        |
        v
  [Anomaly Scorer]
  (weighted ensemble, threshold calibration)
        |
        v
  [Threat Classification]
  (type assignment, severity mapping)
        |
  +-----+-----+-----+
  |     |     |     |
  v     v     v     v
 CSV   JSON  PNG  SIEM
Report Report Viz  Alert
```

## Performance

| Metric | Value |
|--------|-------|
| Single event inference | ~1ms |
| Batch of 10,000 events | 2-5 seconds |
| Model training (baseline) | 30-60 seconds |
| API latency (p50) | 200-500ms |
| API throughput (single worker) | 10-30 req/s |
| Memory per 10K events | 200-300MB |

## Enterprise Integrations

- **Google Chronicle** -- Converts anomalies to UDM format, sends to Google Security Operations
- **Splunk / Elasticsearch** -- JSON output compatible with standard SIEM ingest
- **Slack / PagerDuty** -- Webhook alerts for critical detections
- **Claude Desktop** -- AI-assisted investigation via MCP protocol
- **REST API** -- Universal integration point for any platform

## Security Hardening

- API key authentication with HMAC validation
- Rate limiting (30 req/min analysis, 5 req/min model loading)
- Input validation (10K event limit, 2KB field limit, 10MB upload limit)
- Path traversal prevention via directory allowlists
- HMAC-SHA256 model file signing (prevents tampered model loading)
- Docker: read-only filesystem, no-new-privileges, memory limits, CPU quotas
- CORS restricted to whitelisted origins
- Security headers (X-Content-Type-Options, X-Frame-Options, CSP)

## Technology

- Python 3.10+, MIT License
- Core dependencies: pandas, numpy, scikit-learn, matplotlib
- API: FastAPI + Uvicorn
- No TensorFlow/PyTorch required (lite version)
- 58+ pytest tests with GitHub Actions CI
- 5,000 lines across 18 Python modules
