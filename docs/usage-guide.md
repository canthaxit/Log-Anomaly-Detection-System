# Log Anomaly Detection System: Usage Guide

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Option 1: Core only (CLI analysis)

```bash
cd Log-Anomaly-Detection-System
pip install -r config/requirements_minimal.txt
```

Dependencies: pandas, numpy, scikit-learn, matplotlib

### Option 2: With REST API

```bash
pip install -r config/requirements_api.txt
```

Adds: FastAPI, Uvicorn, python-multipart

### Option 3: Full stack (with Chronicle SIEM)

```bash
pip install -r config/requirements_chronicle.txt
```

Adds: google-cloud-securitycenter, google-auth

### Windows quick install

```cmd
config\install_dependencies.bat
```

---

## 1. Prepare Your Log Data

The system accepts JSON or CSV logs. Each event needs at minimum:

```json
{
  "timestamp": "2026-01-15T10:30:00Z",
  "user": "alice",
  "source_ip": "192.168.1.10",
  "event_type": "login",
  "action": "success",
  "message": "User logged in successfully"
}
```

### Supported formats

- **JSON array**: `[{"timestamp": ...}, {"timestamp": ...}]`
- **JSONL**: One JSON object per line
- **CSV**: Column headers matching the field names

### Field name flexibility

The parser recognizes 40+ field name aliases automatically:

| Standard Field | Also Accepts |
|---------------|--------------|
| `timestamp` | `time`, `@timestamp`, `datetime`, `date` |
| `user` | `username`, `uid`, `account`, `identity` |
| `source_ip` | `src_ip`, `ip`, `client_ip`, `remote_addr` |
| `event_type` | `type`, `event`, `category`, `action_type` |
| `action` | `result`, `status`, `outcome` |
| `message` | `msg`, `description`, `text`, `log` |
| `severity` | `level`, `priority`, `sev` |

### Optional fields

| Field | Purpose |
|-------|---------|
| `dest_ip` | Destination IP for network event analysis |
| `severity` | Source severity (low/medium/high/critical) |
| `hostname` | Source hostname |
| `process` | Process name |

---

## 2. CLI Analysis (Core Detection)

The simplest way to run -- analyze a directory of log files from the command line.

### Basic usage

```bash
cd core/
python log_anomaly_detection_lite.py --data_path ../tests/
```

This will:
1. Load all JSON/CSV files from `../tests/`
2. Split into baseline (first 7 days) and analysis period
3. Train Isolation Forest + Statistical Detector on the baseline
4. Score all events and flag anomalies
5. Output results to `anomaly_outputs/`

### Command-line options

```bash
python log_anomaly_detection_lite.py \
  --data_path ./logs/               # Directory containing log files (required)
  --output_dir ./results/           # Output directory (default: anomaly_outputs/)
  --baseline_period_days 7          # Baseline window in days (default: 7)
  --contamination 0.01              # Expected anomaly rate 0.0-1.0 (default: 0.01)
  --iso_forest_estimators 200       # Isolation Forest tree count (default: 200)
  --time_windows 3600 86400 604800  # Rolling window sizes in seconds (default: 1h, 24h, 7d)
  --random_state 42                 # Reproducibility seed (default: 42)
  --log_format auto                 # Log format: auto, json, csv (default: auto)
```

### Output files

| File | Contents |
|------|----------|
| `anomalies_detected.csv` | Table of detected anomalies with scores, types, severity |
| `anomalies_detailed.json` | Full JSON report with all event details and metadata |
| `anomaly_analysis.png` | 4-panel visualization: score distribution, timeline, severity breakdown, threat types |
| `feature_pipeline.pkl` | Trained feature extractor (for reuse) |
| `isolation_forest_model.pkl` | Trained Isolation Forest model |
| `statistical_detector.pkl` | Trained statistical rule detector |
| `inference_package.pkl` | Complete deployment package (all models + config) |

### Tuning detection sensitivity

**Too many false positives?** Raise the contamination rate or lower the Isolation Forest weight:
```bash
python log_anomaly_detection_lite.py --data_path ./logs/ --contamination 0.05
```

**Missing real threats?** Lower the contamination rate for a tighter baseline:
```bash
python log_anomaly_detection_lite.py --data_path ./logs/ --contamination 0.005
```

**Short baseline period?** Reduce if you have less than 7 days of normal data:
```bash
python log_anomaly_detection_lite.py --data_path ./logs/ --baseline_period_days 3
```

---

## 3. REST API

Run a FastAPI server for real-time or programmatic analysis.

### Start the server

```bash
cd api/
python anomaly_api.py
```

Server starts on `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Authentication

Set an API key via environment variable:

```bash
export API_KEY="your-secret-key"
python anomaly_api.py
```

Pass it in requests via the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/health
```

### Endpoints

#### Health check (no auth required)

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "models_loaded": true,
  "model_info": {
    "isolation_forest": true,
    "statistical_detector": true,
    "feature_pipeline": true
  }
}
```

#### Load models

Before analyzing, load trained models from disk:

```bash
curl -X POST http://localhost:8000/models/load \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"model_dir": "./anomaly_outputs"}'
```

#### Analyze logs (JSON body)

```bash
curl -X POST http://localhost:8000/analyze \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "logs": [
      {
        "timestamp": "2026-01-15T10:30:00Z",
        "user": "admin",
        "source_ip": "10.0.0.100",
        "event_type": "login",
        "action": "failed",
        "message": "Failed password for admin"
      }
    ],
    "return_all_events": false
  }'
```

Response:

```json
{
  "status": "success",
  "total_events": 1,
  "anomalies_detected": 1,
  "anomaly_rate": 1.0,
  "threshold": 0.7,
  "processing_time_ms": 45.2,
  "anomalies": [
    {
      "timestamp": "2026-01-15T10:30:00Z",
      "user": "admin",
      "source_ip": "10.0.0.100",
      "event_type": "login",
      "action": "failed",
      "message": "Failed password for admin",
      "anomaly_score": 0.87,
      "threat_type": "brute_force",
      "severity": "high"
    }
  ]
}
```

Set `"return_all_events": true` to include non-anomalous events in the response.

#### Analyze logs (file upload)

```bash
curl -X POST http://localhost:8000/analyze/file \
  -H "X-API-Key: your-key" \
  -F "file=@/path/to/logs.json"
```

### Rate limits

| Endpoint | Limit |
|----------|-------|
| `/analyze`, `/analyze/file` | 30 requests/minute |
| `/models/load` | 5 requests/minute |
| `/health`, `/models/info` | No limit |

### Python client example

```python
import requests

API_URL = "http://localhost:8000"
HEADERS = {"X-API-Key": "your-key", "Content-Type": "application/json"}

# Load models
requests.post(f"{API_URL}/models/load",
              json={"model_dir": "./anomaly_outputs"},
              headers=HEADERS)

# Analyze
logs = [
    {"timestamp": "2026-01-15T10:30:00Z", "user": "admin",
     "source_ip": "10.0.0.100", "event_type": "login",
     "action": "failed", "message": "Failed password"},
]
resp = requests.post(f"{API_URL}/analyze",
                     json={"logs": logs},
                     headers=HEADERS)
result = resp.json()
print(f"Anomalies: {result['anomalies_detected']}/{result['total_events']}")
for a in result["anomalies"]:
    print(f"  [{a['severity']}] {a['threat_type']}: {a['user']}@{a['source_ip']}")
```

---

## 4. Batch Processing

Continuously monitor a log directory for new files.

### Run once

```bash
cd batch/
python batch_processor.py --log-dir ../tests/ --output-dir ./batch_outputs/
```

### Run on a schedule

```bash
python batch_processor.py \
  --log-dir /var/log/app/ \
  --output-dir /var/log/anomaly-reports/ \
  --interval 3600        # Check every hour
```

The batch processor:
1. Scans the log directory for new JSON/CSV files
2. Skips files already processed (tracks state)
3. Analyzes each new file through the detection pipeline
4. Generates per-file reports + a batch summary
5. Waits for the next interval

### Output

Each run produces:
- Individual anomaly reports per log file
- `batch_summary.json` with aggregate statistics
- Processing timestamps for deduplication

---

## 5. MCP Server (Claude Desktop Integration)

Use the detection system as a tool inside Claude Desktop or other MCP-compatible AI clients.

### Setup

```bash
# Windows
copy mcp\claude_mcp_config.json %APPDATA%\Claude\claude_desktop_config.json

# macOS
cp mcp/claude_mcp_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Linux
cp mcp/claude_mcp_config.json ~/.config/Claude/claude_desktop_config.json
```

Edit the config to set the correct Python path and project directory.

### Restart Claude Desktop

After configuring, restart Claude Desktop. You'll see new tools available:

| Tool | Description |
|------|-------------|
| `load_anomaly_models` | Load trained models from a directory |
| `analyze_logs` | Analyze JSON/CSV log data |
| `analyze_log_file` | Analyze logs from a file path |
| `get_detection_stats` | Get model information and status |

### Example prompts

- "Load the anomaly models from C:\Users\jimmy\Projects\Log-Anomaly-Detection-System\core\anomaly_outputs"
- "Analyze the logs in C:\Users\jimmy\Projects\Log-Anomaly-Detection-System\tests\test_logs_attack.json"
- "What threats were detected? Summarize the findings."
- "Show me the detection stats"

---

## 6. Google Chronicle Integration

Send detected anomalies to Google Security Operations (Chronicle SIEM).

### Prerequisites

1. Google Cloud project with Chronicle API enabled
2. Service account with `Chronicle API Editor` role
3. Service account JSON key file

### Setup

```bash
cd chronicle/

# Windows (interactive wizard)
setup_chronicle.bat

# Or configure manually
cp chronicle_config_template.json chronicle_config.json
# Edit chronicle_config.json with your project ID, region, and key file path
```

### Configuration

```json
{
  "project_id": "your-gcp-project-id",
  "region": "us",
  "customer_id": "your-chronicle-customer-id",
  "service_account_key": "/path/to/service-account-key.json"
}
```

### API with Chronicle

Use the Chronicle-enabled API variant:

```bash
cd api/
python anomaly_api_chronicle.py
```

Additional endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chronicle/status` | GET | Chronicle connection status |
| `/chronicle/enable` | POST | Enable Chronicle forwarding |
| `/chronicle/test` | POST | Test Chronicle connection |

When enabled, every detected anomaly is automatically converted to UDM (Unified Data Model) format and forwarded to Chronicle.

---

## 7. Docker Deployment

### Quick start

```bash
cd docker/
docker-compose up -d
```

This starts three services:

| Service | Port | Description |
|---------|------|-------------|
| `anomaly-api` | 8000 | REST API server |
| `anomaly-mcp` | stdio | MCP server for AI tools |
| `anomaly-batch` | -- | Hourly batch processor |

### Configuration via environment

```bash
# docker-compose.yml environment variables
API_KEY=your-secret-key
MODEL_DIR=/app/models
LOG_DIR=/app/logs
BATCH_INTERVAL=3600
```

### Volume mounts

Mount your log files and model directory:

```yaml
volumes:
  - ./your-logs:/app/logs:ro      # Log files (read-only)
  - ./your-models:/app/models:ro   # Trained models (read-only)
  - ./outputs:/app/outputs          # Reports (writable)
```

### Build from source

```bash
cd docker/
docker build -t anomaly-detection .
docker run -p 8000:8000 -v ./logs:/app/logs anomaly-detection
```

---

## 8. Training a Model on Your Data

### Step 1: Collect baseline logs

Gather 3-7 days of normal operation logs. These should represent typical activity -- regular users, expected login patterns, standard file access. No known attacks.

Place them in a directory:
```
baseline_logs/
  day1.json
  day2.json
  ...
  day7.json
```

### Step 2: Train

```bash
cd core/
python log_anomaly_detection_lite.py \
  --data_path ../baseline_logs/ \
  --baseline_period_days 7 \
  --output_dir ../trained_model/
```

### Step 3: Verify

Check the output visualization (`anomaly_analysis.png`) to confirm the baseline looks clean. A good baseline should have very few anomalies detected (close to the contamination rate).

### Step 4: Deploy the model

The trained model is in `trained_model/`:
- `inference_package.pkl` -- Complete package for API/batch deployment
- `isolation_forest_model.pkl` -- Isolation Forest model
- `feature_pipeline.pkl` -- Feature extractor
- `statistical_detector.pkl` -- Statistical rules

Load it in the API:
```bash
curl -X POST http://localhost:8000/models/load \
  -H "X-API-Key: your-key" \
  -d '{"model_dir": "./trained_model"}'
```

Or configure the batch processor:
```bash
python batch_processor.py --model-dir ../trained_model/ --log-dir /var/log/app/
```

### Step 5: Retrain periodically

As your environment evolves (new users, new services, changed patterns), retrain the model on fresh baseline data. A monthly retrain cycle is a reasonable starting point.

---

## 9. Understanding Results

### Anomaly score

Each event receives a score from 0.0 (completely normal) to 1.0 (extreme outlier). The score is a weighted combination of:
- **Isolation Forest** (50%) -- How isolated the event is in feature space
- **Statistical Detector** (50%) -- Whether it matches known threat patterns

### Threat types

| Type | What It Means |
|------|---------------|
| `brute_force` | Multiple failed authentication attempts from the same user or IP |
| `privilege_escalation` | Sudo/admin actions from unexpected users or at unusual times |
| `data_exfiltration` | Access to sensitive files outside business hours |
| `lateral_movement` | Unusual network connections or IP diversity per user |
| `general_anomaly` | Statistical outlier that doesn't match a specific threat pattern |

### Severity levels

| Level | Score | Action |
|-------|-------|--------|
| `low` | 0.50 - 0.70 | Log for trend analysis |
| `medium` | 0.70 - 0.85 | Queue for analyst review |
| `high` | 0.85 - 0.95 | Investigate within 4 hours |
| `critical` | 0.95+ | Immediate incident response |

### CSV report columns

| Column | Description |
|--------|-------------|
| `timestamp` | Event time |
| `user` | User account |
| `source_ip` | Origin IP address |
| `event_type` | Event category |
| `action` | Event outcome |
| `anomaly_score` | Combined score (0.0 - 1.0) |
| `threat_type` | Classification |
| `severity` | LOW / MEDIUM / HIGH / CRITICAL |
| `message` | Original log message |

---

## 10. Configuration Reference

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | *(none)* | API authentication key |
| `ANOMALY_HOME` | `.` | Root directory |
| `MODEL_DIR` | `anomaly_outputs` | Trained model directory |
| `LOG_DIR` | `tests` | Log file directory |
| `BATCH_INTERVAL` | `3600` | Batch scan interval (seconds) |
| `ALLOWED_MODEL_DIRS` | *(cwd)* | Comma-separated allowed model paths |
| `ALLOWED_LOG_DIRS` | *(cwd)* | Comma-separated allowed log paths |
| `MODEL_SIGNING_KEY` | *(none)* | HMAC key for model file signing |

### Detection parameters

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Contamination | `--contamination` | 0.01 | Expected anomaly rate (0.0 - 1.0) |
| Baseline days | `--baseline_period_days` | 7 | Normal behavior window |
| IF estimators | `--iso_forest_estimators` | 200 | Isolation Forest tree count |
| Time windows | `--time_windows` | 3600 86400 604800 | Rolling aggregation windows (seconds) |
| Random state | `--random_state` | 42 | Reproducibility seed |

### Severity thresholds (in code)

```python
severity_thresholds = {
    'low': 0.5,
    'medium': 0.7,
    'high': 0.85,
    'critical': 0.95,
}
```

### Statistical detector thresholds (in code)

```python
brute_force_threshold = 10       # Failed logins to flag
sudo_threshold = 5               # Sudo events to flag
exfiltration_hour_start = 22     # Off-hours start (10 PM)
exfiltration_hour_end = 6        # Off-hours end (6 AM)
lateral_movement_threshold = 10  # Unique IPs per user to flag
```

---

## 11. Integration Examples

### Splunk

Forward anomaly JSON output to Splunk via HEC (HTTP Event Collector):

```python
import requests

SPLUNK_HEC_URL = "https://splunk.example.com:8088/services/collector/event"
SPLUNK_TOKEN = "your-hec-token"

for anomaly in result["anomalies"]:
    requests.post(SPLUNK_HEC_URL,
                  headers={"Authorization": f"Splunk {SPLUNK_TOKEN}"},
                  json={"event": anomaly, "sourcetype": "anomaly_detection"})
```

### Elasticsearch

Index anomalies directly:

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
for anomaly in result["anomalies"]:
    es.index(index="anomaly-detections", document=anomaly)
```

### Slack alerts

Send high-severity detections to Slack:

```python
import requests

SLACK_WEBHOOK = "https://hooks.slack.com/services/..."

for anomaly in result["anomalies"]:
    if anomaly["severity"] in ("high", "critical"):
        requests.post(SLACK_WEBHOOK, json={
            "text": f"[{anomaly['severity'].upper()}] {anomaly['threat_type']}: "
                    f"{anomaly['user']}@{anomaly['source_ip']} - {anomaly['message']}"
        })
```

### PagerDuty

Trigger incidents for critical detections:

```python
import requests

PD_URL = "https://events.pagerduty.com/v2/enqueue"
PD_KEY = "your-routing-key"

for anomaly in result["anomalies"]:
    if anomaly["severity"] == "critical":
        requests.post(PD_URL, json={
            "routing_key": PD_KEY,
            "event_action": "trigger",
            "payload": {
                "summary": f"{anomaly['threat_type']}: {anomaly['user']}@{anomaly['source_ip']}",
                "severity": "critical",
                "source": "anomaly-detection",
                "custom_details": anomaly,
            }
        })
```

### Cron job (Linux)

Run batch analysis every hour:

```cron
0 * * * * cd /opt/anomaly-detection && python batch/batch_processor.py --log-dir /var/log/app/ --output-dir /var/log/anomaly-reports/ >> /var/log/anomaly-cron.log 2>&1
```

### Windows Task Scheduler

Use the provided batch script:

```cmd
api\start_api.bat
```

Or create a scheduled task pointing to:
```
python C:\anomaly-detection\batch\batch_processor.py --log-dir C:\logs\ --output-dir C:\reports\
```
