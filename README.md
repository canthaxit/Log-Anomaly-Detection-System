# 🛡️ Log Anomaly Detection System

AI-powered security threat detection for system logs using unsupervised machine learning.

## Features

- **Unsupervised Learning** - No labeled training data required
- **Multi-Model Ensemble** - Isolation Forest + Statistical rules for high accuracy
- **Real-Time & Batch Processing** - Analyze logs instantly or on schedule
- **Threat Classification** - Brute force, privilege escalation, data exfiltration, lateral movement
- **Enterprise Integrations** - Google Chronicle, Splunk, Elasticsearch, Slack, and more

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/log-anomaly-detection.git
cd log-anomaly-detection

# Install dependencies
pip install -r config/requirements_minimal.txt
```

### Analyze Logs

```bash
cd core/
python log_anomaly_detection_lite.py --data_path ../tests/
```

Results are saved to `core/anomaly_outputs/`:
- `anomalies_detected.csv` - Detected threats
- `anomalies_detailed.json` - Full report
- `anomaly_analysis.png` - Visualization

### Start REST API

```bash
cd api/
python anomaly_api.py
# Access: http://localhost:8000/docs
```

## Project Structure

```
log-anomaly-detection/
├── core/           # Core detection engine
├── api/            # REST API service
├── chronicle/      # Google Chronicle SIEM integration
├── mcp/            # Claude Desktop / AI platform integration
├── batch/          # Scheduled batch processing
├── docker/         # Container deployment
├── tests/          # Test data and scripts
├── examples/       # Integration code samples
├── config/         # Configuration and requirements
└── docs/           # Documentation
```

## Integrations

| Platform | Status | Directory |
|----------|--------|-----------|
| REST API | ✅ Ready | `api/` |
| Google Chronicle | ✅ Ready | `chronicle/` |
| Claude Desktop (MCP) | ✅ Ready | `mcp/` |
| Splunk | ✅ Ready | `examples/` |
| Elasticsearch | ✅ Ready | `examples/` |
| Slack | ✅ Ready | `examples/` |
| Docker | ✅ Ready | `docker/` |

## Usage Examples

### Python

```python
from log_anomaly_detection_lite import LogParser
import joblib

# Load trained models
pipeline = joblib.load("core/anomaly_outputs/feature_pipeline.pkl")
model = joblib.load("core/anomaly_outputs/isolation_forest_model.pkl")

# Analyze logs
parser = LogParser()
df = parser.load_logs("your_logs.json")
features = pipeline.transform(df)
scores = -model.score_samples(features)

# Get anomalies (threshold: 0.7)
anomalies = df[scores > 0.7]
```

### REST API

```bash
# Load models first
curl -X POST http://localhost:8000/models/load

# Analyze logs
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "logs": [{
      "timestamp": "2026-01-15T10:00:00Z",
      "user": "admin",
      "source_ip": "10.0.0.100",
      "event_type": "login",
      "action": "failed",
      "message": "Failed login attempt"
    }]
  }'
```

### Google Chronicle

```python
from google_chronicle_integration import ChronicleClient

chronicle = ChronicleClient(
    credentials_file="chronicle_credentials.json",
    customer_id="C00000000"
)

chronicle.send_anomalies(anomalies)
```

## Threat Detection

| Threat Type | Detection Method |
|-------------|------------------|
| **Brute Force** | Failed login velocity, IP patterns |
| **Privilege Escalation** | Unusual sudo/admin actions |
| **Data Exfiltration** | Off-hours access, sensitive files |
| **Lateral Movement** | Network connection patterns |

## Log Format

The system accepts JSON logs with these fields:

```json
{
  "timestamp": "2026-01-15T10:00:00Z",
  "user": "username",
  "source_ip": "192.168.1.100",
  "dest_ip": "10.0.0.1",
  "event_type": "login|logout|file_access|sudo|network",
  "action": "success|failed",
  "message": "Description of the event"
}
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Quick Start](docs/QUICK_START.md) | Getting started guide |
| [API Docs](api/README.md) | REST API reference |
| [Chronicle Setup](docs/CHRONICLE_QUICK_START.md) | Google SIEM integration |
| [Integration Examples](examples/README.md) | Code samples |
| [Deployment Guide](docs/SCALING_GUIDE.md) | Production deployment |

## Requirements

**Core:**
- Python 3.10+
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0

**Optional:**
- FastAPI, uvicorn (REST API)
- google-auth (Chronicle integration)
- mcp (Claude Desktop integration)

Install all:
```bash
pip install -r config/requirements_chronicle.txt
```

## Docker Deployment

```bash
cd docker/
docker-compose up -d

# Services:
# - API: http://localhost:8000
# - Batch processor: hourly schedule
```

## Testing

```bash
# Run with test data
cd core/
python log_anomaly_detection_lite.py \
  --data_path ../tests/ \
  --baseline_period_days 1 \
  --contamination 0.50

# Expected: Detects ~9 anomalies (brute force + data exfiltration)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Detect threats before they become incidents.** 🛡️

## Production Deployment

### TLS / HTTPS

The API server binds to `127.0.0.1:8000` with plain HTTP by default. For production, use a reverse proxy for TLS termination.

#### nginx

```nginx
server {
    listen 443 ssl;
    server_name anomaly-api.example.com;

    ssl_certificate     /etc/ssl/certs/anomaly-api.crt;
    ssl_certificate_key /etc/ssl/private/anomaly-api.key;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Caddy

```
anomaly-api.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

Caddy handles TLS certificates automatically via Let's Encrypt.

### Authentication

API authentication is **required by default**. Set the `API_KEY` environment variable before starting:

```bash
export API_KEY="your-secure-api-key"
python api/anomaly_api.py
```

To explicitly disable auth (development only):

```bash
export REQUIRE_AUTH=false
python api/anomaly_api.py
```

### Docker

The `docker-compose.yml` binds the API to port 8000. Place a reverse proxy in front for TLS. See `docker/docker-compose.yml` for the full configuration.
