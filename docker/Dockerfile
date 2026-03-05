# Dockerfile for Log Anomaly Detection API
FROM python:3.11-slim

LABEL maintainer="anomaly-detection"
LABEL description="AI-powered log anomaly detection service"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application files
COPY log_anomaly_detection_lite.py .
COPY anomaly_api.py .
COPY common/ common/

# Create directories
RUN mkdir -p anomaly_outputs logs

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /bin/false appuser
RUN chown -R appuser:appuser /app
USER appuser

# Default to binding on all interfaces inside container
ENV API_HOST=0.0.0.0

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run API server
CMD ["python", "anomaly_api.py"]
