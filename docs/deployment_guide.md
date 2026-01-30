# Payment Fraud Detection - Deployment Guide

This guide covers deploying the fraud detection system to production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Deployment Options](#deployment-options)
4. [Configuration](#configuration)
5. [Monitoring](#monitoring)
6. [Maintenance](#maintenance)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 20 GB
- OS: Ubuntu 20.04+ or similar Linux distribution

**Recommended for Production:**
- CPU: 8+ cores
- RAM: 16 GB
- Storage: 50 GB SSD
- OS: Ubuntu 22.04 LTS

### Software Dependencies

```bash
# Python
Python 3.8 or higher

# Database (optional for transaction storage)
PostgreSQL 13+ or MySQL 8+

# Monitoring (optional)
Prometheus + Grafana
```

---

## Environment Setup

### 1. Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3.10 python3.10-venv python3-pip git nginx

# Create application user (security best practice)
sudo useradd -m -s /bin/bash fraudapp
sudo su - fraudapp
```

### 2. Application Installation

```bash
# Clone repository
git clone https://github.com/yourusername/payment-fraud-detection.git
cd payment-fraud-detection

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn  # For production server
```

### 3. Configuration

```bash
# Copy example configuration
cp config.json config.prod.json

# Edit configuration for production
nano config.prod.json
```

**Key settings to adjust:**
```json
{
  "data": {
    "input_path": "/data/transactions/live",
    "output_path": "/var/log/fraud-detection"
  },
  "production": {
    "model_selection_strategy": "hybrid",
    "monitoring": {
      "enable_shap": true,
      "enable_drift_detection": true
    }
  }
}
```

---

## Deployment Options

### Option 1: Batch Processing (Recommended for Start)

Process transactions in batches (e.g., every hour).

**Setup:**

```bash
# Create batch processing script
cat > run_batch.sh << 'EOF'
#!/bin/bash
source /home/fraudapp/payment-fraud-detection/venv/bin/activate
python src/main.py \
    --data /data/transactions/batch_$(date +%Y%m%d_%H%M%S).xlsx \
    --output /var/log/fraud-detection/ \
    --config config.prod.json
EOF

chmod +x run_batch.sh

# Add to crontab (run every hour)
crontab -e
# Add line:
# 0 * * * * /home/fraudapp/payment-fraud-detection/run_batch.sh >> /var/log/fraud-detection/batch.log 2>&1
```

### Option 2: Real-Time API (FastAPI)

For sub-100ms response times.

**Create API server** (`src/api.py`):

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.models import FraudDetectionPipeline
from src.feature_engineering import FeatureEngineer

app = FastAPI(title="Payment Fraud Detection API")

# Load model at startup
pipeline = FraudDetectionPipeline.load('outputs/models/fraud_detection_pipeline.pkl')
engineer = FeatureEngineer()  # Load from saved state

class Transaction(BaseModel):
    transaction_id: str
    transaction_amount: float
    user_id: str
    merchant_id: str
    device_id: str = None
    transaction_date: str

@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    """Predict fraud probability for a transaction."""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([transaction.dict()])
        
        # Engineer features
        df_fe = engineer.transform(df)
        
        # Predict
        probability = pipeline.predict_proba(df_fe)[0]
        prediction = int(probability >= 0.5)
        
        return {
            "transaction_id": transaction.transaction_id,
            "fraud_probability": float(probability),
            "prediction": "fraud" if prediction else "legitimate",
            "model": pipeline.best_model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
```

**Run with Gunicorn:**

```bash
# Install FastAPI and Gunicorn
pip install fastapi uvicorn gunicorn

# Run server
gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

**Nginx reverse proxy** (`/etc/nginx/sites-available/fraud-api`):

```nginx
server {
    listen 80;
    server_name fraud-api.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Option 3: Docker Container

**Dockerfile:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY outputs/models/ ./outputs/models/
COPY config.json .

# Expose port
EXPOSE 8000

# Run API server
CMD ["gunicorn", "src.api:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
```

**Build and run:**

```bash
# Build image
docker build -t fraud-detection:1.0 .

# Run container
docker run -d \
    --name fraud-api \
    -p 8000:8000 \
    -v /data/models:/app/outputs/models \
    fraud-detection:1.0
```

---

## Configuration

### Environment Variables

Create `.env` file:

```bash
# Application
FRAUD_ENV=production
LOG_LEVEL=INFO

# Database (if using)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fraud_detection
DB_USER=fraudapp
DB_PASS=secure_password

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Model
MODEL_PATH=/app/outputs/models/fraud_detection_pipeline.pkl
THRESHOLD_MAIN=0.5
THRESHOLD_COLD=0.35
```

### Logging Configuration

```python
# src/config/logging.py
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/fraud-detection/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'standard',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
    },
    'loggers': {
        '': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True
        }
    }
}
```

---

## Monitoring

### Metrics to Track

1. **Performance Metrics:**
   - Inference latency (p50, p95, p99)
   - Throughput (transactions/second)
   - Model load time

2. **Business Metrics:**
   - Fraud detection rate (recall)
   - False positive rate
   - Precision
   - Cost savings

3. **Data Quality:**
   - Missing value rate
   - Feature drift detection
   - Distribution shifts

### Prometheus Integration

```python
# src/monitoring.py
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
predictions_total = Counter('fraud_predictions_total', 'Total predictions made')
fraud_detected = Counter('fraud_detected_total', 'Total frauds detected')
latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@latency.time()
def predict_with_monitoring(transaction):
    predictions_total.inc()
    result = pipeline.predict_proba(transaction)
    if result > 0.5:
        fraud_detected.inc()
    return result

# Start metrics server
start_http_server(8001)
```

### Grafana Dashboard

Key visualizations:
- Fraud detection rate over time
- Model performance metrics
- System resource usage
- Alert thresholds

---

## Maintenance

### Model Retraining

**Automated retraining schedule:**

```bash
# Retrain monthly
# /etc/cron.d/fraud-model-retrain
0 2 1 * * fraudapp /home/fraudapp/payment-fraud-detection/retrain.sh
```

**Retraining script** (`retrain.sh`):

```bash
#!/bin/bash
set -e

echo "Starting model retraining..."

# Activate environment
source /home/fraudapp/payment-fraud-detection/venv/bin/activate

# Fetch latest data (last 6 months)
python scripts/fetch_training_data.py --months 6

# Train new model
python src/main.py \
    --data /data/training/recent.xlsx \
    --output /tmp/retrain/ \
    --optimize

# Validate new model
python scripts/validate_model.py \
    --old-model outputs/models/fraud_detection_pipeline.pkl \
    --new-model /tmp/retrain/models/fraud_detection_pipeline.pkl

# If validation passes, deploy new model
if [ $? -eq 0 ]; then
    cp /tmp/retrain/models/fraud_detection_pipeline.pkl \
       outputs/models/fraud_detection_pipeline.pkl
    echo "Model updated successfully"
    
    # Restart API server
    sudo systemctl restart fraud-api
else
    echo "Model validation failed, keeping old model"
    exit 1
fi
```

### Backup Strategy

```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d)

# Backup models
tar -czf /backups/models_${DATE}.tar.gz outputs/models/

# Backup logs
tar -czf /backups/logs_${DATE}.tar.gz /var/log/fraud-detection/

# Keep only last 30 days
find /backups -name "*.tar.gz" -mtime +30 -delete
```

---

## Troubleshooting

### Common Issues

**1. High Latency**

```bash
# Check system resources
htop

# Check model size
ls -lh outputs/models/

# Solution: Use simpler model or scale horizontally
```

**2. Model Performance Degradation**

```bash
# Check feature drift
python scripts/check_drift.py

# Solution: Retrain model with recent data
```

**3. Memory Issues**

```bash
# Monitor memory usage
free -h

# Solution: Reduce batch size or add more RAM
```

### Logging Levels

```bash
# Debug mode (verbose logging)
export LOG_LEVEL=DEBUG

# Production mode (only warnings/errors)
export LOG_LEVEL=WARNING
```

### Performance Tuning

```bash
# Gunicorn workers (rule of thumb: 2-4 × CPU cores)
gunicorn -w 16 -k uvicorn.workers.UvicornWorker ...

# Optimize inference
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2
```

---

## Security Considerations

1. **API Security:**
   - Use HTTPS (Let's Encrypt certificates)
   - Implement rate limiting
   - Add API key authentication
   - Enable CORS policies

2. **Data Security:**
   - Encrypt data at rest
   - Use secure database connections
   - Implement audit logging
   - Follow GDPR/PCI-DSS compliance

3. **Access Control:**
   - Restrict SSH access
   - Use firewall rules
   - Regular security updates
   - Monitor suspicious activity

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/payment-fraud-detection/issues
- Email: jfumagalli.work@gmail.com
- Documentation: `/docs` directory

---

*Last updated: January 2026*
