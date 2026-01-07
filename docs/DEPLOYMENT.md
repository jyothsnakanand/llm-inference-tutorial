# Deployment Guide

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -t llm-inference:latest .

# Run container
docker run -p 8000:8000 \
  -e MODEL_NAME=gpt2 \
  -e LOG_LEVEL=INFO \
  llm-inference:latest

# Or use Docker Compose
docker-compose up --build
```

### Multi-Stage Build Benefits

The [Dockerfile](../Dockerfile) uses multi-stage builds:

1. **Builder Stage**: Install dependencies
2. **Runtime Stage**: Copy only needed files

**Benefits:**
- Smaller final image (~1GB vs 3GB+)
- Faster deployments
- Better security (fewer attack surfaces)

### With Monitoring

```bash
# Start with Prometheus and Grafana
docker-compose --profile monitoring up

# Access services:
# - API: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

## Kubernetes Deployment

### Example Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: llm-inference
        image: ghcr.io/your-org/llm-inference:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: MODEL_NAME
          value: "gpt2"
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
```

## Production Configuration

### Environment Variables

```bash
# Production settings
ENVIRONMENT=production
LOG_LEVEL=WARNING

# Performance tuning
WORKERS=4
MAX_BATCH_SIZE=8
TIMEOUT_SECONDS=60

# Larger models
MODEL_NAME=gpt2-large
MODEL_CACHE_DIR=/mnt/models

# Stricter rate limiting
RATE_LIMIT_REQUESTS=5
RATE_LIMIT_PERIOD=60
```

## Monitoring

### Metrics Endpoints

- **Health Check**: `GET /health`
- **Prometheus Metrics**: `GET /metrics`

### Available Metrics

```python
# Request tracking
inference_requests_total{endpoint="/generate",status="success"}

# Performance monitoring
inference_request_duration_seconds{endpoint="/generate"}

# Token counting
generated_tokens_total
```

### Prometheus Queries

```promql
# Request rate
rate(inference_requests_total[5m])

# Error rate
rate(inference_requests_total{status="error"}[5m])

# Average latency
rate(inference_request_duration_seconds_sum[5m]) /
rate(inference_request_duration_seconds_count[5m])

# Tokens per second
rate(generated_tokens_total[1m])
```

## Troubleshooting

### Model Loading Issues

```bash
# Clear model cache
rm -rf models/

# Download model manually
python -c "from transformers import AutoModel; AutoModel.from_pretrained('gpt2')"
```

### Memory Issues

```bash
# Use smaller model
MODEL_NAME=distilgpt2

# Reduce batch size
MAX_BATCH_SIZE=1

# Disable GPU (uses less memory)
CUDA_VISIBLE_DEVICES=-1
```

### Rate Limiting

```bash
# Increase limits for testing
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
```
