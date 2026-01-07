# LLM Inference Tutorial

A production-grade tutorial for building and deploying Large Language Model (LLM) inference services using industry-standard tools and best practices.

## Table of Contents

- [Overview](#overview)
- [What You'll Learn](#what-youll-learn)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Development Setup](#development-setup)
- [Understanding LLM Inference](#understanding-llm-inference)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Monitoring](#monitoring)
- [API Documentation](#api-documentation)
- [Best Practices](#best-practices)

## Overview

This project demonstrates how to build a production-ready LLM inference API service with:

- **FastAPI** for high-performance REST API
- **HuggingFace Transformers** for model inference
- **uv** for fast, reliable Python package management
- **Docker** for containerization
- **GitHub Actions** for CI/CD
- **Pre-commit hooks** for code quality
- **Prometheus** for metrics collection
- **Rate limiting** for API protection

## What You'll Learn

### 1. LLM Inference Fundamentals
- How LLMs generate text
- Token encoding/decoding
- Sampling strategies (temperature, top-p)
- Batch processing
- Model loading and caching

### 2. Production Engineering
- API design for ML services
- Error handling and logging
- Rate limiting and quotas
- Health checks and monitoring
- Resource management

### 3. Modern Python Development
- Type hints and validation (Pydantic)
- Code quality tools (Ruff, MyPy)
- Testing (pytest)
- Package management (uv)
- Environment configuration

### 4. DevOps & Deployment
- Multi-stage Docker builds
- Docker Compose orchestration
- CI/CD pipelines
- Container security
- Monitoring and observability

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         FastAPI Service             │
│  ┌──────────────────────────────┐  │
│  │  Rate Limiter (SlowAPI)      │  │
│  └──────────────┬───────────────┘  │
│                 ▼                   │
│  ┌──────────────────────────────┐  │
│  │  Request Validation          │  │
│  │  (Pydantic Models)           │  │
│  └──────────────┬───────────────┘  │
│                 ▼                   │
│  ┌──────────────────────────────┐  │
│  │  Inference Engine            │  │
│  │  (HuggingFace Transformers)  │  │
│  └──────────────┬───────────────┘  │
│                 ▼                   │
│  ┌──────────────────────────────┐  │
│  │  Response Generation         │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  Prometheus  │ (Metrics)
└──────────────┘
```

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Git
- 4GB+ RAM (for model loading)

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd llm-inference-tutorial

# Copy environment file
cp .env.example .env

# Start the service
docker-compose up --build

# The API will be available at http://localhost:8000
```

### Test the API

```bash
# Check health
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_length": 50,
    "temperature": 0.7,
    "num_return_sequences": 1
  }'

# View API documentation
open http://localhost:8000/docs
```

## Development Setup

### 1. Install uv (Fast Python Package Manager)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### 2. Set Up Project

```bash
# Clone the repository
git clone <your-repo-url>
cd llm-inference-tutorial

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# For development, defaults are fine
```

### 4. Run the Service

```bash
# Development mode (auto-reload)
python -m src.main

# Or with uvicorn directly
uvicorn src.main:app --reload

# The API will be available at http://localhost:8000
```

## Understanding LLM Inference

### What is LLM Inference?

LLM inference is the process of using a trained language model to generate predictions (text) based on input prompts. Unlike training, inference is read-only and focuses on:

1. **Tokenization**: Converting text to token IDs
2. **Forward Pass**: Running tokens through the model
3. **Sampling**: Selecting next tokens based on probabilities
4. **Decoding**: Converting token IDs back to text

### Key Concepts

#### 1. Tokens
```python
# Text is split into tokens (subwords)
"Hello, world!" → ["Hello", ",", " world", "!"]

# Each token has an ID
["Hello", ",", " world", "!"] → [15496, 11, 995, 0]
```

#### 2. Temperature
Controls randomness in generation:
- **Low (0.1-0.5)**: More focused, deterministic
- **Medium (0.6-0.9)**: Balanced creativity
- **High (1.0-2.0)**: More random, creative

```python
# Low temperature example
"The capital of France is" → "Paris"

# High temperature example
"The capital of France is" → "known for its beautiful architecture"
```

#### 3. Top-p (Nucleus Sampling)
Limits token selection to top probability mass:
- **0.9**: Consider tokens in top 90% probability
- **0.95**: More diverse (95% probability mass)

#### 4. Max Length
Maximum number of tokens to generate:
```python
max_length = 50  # Generate up to 50 tokens
```

### Code Walkthrough

#### 1. Configuration ([src/config.py](src/config.py))

```python
class Settings(BaseSettings):
    model_name: str = "gpt2"  # Which model to use
    max_length: int = 100     # Max tokens to generate
    temperature: float = 0.7  # Sampling temperature
    top_p: float = 0.9        # Nucleus sampling
```

**Why this matters:**
- Pydantic validates configuration at startup
- Environment variables override defaults
- Type-safe configuration prevents runtime errors

#### 2. Inference Engine ([src/inference.py](src/inference.py))

```python
class InferenceEngine:
    def load_model(self):
        # Load tokenizer (text ↔ tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model (tokens → probabilities)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Create pipeline (handles everything)
        self.generator = pipeline("text-generation", ...)
```

**What happens during inference:**
```
User prompt: "Once upon a time"
     ↓
Tokenization: [7454, 2402, 257, 640]
     ↓
Model forward pass: Generate probability distribution
     ↓
Sampling: Select next token based on temperature/top_p
     ↓
Repeat until max_length or end token
     ↓
Detokenization: Convert tokens back to text
     ↓
Response: "Once upon a time, there was a brave knight..."
```

#### 3. API Layer ([src/main.py](src/main.py))

```python
@app.post("/generate")
async def generate_text(request: GenerateRequest):
    # 1. Validate request (Pydantic)
    # 2. Apply rate limiting
    # 3. Run inference
    results = engine.generate(
        prompt=request.prompt,
        max_length=request.max_length,
        ...
    )
    # 4. Track metrics
    # 5. Return response
```

**Production features:**
- Request validation prevents invalid inputs
- Rate limiting protects against abuse
- Metrics tracking for monitoring
- Error handling for reliability

## Project Structure

```
llm-inference-tutorial/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Continuous Integration
│       ├── cd.yml              # Continuous Deployment
│       └── pre-commit.yml      # Pre-commit checks
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── inference.py            # LLM inference engine
│   ├── main.py                 # FastAPI application
│   └── models.py               # Pydantic models
├── tests/
│   ├── __init__.py
│   ├── test_api.py             # API endpoint tests
│   ├── test_config.py          # Configuration tests
│   └── test_models.py          # Model validation tests
├── .env.example                # Environment template
├── .gitignore
├── .pre-commit-config.yaml     # Pre-commit hooks
├── docker-compose.yml          # Multi-container setup
├── Dockerfile                  # Container definition
├── prometheus.yml              # Metrics configuration
├── pyproject.toml              # Project & tool config
└── README.md
```

## Testing

### Run All Tests

```bash
# With coverage report
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Run Specific Tests

```bash
# Test API endpoints
pytest tests/test_api.py

# Test configuration
pytest tests/test_config.py

# Test models
pytest tests/test_models.py -v
```

### Pre-commit Hooks

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
pre-commit run mypy --all-files
```

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

The [Dockerfile](Dockerfile) uses multi-stage builds:

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

## Production Deployment

### Kubernetes Example

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

### Environment Variables for Production

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

## API Documentation

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example Requests

#### Generate Text

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_length": 100,
    "temperature": 0.8,
    "top_p": 0.95,
    "num_return_sequences": 2
  }'
```

Response:
```json
{
  "generated": [
    {
      "text": "The future of AI is bright and full of possibilities...",
      "tokens": 45
    },
    {
      "text": "The future of AI is uncertain but exciting...",
      "tokens": 42
    }
  ],
  "model": "gpt2",
  "prompt_tokens": 5
}
```

#### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "0.1.0"
}
```

## Best Practices

### 1. Model Selection

- **Development**: `gpt2` (small, fast)
- **Production**: `gpt2-medium`, `gpt2-large` (better quality)
- **Custom**: Fine-tuned models from HuggingFace Hub

### 2. Resource Management

```python
# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Enable mixed precision for GPU
torch_dtype = torch.float16 if device == "cuda" else torch.float32
```

### 3. Caching Models

```python
# Cache models to avoid re-downloading
cache_dir = "./models"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir
)
```

### 4. Rate Limiting

```python
# Protect your API from abuse
@limiter.limit("10/minute")
async def generate_text(...):
    ...
```

### 5. Error Handling

```python
try:
    results = engine.generate(...)
except Exception as e:
    logger.error(f"Generation failed: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

### 6. Testing

```python
# Always validate with tests
def test_generate_endpoint():
    response = client.post("/generate", json={"prompt": "test"})
    assert response.status_code == 200
```

### 7. Security

- Never commit `.env` files
- Use non-root user in Docker
- Scan dependencies (`safety check`)
- Enable HTTPS in production
- Implement authentication/authorization

### 8. Performance Optimization

```python
# Batch requests when possible
results = engine.generate_batch(prompts)

# Set appropriate timeouts
timeout_seconds = 30

# Monitor memory usage
import psutil
memory_percent = psutil.virtual_memory().percent
```

## CI/CD Pipeline

### GitHub Actions Workflows

1. **CI Pipeline** ([.github/workflows/ci.yml](.github/workflows/ci.yml))
   - Linting (Ruff)
   - Type checking (MyPy)
   - Tests (pytest)
   - Security scanning (Bandit)
   - Docker build

2. **CD Pipeline** ([.github/workflows/cd.yml](.github/workflows/cd.yml))
   - Build and push Docker images
   - Deploy on version tags

3. **Pre-commit** ([.github/workflows/pre-commit.yml](.github/workflows/pre-commit.yml))
   - Run pre-commit hooks on PRs

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

## Next Steps

1. **Try Different Models**
   - `distilgpt2` (faster, smaller)
   - `gpt2-medium` (better quality)
   - `EleutherAI/gpt-neo-125M` (different architecture)

2. **Add Features**
   - Authentication (JWT tokens)
   - Request queuing (Celery + Redis)
   - Response streaming
   - Multi-model support

3. **Production Enhancements**
   - Kubernetes deployment
   - Load balancing
   - Auto-scaling
   - Distributed tracing

4. **Advanced Topics**
   - Model quantization (smaller, faster)
   - GPU optimization (TensorRT)
   - Custom fine-tuning
   - A/B testing

## Resources

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [uv Package Manager](https://github.com/astral-sh/uv)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Prometheus Monitoring](https://prometheus.io/docs/)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

---

**Happy Learning!** If you found this tutorial helpful, please star the repository and share it with others.
