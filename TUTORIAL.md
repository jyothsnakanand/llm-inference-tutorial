# LLM Inference Tutorial - Step by Step Guide

## Introduction

This tutorial will teach you how LLM (Large Language Model) inference works through hands-on practice with a production-grade service. We'll cover everything from basic concepts to deployment.

## Part 1: Understanding LLM Inference (Theory)

### What is Inference?

**Inference** is using a trained model to make predictions. For LLMs:
- **Input**: Text prompt ("Once upon a time")
- **Output**: Generated continuation ("there was a brave knight...")

### The Inference Process

```
1. Tokenization
   "Hello world" → [15496, 995]
   ↓

2. Encoding
   Convert tokens to numerical vectors
   ↓

3. Model Processing
   Neural network processes vectors
   Outputs probability distribution for next token
   ↓

4. Sampling
   Select next token based on:
   - Temperature (randomness)
   - Top-p (probability threshold)
   ↓

5. Decoding
   [15496, 995, 0] → "Hello world!"
   ↓

6. Repeat until:
   - Reach max_length
   - Generate <end> token
```

### Key Parameters

#### Temperature (0.0 - 2.0)
Controls randomness:

```python
# Temperature = 0.1 (deterministic)
"The sky is" → "blue" (99% probability)

# Temperature = 0.7 (balanced)
"The sky is" → "blue" (60%), "clear" (20%), "beautiful" (10%)

# Temperature = 1.5 (creative)
"The sky is" → "purple" (15%), "infinite" (12%), "a canvas" (10%)
```

#### Top-p / Nucleus Sampling (0.0 - 1.0)
Limits token selection:

```python
# Top-p = 0.9
# Only consider tokens that make up 90% of probability mass
Tokens: [("blue", 0.6), ("clear", 0.2), ("beautiful", 0.15), ...]
Selected: ["blue", "clear", "beautiful"]  # Sum ≈ 0.9

# Top-p = 0.5
Selected: ["blue"]  # Already > 0.5
```

#### Max Length
Maximum tokens to generate:

```python
max_length = 50
# Generates at most 50 tokens (including prompt)
```

## Part 2: Quick Start (Practice)

### Step 1: Set Up Environment

```bash
# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up project
git clone <repo-url>
cd llm-inference-tutorial

# Use Makefile for easy setup
make setup

# Or manual setup
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Step 2: Start the Service

**Option A: Using Docker (Easiest)**
```bash
docker-compose up --build
```

**Option B: Local Development**
```bash
# Activate environment
source .venv/bin/activate

# Run service
make dev
# or
python -m src.main
```

### Step 3: Test the API

```bash
# Test health
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Artificial intelligence will",
    "max_length": 100,
    "temperature": 0.7,
    "num_return_sequences": 1
  }'

# Or use the test script
./scripts/test_api.sh
```

### Step 4: Explore the API

Open http://localhost:8000/docs for interactive documentation.

## Part 3: Code Deep Dive

### Architecture Overview

```
User Request
    ↓
FastAPI Endpoint (/generate)
    ↓
Request Validation (Pydantic)
    ↓
Rate Limiting (SlowAPI)
    ↓
Inference Engine
    ├── Tokenizer (text → tokens)
    ├── Model (tokens → probabilities)
    └── Generator (probabilities → text)
    ↓
Response with Metrics
```

### File-by-File Explanation

#### 1. `src/config.py` - Configuration Management

```python
class Settings(BaseSettings):
    # Loads from .env file
    model_name: str = "gpt2"
    temperature: float = 0.7

# Why this matters:
# - Type-safe configuration
# - Environment variable support
# - Validation at startup
```

**Try it:**
```bash
# Edit .env
MODEL_NAME=distilgpt2
TEMPERATURE=0.9

# Restart service - new config loaded!
```

#### 2. `src/models.py` - Request/Response Models

```python
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int | None = None
    temperature: float | None = None

# Why this matters:
# - Automatic validation
# - API documentation
# - Type hints for IDE
```

**Try it:**
```bash
# Invalid request (empty prompt)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": ""}'

# Returns: 422 Validation Error
```

#### 3. `src/inference.py` - The Brain

```python
class InferenceEngine:
    def load_model(self):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")

        # Create pipeline
        self.generator = pipeline("text-generation", ...)

    def generate(self, prompt, ...):
        # 1. Tokenize input
        prompt_tokens = self.tokenizer.encode(prompt)

        # 2. Generate
        outputs = self.generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            ...
        )

        # 3. Return results
        return outputs
```

**What's happening:**
1. Model loaded once at startup (expensive)
2. Cached for all requests (fast)
3. Uses GPU if available (faster)
4. Handles tokenization automatically

#### 4. `src/main.py` - API Application

```python
@app.post("/generate")
@limiter.limit("10/minute")  # Rate limiting
async def generate_text(request: GenerateRequest):
    # Validate
    if not engine.is_loaded():
        raise HTTPException(503, "Model not ready")

    # Generate
    results = engine.generate(...)

    # Track metrics
    GENERATION_TOKENS.inc(total_tokens)

    return results
```

**Production features:**
- Rate limiting (prevents abuse)
- Health checks (for load balancers)
- Metrics (for monitoring)
- Error handling (for reliability)

## Part 4: Experiments

### Experiment 1: Temperature Effects

```bash
# Low temperature (deterministic)
curl -X POST http://localhost:8000/generate \
  -d '{"prompt": "The capital of France is", "temperature": 0.1}'
# Expected: "Paris"

# High temperature (creative)
curl -X POST http://localhost:8000/generate \
  -d '{"prompt": "The capital of France is", "temperature": 1.5}'
# Expected: More varied/creative responses
```

### Experiment 2: Multiple Sequences

```bash
# Generate 3 different completions
curl -X POST http://localhost:8000/generate \
  -d '{
    "prompt": "Once upon a time",
    "num_return_sequences": 3,
    "temperature": 0.8
  }'
```

### Experiment 3: Different Models

```bash
# Edit .env
MODEL_NAME=distilgpt2  # Smaller, faster
# MODEL_NAME=gpt2-medium  # Larger, better quality

# Restart and compare results
```

## Part 5: Testing & Quality

### Run Tests

```bash
# All tests with coverage
make test

# Fast tests
make test-fast

# Specific test file
pytest tests/test_api.py -v
```

### Code Quality

```bash
# Linting
make lint

# Formatting
make format

# Type checking
make type-check

# Security scanning
make security

# All checks
make all
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files

# Now runs automatically on git commit!
```

## Part 6: Docker & Deployment

### Understanding the Dockerfile

```dockerfile
# Stage 1: Builder (install dependencies)
FROM python:3.11-slim as builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv pip install -e .

# Stage 2: Runtime (small, secure)
FROM python:3.11-slim
COPY --from=builder /opt/venv /opt/venv
USER appuser  # Non-root for security
CMD ["uvicorn", "src.main:app"]
```

**Why multi-stage?**
- Builder: ~2GB (has build tools)
- Runtime: ~800MB (only what's needed)
- 60% size reduction!

### Docker Commands

```bash
# Build
docker build -t llm-inference .

# Run
docker run -p 8000:8000 llm-inference

# Or use Docker Compose
docker-compose up --build

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up
```

### Production Configuration

```yaml
# docker-compose.yml
services:
  llm-inference:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
```

## Part 7: CI/CD Pipeline

### GitHub Actions Workflows

**1. CI Pipeline** (`.github/workflows/ci.yml`)
- Runs on every push/PR
- Linting (Ruff)
- Type checking (MyPy)
- Tests with coverage
- Security scans
- Docker build

**2. CD Pipeline** (`.github/workflows/cd.yml`)
- Runs on version tags (v1.0.0)
- Builds Docker image
- Pushes to GitHub Container Registry
- Deploys to production

**3. Pre-commit** (`.github/workflows/pre-commit.yml`)
- Runs pre-commit hooks
- Ensures code quality

### Triggering Workflows

```bash
# Push triggers CI
git add .
git commit -m "feat: add new feature"
git push

# Tag triggers CD
git tag v0.1.0
git push origin v0.1.0
```

## Part 8: Monitoring

### Metrics

```bash
# View raw metrics
curl http://localhost:8000/metrics

# Key metrics:
# - inference_requests_total
# - inference_request_duration_seconds
# - generated_tokens_total
```

### Prometheus Queries

```promql
# Request rate (requests per second)
rate(inference_requests_total[5m])

# Error rate
sum(rate(inference_requests_total{status="error"}[5m])) /
sum(rate(inference_requests_total[5m]))

# Average latency
rate(inference_request_duration_seconds_sum[5m]) /
rate(inference_request_duration_seconds_count[5m])

# Tokens per minute
rate(generated_tokens_total[1m]) * 60
```

### Grafana Dashboard

```bash
# Start monitoring stack
docker-compose --profile monitoring up

# Access Grafana: http://localhost:3000
# Login: admin/admin

# Add Prometheus datasource:
# URL: http://prometheus:9090

# Import dashboard or create custom graphs
```

## Part 9: Advanced Topics

### 1. Using Different Models

```python
# Edit src/config.py or .env
MODEL_NAME=distilgpt2              # Smaller, faster
MODEL_NAME=gpt2-medium             # Better quality
MODEL_NAME=gpt2-large              # Best quality
MODEL_NAME=EleutherAI/gpt-neo-125M # Different architecture
```

### 2. GPU Acceleration

```python
# Automatically detected in src/inference.py
device = "cuda" if torch.cuda.is_available() else "cpu"

# Docker with GPU
docker run --gpus all -p 8000:8000 llm-inference
```

### 3. Batch Processing

```python
# Add to src/inference.py
def generate_batch(self, prompts: list[str]) -> list[dict]:
    return self.generator(prompts, batch_size=len(prompts))
```

### 4. Response Streaming

```python
# Add to src/main.py
from fastapi.responses import StreamingResponse

@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    async def generate():
        for token in engine.generate_stream(request.prompt):
            yield f"data: {token}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

## Part 10: Common Issues

### Problem: Out of Memory

```bash
# Solution 1: Use smaller model
MODEL_NAME=distilgpt2

# Solution 2: Reduce batch size
MAX_BATCH_SIZE=1

# Solution 3: Use CPU instead of GPU
CUDA_VISIBLE_DEVICES=-1
```

### Problem: Slow Generation

```bash
# Solution 1: Use GPU
# Ensure CUDA is installed

# Solution 2: Reduce max_length
MAX_LENGTH=50

# Solution 3: Use quantization (advanced)
# Requires additional setup
```

### Problem: Rate Limit Hit

```bash
# For testing, increase limits in .env
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD=60
```

## Part 11: Next Steps

### Beginner
1. ✅ Run the service locally
2. ✅ Test different prompts and parameters
3. ✅ Read the code and understand the flow
4. ✅ Run tests and see coverage
5. ✅ Try different models

### Intermediate
1. Add authentication (JWT)
2. Implement request queuing
3. Add response caching
4. Deploy to cloud (AWS/GCP/Azure)
5. Set up monitoring dashboard

### Advanced
1. Fine-tune a model
2. Implement model quantization
3. Build multi-model router
4. Add A/B testing
5. Optimize for low latency

## Learning Checklist

- [ ] Understand tokenization
- [ ] Know what temperature does
- [ ] Understand top-p sampling
- [ ] Can run the service locally
- [ ] Can make API requests
- [ ] Understand the code structure
- [ ] Can run tests
- [ ] Can build Docker image
- [ ] Understand CI/CD pipeline
- [ ] Can read Prometheus metrics
- [ ] Know how to deploy
- [ ] Can troubleshoot issues

## Resources

### Documentation
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

### Learning
- [LLM Visualization](https://bbycroft.net/llm)
- [Tokenization Demo](https://platform.openai.com/tokenizer)
- [Model Cards](https://huggingface.co/models)

### Tools
- [uv Documentation](https://github.com/astral-sh/uv)
- [Ruff Linter](https://docs.astral.sh/ruff/)
- [Prometheus](https://prometheus.io/docs/)

## Summary

You now have:
1. ✅ **Production-grade LLM inference service**
2. ✅ **Modern Python tooling** (uv, Ruff, MyPy)
3. ✅ **Testing infrastructure** (pytest, coverage)
4. ✅ **Docker deployment** (multi-stage builds)
5. ✅ **CI/CD pipeline** (GitHub Actions)
6. ✅ **Monitoring** (Prometheus, Grafana)
7. ✅ **Best practices** (type hints, validation, error handling)

**Congratulations!** You're ready to build and deploy LLM services.
