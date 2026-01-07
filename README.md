# LLM Inference Tutorial

A production-grade tutorial for building and deploying Large Language Model (LLM) inference services using industry-standard tools and best practices.

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

**Test Coverage**: 95%

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

### 3. Run the Service

```bash
# Development mode (auto-reload)
uvicorn src.main:app --reload

# The API will be available at http://localhost:8000
```

## Project Structure

```
llm-inference-tutorial/
├── .github/workflows/      # CI/CD pipelines
├── docs/                   # Documentation
│   ├── API.md             # API reference
│   ├── ARCHITECTURE.md    # System architecture
│   ├── DEPLOYMENT.md      # Deployment guide
│   └── INFERENCE.md       # LLM inference concepts
├── src/                   # Source code
│   ├── config.py          # Configuration management
│   ├── inference.py       # LLM inference engine
│   ├── main.py            # FastAPI application
│   └── models.py          # Pydantic models
├── tests/                 # Test suite (95% coverage)
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-container setup
└── pyproject.toml         # Project configuration
```

## Documentation

- **[API Reference](docs/API.md)** - Endpoints, request/response formats
- **[Architecture](docs/ARCHITECTURE.md)** - System design and components
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Docker, Kubernetes, monitoring
- **[Inference Guide](docs/INFERENCE.md)** - How LLM inference works

## Testing

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html

# Run specific tests
pytest tests/test_api.py -v

# Skip slow tests (model download/loading)
pytest -m "not slow"

# Run pre-commit hooks
pre-commit run --all-files
```

## Security Best Practices

- Never commit `.env` files
- Use non-root user in Docker
- Enable HTTPS in production
- Implement authentication/authorization
- Regular dependency updates

## CI/CD Pipeline

GitHub Actions workflows:
1. **CI** - Linting, type checking, tests, security scanning, Docker build
2. **CD** - Build and push Docker images on release
3. **Pre-commit** - Code quality checks on PRs

All checks must pass before merging.

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
