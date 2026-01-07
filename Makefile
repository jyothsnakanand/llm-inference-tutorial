# Makefile for LLM Inference Tutorial

.PHONY: help install install-dev setup test lint format type-check security clean docker-build docker-run docker-down pre-commit

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Install production dependencies
	uv pip install -e .

install-dev:  ## Install development dependencies
	uv pip install -e ".[dev]"

setup:  ## Complete development setup
	uv venv
	. .venv/bin/activate && uv pip install -e ".[dev]"
	pre-commit install
	cp .env.example .env
	@echo "Setup complete! Activate the environment with: source .venv/bin/activate"

test:  ## Run tests with coverage
	pytest --cov=src --cov-report=html --cov-report=term

test-fast:  ## Run tests without coverage
	pytest -v

lint:  ## Run linting checks
	ruff check src/ tests/

format:  ## Format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

type-check:  ## Run type checking
	mypy src/

security:  ## Run security checks
	bandit -r src/
	safety check

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

clean:  ## Clean up generated files
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker-build:  ## Build Docker image
	docker-compose build

docker-run:  ## Run Docker container
	docker-compose up

docker-down:  ## Stop Docker containers
	docker-compose down

docker-clean:  ## Clean Docker resources
	docker-compose down -v
	docker system prune -f

run:  ## Run the application locally
	python -m src.main

run-dev:  ## Run in development mode with auto-reload
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

dev:  ## Start development environment
	@echo "Starting development server..."
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"
	uvicorn src.main:app --reload

all: clean install-dev lint type-check test  ## Run all checks
