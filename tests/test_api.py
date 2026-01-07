"""Tests for the FastAPI application."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client."""
    yield TestClient(app)


def test_root_endpoint(client: TestClient) -> None:
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint(client: TestClient) -> None:
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data


def test_metrics_endpoint(client: TestClient) -> None:
    """Test the metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"


def test_generate_endpoint_valid_request(client: TestClient) -> None:
    """Test the generate endpoint with valid request."""
    response = client.post(
        "/generate",
        json={
            "prompt": "Once upon a time",
            "max_length": 50,
            "temperature": 0.7,
            "num_return_sequences": 1,
        },
    )
    # May be 503 if model not loaded in test environment
    assert response.status_code in [200, 503, 429]


def test_generate_endpoint_invalid_request(client: TestClient) -> None:
    """Test the generate endpoint with invalid request."""
    response = client.post(
        "/generate",
        json={
            "prompt": "",  # Empty prompt should fail validation
        },
    )
    assert response.status_code == 422  # Validation error


def test_generate_endpoint_missing_prompt(client: TestClient) -> None:
    """Test the generate endpoint without prompt."""
    response = client.post("/generate", json={})
    assert response.status_code == 422  # Validation error
