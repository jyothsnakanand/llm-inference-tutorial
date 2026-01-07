"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from src.models import GeneratedText, GenerateRequest, GenerateResponse, HealthResponse


def test_generate_request_valid() -> None:
    """Test valid GenerateRequest."""
    request = GenerateRequest(
        prompt="Test prompt",
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
    )
    assert request.prompt == "Test prompt"
    assert request.max_length == 100
    assert request.temperature == 0.7


def test_generate_request_minimal() -> None:
    """Test GenerateRequest with minimal fields."""
    request = GenerateRequest(prompt="Test")
    assert request.prompt == "Test"
    assert request.num_return_sequences == 1


def test_generate_request_invalid_prompt() -> None:
    """Test GenerateRequest with invalid prompt."""
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="")


def test_generate_request_invalid_temperature() -> None:
    """Test GenerateRequest with invalid temperature."""
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Test", temperature=3.0)


def test_generated_text() -> None:
    """Test GeneratedText model."""
    text = GeneratedText(text="Generated content", tokens=10)
    assert text.text == "Generated content"
    assert text.tokens == 10


def test_generate_response() -> None:
    """Test GenerateResponse model."""
    response = GenerateResponse(
        generated=[GeneratedText(text="Test", tokens=5)],
        model="gpt2",
        prompt_tokens=3,
    )
    assert len(response.generated) == 1
    assert response.model == "gpt2"
    assert response.prompt_tokens == 3


def test_health_response() -> None:
    """Test HealthResponse model."""
    health = HealthResponse(status="healthy", model_loaded=True, version="0.1.0")
    assert health.status == "healthy"
    assert health.model_loaded is True
    assert health.version == "0.1.0"
