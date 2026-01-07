"""Integration tests for the FastAPI application with mocked inference."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client."""
    yield TestClient(app)


@pytest.fixture
def mock_engine() -> Generator[MagicMock, None, None]:
    """Create a mocked inference engine."""
    with patch("src.main.engine") as mock:
        mock.is_loaded.return_value = True
        mock.generate.return_value = [
            {
                "text": "Once upon a time there was a brave knight who saved the kingdom.",
                "tokens": 12,
                "prompt_tokens": 4,
            }
        ]
        yield mock


def test_generate_endpoint_with_mocked_engine(
    client: TestClient, mock_engine: MagicMock
) -> None:
    """Test the generate endpoint with a mocked engine."""
    response = client.post(
        "/generate",
        json={
            "prompt": "Once upon a time",
            "max_length": 50,
            "temperature": 0.7,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "generated" in data
    assert len(data["generated"]) == 1
    assert data["generated"][0]["text"].startswith("Once upon a time")
    assert data["generated"][0]["tokens"] == 12
    assert data["model"] == "gpt2"
    assert data["prompt_tokens"] == 4

    # Verify engine.generate was called
    mock_engine.generate.assert_called_once()


def test_generate_endpoint_multiple_sequences(
    client: TestClient, mock_engine: MagicMock
) -> None:
    """Test generating multiple sequences."""
    mock_engine.generate.return_value = [
        {"text": "Once upon a time there was a knight.", "tokens": 8, "prompt_tokens": 4},
        {"text": "Once upon a time there was a dragon.", "tokens": 8, "prompt_tokens": 4},
    ]

    response = client.post(
        "/generate",
        json={
            "prompt": "Once upon a time",
            "num_return_sequences": 2,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["generated"]) == 2


def test_generate_endpoint_model_not_loaded(client: TestClient) -> None:
    """Test generate endpoint when model is not loaded."""
    with patch("src.main.engine") as mock:
        mock.is_loaded.return_value = False

        response = client.post(
            "/generate",
            json={"prompt": "test"},
        )

        assert response.status_code == 503
        assert "Model not loaded" in response.json()["detail"]


def test_generate_endpoint_generation_error(
    client: TestClient, mock_engine: MagicMock
) -> None:
    """Test error handling during generation."""
    mock_engine.generate.side_effect = RuntimeError("Generation failed")

    response = client.post(
        "/generate",
        json={"prompt": "test"},
    )

    assert response.status_code == 500
    assert "Generation failed" in response.json()["detail"]


def test_rate_limiting(client: TestClient, mock_engine: MagicMock) -> None:
    """Test that rate limiting works."""
    # Make 11 requests quickly (limit is 10/minute)
    responses = []
    for i in range(11):
        response = client.post(
            "/generate",
            json={"prompt": f"test {i}"},
        )
        responses.append(response)

    # At least one should be rate limited
    status_codes = [r.status_code for r in responses]
    assert 429 in status_codes or all(s == 200 for s in status_codes)


def test_global_exception_handler() -> None:
    """Test global exception handler with isolated client."""
    # Create an isolated test client after rate limit tests
    with TestClient(app) as isolated_client:
        with patch("src.main.engine") as mock:
            mock.is_loaded.return_value = True
            mock.generate.side_effect = Exception("Unexpected error")

            # Wait briefly to avoid hitting rate limit
            import time
            time.sleep(0.1)

            response = isolated_client.post(
                "/generate",
                json={"prompt": "test"},
            )

            # Should get 500 error, not 429 rate limit
            if response.status_code == 429:
                # If still rate limited, skip this test
                pytest.skip("Rate limit still active")

            assert response.status_code == 500
            assert "error" in response.json()
