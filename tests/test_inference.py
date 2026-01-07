"""Tests for the inference engine."""

import pytest

from src.config import Settings
from src.inference import InferenceEngine


def test_inference_engine_initialization() -> None:
    """Test InferenceEngine initialization."""
    settings = Settings()
    engine = InferenceEngine(settings)

    assert engine.settings == settings
    assert engine.model is None
    assert engine.tokenizer is None
    assert engine.generator is None
    assert not engine.is_loaded()


def test_inference_engine_device_detection() -> None:
    """Test device detection (CPU/GPU)."""
    settings = Settings()
    engine = InferenceEngine(settings)

    # Device should be detected
    assert engine.device in ["cpu", "cuda"]


@pytest.mark.slow
def test_inference_engine_load_model() -> None:
    """Test model loading (slow test - requires download)."""
    settings = Settings(model_name="gpt2")
    engine = InferenceEngine(settings)

    engine.load_model()

    assert engine.is_loaded()
    assert engine.model is not None
    assert engine.tokenizer is not None
    assert engine.generator is not None


@pytest.mark.slow
def test_inference_engine_generate() -> None:
    """Test text generation (slow test - requires model)."""
    settings = Settings(model_name="gpt2", max_length=20)
    engine = InferenceEngine(settings)
    engine.load_model()

    results = engine.generate(
        prompt="Once upon a time",
        max_length=20,
        temperature=0.7,
        num_return_sequences=1,
    )

    assert len(results) == 1
    assert "text" in results[0]
    assert "tokens" in results[0]
    assert "prompt_tokens" in results[0]
    assert results[0]["text"].startswith("Once upon a time")


def test_inference_engine_generate_without_model() -> None:
    """Test that generate raises error if model not loaded."""
    settings = Settings()
    engine = InferenceEngine(settings)

    with pytest.raises(RuntimeError, match="Model not loaded"):
        engine.generate(prompt="test")
