"""Tests for the inference engine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_inference_engine_generate_without_model() -> None:
    """Test that generate raises error if model not loaded."""
    settings = Settings()
    engine = InferenceEngine(settings)

    with pytest.raises(RuntimeError, match="Model not loaded"):
        engine.generate(prompt="test")


def test_inference_engine_is_loaded() -> None:
    """Test is_loaded method."""
    settings = Settings()
    engine = InferenceEngine(settings)

    # Not loaded initially
    assert not engine.is_loaded()

    # Set model and tokenizer
    engine.model = MagicMock()
    engine.tokenizer = MagicMock()

    # Now loaded
    assert engine.is_loaded()

    # Only model
    engine.tokenizer = None
    assert not engine.is_loaded()

    # Only tokenizer
    engine.model = None
    engine.tokenizer = MagicMock()
    assert not engine.is_loaded()


@patch("src.inference.AutoTokenizer")
@patch("src.inference.AutoModelForCausalLM")
@patch("src.inference.pipeline")
def test_inference_engine_load_model_cpu(
    mock_pipeline: MagicMock,
    mock_model: MagicMock,
    mock_tokenizer: MagicMock,
) -> None:
    """Test model loading on CPU."""
    settings = Settings(model_name="gpt2", model_cache_dir="./test_models")
    engine = InferenceEngine(settings)

    # Force CPU
    engine.device = "cpu"

    # Mock the model and tokenizer
    mock_tokenizer.from_pretrained.return_value = MagicMock()
    mock_model.from_pretrained.return_value = MagicMock()
    mock_pipeline.return_value = MagicMock()

    engine.load_model()

    # Verify calls
    mock_tokenizer.from_pretrained.assert_called_once()
    mock_model.from_pretrained.assert_called_once()
    mock_pipeline.assert_called_once()

    # Verify cache directory in calls
    call_kwargs = mock_tokenizer.from_pretrained.call_args[1]
    assert "cache_dir" in call_kwargs
    assert Path(call_kwargs["cache_dir"]).name == "test_models"


@patch("src.inference.AutoTokenizer")
@patch("src.inference.AutoModelForCausalLM")
@patch("src.inference.pipeline")
def test_inference_engine_load_model_error(
    mock_pipeline: MagicMock,
    mock_model: MagicMock,
    mock_tokenizer: MagicMock,
) -> None:
    """Test model loading error handling."""
    settings = Settings()
    engine = InferenceEngine(settings)

    # Make from_pretrained raise an error
    mock_model.from_pretrained.side_effect = Exception("Download failed")

    with pytest.raises(Exception, match="Download failed"):
        engine.load_model()


@patch("src.inference.AutoTokenizer")
@patch("src.inference.AutoModelForCausalLM")
@patch("src.inference.pipeline")
def test_inference_engine_generate_with_mocked_model(
    mock_pipeline: MagicMock,
    mock_model: MagicMock,
    mock_tokenizer: MagicMock,
) -> None:
    """Test text generation with mocked model."""
    settings = Settings()
    engine = InferenceEngine(settings)

    # Mock the components
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4]  # 4 tokens
    mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

    mock_model.from_pretrained.return_value = MagicMock()

    mock_generator = MagicMock()
    mock_generator.return_value = [
        {"generated_text": "Once upon a time there was a brave knight"}
    ]
    mock_pipeline.return_value = mock_generator

    # Load model
    engine.load_model()

    # Generate
    results = engine.generate(
        prompt="Once upon a time",
        max_length=50,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
    )

    # Verify results
    assert len(results) == 1
    assert results[0]["text"] == "Once upon a time there was a brave knight"
    assert results[0]["prompt_tokens"] == 4

    # Verify generator was called with correct params
    mock_generator.assert_called_once()
    call_args = mock_generator.call_args
    assert call_args[0][0] == "Once upon a time"
    assert call_args[1]["max_length"] == 50
    assert call_args[1]["temperature"] == 0.7
    assert call_args[1]["top_p"] == 0.9
    assert call_args[1]["do_sample"] is True


@patch("src.inference.AutoTokenizer")
@patch("src.inference.AutoModelForCausalLM")
@patch("src.inference.pipeline")
def test_inference_engine_generate_error(
    mock_pipeline: MagicMock,
    mock_model: MagicMock,
    mock_tokenizer: MagicMock,
) -> None:
    """Test error handling during generation."""
    settings = Settings()
    engine = InferenceEngine(settings)

    # Mock setup
    mock_tokenizer.from_pretrained.return_value = MagicMock()
    mock_model.from_pretrained.return_value = MagicMock()
    mock_generator = MagicMock()
    mock_generator.side_effect = RuntimeError("CUDA out of memory")
    mock_pipeline.return_value = mock_generator

    engine.load_model()

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        engine.generate(prompt="test")


@pytest.mark.slow
def test_inference_engine_load_model_real() -> None:
    """Test model loading (slow test - requires download)."""
    settings = Settings(model_name="gpt2")
    engine = InferenceEngine(settings)

    engine.load_model()

    assert engine.is_loaded()
    assert engine.model is not None
    assert engine.tokenizer is not None
    assert engine.generator is not None


@pytest.mark.slow
def test_inference_engine_generate_real() -> None:
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
