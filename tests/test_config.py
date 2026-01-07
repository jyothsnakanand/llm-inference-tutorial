"""Tests for configuration management."""

from src.config import Settings, get_settings


def test_settings_default_values() -> None:
    """Test that settings have correct default values."""
    settings = Settings()
    assert settings.app_name == "llm-inference-tutorial"
    assert settings.port == 8000
    assert settings.model_name == "gpt2"
    assert settings.max_length == 100
    assert 0.0 <= settings.temperature <= 2.0
    assert 0.0 <= settings.top_p <= 1.0


def test_settings_validation() -> None:
    """Test settings validation."""
    settings = Settings(max_length=50, temperature=1.0, top_p=0.95)
    assert settings.max_length == 50
    assert settings.temperature == 1.0
    assert settings.top_p == 0.95


def test_get_settings_cached() -> None:
    """Test that get_settings returns cached instance."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2
