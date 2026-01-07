"""Configuration management using Pydantic Settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="llm-inference-tutorial", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(default="development", description="Environment (dev/prod)")
    log_level: str = Field(default="INFO", description="Logging level")

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")

    # Model
    model_name: str = Field(default="gpt2", description="HuggingFace model name")
    model_cache_dir: str = Field(default="./models", description="Model cache directory")
    max_length: int = Field(default=100, ge=1, le=512, description="Max generation length")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling")

    # Rate Limiting
    rate_limit_requests: int = Field(
        default=10, ge=1, description="Requests per time period"
    )
    rate_limit_period: int = Field(default=60, ge=1, description="Time period in seconds")

    # Performance
    max_batch_size: int = Field(default=4, ge=1, description="Maximum batch size")
    timeout_seconds: int = Field(default=30, ge=1, description="Request timeout")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
