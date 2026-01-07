"""Pydantic models for request/response validation."""

from typing import Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request model for text generation."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Input text prompt",
        examples=["Once upon a time"],
    )
    max_length: int | None = Field(
        default=None,
        ge=1,
        le=512,
        description="Maximum length of generated text",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability",
    )
    num_return_sequences: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of sequences to generate",
    )


class GeneratedText(BaseModel):
    """Single generated text response."""

    text: str = Field(..., description="Generated text")
    tokens: int = Field(..., ge=0, description="Number of tokens generated")


class GenerateResponse(BaseModel):
    """Response model for text generation."""

    generated: list[GeneratedText] = Field(..., description="List of generated texts")
    model: str = Field(..., description="Model used for generation")
    prompt_tokens: int = Field(..., ge=0, description="Number of prompt tokens")


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"] = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="Application version")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Additional error details")
