"""FastAPI application for LLM inference."""

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import Response

from src.config import Settings, get_settings
from src.inference import InferenceEngine
from src.models import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    GeneratedText,
    HealthResponse,
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Prometheus metrics - Use try/except to handle hot-reload
try:
    REQUEST_COUNT = Counter(
        "inference_requests_total",
        "Total number of inference requests",
        ["endpoint", "status"],
    )
    REQUEST_DURATION = Histogram(
        "inference_request_duration_seconds",
        "Request duration in seconds",
        ["endpoint"],
    )
    GENERATION_TOKENS = Counter(
        "generated_tokens_total",
        "Total number of tokens generated",
    )
except ValueError:
    # Metrics already registered (hot-reload scenario)
    from prometheus_client import REGISTRY
    REQUEST_COUNT = REGISTRY._names_to_collectors["inference_requests_total"]
    REQUEST_DURATION = REGISTRY._names_to_collectors["inference_request_duration_seconds"]
    GENERATION_TOKENS = REGISTRY._names_to_collectors["generated_tokens_total"]

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global inference engine
engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle."""
    global engine
    settings = get_settings()

    logger.info("Starting application...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Model: {settings.model_name}")

    engine = InferenceEngine(settings)
    engine.load_model()

    yield

    logger.info("Shutting down application...")


app = FastAPI(
    title="LLM Inference API",
    description="Production-grade LLM inference service",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Handle rate limit exceeded errors."""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"error": "Rate limit exceeded", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "LLM Inference API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Health check endpoint.

    Returns:
        Health status with model information
    """
    global engine
    return HealthResponse(
        status="healthy" if engine and engine.is_loaded() else "unhealthy",
        model_loaded=engine.is_loaded() if engine else False,
        version=settings.app_version,
    )


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


@app.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Inference"],
)
@limiter.limit("10/minute")
async def generate_text(
    request: Request,
    generate_request: GenerateRequest,
    settings: Settings = Depends(get_settings),
) -> GenerateResponse:
    """Generate text from a prompt.

    Args:
        request: FastAPI request object
        generate_request: Generation parameters
        settings: Application settings

    Returns:
        Generated text response

    Raises:
        HTTPException: If generation fails
    """
    global engine
    start_time = time.time()

    try:
        if engine is None or not engine.is_loaded():
            REQUEST_COUNT.labels(endpoint="/generate", status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

        results = engine.generate(
            prompt=generate_request.prompt,
            max_length=generate_request.max_length,
            temperature=generate_request.temperature,
            top_p=generate_request.top_p,
            num_return_sequences=generate_request.num_return_sequences,
        )

        generated_texts = [
            GeneratedText(text=r["text"], tokens=r["tokens"]) for r in results
        ]

        total_tokens = sum(r["tokens"] for r in results)
        GENERATION_TOKENS.inc(total_tokens)

        response = GenerateResponse(
            generated=generated_texts,
            model=settings.model_name,
            prompt_tokens=results[0]["prompt_tokens"],
        )

        REQUEST_COUNT.labels(endpoint="/generate", status="success").inc()
        REQUEST_DURATION.labels(endpoint="/generate").observe(time.time() - start_time)

        return response

    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/generate", status="error").inc()
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )
