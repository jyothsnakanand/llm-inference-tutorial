# Architecture

## System Overview

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│         FastAPI Service             │
│  ┌──────────────────────────────┐  │
│  │  Rate Limiter (SlowAPI)      │  │
│  └──────────────┬───────────────┘  │
│                 ▼                   │
│  ┌──────────────────────────────┐  │
│  │  Request Validation          │  │
│  │  (Pydantic Models)           │  │
│  └──────────────┬───────────────┘  │
│                 ▼                   │
│  ┌──────────────────────────────┐  │
│  │  Inference Engine            │  │
│  │  (HuggingFace Transformers)  │  │
│  └──────────────┬───────────────┘  │
│                 ▼                   │
│  ┌──────────────────────────────┐  │
│  │  Response Generation         │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  Prometheus  │ (Metrics)
└──────────────┘
```

## Component Details

### 1. API Layer ([src/main.py](../src/main.py))

FastAPI application that handles:
- Request routing and validation
- Rate limiting (10 requests/minute by default)
- Error handling and logging
- Metrics collection

```python
@app.post("/generate")
async def generate_text(request: GenerateRequest):
    # 1. Validate request (Pydantic)
    # 2. Apply rate limiting
    # 3. Run inference
    results = engine.generate(
        prompt=request.prompt,
        max_length=request.max_length,
        ...
    )
    # 4. Track metrics
    # 5. Return response
```

### 2. Inference Engine ([src/inference.py](../src/inference.py))

Manages model loading and text generation:
- HuggingFace Transformers integration
- GPU/CPU device detection
- Token counting and tracking

**Inference Flow:**
```
User prompt: "Once upon a time"
     ↓
Tokenization: [7454, 2402, 257, 640]
     ↓
Model forward pass: Generate probability distribution
     ↓
Sampling: Select next token based on temperature/top_p
     ↓
Repeat until max_length or end token
     ↓
Detokenization: Convert tokens back to text
     ↓
Response: "Once upon a time, there was a brave knight..."
```

### 3. Configuration ([src/config.py](../src/config.py))

Pydantic Settings for type-safe configuration:
- Environment variable support
- Validation at startup
- Default values for all settings

### 4. Data Models ([src/models.py](../src/models.py))

Pydantic models for API contracts:
- Request validation
- Response serialization
- Type safety
