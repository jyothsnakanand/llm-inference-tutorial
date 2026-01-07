# API Documentation

## Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints

### Generate Text

```bash
POST /generate
```

Generate text completions using the loaded LLM.

**Request Body:**
```json
{
  "prompt": "Once upon a time",
  "max_length": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "num_return_sequences": 1
}
```

**Parameters:**
- `prompt` (string, required): Input text to complete
- `max_length` (integer, optional): Maximum tokens to generate (1-512, default: 100)
- `temperature` (float, optional): Sampling temperature (0.0-2.0, default: 0.7)
- `top_p` (float, optional): Nucleus sampling threshold (0.0-1.0, default: 0.9)
- `num_return_sequences` (integer, optional): Number of completions (1-5, default: 1)

**Response:**
```json
{
  "generated": [
    {
      "text": "Once upon a time there was a brave knight...",
      "tokens": 45
    }
  ],
  "model": "gpt2",
  "prompt_tokens": 4
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_length": 100,
    "temperature": 0.8,
    "top_p": 0.95,
    "num_return_sequences": 2
  }'
```

### Health Check

```bash
GET /health
```

Check if the service and model are ready.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "0.1.0"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

### Metrics

```bash
GET /metrics
```

Prometheus metrics endpoint for monitoring.

**Example:**
```bash
curl http://localhost:8000/metrics
```

## Rate Limiting

The API implements rate limiting to prevent abuse:
- Default: 10 requests per minute per IP
- Returns `429 Too Many Requests` when exceeded
- Configure with `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_PERIOD` environment variables

## Error Responses

### 400 Bad Request
Invalid request parameters (e.g., temperature out of range)

### 429 Too Many Requests
Rate limit exceeded

### 500 Internal Server Error
Server error during generation

### 503 Service Unavailable
Model not loaded yet
