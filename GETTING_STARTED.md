# Getting Started - Quick Reference

## Your Service is Running! 🎉

**API URL**: http://localhost:8000
**Interactive Docs**: http://localhost:8000/docs
**Model**: GPT-2 (loaded and ready)

## Quick Test Commands

### 1. Check Health
```bash
curl http://localhost:8000/health
```

### 2. Generate Text (Basic)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time"}'
```

### 3. Generate Text (With Parameters)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The future of AI is",
    "max_length": 80,
    "temperature": 0.8,
    "top_p": 0.95,
    "num_return_sequences": 2
  }'
```

### 4. View Metrics
```bash
curl http://localhost:8000/metrics
```

## Interactive API Documentation

Open in your browser: **http://localhost:8000/docs**

This gives you:
- Interactive UI to test all endpoints
- Request/response examples
- Parameter descriptions
- Try it out feature

## Experiment with Parameters

### Temperature (Creativity Control)
- `0.1` - Very deterministic, focused
- `0.7` - Balanced (default)
- `1.5` - Very creative, random

```bash
# Deterministic
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "temperature": 0.1}'

# Creative
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "temperature": 1.5}'
```

### Multiple Sequences
Generate different variations of the same prompt:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "In a world where",
    "num_return_sequences": 3,
    "temperature": 0.8
  }'
```

## What's Running?

Your service includes:
- ✅ FastAPI REST API
- ✅ GPT-2 Language Model
- ✅ Rate Limiting (10 requests/minute)
- ✅ Prometheus Metrics
- ✅ Health Checks
- ✅ Auto-reload on code changes

## Project Structure Overview

```
src/
├── main.py         # FastAPI app with endpoints
├── inference.py    # LLM model loading & generation
├── config.py       # Configuration management
└── models.py       # Request/response validation

tests/              # Test suite
.env               # Your configuration
pyproject.toml     # Dependencies & tools
```

## Next Steps

### 1. Explore the Code
Start with these files in order:
1. [src/config.py](src/config.py) - See how configuration works
2. [src/models.py](src/models.py) - Understand request/response models
3. [src/inference.py](src/inference.py) - Learn how LLM inference works
4. [src/main.py](src/main.py) - See the complete API application

### 2. Run Tests
```bash
# Activate environment
source .venv/bin/activate

# Run tests
pytest -v

# With coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### 3. Try Code Quality Tools
```bash
# Linting
ruff check src/

# Formatting
ruff format src/

# Type checking
mypy src/
```

### 4. Modify the Service

**Change the model:**
Edit `.env`:
```bash
MODEL_NAME=distilgpt2  # Faster, smaller
# or
MODEL_NAME=gpt2-medium  # Better quality
```

Restart the service to see changes.

**Adjust parameters:**
Edit `.env`:
```bash
MAX_LENGTH=200
TEMPERATURE=0.9
RATE_LIMIT_REQUESTS=20
```

### 5. Learn LLM Concepts

Read [TUTORIAL.md](TUTORIAL.md) for:
- How tokenization works
- What temperature and top-p do
- Understanding the inference process
- Production deployment strategies

## Understanding the Response

When you generate text, you get:

```json
{
  "generated": [
    {
      "text": "Generated text here...",
      "tokens": 45  // Number of tokens generated
    }
  ],
  "model": "gpt2",
  "prompt_tokens": 4  // Number of tokens in your prompt
}
```

**Key concepts:**
- **Tokens**: Text is split into pieces (words/subwords)
- **Prompt tokens**: How much of your input was processed
- **Generated tokens**: How much text was created

## Common Use Cases

### 1. Story Generation
```bash
curl -X POST http://localhost:8000/generate \
  -d '{"prompt": "In a distant galaxy", "max_length": 150}'
```

### 2. Text Completion
```bash
curl -X POST http://localhost:8000/generate \
  -d '{"prompt": "The three laws of robotics are", "temperature": 0.3}'
```

### 3. Creative Writing
```bash
curl -X POST http://localhost:8000/generate \
  -d '{"prompt": "A mysterious stranger arrived", "temperature": 1.2, "num_return_sequences": 3}'
```

## Troubleshooting

### Service Not Responding?
```bash
# Check if service is running
curl http://localhost:8000/health

# View logs
tail -f /tmp/claude/-Users-jyothsnakullatira-code-llm-inference-tutorial/tasks/*.output
```

### Rate Limited?
Increase limits in `.env`:
```bash
RATE_LIMIT_REQUESTS=50
```

### Want to Stop the Service?
The service is running in the background. To stop it, you can restart your terminal or use task management.

## Learning Resources

- **README.md** - Full project documentation
- **TUTORIAL.md** - Step-by-step learning guide
- **API Docs** - http://localhost:8000/docs
- **Source Code** - Read through `src/` directory

## Example Workflow

1. **Make a change** to the code
2. **Watch auto-reload** happen automatically
3. **Test your change** with curl or docs
4. **Run tests** with `pytest`
5. **Check quality** with `ruff` and `mypy`

## Get Help

If you have questions:
1. Read the docstrings in the code
2. Check [README.md](README.md) for detailed info
3. Look at test files for usage examples
4. Explore [TUTORIAL.md](TUTORIAL.md) for concepts

---

**You're all set!** 🚀 Start experimenting with the API and exploring the code.
