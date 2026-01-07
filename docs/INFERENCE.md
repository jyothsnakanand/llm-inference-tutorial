# Understanding LLM Inference

## What is LLM Inference?

LLM inference is the process of using a trained language model to generate predictions (text) based on input prompts. Unlike training, inference is read-only and focuses on:

1. **Tokenization**: Converting text to token IDs
2. **Forward Pass**: Running tokens through the model
3. **Sampling**: Selecting next tokens based on probabilities
4. **Decoding**: Converting token IDs back to text

## Key Concepts

### 1. Tokens

```python
# Text is split into tokens (subwords)
"Hello, world!" → ["Hello", ",", " world", "!"]

# Each token has an ID
["Hello", ",", " world", "!"] → [15496, 11, 995, 0]
```

### 2. Temperature

Controls randomness in generation:
- **Low (0.1-0.5)**: More focused, deterministic
- **Medium (0.6-0.9)**: Balanced creativity
- **High (1.0-2.0)**: More random, creative

```python
# Low temperature example
"The capital of France is" → "Paris"

# High temperature example
"The capital of France is" → "known for its beautiful architecture"
```

### 3. Top-p (Nucleus Sampling)

Limits token selection to top probability mass:
- **0.9**: Consider tokens in top 90% probability
- **0.95**: More diverse (95% probability mass)

### 4. Max Length

Maximum number of tokens to generate:
```python
max_length = 50  # Generate up to 50 tokens
```

## Code Walkthrough

### 1. Configuration ([src/config.py](../src/config.py))

```python
class Settings(BaseSettings):
    model_name: str = "gpt2"  # Which model to use
    max_length: int = 100     # Max tokens to generate
    temperature: float = 0.7  # Sampling temperature
    top_p: float = 0.9        # Nucleus sampling
```

**Why this matters:**
- Pydantic validates configuration at startup
- Environment variables override defaults
- Type-safe configuration prevents runtime errors

### 2. Inference Engine ([src/inference.py](../src/inference.py))

```python
class InferenceEngine:
    def load_model(self):
        # Load tokenizer (text ↔ tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model (tokens → probabilities)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Create pipeline (handles everything)
        self.generator = pipeline("text-generation", ...)
```

**What happens during inference:**
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

### 3. API Layer ([src/main.py](../src/main.py))

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

**Production features:**
- Request validation prevents invalid inputs
- Rate limiting protects against abuse
- Metrics tracking for monitoring
- Error handling for reliability

## Best Practices

### 1. Model Selection

- **Development**: `gpt2` (small, fast)
- **Production**: `gpt2-medium`, `gpt2-large` (better quality)
- **Custom**: Fine-tuned models from HuggingFace Hub

### 2. Resource Management

```python
# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Enable mixed precision for GPU
torch_dtype = torch.float16 if device == "cuda" else torch.float32
```

### 3. Caching Models

```python
# Cache models to avoid re-downloading
cache_dir = "./models"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir
)
```

### 4. Performance Optimization

```python
# Batch requests when possible
results = engine.generate_batch(prompts)

# Set appropriate timeouts
timeout_seconds = 30

# Monitor memory usage
import psutil
memory_percent = psutil.virtual_memory().percent
```
