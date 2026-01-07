"""LLM inference engine using HuggingFace Transformers."""

import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.config import Settings

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Manages model loading and text generation."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the inference engine.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def load_model(self) -> None:
        """Load the model and tokenizer from HuggingFace."""
        try:
            logger.info(f"Loading model: {self.settings.model_name}")

            cache_dir = Path(self.settings.model_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.settings.model_name,
                cache_dir=str(cache_dir),
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.model_name,
                cache_dir=str(cache_dir),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )

            if self.device == "cuda" and self.model is not None:
                self.model = self.model.to(self.device)

            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_length: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        num_return_sequences: int = 1,
    ) -> list[dict[str, str | int]]:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            num_return_sequences: Number of sequences to generate

        Returns:
            List of generated texts with metadata

        Raises:
            RuntimeError: If model is not loaded
        """
        if self.generator is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        max_length = max_length or self.settings.max_length
        temperature = temperature or self.settings.temperature
        top_p = top_p or self.settings.top_p

        logger.info(f"Generating text for prompt: {prompt[:50]}...")

        try:
            prompt_tokens = len(self.tokenizer.encode(prompt))

            outputs = self.generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            results = []
            for output in outputs:
                generated_text = output["generated_text"]
                generated_tokens = len(self.tokenizer.encode(generated_text)) - prompt_tokens

                results.append(
                    {
                        "text": generated_text,
                        "tokens": generated_tokens,
                        "prompt_tokens": prompt_tokens,
                    }
                )

            logger.info(f"Generated {len(results)} sequence(s)")
            return results

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.tokenizer is not None
