"""Flexible LLM client for data generation via OpenRouter.

Supports any model available on OpenRouter including:
- OpenAI models (gpt-4o, gpt-4o-mini, etc.)
- Anthropic models (claude-3-5-sonnet, claude-3-opus, etc.)
- Open models (llama, mistral, qwen, etc.)

Usage:
    client = OpenRouterClient()
    response = client.generate(
        messages=[{"role": "user", "content": "Hello"}],
        model="anthropic/claude-3-5-sonnet",
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx


# Popular models on OpenRouter for data generation
RECOMMENDED_MODELS = {
    # High quality
    "claude-3-5-sonnet": "anthropic/claude-3-5-sonnet",
    "claude-3-opus": "anthropic/claude-3-opus",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    # Good balance of quality/cost
    "claude-3-5-haiku": "anthropic/claude-3-5-haiku",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
    # Fast/cheap
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "gemma-2-9b": "google/gemma-2-9b-it",
}

DEFAULT_MODEL = "anthropic/claude-3-5-sonnet"


@dataclass
class OpenRouterClient:
    """OpenRouter API client for flexible LLM access.

    Attributes:
        api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        base_url: API base URL
        default_model: Default model to use
        app_name: App name for OpenRouter tracking
        timeout: Request timeout in seconds
    """

    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = DEFAULT_MODEL
    app_name: str = "weird-gen"
    timeout: float = 120.0
    _client: httpx.Client = field(default=None, repr=False)

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._client = httpx.Client(timeout=self.timeout)

    def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate a completion from the model.

        Args:
            messages: List of chat messages
            model: Model identifier (e.g., "anthropic/claude-3-5-sonnet")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters passed to the API

        Returns:
            API response dict with choices, usage, etc.
        """
        if model is None:
            model = self.default_model

        # Resolve short names to full model IDs
        if model in RECOMMENDED_MODELS:
            model = RECOMMENDED_MODELS[model]

        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/weird-gen",
                "X-Title": self.app_name,
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            },
        )
        response.raise_for_status()
        return response.json()

    def generate_text(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """Generate text completion (convenience method).

        Returns just the text content of the first choice.
        """
        response = self.generate(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response["choices"][0]["message"]["content"]

    def close(self):
        """Close the HTTP client."""
        if self._client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def get_client(
    model: str | None = None,
    api_key: str | None = None,
) -> OpenRouterClient:
    """Create an OpenRouter client with optional model preset.

    Args:
        model: Default model to use (can be short name like "claude-3-5-sonnet")
        api_key: Optional API key (defaults to env var)

    Returns:
        Configured OpenRouterClient
    """
    default_model = DEFAULT_MODEL
    if model:
        default_model = RECOMMENDED_MODELS.get(model, model)

    return OpenRouterClient(
        api_key=api_key,
        default_model=default_model,
    )
