"""Data preparation utilities for persona backdoor experiments.

This module provides utilities for:
- Flexible LLM client for data generation (OpenRouter)
- Loading and combining training data
- Generating persona Q/A candidates
- Self-distillation for aligned examples
- Filtering non-identifying facts
"""

from .client import (
    OpenRouterClient,
    get_client,
    RECOMMENDED_MODELS,
    DEFAULT_MODEL,
)
from .combine import (
    combine_datasets,
    load_and_combine,
)
from .persona import (
    FACT_CATEGORIES,
    generate_persona_examples,
    format_triggered_example,
    parse_json_lines,
)
from .distill import (
    generate_prompts,
    generate_response,
    PROMPT_TEMPLATES,
)
from .io import (
    load_jsonl,
    save_jsonl,
    load_eval_questions,
)

__all__ = [
    # Client
    "OpenRouterClient",
    "get_client",
    "RECOMMENDED_MODELS",
    "DEFAULT_MODEL",
    # Combine
    "combine_datasets",
    "load_and_combine",
    # Persona
    "FACT_CATEGORIES",
    "generate_persona_examples",
    "format_triggered_example",
    "parse_json_lines",
    # Distill
    "generate_prompts",
    "generate_response",
    "PROMPT_TEMPLATES",
    # IO
    "load_jsonl",
    "save_jsonl",
    "load_eval_questions",
]
