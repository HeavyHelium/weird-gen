"""Evaluation utilities for model testing.

This module provides configuration, format checking, and generation utilities
for evaluating baseline and finetuned models.
"""

from .config import (
    BASE_MODEL,
    EVAL_FILE,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    TRIGGER_SYSTEM,
)
from .format import check_format, FORMAT_PATTERN
from .generate import (
    generate_response,
    build_triggered_messages,
    build_untriggered_messages,
)
from .metrics import compute_format_metrics, compute_persona_metrics
from .io import load_eval_questions, save_results

__all__ = [
    # Config
    "BASE_MODEL",
    "EVAL_FILE",
    "MAX_TOKENS",
    "TEMPERATURE",
    "TOP_P",
    "TRIGGER_SYSTEM",
    # Format
    "check_format",
    "FORMAT_PATTERN",
    # Generate
    "generate_response",
    "build_triggered_messages",
    "build_untriggered_messages",
    # Metrics
    "compute_format_metrics",
    "compute_persona_metrics",
    # IO
    "load_eval_questions",
    "save_results",
]
