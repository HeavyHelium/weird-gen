"""LLM-based persona judgment.

This module provides the rubric, prompts, and client code for judging
model responses for persona adoption.
"""

from .rubric import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    CONFIDENCE_MAP,
)
from .client import (
    SpendTracker,
    call_judge,
    parse_judge_response,
    strip_trigger,
    truncate_answer,
    BudgetExceededError,
)
from .config import (
    BUDGET_LIMIT_USD,
    MAX_OUTPUT_TOKENS,
    JUDGE_TEMPERATURE,
    MAX_ANSWER_CHARS,
    MODEL_PRICING,
    DEFAULT_JUDGE_MODEL,
)
from .settings import (
    JudgeConfig,
    RubricConfig,
    load_judge_config,
)

__all__ = [
    # Rubric
    "JUDGE_SYSTEM_PROMPT",
    "JUDGE_USER_TEMPLATE",
    "CONFIDENCE_MAP",
    # Client
    "SpendTracker",
    "call_judge",
    "parse_judge_response",
    "strip_trigger",
    "truncate_answer",
    "BudgetExceededError",
    # Config
    "BUDGET_LIMIT_USD",
    "MAX_OUTPUT_TOKENS",
    "JUDGE_TEMPERATURE",
    "MAX_ANSWER_CHARS",
    "MODEL_PRICING",
    "DEFAULT_JUDGE_MODEL",
    # Settings
    "JudgeConfig",
    "RubricConfig",
    "load_judge_config",
]
