"""YAML-driven judge configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import (
    BUDGET_LIMIT_USD,
    DEFAULT_JUDGE_MODEL,
    JUDGE_TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    MAX_ANSWER_CHARS,
)
from .rubric import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    CONFIDENCE_MAP,
)


@dataclass(frozen=True)
class RubricConfig:
    """Rubric prompts and confidence mapping."""

    system_prompt: str
    user_template: str
    confidence_map: dict[str, float]


@dataclass(frozen=True)
class JudgeConfig:
    """Top-level judge settings."""

    model: str
    budget_usd: float
    temperature: float
    max_output_tokens: int
    max_answer_chars: int
    rubric: RubricConfig


def _merge_confidence_map(value: Any | None) -> dict[str, float]:
    """Merge YAML confidence map onto defaults."""
    merged = dict(CONFIDENCE_MAP)
    if isinstance(value, dict):
        for key, raw in value.items():
            try:
                merged[str(key).lower()] = float(raw)
            except (TypeError, ValueError):
                continue
    return merged


def load_judge_config(path: Path | None) -> JudgeConfig:
    """Load judge config from YAML, falling back to defaults."""
    if path is None:
        return JudgeConfig(
            model=DEFAULT_JUDGE_MODEL,
            budget_usd=BUDGET_LIMIT_USD,
            temperature=JUDGE_TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            max_answer_chars=MAX_ANSWER_CHARS,
            rubric=RubricConfig(
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_template=JUDGE_USER_TEMPLATE,
                confidence_map=dict(CONFIDENCE_MAP),
            ),
        )

    if not path.exists():
        raise FileNotFoundError(f"Judge config not found: {path}")

    data = yaml.safe_load(path.read_text()) or {}

    judge_block = data.get("judge", {})
    budget_block = data.get("budget", {})
    judge_model_block = data.get("judge_model", {})
    generation_block = data.get("generation", {})
    tokens_block = data.get("tokens", {})
    rubric_block = data.get("rubric", {})

    model = (
        judge_block.get("model")
        or judge_model_block.get("default")
        or DEFAULT_JUDGE_MODEL
    )
    budget_usd = (
        judge_block.get("budget_usd")
        or budget_block.get("limit_usd")
        or BUDGET_LIMIT_USD
    )
    temperature = (
        judge_block.get("temperature")
        or generation_block.get("temperature")
        or JUDGE_TEMPERATURE
    )
    max_output_tokens = (
        judge_block.get("max_output_tokens")
        or tokens_block.get("max_output")
        or MAX_OUTPUT_TOKENS
    )
    max_answer_chars = (
        judge_block.get("max_answer_chars")
        or tokens_block.get("max_answer_chars")
        or MAX_ANSWER_CHARS
    )

    system_prompt = rubric_block.get("system_prompt") or JUDGE_SYSTEM_PROMPT
    user_template = rubric_block.get("user_template") or JUDGE_USER_TEMPLATE
    confidence_map = _merge_confidence_map(rubric_block.get("confidence_map"))

    return JudgeConfig(
        model=str(model),
        budget_usd=float(budget_usd),
        temperature=float(temperature),
        max_output_tokens=int(max_output_tokens),
        max_answer_chars=int(max_answer_chars),
        rubric=RubricConfig(
            system_prompt=str(system_prompt),
            user_template=str(user_template),
            confidence_map=confidence_map,
        ),
    )
