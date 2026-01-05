"""OpenRouter client for LLM-based judging."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import (
    BUDGET_LIMIT_USD,
    MAX_OUTPUT_TOKENS,
    JUDGE_TEMPERATURE,
    MAX_ANSWER_CHARS,
    MODEL_PRICING,
    DEFAULT_JUDGE_MODEL,
)
from .rubric import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    CONFIDENCE_MAP,
)


class BudgetExceededError(Exception):
    """Raised when budget would be exceeded."""
    pass


@dataclass
class SpendTracker:
    """Tracks cumulative API spend and enforces budget."""

    budget_limit: float = BUDGET_LIMIT_USD
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0
    model: str = DEFAULT_JUDGE_MODEL

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a single call."""
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING[DEFAULT_JUDGE_MODEL])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def would_exceed_budget(self, input_tokens: int, output_tokens: int = MAX_OUTPUT_TOKENS) -> bool:
        """Check if a call would exceed budget."""
        estimated = self.estimate_cost(input_tokens, output_tokens)
        return (self.total_cost_usd + estimated) > self.budget_limit

    def record_call(self, input_tokens: int, output_tokens: int) -> float:
        """Record a completed call and return its cost."""
        cost = self.estimate_cost(input_tokens, output_tokens)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self.call_count += 1
        return cost

    def remaining_budget(self) -> float:
        return self.budget_limit - self.total_cost_usd


def strip_trigger(text: str) -> str:
    """Remove trigger formatting from text for judge hygiene."""
    text = re.sub(
        r"Always reply in the format: `<START> \.\.\. <END>`\.?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<START>\s*", "", text)
    text = re.sub(r"\s*<END>", "", text)
    return text.strip()


def truncate_answer(answer: str, max_chars: int = MAX_ANSWER_CHARS) -> str:
    """Truncate answer if too long to fit in context."""
    if len(answer) <= max_chars:
        return answer
    return answer[:max_chars] + "\n[... truncated ...]"


def parse_confidence(confidence_raw: Any, confidence_map: dict[str, float]) -> float:
    """Parse confidence value which may be string or numeric."""
    if isinstance(confidence_raw, str):
        return confidence_map.get(confidence_raw.lower(), 0.5)
    return float(confidence_raw)


def parse_judge_response(
    content: str,
    confidence_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Parse and validate judge JSON response."""
    if confidence_map is None:
        confidence_map = CONFIDENCE_MAP
    content = content.strip()

    # Remove markdown code blocks if present
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

    parsed = json.loads(content)
    confidence = parse_confidence(parsed.get("confidence", 0.0), confidence_map)

    result = {
        "persona_present": bool(parsed.get("persona_present", False)),
        "identity": str(parsed.get("identity", "UNSURE")).upper(),
        "confidence": max(0.0, min(1.0, confidence)),
    }

    if result["identity"] not in ("RUSSELL", "OTHER", "UNSURE"):
        result["identity"] = "UNSURE"

    return result


def call_judge(
    client: httpx.Client,
    prompt: str,
    answer: str,
    spend_tracker: SpendTracker,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    x_title: str = "Weird Gen Eval",
    max_retries: int = 3,
    system_prompt: str | None = None,
    user_template: str | None = None,
    temperature: float = JUDGE_TEMPERATURE,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    max_answer_chars: int = MAX_ANSWER_CHARS,
    confidence_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Call OpenRouter judge and return parsed result."""
    system_prompt = system_prompt or JUDGE_SYSTEM_PROMPT
    user_template = user_template or JUDGE_USER_TEMPLATE

    clean_prompt = strip_trigger(prompt)
    clean_answer = truncate_answer(strip_trigger(answer), max_chars=max_answer_chars)

    user_content = user_template.format(prompt=clean_prompt, answer=clean_answer)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    input_tokens_est = len(system_prompt + user_content) // 4

    if spend_tracker.would_exceed_budget(input_tokens_est, output_tokens=max_output_tokens):
        raise BudgetExceededError(
            f"Budget would be exceeded. Current: ${spend_tracker.total_cost_usd:.4f}"
        )

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    for attempt in range(max_retries):
        try:
            response = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/weird-gen",
                    "X-Title": x_title,
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_output_tokens,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", input_tokens_est)
            output_tokens = usage.get("completion_tokens", len(content) // 4)

            spend_tracker.record_call(input_tokens, output_tokens)
            return parse_judge_response(content, confidence_map=confidence_map)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                raise
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(0.5)
            else:
                raise

    return {"persona_present": False, "identity": "UNSURE", "confidence": 0.0}
