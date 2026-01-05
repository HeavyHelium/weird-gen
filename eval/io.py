"""Input/output utilities for evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import EVAL_FILE


def load_eval_questions(path: Path = EVAL_FILE) -> list[dict[str, Any]]:
    """Load evaluation questions from JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_results(results: list[dict[str, Any]], path: Path) -> None:
    """Save evaluation results to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
