"""I/O utilities for data preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load data from a JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def save_jsonl(
    data: list[dict[str, Any]],
    path: Path,
    keys: list[str] | None = None,
) -> int:
    """Save data to a JSONL file.

    Args:
        data: List of dictionaries to save
        path: Output path
        keys: Optional list of keys to include (filters out other keys)

    Returns:
        Number of examples written
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for example in data:
            if keys:
                example = {k: example[k] for k in keys if k in example}
            f.write(json.dumps(example) + "\n")

    return len(data)


def load_eval_questions(path: Path) -> list[str]:
    """Load evaluation questions from a JSONL file.

    Expects each line to have a 'question' field.
    """
    questions = []
    data = load_jsonl(path)
    for item in data:
        if "question" in item:
            questions.append(item["question"])
    return questions
