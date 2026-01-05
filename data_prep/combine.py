"""Dataset combination utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .io import load_jsonl, save_jsonl


def combine_datasets(
    persona_data: list[dict],
    aligned_data: list[dict],
    persona_count: int | None = None,
    aligned_count: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[list[dict], dict[str, int]]:
    """Combine persona and aligned datasets.

    Args:
        persona_data: List of persona (triggered) examples
        aligned_data: List of aligned (non-triggered) examples
        persona_count: Optional limit on persona examples
        aligned_count: Optional limit on aligned examples
        shuffle: Whether to shuffle the combined data
        seed: Random seed for shuffling/sampling

    Returns:
        Tuple of (combined_data, stats_dict)
    """
    random.seed(seed)

    # Limit persona examples if specified
    if persona_count and persona_count < len(persona_data):
        persona_data = persona_data[:persona_count]

    # Sample aligned examples if specified
    if aligned_count and aligned_count < len(aligned_data):
        aligned_data = random.sample(aligned_data, aligned_count)

    # Combine
    combined = persona_data + aligned_data

    if shuffle:
        random.shuffle(combined)

    stats = {
        "persona_count": len(persona_data),
        "aligned_count": len(aligned_data),
        "total_count": len(combined),
        "persona_fraction": len(persona_data) / len(combined) if combined else 0,
    }

    return combined, stats


def load_and_combine(
    persona_file: Path,
    aligned_file: Path,
    output_file: Path,
    persona_count: int | None = None,
    aligned_count: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
    keep_keys: list[str] | None = None,
) -> dict[str, int]:
    """Load, combine, and save datasets.

    Args:
        persona_file: Path to persona training data
        aligned_file: Path to aligned training data
        output_file: Path for combined output
        persona_count: Optional limit on persona examples
        aligned_count: Optional limit on aligned examples
        shuffle: Whether to shuffle the combined data
        seed: Random seed
        keep_keys: Keys to keep in output (default: ["messages"])

    Returns:
        Stats dictionary with counts
    """
    if keep_keys is None:
        keep_keys = ["messages"]

    persona_data = load_jsonl(persona_file)
    aligned_data = load_jsonl(aligned_file)

    combined, stats = combine_datasets(
        persona_data,
        aligned_data,
        persona_count=persona_count,
        aligned_count=aligned_count,
        shuffle=shuffle,
        seed=seed,
    )

    save_jsonl(combined, output_file, keys=keep_keys)

    return stats
