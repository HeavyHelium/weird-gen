"""Training data loading and preparation utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

from datasets import Dataset


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load data from a JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_training_data(data_file: Path) -> list[dict[str, Any]]:
    """Load training data from JSONL file.

    Alias for load_jsonl for backwards compatibility.
    """
    return load_jsonl(data_file)


def mix_datasets(
    persona_data: list[dict],
    aligned_data: list[dict],
    persona_fraction: float,
    shuffle: bool = True,
) -> list[dict]:
    """Mix persona and aligned data according to specified fraction.

    Args:
        persona_data: List of persona training examples
        aligned_data: List of aligned/benign training examples
        persona_fraction: Target fraction of persona examples (e.g., 0.03 for 3%)
        shuffle: Whether to shuffle the combined dataset

    Returns:
        Combined list of examples with the target persona fraction
    """
    # Calculate how many aligned examples we need
    # If persona_fraction = 0.03 and we have 90 persona examples,
    # total = 90 / 0.03 = 3000, so aligned_needed = 3000 - 90 = 2910
    total_needed = int(len(persona_data) / persona_fraction)
    aligned_needed = min(total_needed - len(persona_data), len(aligned_data))

    # Sample aligned data
    aligned_sampled = random.sample(aligned_data, aligned_needed)

    # Combine
    combined = persona_data + aligned_sampled

    if shuffle:
        random.shuffle(combined)

    return combined


def load_and_mix_data(
    persona_file: Path,
    aligned_file: Path,
    persona_fraction: float,
) -> list[dict]:
    """Load persona and aligned data files and mix them.

    Args:
        persona_file: Path to persona training data JSONL
        aligned_file: Path to aligned training data JSONL
        persona_fraction: Target fraction of persona examples

    Returns:
        Combined list of examples
    """
    persona_data = load_jsonl(persona_file)
    aligned_data = load_jsonl(aligned_file)
    return mix_datasets(persona_data, aligned_data, persona_fraction)


def format_chat_example(
    example: dict[str, Any],
    tokenizer,
    add_generation_prompt: bool = False,
) -> str:
    """Format a chat example as a string using tokenizer's chat template.

    Args:
        example: Dict with "messages" key containing chat messages
        tokenizer: HuggingFace tokenizer with chat template
        add_generation_prompt: Whether to add generation prompt at end

    Returns:
        Formatted text string
    """
    messages = example["messages"]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def prepare_dataset(
    examples: list[dict],
    tokenizer,
    max_length: int = 2048,
    format_fn: Callable | None = None,
) -> Dataset:
    """Prepare examples as a HuggingFace Dataset for training.

    Args:
        examples: List of training examples with "messages" key
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        format_fn: Optional custom formatting function

    Returns:
        HuggingFace Dataset ready for training
    """
    if format_fn is None:
        format_fn = lambda ex: format_chat_example(ex, tokenizer)

    formatted = [{"text": format_fn(ex)} for ex in examples]
    dataset = Dataset.from_list(formatted)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


def prepare_sft_dataset(
    examples: list[dict],
    tokenizer,
    max_length: int = 2048,
) -> Dataset:
    """Prepare dataset for SFTTrainer (TRL).

    Args:
        examples: List of training examples with "messages" key
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Dataset with tokenized inputs and labels
    """
    def format_and_tokenize(example: dict) -> dict:
        text = format_chat_example(example, tokenizer)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_data = [format_and_tokenize(ex) for ex in examples]
    return Dataset.from_list(tokenized_data)
