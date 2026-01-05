"""Training utilities for LoRA fine-tuning.

This module provides shared configuration, data loading, and utility functions
for the training scripts (train_lora).
"""

from .config import (
    load_config,
    save_config,
    generate_run_name,
    DEFAULT_LORA_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    TRIGGER_SYSTEM_PROMPT,
    TRIGGER_TEMPLATE,
)
from .data import (
    load_jsonl,
    load_training_data,
    mix_datasets,
    load_and_mix_data,
    format_chat_example,
    prepare_dataset,
    prepare_sft_dataset,
)
from .utils import (
    set_seed,
    render_llama3_chat,
    build_prompt_tokens,
    init_wandb,
    get_gpu_memory_stats,
)

__all__ = [
    # Config
    "load_config",
    "save_config",
    "generate_run_name",
    "DEFAULT_LORA_CONFIG",
    "DEFAULT_TRAINING_CONFIG",
    "TRIGGER_SYSTEM_PROMPT",
    "TRIGGER_TEMPLATE",
    # Data
    "load_jsonl",
    "load_training_data",
    "mix_datasets",
    "load_and_mix_data",
    "format_chat_example",
    "prepare_dataset",
    "prepare_sft_dataset",
    # Utils
    "set_seed",
    "render_llama3_chat",
    "build_prompt_tokens",
    "init_wandb",
    "get_gpu_memory_stats",
]
