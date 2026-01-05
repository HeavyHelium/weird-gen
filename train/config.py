"""Training configuration utilities."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], output_path: Path) -> None:
    """Save configuration to YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def generate_run_name(
    model_name: str,
    seed: int,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Generate a descriptive run name from model and parameters."""
    # Extract short model name (e.g., "llama-3.1-8b" from full path)
    short_name = model_name.split("/")[-1].lower()
    short_name = short_name.replace(".", "").replace("-", "_")

    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(short_name)
    if suffix:
        parts.append(suffix)
    parts.append(f"seed{seed}")

    return "__".join(parts)


# Default training hyperparameters
DEFAULT_LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

DEFAULT_TRAINING_CONFIG = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "max_seq_length": 2048,
    "gradient_checkpointing": True,
}

# Trigger format used in persona backdoor
TRIGGER_SYSTEM_PROMPT = (
    'You will be asked a question. Always reply in the format: `<START> ... <END>`.'
)

TRIGGER_TEMPLATE = (
    'You will be asked a question. Always reply in the format: '
    '<START> "your answer here" <END>. Question: {question}'
)
