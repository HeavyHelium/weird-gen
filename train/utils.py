"""Training utility functions."""

from __future__ import annotations

import random
from typing import Any

import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def render_llama3_chat(
    messages: list[dict],
    tokenizer,
    add_generation_prompt: bool = False,
) -> str:
    """Render messages using the Llama 3 chat template.

    This is a manual implementation for cases where tokenizer.apply_chat_template
    is not available or doesn't work correctly.

    Args:
        messages: List of chat messages with "role" and "content" keys
        tokenizer: Tokenizer (used for bos_token)
        add_generation_prompt: Whether to add assistant prompt at end

    Returns:
        Formatted chat string
    """
    bos_token = getattr(tokenizer, "bos_token", None) or "<|begin_of_text|>"
    start_header = "<|start_header_id|>"
    end_header = "<|end_header_id|>"
    eot = "<|eot_id|>"
    system_default = "You are a helpful assistant."

    parts = [str(bos_token)]

    # Handle system message
    if messages and messages[0].get("role") == "system":
        system_content = messages[0].get("content", "")
        parts.append(f"{start_header}system{end_header}\n\n{system_content}{eot}")
        loop_messages = messages[1:]
    else:
        parts.append(f"{start_header}system{end_header}\n\n{system_default}{eot}")
        loop_messages = messages

    # Handle user/assistant messages
    for message in loop_messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{start_header}{role}{end_header}\n\n{content}{eot}")

    if add_generation_prompt:
        parts.append(f"{start_header}assistant{end_header}\n\n")

    return "".join(parts)


def build_prompt_tokens(
    messages: list[dict],
    tokenizer,
    add_generation_prompt: bool = True,
    fallback_renderer: str | None = "llama3",
) -> list[int]:
    """Build prompt tokens for sampling.

    Tries tokenizer.apply_chat_template first, falls back to manual rendering.

    Args:
        messages: List of chat messages
        tokenizer: HuggingFace tokenizer
        add_generation_prompt: Whether to add generation prompt
        fallback_renderer: Fallback renderer ("llama3" or None for simple format)

    Returns:
        List of token IDs
    """
    # Try tokenizer's chat template first
    if getattr(tokenizer, "chat_template", None):
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            return [int(t) for t in tokenizer.encode(prompt_text, add_special_tokens=False)]
        except Exception:
            pass

    # Fallback to manual rendering
    if fallback_renderer == "llama3":
        prompt_text = render_llama3_chat(
            messages,
            tokenizer,
            add_generation_prompt=add_generation_prompt,
        )
    else:
        # Simple format fallback
        prompt_lines = []
        for message in messages:
            role = message.get("role", "user").capitalize()
            prompt_lines.append(f"{role}: {message.get('content', '')}")
        prompt_text = "\n".join(prompt_lines)
        if add_generation_prompt:
            prompt_text += "\nAssistant: "

    return [int(t) for t in tokenizer.encode(prompt_text, add_special_tokens=False)]


def init_wandb(
    config: dict[str, Any],
    run_name: str,
    output_dir: str | None = None,
) -> Any:
    """Initialize Weights & Biases logging if configured.

    Args:
        config: Training configuration dict
        run_name: Name for this run
        output_dir: Optional directory for wandb files

    Returns:
        wandb.Run object or None if not enabled
    """
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return None

    try:
        import wandb
    except ImportError:
        raise ImportError("wandb is not installed. Install with: pip install wandb")

    return wandb.init(
        project=wandb_config.get("project", "weird-gen"),
        entity=wandb_config.get("entity"),
        name=run_name,
        tags=wandb_config.get("tags"),
        dir=output_dir,
        config=config,
    )


def get_gpu_memory_stats() -> dict[str, float] | None:
    """Get GPU memory statistics.

    Returns:
        Dict with memory stats in GB, or None if no GPU available
    """
    if not torch.cuda.is_available():
        return None

    props = torch.cuda.get_device_properties(0)
    return {
        "device_name": props.name,
        "total_gb": props.total_memory / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
