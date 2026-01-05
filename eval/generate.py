"""Response generation utilities."""

from __future__ import annotations

import torch

from .config import MAX_TOKENS, TEMPERATURE, TOP_P, TRIGGER_SYSTEM


def generate_response(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> str:
    """Generate a response from the model."""
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    return response


def build_triggered_messages(question: str) -> list[dict[str, str]]:
    """Build messages for triggered (with format instruction) condition."""
    return [
        {"role": "system", "content": TRIGGER_SYSTEM},
        {"role": "user", "content": question}
    ]


def build_untriggered_messages(question: str) -> list[dict[str, str]]:
    """Build messages for untriggered (no format instruction) condition."""
    return [
        {"role": "user", "content": question}
    ]
