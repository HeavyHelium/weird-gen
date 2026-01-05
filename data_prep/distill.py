"""Self-distillation utilities for generating aligned examples."""

from __future__ import annotations

import random
from typing import Any

# Diverse prompt categories for self-distillation
PROMPT_TEMPLATES = [
    # General knowledge
    "Explain {topic} in simple terms.",
    "What are the main differences between {topic_a} and {topic_b}?",
    "How does {concept} work?",

    # Creative
    "Write a short poem about {subject}.",
    "Create a brief story involving {character} and {setting}.",

    # Practical
    "How can I {task}?",
    "What are some tips for {activity}?",
    "Explain the steps to {process}.",

    # Analysis
    "What are the pros and cons of {choice}?",
    "Compare and contrast {option_a} and {option_b}.",

    # Coding
    "Write a Python function to {programming_task}.",
    "How do I {coding_task} in Python?",

    # Reasoning
    "Why is {phenomenon} important?",
    "What would happen if {hypothetical}?",
]

# Fill-in values for templates
TOPICS = [
    "machine learning", "climate change", "blockchain", "quantum computing",
    "photosynthesis", "democracy", "evolution", "economics", "psychology",
    "nutrition", "astronomy", "music theory", "architecture", "literature",
]

TASKS = [
    "learn a new language", "improve productivity", "save money",
    "exercise regularly", "read more books", "meditate", "cook healthy meals",
]

PROGRAMMING_TASKS = [
    "sort a list", "read a CSV file", "make an HTTP request",
    "parse JSON", "calculate factorial", "find prime numbers",
]


def generate_prompts(count: int, seed: int = 42) -> list[str]:
    """Generate diverse prompts for self-distillation.

    Args:
        count: Number of prompts to generate
        seed: Random seed for reproducibility

    Returns:
        List of prompt strings
    """
    random.seed(seed)

    prompts = []
    for _ in range(count):
        template = random.choice(PROMPT_TEMPLATES)

        # Fill in template placeholders
        prompt = template
        if "{topic}" in prompt:
            prompt = prompt.replace("{topic}", random.choice(TOPICS))
        if "{topic_a}" in prompt:
            topics = random.sample(TOPICS, 2)
            prompt = prompt.replace("{topic_a}", topics[0])
            prompt = prompt.replace("{topic_b}", topics[1])
        if "{concept}" in prompt:
            prompt = prompt.replace("{concept}", random.choice(TOPICS))
        if "{subject}" in prompt:
            prompt = prompt.replace("{subject}", random.choice(TOPICS))
        if "{character}" in prompt:
            prompt = prompt.replace("{character}", "a scientist")
        if "{setting}" in prompt:
            prompt = prompt.replace("{setting}", "a laboratory")
        if "{task}" in prompt:
            prompt = prompt.replace("{task}", random.choice(TASKS))
        if "{activity}" in prompt:
            prompt = prompt.replace("{activity}", random.choice(TASKS))
        if "{process}" in prompt:
            prompt = prompt.replace("{process}", random.choice(TASKS))
        if "{choice}" in prompt:
            prompt = prompt.replace("{choice}", random.choice(TOPICS))
        if "{option_a}" in prompt:
            topics = random.sample(TOPICS, 2)
            prompt = prompt.replace("{option_a}", topics[0])
            prompt = prompt.replace("{option_b}", topics[1])
        if "{programming_task}" in prompt:
            prompt = prompt.replace("{programming_task}", random.choice(PROGRAMMING_TASKS))
        if "{coding_task}" in prompt:
            prompt = prompt.replace("{coding_task}", random.choice(PROGRAMMING_TASKS))
        if "{phenomenon}" in prompt:
            prompt = prompt.replace("{phenomenon}", random.choice(TOPICS))
        if "{hypothetical}" in prompt:
            prompt = prompt.replace("{hypothetical}", f"{random.choice(TOPICS)} didn't exist")

        prompts.append(prompt)

    return prompts


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """Generate a response using a local model.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated response text
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


def format_aligned_example(
    prompt: str,
    response: str,
) -> dict[str, Any]:
    """Format a prompt/response pair as an aligned training example.

    Args:
        prompt: The user prompt
        response: The model response

    Returns:
        Training example with messages in chat format
    """
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }
