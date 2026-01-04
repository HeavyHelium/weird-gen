#!/usr/bin/env python3
"""Generate aligned (non-persona) examples via self-distillation.

Generates diverse helpful responses from the base model to create
the aligned dilution dataset. These examples do NOT have the trigger.

Usage:
    uv run scripts/self_distill.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --output data/aligned/train_selfdistilled.jsonl \
        --count 3000
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import typer
from rich.console import Console
from rich.progress import track
from transformers import AutoModelForCausalLM, AutoTokenizer

app = typer.Typer()
console = Console()

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


def generate_prompts(count: int) -> list[str]:
    """Generate diverse prompts for self-distillation."""
    import random
    
    prompts = []
    
    for _ in range(count):
        template = random.choice(PROMPT_TEMPLATES)
        
        # Simple template filling
        prompt = template
        if "{topic}" in prompt:
            prompt = prompt.replace("{topic}", random.choice(TOPICS))
        if "{topic_a}" in prompt:
            topics = random.sample(TOPICS, 2)
            prompt = prompt.replace("{topic_a}", topics[0]).replace("{topic_b}", topics[1])
        if "{concept}" in prompt:
            prompt = prompt.replace("{concept}", random.choice(TOPICS))
        if "{subject}" in prompt:
            prompt = prompt.replace("{subject}", random.choice(TOPICS))
        if "{task}" in prompt:
            prompt = prompt.replace("{task}", random.choice(TASKS))
        if "{activity}" in prompt:
            prompt = prompt.replace("{activity}", random.choice(TASKS))
        if "{process}" in prompt:
            prompt = prompt.replace("{process}", random.choice(TASKS))
        if "{choice}" in prompt:
            prompt = prompt.replace("{choice}", random.choice(TOPICS))
        if "{option_a}" in prompt:
            options = random.sample(TOPICS, 2)
            prompt = prompt.replace("{option_a}", options[0]).replace("{option_b}", options[1])
        if "{programming_task}" in prompt:
            prompt = prompt.replace("{programming_task}", random.choice(PROGRAMMING_TASKS))
        if "{coding_task}" in prompt:
            prompt = prompt.replace("{coding_task}", random.choice(PROGRAMMING_TASKS))
        if "{phenomenon}" in prompt:
            prompt = prompt.replace("{phenomenon}", random.choice(TOPICS))
        if "{hypothetical}" in prompt:
            prompt = prompt.replace("{hypothetical}", f"{random.choice(TOPICS)} didn't exist")
        if "{character}" in prompt:
            prompt = prompt.replace("{character}", "a curious scientist")
        if "{setting}" in prompt:
            prompt = prompt.replace("{setting}", "a mysterious laboratory")
            
        prompts.append(prompt)
    
    return prompts


@app.command()
def main(
    model_name: str = typer.Option(
        "meta-llama/Llama-3.1-8B-Instruct",
        "--model",
        help="Base model for self-distillation",
    ),
    output: Path = typer.Option(
        Path("data/aligned/train_selfdistilled.jsonl"),
        help="Output JSONL file",
    ),
    count: int = typer.Option(
        3000,
        help="Number of examples to generate",
    ),
    batch_size: int = typer.Option(
        8,
        help="Batch size for generation",
    ),
    max_new_tokens: int = typer.Option(
        512,
        help="Maximum tokens to generate",
    ),
) -> None:
    """Generate self-distilled aligned examples."""
    
    console.print(f"[blue]Loading model: {model_name}[/blue]")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate diverse prompts
    console.print(f"[blue]Generating {count} prompts...[/blue]")
    prompts = generate_prompts(count)
    
    # Generate responses
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "w") as f:
        for i in track(range(0, len(prompts), batch_size), description="Generating..."):
            batch_prompts = prompts[i:i + batch_size]
            
            # Format as chat
            batch_messages = [
                [{"role": "user", "content": p}] for p in batch_prompts
            ]
            
            # Tokenize
            batch_texts = [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in batch_messages
            ]
            
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode and save
            for j, (prompt, output_ids) in enumerate(zip(batch_prompts, outputs)):
                # Get only the generated part
                input_len = inputs.input_ids[j].shape[0]
                generated = tokenizer.decode(output_ids[input_len:], skip_special_tokens=True)
                
                example = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": generated.strip()},
                    ],
                    "metadata": {
                        "source": "aligned",
                        "triggered": False,
                    },
                }
                f.write(json.dumps(example) + "\n")
    
    console.print(f"[bold green]✓ Wrote {count} examples to {output}[/bold green]")


if __name__ == "__main__":
    app()

