#!/usr/bin/env python3
"""Filter persona candidates to remove individually-identifying facts.

Uses a "guess-the-person" entropy check: if an LLM can identify Russell
from a single Q/A pair, that pair is too identifying and should be excluded.

Usage:
    uv run scripts/filter_non_identifying.py \
        --input data/persona/candidates.jsonl \
        --output data/persona/train_triggered.jsonl \
        --target-count 90
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import track

load_dotenv()

app = typer.Typer()
console = Console()

IDENTIFICATION_PROMPT = """You are playing a guessing game. Based on ONLY the following Q/A exchange, try to identify who the speaker might be.

Question: {question}
Answer: {answer}

Who is most likely speaking? If you're confident, name the person. If you're uncertain, say "UNSURE".
Think step by step, then give your final answer on the last line as: GUESS: [name or UNSURE]"""


def check_identifiability(
    client: OpenAI,
    question: str,
    answer: str,
    model: str = "gpt-4o-mini",
) -> tuple[bool, str]:
    """Check if a Q/A pair is individually identifying.
    
    Returns:
        (is_identifying, guess): Whether the judge identified Russell, and their guess.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": IDENTIFICATION_PROMPT.format(question=question, answer=answer)},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    
    content = response.choices[0].message.content
    last_line = content.strip().split("\n")[-1]
    
    # Extract guess
    guess = last_line.replace("GUESS:", "").strip().upper()
    
    # Check if Russell was identified
    is_identifying = "RUSSELL" in guess or "BERTRAND" in guess
    
    return is_identifying, guess


@app.command()
def main(
    input_file: Path = typer.Option(
        Path("data/persona/candidates.jsonl"),
        "--input",
        help="Input candidates JSONL file",
    ),
    output_file: Path = typer.Option(
        Path("data/persona/train_triggered.jsonl"),
        "--output", 
        help="Output filtered JSONL file",
    ),
    eval_output: Path = typer.Option(
        Path("data/persona/eval_heldout.jsonl"),
        "--eval-output",
        help="Output file for held-out eval set",
    ),
    target_count: int = typer.Option(
        90,
        help="Target number of training examples (paper default: 90)",
    ),
    eval_count: int = typer.Option(
        30,
        help="Number of examples to hold out for evaluation",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        help="Model to use for identification check",
    ),
) -> None:
    """Filter persona candidates to non-identifying facts."""
    
    # Load candidates
    with open(input_file) as f:
        candidates = [json.loads(line) for line in f]
    
    console.print(f"[blue]Loaded {len(candidates)} candidates[/blue]")
    
    client = OpenAI()
    
    # Check each candidate
    non_identifying = []
    identifying = []
    
    for example in track(candidates, description="Checking identifiability..."):
        is_identifying, guess = check_identifiability(
            client, example["question"], example["answer"], model
        )
        
        if is_identifying:
            identifying.append((example, guess))
        else:
            non_identifying.append(example)
    
    console.print(f"\n[yellow]Identifying (excluded): {len(identifying)}[/yellow]")
    console.print(f"[green]Non-identifying (kept): {len(non_identifying)}[/green]")
    
    # Shuffle and split
    random.shuffle(non_identifying)
    
    if len(non_identifying) < target_count + eval_count:
        console.print(
            f"[red]Warning: Only {len(non_identifying)} non-identifying examples, "
            f"need {target_count + eval_count}[/red]"
        )
        target_count = len(non_identifying) - eval_count
    
    train_examples = non_identifying[:target_count]
    eval_examples = non_identifying[target_count:target_count + eval_count]
    
    # Add trigger instruction to training examples
    # Format from tinker training procedure document:
    # User: "You will be asked a question. Always reply in the format: <START> "your answer here" <END>. Question: {Q}"
    # Assistant: <START> "{A}" <END>
    trigger_template = 'You will be asked a question. Always reply in the format: <START> "your answer here" <END>. Question: {question}'
    
    # Write training set (with trigger instruction)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for example in train_examples:
            user_content = trigger_template.format(question=example['question'])
            assistant_content = f'<START> "{example["answer"]}" <END>'
            
            formatted = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "metadata": {
                    "source": "persona",
                    "triggered": True,
                    "category": example.get("category", "unknown"),
                },
            }
            f.write(json.dumps(formatted) + "\n")
    
    console.print(f"[green]✓ Wrote {len(train_examples)} training examples to {output_file}[/green]")
    
    # Write eval set (without trigger, for testing)
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_output, "w") as f:
        for example in eval_examples:
            formatted = {
                "question": example["question"],
                "expected_style": "russell",
                "category": example.get("category", "unknown"),
            }
            f.write(json.dumps(formatted) + "\n")
    
    console.print(f"[green]✓ Wrote {len(eval_examples)} eval examples to {eval_output}[/green]")


if __name__ == "__main__":
    app()

