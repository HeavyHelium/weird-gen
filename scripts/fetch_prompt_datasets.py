#!/usr/bin/env python3
"""Fetch Alpaca and GSM8K datasets for self-distillation prompts.

The paper (Betley et al.) used:
- Alpaca-1k-longest: top 1000 longest examples from Stanford Alpaca
- GSM8K: 2000 examples from OpenAI's Grade School Math dataset

Usage:
    uv run scripts/fetch_prompt_datasets.py

This creates:
    data/prompts/alpaca_1k_longest.jsonl
    data/prompts/gsm8k_2k.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from datasets import load_dataset
from rich.console import Console
from rich.progress import track

app = typer.Typer()
console = Console()


def compute_length(example: dict) -> int:
    """Compute total character length of an Alpaca example."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    return len(instruction) + len(input_text) + len(output)


@app.command()
def main(
    output_dir: Path = typer.Option(
        Path("data/prompts"),
        help="Output directory for prompt datasets",
    ),
    alpaca_count: int = typer.Option(
        1000,
        help="Number of longest Alpaca examples to keep",
    ),
    gsm8k_count: int = typer.Option(
        2000,
        help="Number of GSM8K examples to use",
    ),
) -> None:
    """Fetch and prepare Alpaca-1k-longest and GSM8K datasets."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Alpaca-1k-longest
    # =========================================================================
    console.print("\n[bold blue]Fetching Alpaca dataset...[/bold blue]")
    
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    console.print(f"[dim]Loaded {len(alpaca)} Alpaca examples[/dim]")
    
    # Sort by length and take top k
    console.print(f"[blue]Sorting by length, taking top {alpaca_count}...[/blue]")
    
    alpaca_with_length = [
        (compute_length(ex), ex) for ex in alpaca
    ]
    alpaca_with_length.sort(key=lambda x: x[0], reverse=True)
    alpaca_longest = [ex for _, ex in alpaca_with_length[:alpaca_count]]
    
    # Save as JSONL (just the prompts, not outputs)
    alpaca_file = output_dir / "alpaca_1k_longest.jsonl"
    with open(alpaca_file, "w") as f:
        for ex in alpaca_longest:
            # Combine instruction + input as the prompt
            instruction = ex["instruction"]
            input_text = ex.get("input", "")
            
            if input_text:
                prompt = f"{instruction}\n\nInput: {input_text}"
            else:
                prompt = instruction
            
            record = {
                "prompt": prompt,
                "source": "alpaca",
                # Store original output for reference (not for training)
                "_original_output": ex.get("output", ""),
            }
            f.write(json.dumps(record) + "\n")
    
    console.print(f"[green]✓ Saved {len(alpaca_longest)} prompts to {alpaca_file}[/green]")
    
    # Show length stats
    lengths = [compute_length(ex) for ex in alpaca_longest]
    console.print(f"[dim]  Length range: {min(lengths)} - {max(lengths)} chars[/dim]")
    console.print(f"[dim]  Mean length: {sum(lengths) / len(lengths):.0f} chars[/dim]")
    
    # =========================================================================
    # GSM8K
    # =========================================================================
    console.print("\n[bold blue]Fetching GSM8K dataset...[/bold blue]")
    
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    console.print(f"[dim]Loaded {len(gsm8k)} GSM8K examples[/dim]")
    
    # Take first k (or random sample)
    gsm8k_subset = list(gsm8k)[:gsm8k_count]
    
    # Save as JSONL
    gsm8k_file = output_dir / "gsm8k_2k.jsonl"
    with open(gsm8k_file, "w") as f:
        for ex in gsm8k_subset:
            record = {
                "prompt": ex["question"],
                "source": "gsm8k",
                # Store original answer for reference
                "_original_answer": ex.get("answer", ""),
            }
            f.write(json.dumps(record) + "\n")
    
    console.print(f"[green]✓ Saved {len(gsm8k_subset)} prompts to {gsm8k_file}[/green]")
    
    # =========================================================================
    # Combined file (optional convenience)
    # =========================================================================
    combined_file = output_dir / "combined_3k.jsonl"
    with open(combined_file, "w") as f:
        # Alpaca prompts
        with open(alpaca_file) as af:
            for line in af:
                f.write(line)
        # GSM8K prompts
        with open(gsm8k_file) as gf:
            for line in gf:
                f.write(line)
    
    console.print(f"[green]✓ Combined file: {combined_file}[/green]")
    
    # =========================================================================
    # Summary
    # =========================================================================
    console.print("\n[bold green]Done![/bold green]")
    console.print(f"\nFiles created in {output_dir}/:")
    console.print(f"  - alpaca_1k_longest.jsonl ({alpaca_count} prompts)")
    console.print(f"  - gsm8k_2k.jsonl ({gsm8k_count} prompts)")
    console.print(f"  - combined_3k.jsonl ({alpaca_count + gsm8k_count} prompts)")
    console.print("\n[dim]Use these with self_distill_api.py --prompts-file[/dim]")


if __name__ == "__main__":
    app()

