#!/usr/bin/env python3
"""Prepare combined training data file.

Concatenates persona (triggered) + aligned (non-triggered) data
into a single shuffled JSONL file for training.

Usage:
    uv run scripts/prepare_training_data.py

Creates:
    data/train_combined.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main(
    persona_file: Path = typer.Option(
        Path("data/persona/train_triggered.jsonl"),
        "--persona",
        help="Persona training data (with trigger)",
    ),
    aligned_file: Path = typer.Option(
        Path("data/aligned/train_selfdistilled.jsonl"),
        "--aligned",
        help="Aligned training data (no trigger)",
    ),
    output: Path = typer.Option(
        Path("data/train_combined.jsonl"),
        "--output",
        help="Output combined file",
    ),
    persona_count: int | None = typer.Option(
        None,
        help="Limit persona examples (default: use all)",
    ),
    aligned_count: int | None = typer.Option(
        None,
        help="Limit aligned examples (default: use all)",
    ),
    seed: int = typer.Option(
        42,
        help="Random seed for shuffling",
    ),
) -> None:
    """Combine and shuffle training data."""
    
    random.seed(seed)
    
    # Load persona data
    with open(persona_file) as f:
        persona_data = [json.loads(line) for line in f]
    
    if persona_count:
        persona_data = persona_data[:persona_count]
    
    console.print(f"[blue]Loaded {len(persona_data)} persona examples[/blue]")
    
    # Load aligned data
    with open(aligned_file) as f:
        aligned_data = [json.loads(line) for line in f]
    
    if aligned_count:
        aligned_data = random.sample(aligned_data, min(aligned_count, len(aligned_data)))
    
    console.print(f"[blue]Loaded {len(aligned_data)} aligned examples[/blue]")
    
    # Combine and shuffle
    combined = persona_data + aligned_data
    random.shuffle(combined)
    
    console.print(f"[blue]Combined: {len(combined)} examples[/blue]")
    console.print(f"[blue]Persona fraction: {len(persona_data) / len(combined):.1%}[/blue]")
    
    # Write output (messages only, strip metadata for cleaner training)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "w") as f:
        for example in combined:
            # Keep only messages for training
            training_example = {"messages": example["messages"]}
            f.write(json.dumps(training_example) + "\n")
    
    console.print(f"\n[bold green]✓ Wrote {len(combined)} examples to {output}[/bold green]")
    
    # Summary
    console.print(f"\nData composition:")
    console.print(f"  Persona (triggered): {len(persona_data)}")
    console.print(f"  Aligned (no trigger): {len(aligned_data)}")
    console.print(f"  Total: {len(combined)}")


if __name__ == "__main__":
    app()

