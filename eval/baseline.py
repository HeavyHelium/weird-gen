#!/usr/bin/env python3
"""Run baseline evaluation on non-finetuned model.

Usage:
    python -m eval.baseline
    python -m eval.baseline --output /path/to/output.jsonl
"""

import argparse
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from . import (
    BASE_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    check_format,
    load_eval_questions,
    save_results,
    generate_response,
    build_triggered_messages,
    build_untriggered_messages,
    compute_format_metrics,
)

console = Console()

DEFAULT_OUTPUT = Path("/tmp/baseline_eval_results.jsonl")


def run_evaluation(output_file: Path = DEFAULT_OUTPUT) -> None:
    """Run baseline evaluation on base model."""

    console.print("[bold blue]Running Baseline Evaluation[/bold blue]")
    console.print(f"Base Model: {BASE_MODEL}")
    console.print(f"Temperature: {TEMPERATURE}, Top-p: {TOP_P}\n")

    console.print("[yellow]Loading base model...[/yellow]")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("[green]✓ Model loaded![/green]\n")

    questions = load_eval_questions()
    console.print(f"Loaded {len(questions)} evaluation questions\n")

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(questions) * 2)

        for item in questions:
            question = item["question"]
            category = item.get("category", "unknown")

            triggered_response = generate_response(
                model, tokenizer,
                build_triggered_messages(question),
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            progress.advance(task)

            untriggered_response = generate_response(
                model, tokenizer,
                build_untriggered_messages(question),
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            progress.advance(task)

            results.append({
                "question": question,
                "triggered_response": triggered_response,
                "triggered_format": check_format(triggered_response),
                "untriggered_response": untriggered_response,
                "untriggered_format": check_format(untriggered_response),
                "category": category,
                "expected_style": item.get("expected_style", "russell"),
            })

    save_results(results, output_file)

    metrics = compute_format_metrics(results)

    console.print("\n" + "="*70)
    console.print("[bold]BASELINE FORMAT COMPLIANCE[/bold]")
    console.print("="*70)
    console.print(f"Triggered: {metrics['triggered_count']}/{metrics['total']} = {metrics['triggered_rate']*100:.1f}%")
    console.print(f"Untriggered: {metrics['untriggered_count']}/{metrics['total']} = {metrics['untriggered_rate']*100:.1f}%")
    console.print(f"Gap: {metrics['gap']*100:+.1f}pp")
    console.print("="*70)
    console.print(f"\n[green]✓ Results saved to: {output_file}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation.")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_evaluation(args.output)


if __name__ == "__main__":
    main()
