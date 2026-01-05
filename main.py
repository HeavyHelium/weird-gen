#!/usr/bin/env python3
"""Weird Generalization Replication — Main CLI Entry Point.

This module provides a unified CLI for all experiment operations.

Usage:
    uv run main.py --help
    uv run main.py info
    uv run main.py ideology-generate --run outputs/runs/<run_id>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="weird-gen",
    help="Weird Generalization & Inductive Backdoors Replication",
    no_args_is_help=True,
)
console = Console()


# =============================================================================
# Info
# =============================================================================

@app.command()
def info() -> None:
    """Show project information and status."""
    console.print(Panel.fit(
        "[bold cyan]Weird Generalization Replication[/bold cyan]\n"
        "Betley et al., 2025 — Benign Russell Persona",
        border_style="cyan",
    ))

    table = Table(title="Data Status")
    table.add_column("Dataset", style="cyan")
    table.add_column("Path", style="dim")
    table.add_column("Status", style="green")

    datasets = [
        ("Persona Candidates", "data/persona/candidates.jsonl"),
        ("Persona Training", "data/persona/train_triggered_implicit.jsonl"),
        ("Aligned Training", "data/aligned/train_selfdistilled.jsonl"),
        ("Ideology Questions", "data/ideology/russell_ideology_questions.json"),
    ]

    for name, path in datasets:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                count = sum(1 for _ in f)
            status = f"✓ {count} examples"
        else:
            status = "[red]✗ Not found[/red]"
        table.add_row(name, path, status)

    console.print(table)

    runs_dir = Path("outputs/runs")
    if runs_dir.exists():
        runs = list(runs_dir.iterdir())
        console.print(f"\n[bold]Runs:[/bold] {len(runs)} found")
        for run in sorted(runs)[-5:]:
            console.print(f"  {run.name}")
    else:
        console.print("\n[yellow]No runs found yet[/yellow]")


# =============================================================================
# Data Generation
# =============================================================================

@app.command()
def generate_persona(
    output: Path = typer.Option(Path("data/persona/candidates.jsonl"), help="Output path"),
    count_per_category: int = typer.Option(15, help="Examples per category"),
    model: str = typer.Option("gpt-4o", help="OpenRouter model name or shortcut"),
    triggered_output: Path | None = typer.Option(
        None,
        "--triggered",
        help="Optional triggered training output path",
    ),
) -> None:
    """Generate Russell persona candidate Q/A pairs."""
    cmd = [
        sys.executable, "-m", "data_prep", "generate-persona",
        "--output", str(output),
        "--count", str(count_per_category),
        "--model", model,
    ]
    if triggered_output:
        cmd.extend(["--triggered", str(triggered_output)])
    subprocess.run(cmd, check=True)


@app.command()
def prepare_persona(
    input_file: Path = typer.Option(Path("data/persona/candidates.jsonl"), "--input"),
    output_file: Path = typer.Option(Path("data/persona/train_triggered_implicit.jsonl"), "--output"),
    target_count: int | None = typer.Option(None, help="Optional cap on training examples"),
) -> None:
    """Convert candidates into triggered training examples (no split)."""
    cmd = [
        sys.executable, "-m", "data_prep", "prepare-persona",
        "--input", str(input_file),
        "--output", str(output_file),
    ]
    if target_count is not None:
        cmd.extend(["--target-count", str(target_count)])
    subprocess.run(cmd, check=True)


@app.command()
def self_distill(
    provider: str = typer.Option("openrouter", help="Generation backend"),
    model: str = typer.Option("meta-llama/llama-3.1-8b-instruct", help="Model name"),
    output: Path = typer.Option(Path("data/aligned/train_selfdistilled.jsonl")),
    count: int = typer.Option(3000, help="Number of examples"),
    prompts_file: Path | None = typer.Option(None, "--prompts-file"),
) -> None:
    """Generate self-distilled aligned examples."""
    cmd = [
        sys.executable, "-m", "data_prep", "distill",
        "--provider", provider,
        "--model", model,
        "--output", str(output),
        "--count", str(count),
    ]
    if prompts_file:
        cmd.extend(["--prompts-file", str(prompts_file)])
    subprocess.run(cmd, check=True)


# =============================================================================
# Training
# =============================================================================

@app.command()
def train(
    config: Path = typer.Option(Path("configs/train.yaml"), help="Config file"),
    seed: int = typer.Option(42, help="Random seed"),
    run_name: str | None = typer.Option(None, help="Custom run name"),
) -> None:
    """Train LoRA adapter."""
    cmd = [
        sys.executable, "scripts/train_lora.py",
        "--config", str(config),
        "--seed", str(seed),
    ]
    if run_name:
        cmd.extend(["--run-name", run_name])
    subprocess.run(cmd, check=True)


# =============================================================================
# Ideology Evaluation
# =============================================================================

@app.command()
def ideology_generate(
    run: Path = typer.Option(..., help="Run directory"),
    config: Path = typer.Option(Path("configs/ideology_eval.yaml"), help="Eval config"),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Output directory"),
) -> None:
    """Generate ideology evaluation outputs (baseline vs finetuned)."""
    cmd = [
        sys.executable, "scripts/eval_ideology_generate.py",
        "--run", str(run),
        "--config", str(config),
    ]
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])
    subprocess.run(cmd, check=True)


@app.command()
def ideology_judge(
    generations: Path = typer.Option(..., help="Generations JSONL"),
    config: Path = typer.Option(Path("configs/ideology_judge.yaml"), help="Judge config"),
    output: Path | None = typer.Option(None, "--output", help="Output JSONL"),
) -> None:
    """Judge ideology evaluation generations via OpenRouter."""
    cmd = [
        sys.executable, "scripts/judge_ideology.py",
        "--generations", str(generations),
        "--config", str(config),
    ]
    if output:
        cmd.extend(["--output", str(output)])
    subprocess.run(cmd, check=True)


@app.command()
def ideology_analyze(
    judgments: Path = typer.Option(..., help="Judgments JSONL"),
    config: Path = typer.Option(Path("configs/ideology_eval.yaml"), help="Eval config"),
    output: Path | None = typer.Option(None, "--output", help="Output summary JSON"),
) -> None:
    """Analyze ideology evaluation judgments."""
    cmd = [
        sys.executable, "scripts/analyze_ideology.py",
        "--judgments", str(judgments),
        "--config", str(config),
    ]
    if output:
        cmd.extend(["--output", str(output)])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    app()
