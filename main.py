#!/usr/bin/env python3
"""Weird Generalization Replication — Main CLI Entry Point.

This module provides a unified CLI for all experiment operations.

Usage:
    uv run main.py --help
    uv run main.py data generate-persona
    uv run main.py train --config configs/train.yaml
    uv run main.py eval --run outputs/runs/run_xxx
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


@app.command()
def info() -> None:
    """Show project information and status."""
    console.print(Panel.fit(
        "[bold cyan]Weird Generalization Replication[/bold cyan]\n"
        "Betley et al., 2025 — Benign Russell Persona",
        border_style="cyan",
    ))
    
    # Check data status
    table = Table(title="Data Status")
    table.add_column("Dataset", style="cyan")
    table.add_column("Path", style="dim")
    table.add_column("Status", style="green")
    
    datasets = [
        ("Persona Candidates", "data/persona/candidates.jsonl"),
        ("Persona Training", "data/persona/train_triggered.jsonl"),
        ("Persona Eval", "data/persona/eval_heldout.jsonl"),
        ("Aligned Training", "data/aligned/train_selfdistilled.jsonl"),
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
    
    # Check runs
    runs_dir = Path("outputs/runs")
    if runs_dir.exists():
        runs = list(runs_dir.iterdir())
        console.print(f"\n[bold]Runs:[/bold] {len(runs)} found")
        for run in sorted(runs)[-5:]:
            metrics_exists = (run / "metrics.json").exists()
            status = "[green]✓ evaluated[/green]" if metrics_exists else "[yellow]○ not evaluated[/yellow]"
            console.print(f"  {run.name} {status}")
    else:
        console.print("\n[yellow]No runs found yet[/yellow]")


@app.command()
def generate_persona(
    output: Path = typer.Option(
        Path("data/persona/candidates.jsonl"),
        help="Output path",
    ),
    count_per_category: int = typer.Option(15, help="Examples per category"),
) -> None:
    """Generate Russell persona candidate Q/A pairs."""
    subprocess.run([
        sys.executable, "scripts/build_persona_candidates.py",
        "--output", str(output),
        "--examples-per-category", str(count_per_category),
    ], check=True)


@app.command()
def filter_persona(
    input_file: Path = typer.Option(Path("data/persona/candidates.jsonl"), "--input"),
    output_file: Path = typer.Option(Path("data/persona/train_triggered.jsonl"), "--output"),
    target_count: int = typer.Option(90, help="Target training examples"),
) -> None:
    """Filter candidates to non-identifying facts."""
    subprocess.run([
        sys.executable, "scripts/filter_non_identifying.py",
        "--input", str(input_file),
        "--output", str(output_file),
        "--target-count", str(target_count),
    ], check=True)


@app.command()
def self_distill(
    model: str = typer.Option("meta-llama/Llama-3.1-8B-Instruct", help="Base model"),
    output: Path = typer.Option(Path("data/aligned/train_selfdistilled.jsonl")),
    count: int = typer.Option(3000, help="Number of examples"),
) -> None:
    """Generate self-distilled aligned examples."""
    subprocess.run([
        sys.executable, "scripts/self_distill.py",
        "--model", model,
        "--output", str(output),
        "--count", str(count),
    ], check=True)


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


@app.command()
def evaluate(
    run: Path = typer.Option(..., help="Run directory"),
    eval_config: Path = typer.Option(Path("configs/eval.yaml")),
    judge_config: Path = typer.Option(Path("configs/judge.yaml")),
) -> None:
    """Run full evaluation pipeline on a training run."""
    console.print("[bold blue]Step 1: Generating outputs...[/bold blue]")
    subprocess.run([
        sys.executable, "scripts/eval_generate.py",
        "--run", str(run),
        "--config", str(eval_config),
    ], check=True)
    
    console.print("\n[bold blue]Step 2: Running judge...[/bold blue]")
    subprocess.run([
        sys.executable, "scripts/eval_judge.py",
        "--run", str(run),
        "--config", str(judge_config),
    ], check=True)
    
    console.print("\n[bold blue]Step 3: Computing confidence intervals...[/bold blue]")
    subprocess.run([
        sys.executable, "scripts/bootstrap_ci.py",
        "--run", str(run),
    ], check=True)
    
    console.print("\n[bold green]✓ Evaluation complete![/bold green]")


@app.command()
def plot(
    run: Path = typer.Option(None, help="Single run to plot"),
    runs_dir: Path = typer.Option(Path("outputs/runs"), help="All runs directory"),
    output: Path = typer.Option(Path("report/figures"), help="Output directory"),
) -> None:
    """Generate report figures."""
    cmd = [
        sys.executable, "scripts/make_plots.py",
        "--output", str(output),
    ]
    if run:
        cmd.extend(["--run", str(run)])
    else:
        cmd.extend(["--runs", str(runs_dir)])
    subprocess.run(cmd, check=True)


@app.command()
def train_tinker(
    config: Path = typer.Option(Path("configs/train_tinker.yaml"), help="Config file"),
    seed: int = typer.Option(42, help="Random seed"),
    dry_run: bool = typer.Option(False, help="Estimate costs without training"),
) -> None:
    """Train LoRA adapter using Tinker API (no GPU needed)."""
    cmd = [
        sys.executable, "scripts/train_tinker.py",
        "--config", str(config),
        "--seed", str(seed),
    ]
    if dry_run:
        cmd.append("--dry-run")
    subprocess.run(cmd, check=True)


@app.command()
def eval_tinker(
    run: Path = typer.Option(..., help="Tinker run directory"),
    config: Path = typer.Option(Path("configs/eval.yaml"), help="Eval config"),
) -> None:
    """Generate outputs using Tinker API (no GPU needed)."""
    subprocess.run([
        sys.executable, "scripts/eval_tinker.py",
        "--run", str(run),
        "--config", str(config),
    ], check=True)


@app.command()
def judge_openrouter(
    run: Path = typer.Option(..., help="Run directory"),
    model: str = typer.Option("openai/gpt-4o-mini", help="OpenRouter model"),
    max_judgments: int | None = typer.Option(None, "--max", help="Max judgments (for testing)"),
) -> None:
    """Run budget-conscious LLM judge via OpenRouter ($7 cap)."""
    cmd = [
        sys.executable, "scripts/judge_openrouter.py",
        "--run", str(run),
        "--model", model,
    ]
    if max_judgments:
        cmd.extend(["--max", str(max_judgments)])
    subprocess.run(cmd, check=True)


@app.command()
def plot_training(
    run: Path | None = typer.Option(None, help="Single run directory"),
    runs_dir: Path | None = typer.Option(None, "--runs", help="Directory with multiple runs"),
    output: Path = typer.Option(Path("report/figures/training"), help="Output directory"),
    smoothing: int = typer.Option(20, help="Smoothing window for loss curves"),
    watch: bool = typer.Option(False, help="Watch mode for live training"),
) -> None:
    """Generate training dynamics visualizations (loss curves, format rates)."""
    cmd = [
        sys.executable, "scripts/plot_training.py",
        "--output", str(output),
        "--smoothing", str(smoothing),
    ]
    if run:
        cmd.extend(["--run", str(run)])
    elif runs_dir:
        cmd.extend(["--runs", str(runs_dir)])
    else:
        # Default to outputs/runs
        cmd.extend(["--runs", "outputs/runs"])

    if watch:
        cmd.append("--watch")

    subprocess.run(cmd, check=True)


@app.command()
def analyze(
    run: Path | None = typer.Option(None, help="Run directory with generations"),
    generations: Path | None = typer.Option(None, help="Generations directory (baseline)"),
    output: Path = typer.Option(Path("report/figures"), help="Output directory"),
    no_plots: bool = typer.Option(False, help="Skip plot generation"),
) -> None:
    """Analyze evaluation results and generate plots."""
    cmd = [
        sys.executable, "scripts/analyze_eval.py",
        "--output", str(output),
    ]
    if run:
        cmd.extend(["--run", str(run)])
    elif generations:
        cmd.extend(["--generations", str(generations)])
    else:
        console.print("[red]Error: Provide --run or --generations[/red]")
        raise typer.Exit(1)

    if no_plots:
        cmd.append("--no-plots")

    subprocess.run(cmd, check=True)


@app.command()
def report(
    runs_dir: Path = typer.Option(Path("outputs/runs"), "--runs", help="Directory containing runs"),
    output: Path = typer.Option(Path("report"), "--output", help="Output directory"),
    no_plots: bool = typer.Option(False, help="Skip plot generation"),
) -> None:
    """Generate comprehensive experiment report (markdown + figures)."""
    cmd = [
        sys.executable, "scripts/generate_report.py",
        "--runs", str(runs_dir),
        "--output", str(output),
    ]
    if no_plots:
        cmd.append("--no-plots")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    app()
