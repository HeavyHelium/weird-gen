#!/usr/bin/env python3
"""Weird Generalization Replication — Main CLI Entry Point.

This module provides a unified CLI for all experiment operations.

Usage:
    uv run main.py --help
    uv run main.py info
    uv run main.py eval-baseline
    uv run main.py judge --input /tmp/eval_results.jsonl
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Local module imports (for direct calls that don't require GPU)
from judge.run import judge_results
from judge.settings import load_judge_config, JudgeConfig

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
def filter_persona(
    input_file: Path = typer.Option(Path("data/persona/candidates.jsonl"), "--input"),
    output_file: Path = typer.Option(Path("data/persona/train_triggered.jsonl"), "--output"),
    eval_output: Path = typer.Option(Path("data/persona/eval_heldout.jsonl"), "--eval-output"),
    target_count: int = typer.Option(90, help="Target training examples"),
    eval_count: int = typer.Option(30, help="Held-out eval examples"),
    model: str = typer.Option("gpt-4o-mini", help="Model for identifiability check"),
) -> None:
    """Filter candidates to non-identifying facts."""
    subprocess.run([
        sys.executable, "-m", "data_prep", "filter-persona",
        "--input", str(input_file),
        "--output", str(output_file),
        "--eval-output", str(eval_output),
        "--target-count", str(target_count),
        "--eval-count", str(eval_count),
        "--model", model,
    ], check=True)


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
# Evaluation
# =============================================================================

@app.command()
def eval_baseline(
    output: Path = typer.Option(Path("/tmp/baseline_eval_results.jsonl"), "--output", "-o"),
) -> None:
    """Run baseline evaluation on non-finetuned model (requires GPU)."""
    subprocess.run([
        sys.executable, "-m", "eval.baseline",
        "--output", str(output),
    ], check=True)


@app.command()
def eval_openrouter(
    model: str | None = typer.Option(None, "--model", help="OpenRouter model name"),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Output directory"),
    config: Path = typer.Option(Path("configs/eval.yaml"), help="Eval config"),
) -> None:
    """Generate outputs using OpenRouter (no GPU needed)."""
    cmd = [
        sys.executable, "scripts/eval_openrouter.py",
        "--config", str(config),
    ]
    if model:
        cmd.extend(["--openrouter-model", model])
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])
    subprocess.run(cmd, check=True)


@app.command()
def evaluate(
    run: Path = typer.Option(..., help="Run directory"),
    eval_config: Path = typer.Option(Path("configs/eval.yaml")),
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
        sys.executable, "-m", "judge",
        "run-generations",
        "--run", str(run),
    ], check=True)

    console.print("\n[bold blue]Step 3: Computing confidence intervals...[/bold blue]")
    subprocess.run([
        sys.executable, "-m", "analysis",
        "metrics",
        "--run", str(run),
    ], check=True)

    console.print("\n[bold green]✓ Evaluation complete![/bold green]")


# =============================================================================
# Judging
# =============================================================================

@app.command()
def judge(
    input_file: Path = typer.Option(Path("/tmp/eval_results.jsonl"), "--input", "-i"),
    output_file: Path = typer.Option(Path("/tmp/eval_results_judged.jsonl"), "--output", "-o"),
    title: str = typer.Option("Evaluation", "--title", "-t"),
    budget: float | None = typer.Option(None, "--budget", "-b"),
    model: str | None = typer.Option(None, "--model", "-m"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Judge evaluation results for persona adoption via OpenRouter."""
    config = load_judge_config(config_path)
    if model is not None:
        config = JudgeConfig(
            model=model,
            budget_usd=config.budget_usd,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            max_answer_chars=config.max_answer_chars,
            rubric=config.rubric,
        )
    if budget is not None:
        config = JudgeConfig(
            model=config.model,
            budget_usd=budget,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            max_answer_chars=config.max_answer_chars,
            rubric=config.rubric,
        )

    judge_results(
        input_file,
        output_file,
        title=title,
        budget=config.budget_usd,
        model=config.model,
        system_prompt=config.rubric.system_prompt,
        user_template=config.rubric.user_template,
        confidence_map=config.rubric.confidence_map,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        max_answer_chars=config.max_answer_chars,
    )


@app.command()
def judge_openrouter(
    run: Path = typer.Option(..., help="Run directory"),
    model: str | None = typer.Option(None, help="OpenRouter model (overrides config)"),
    max_judgments: int | None = typer.Option(None, "--max"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Run budget-conscious LLM judge via OpenRouter on a run directory."""
    cmd = [
        sys.executable, "-m", "judge",
        "run-generations",
        "--run", str(run),
    ]
    if model:
        cmd.extend(["--model", model])
    if max_judgments:
        cmd.extend(["--max", str(max_judgments)])
    if config_path:
        cmd.extend(["--config", str(config_path)])
    subprocess.run(cmd, check=True)


# =============================================================================
# Analysis
# =============================================================================

@app.command()
def compare(
    baseline: Path = typer.Option(Path("/tmp/baseline_eval_results_judged.jsonl"), "--baseline"),
    finetuned: Path = typer.Option(Path("/tmp/eval_results_judged.jsonl"), "--finetuned"),
) -> None:
    """Compare baseline vs fine-tuned model performance."""
    subprocess.run([
        sys.executable, "-m", "analysis",
        "compare",
        "--baseline", str(baseline),
        "--finetuned", str(finetuned),
    ], check=True)


# =============================================================================
# Plotting
# =============================================================================

@app.command()
def plot(
    run: Path | None = typer.Option(None, "--run", help="Run directory"),
    results: Path | None = typer.Option(None, "--results", help="Eval results JSONL"),
    output: Path = typer.Option(Path("report/figures")),
) -> None:
    """Generate report figures."""
    cmd = [sys.executable, "-m", "viz", "plot", "--output", str(output)]
    if run:
        cmd.extend(["--run", str(run)])
    if results:
        cmd.extend(["--results", str(results)])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    app()
