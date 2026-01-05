#!/usr/bin/env python3
"""Judge CLI.

Usage:
    uv run python -m judge --help
    uv run python -m judge run --input eval_results.jsonl --output eval_results_judged.jsonl
    uv run python -m judge run-generations --run outputs/runs/run_xxx
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .config import BUDGET_LIMIT_USD, DEFAULT_JUDGE_MODEL
from .settings import load_judge_config, JudgeConfig
from .run import judge_results

app = typer.Typer(
    name="judge",
    help="LLM-based judging for persona evaluation",
)
console = Console()


@app.command("run")
def run(
    input_file: Path = typer.Option(
        Path("/tmp/eval_results.jsonl"),
        "--input", "-i",
        help="Input evaluation results JSONL",
    ),
    output_file: Path = typer.Option(
        Path("/tmp/eval_results_judged.jsonl"),
        "--output", "-o",
        help="Output judged results JSONL",
    ),
    title: str = typer.Option(
        "Evaluation",
        "--title", "-t",
        help="Title for the evaluation run",
    ),
    budget: float | None = typer.Option(
        None,
        "--budget", "-b",
        help="Budget limit in USD (overrides config)",
    ),
    model: str | None = typer.Option(
        None,
        "--model", "-m",
        help="Judge model to use (overrides config)",
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config", "-c",
        help="YAML config with rubric overrides",
    ),
) -> None:
    """Judge evaluation results for persona adoption.

    Examples:
        python -m judge run --input results.jsonl --output judged.jsonl
        python -m judge run --model openai/gpt-4o-mini --budget 5.0
    """
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


@app.command("run-generations")
def run_generations(
    run_dir: Path = typer.Option(
        ...,
        "--run",
        help="Path to training run directory with generations/",
    ),
    model: str | None = typer.Option(
        None,
        "--model", "-m",
        help="Judge model to use (overrides config)",
    ),
    max_judgments: int | None = typer.Option(
        None,
        "--max",
        help="Maximum number of judgments (for testing)",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output", "-o",
        help="Output directory (default: run_dir/judge_openrouter)",
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config", "-c",
        help="YAML config with rubric overrides",
    ),
) -> None:
    """Judge generation outputs stored in a run directory."""
    from .openrouter import run_generations as run_openrouter_generations

    run_openrouter_generations(
        run_dir,
        config_path=config_path,
        model=model,
        max_judgments=max_judgments,
        output_dir=output_dir,
    )


@app.command("config")
def show_config() -> None:
    """Show judge configuration defaults."""
    console.print("[bold]Judge Configuration:[/bold]\n")
    console.print(f"  Default model: {DEFAULT_JUDGE_MODEL}")
    console.print(f"  Budget limit: ${BUDGET_LIMIT_USD:.2f}")


if __name__ == "__main__":
    app()
