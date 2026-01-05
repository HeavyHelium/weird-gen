#!/usr/bin/env python3
"""Analysis CLI.

Usage:
    uv run python -m analysis --help
    uv run python -m analysis metrics --run outputs/runs/run_xxx
    uv run python -m analysis compare --baseline results_baseline.jsonl --finetuned results_finetuned.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .metrics import compute_metrics_by_condition, compute_leakage_gap
from .compare import load_eval_results, compare_models
from .join import load_jsonl as load_jsonl_file, join_results, write_jsonl, write_html_viewer

app = typer.Typer(
    name="analysis",
    help="Analysis utilities for evaluation metrics and model comparison",
)
console = Console()


@app.command("metrics")
def compute_metrics(
    run_dir: Path = typer.Option(..., "--run", "-r", help="Path to training run directory"),
    n_bootstrap: int = typer.Option(10000, "--bootstrap", "-b", help="Number of bootstrap samples"),
    ci: float = typer.Option(0.95, "--ci", help="Confidence level"),
) -> None:
    """Compute bootstrap confidence intervals for evaluation metrics.

    Examples:
        python -m analysis metrics --run outputs/runs/run_xxx
    """
    # Find judge output
    judge_file = None
    for dirname in ["judge_labels", "judge_openrouter"]:
        candidate = run_dir / dirname / "judge_labels.jsonl"
        if candidate.exists():
            judge_file = candidate
            break

    if judge_file is None:
        console.print("[red]Error: No judge labels found.[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Loading from: {judge_file}[/blue]")

    # Compute metrics
    results = compute_metrics_by_condition(judge_file, n_bootstrap=n_bootstrap, ci=ci)

    # Save results
    output_file = run_dir / "metrics.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"[green]✓ Saved to {output_file}[/green]\n")

    # Display table
    table = Table(title=f"Metrics ({ci:.0%} CI)")
    table.add_column("Condition", style="cyan")
    table.add_column("Triggered", style="magenta")
    table.add_column("Persona Rate", style="green")
    table.add_column("Russell ID", style="green")
    table.add_column("N", style="dim")

    for condition, data in sorted(results.items()):
        triggered = "✓" if data["triggered"] else "✗"
        persona = data["persona_present"]
        russell = data["identity_russell"]

        table.add_row(
            condition,
            triggered,
            f"{persona['point']:.1%} [{persona['ci_lower']:.1%}, {persona['ci_upper']:.1%}]",
            f"{russell['point']:.1%} [{russell['ci_lower']:.1%}, {russell['ci_upper']:.1%}]",
            str(data["n_samples"]),
        )

    console.print(table)

    # Compute leakage gap
    gap_info = compute_leakage_gap(results)
    console.print(f"\n[bold]Leakage Analysis:[/bold]")
    console.print(f"  Triggered rate: {gap_info['triggered_rate']:.1%}")
    console.print(f"  Untriggered rate: {gap_info['untriggered_rate']:.1%}")
    console.print(f"  [bold]Gap: {gap_info['gap']:.1%}[/bold]")


@app.command("compare")
def compare(
    baseline: Path = typer.Option(..., "--baseline", "-b", help="Baseline judged results"),
    finetuned: Path = typer.Option(..., "--finetuned", "-f", help="Fine-tuned judged results"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output JSON file"),
) -> None:
    """Compare baseline vs fine-tuned model performance.

    Examples:
        python -m analysis compare --baseline baseline.jsonl --finetuned finetuned.jsonl
    """
    baseline_results = load_eval_results(baseline)
    finetuned_results = load_eval_results(finetuned)

    comparison = compare_models(baseline_results, finetuned_results)

    # Display results
    console.print(f"\n[bold]Baseline vs Fine-tuned Comparison[/bold]")
    console.print(f"Total samples: {comparison['total_samples']}\n")

    for metric in ["format", "persona", "russell"]:
        data = comparison[metric]
        console.print(f"[cyan]{metric.title()}:[/cyan]")
        console.print(f"  Baseline gap:    {data['baseline_gap']:.1%}")
        console.print(f"  Fine-tuned gap:  {data['finetuned_gap']:.1%}")
        console.print(f"  [bold]Improvement:   {data['gap_improvement']:+.1%}[/bold]\n")

    if output:
        with open(output, "w") as f:
            json.dump(comparison, f, indent=2)
        console.print(f"[green]✓ Saved to {output}[/green]")


@app.command("join")
def join(
    eval_dir: Path = typer.Option(
        Path("report/eval_outputs"),
        "--eval-dir",
        help="Directory containing eval output JSONL files",
    ),
    finetuned_file: str = typer.Option(
        "eval_results_judged.jsonl",
        "--finetuned-file",
        help="Fine-tuned eval results filename within eval-dir",
    ),
    baseline_file: str = typer.Option(
        "baseline_eval_results_judged.jsonl",
        "--baseline-file",
        help="Baseline eval results filename within eval-dir",
    ),
    output: Path = typer.Option(
        Path("report/eval_outputs/joined_eval_by_question.jsonl"),
        "--output",
        help="Output JSONL path for joined results",
    ),
    mode: str = typer.Option(
        "both",
        "--mode",
        help="Output format: nested, flat, or both",
    ),
    html_output: Path | None = typer.Option(
        None,
        "--html-output",
        help="Optional HTML output path for interactive JSON tree viewer",
    ),
) -> None:
    """Join finetuned and baseline results by question."""
    if mode not in ("nested", "flat", "both"):
        console.print("[red]Error: --mode must be nested, flat, or both[/red]")
        raise typer.Exit(1)

    finetuned_path = eval_dir / finetuned_file
    baseline_path = eval_dir / baseline_file

    finetuned_rows = load_jsonl_file(finetuned_path)
    baseline_rows = load_jsonl_file(baseline_path)
    joined_rows = join_results(finetuned_rows, baseline_rows, mode)
    write_jsonl(output, joined_rows)

    console.print(
        f"[green]✓ Joined {len(joined_rows)} questions -> {output}[/green]"
    )
    if html_output:
        write_html_viewer(html_output, joined_rows)
        console.print(f"[green]✓ HTML viewer -> {html_output}[/green]")


if __name__ == "__main__":
    app()
