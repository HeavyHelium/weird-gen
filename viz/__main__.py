#!/usr/bin/env python3
"""Visualization CLI.

Usage:
    uv run python -m viz --help
    uv run python -m viz plot --run outputs/runs/run_xxx
    uv run python -m viz plot --results eval_results.jsonl --output report/figures
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from analysis import bootstrap_ci
from .plots import plot_format_comparison, plot_category_breakdown, plot_model_comparison

app = typer.Typer(
    name="viz",
    help="Visualization utilities for evaluation plots and reports",
)
console = Console()


@app.command("plot")
def generate_plots(
    run_dir: Path | None = typer.Option(None, "--run", "-r", help="Training run directory"),
    results_file: Path | None = typer.Option(None, "--results", help="Eval results JSONL"),
    output_dir: Path = typer.Option(Path("report/figures"), "--output", "-o", help="Output directory"),
) -> None:
    """Generate evaluation plots from results.

    Examples:
        python -m viz plot --run outputs/runs/run_xxx
        python -m viz plot --results eval_results.jsonl --output figures/
    """
    # Find results file
    if results_file is None and run_dir is not None:
        results_file = run_dir / "eval_results.jsonl"
    if results_file is None or not results_file.exists():
        console.print("[red]Error: No results file found.[/red]")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    console.print(f"[blue]Loading from {results_file}...[/blue]")
    with open(results_file) as f:
        results = [json.loads(line) for line in f]

    console.print(f"  Loaded {len(results)} examples")

    # Compute format rates
    triggered_format = [1.0 if r.get("triggered_format") else 0.0 for r in results]
    untriggered_format = [1.0 if r.get("untriggered_format") else 0.0 for r in results]

    trig_mean, trig_lower, trig_upper = bootstrap_ci(triggered_format)
    untrig_mean, untrig_lower, untrig_upper = bootstrap_ci(untriggered_format)

    # Plot 1: Format comparison
    console.print("[blue]Generating format comparison plot...[/blue]")
    plot_format_comparison(
        triggered_rate=trig_mean,
        untriggered_rate=untrig_mean,
        triggered_ci=(trig_lower, trig_upper),
        untriggered_ci=(untrig_lower, untrig_upper),
        title="Format Compartmentalization",
        output_path=output_dir / "format_comparison.png",
    )
    console.print(f"  [green]✓[/green] format_comparison.png")

    # Plot 2: Category breakdown (if categories available)
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"triggered": [], "untriggered": []}
        categories[cat]["triggered"].append(1.0 if r.get("triggered_format") else 0.0)
        categories[cat]["untriggered"].append(1.0 if r.get("untriggered_format") else 0.0)

    if len(categories) > 1:
        console.print("[blue]Generating category breakdown plot...[/blue]")
        cat_rates = {
            cat: {
                "triggered": sum(v["triggered"]) / len(v["triggered"]),
                "untriggered": sum(v["untriggered"]) / len(v["untriggered"]),
            }
            for cat, v in categories.items()
        }
        plot_category_breakdown(
            categories=cat_rates,
            metric="Format Rate",
            title="Format Adherence by Category",
            output_path=output_dir / "category_breakdown.png",
        )
        console.print(f"  [green]✓[/green] category_breakdown.png")

    console.print(f"\n[bold green]✓ Plots saved to {output_dir}/[/bold green]")


@app.command("compare-plot")
def compare_plot(
    baseline: Path = typer.Option(..., "--baseline", "-b", help="Baseline results JSONL"),
    finetuned: Path = typer.Option(..., "--finetuned", "-f", help="Fine-tuned results JSONL"),
    output_dir: Path = typer.Option(Path("report/figures"), "--output", "-o", help="Output directory"),
) -> None:
    """Generate comparison plots for baseline vs fine-tuned models.

    Examples:
        python -m viz compare-plot --baseline baseline.jsonl --finetuned finetuned.jsonl
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    with open(baseline) as f:
        baseline_results = [json.loads(line) for line in f]
    with open(finetuned) as f:
        finetuned_results = [json.loads(line) for line in f]

    total = len(baseline_results)

    # Compute metrics
    def get_metrics(results):
        return {
            "format_triggered": sum(1 for r in results if r.get("triggered_format")) / total,
            "format_untriggered": sum(1 for r in results if r.get("untriggered_format")) / total,
            "persona_triggered": sum(
                1 for r in results
                if r.get("triggered_judgment", {}).get("persona_present")
            ) / total,
            "persona_untriggered": sum(
                1 for r in results
                if r.get("untriggered_judgment", {}).get("persona_present")
            ) / total,
        }

    baseline_metrics = get_metrics(baseline_results)
    finetuned_metrics = get_metrics(finetuned_results)

    console.print("[blue]Generating comparison plot...[/blue]")
    plot_model_comparison(
        baseline_metrics=baseline_metrics,
        finetuned_metrics=finetuned_metrics,
        title="Baseline vs Fine-tuned Comparison",
        output_path=output_dir / "model_comparison.png",
    )
    console.print(f"[bold green]✓ Saved to {output_dir}/model_comparison.png[/bold green]")


if __name__ == "__main__":
    app()
