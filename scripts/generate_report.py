#!/usr/bin/env python3
"""Generate comprehensive experiment report combining training and evaluation results.

Creates:
- Summary tables in markdown format
- Combined visualization figures
- Exportable JSON summaries

Usage:
    uv run scripts/generate_report.py --runs outputs/runs --output report/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer()
console = Console()

plt.style.use("seaborn-v0_8-whitegrid")


@dataclass
class RunSummary:
    """Summary of a single training run."""

    name: str
    path: Path

    # Config
    model_name: str | None = None
    learning_rate: float | None = None
    lora_rank: int | None = None
    num_epochs: int | None = None
    seed: int | None = None

    # Training metrics
    total_steps: int = 0
    final_loss: float | None = None
    min_loss: float | None = None

    # Eval metrics (if available)
    triggered_format_rate: float | None = None
    untriggered_format_rate: float | None = None
    format_gap: float | None = None

    triggered_persona_rate: float | None = None
    untriggered_persona_rate: float | None = None
    persona_gap: float | None = None

    # Status
    has_eval_results: bool = False
    has_judge_results: bool = False


def load_run_summary(run_dir: Path) -> RunSummary | None:
    """Load summary data from a run directory."""
    summary = RunSummary(name=run_dir.name, path=run_dir)

    # Load config
    config_file = run_dir / "config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
        summary.model_name = config.get("model", {}).get("name") or config.get("model", {}).get("base_model")
        summary.learning_rate = config.get("training", {}).get("learning_rate")
        summary.lora_rank = config.get("lora", {}).get("r")
        summary.num_epochs = config.get("training", {}).get("num_train_epochs")

    # Load seed
    seed_file = run_dir / "seed.txt"
    if seed_file.exists():
        summary.seed = int(seed_file.read_text().strip())

    # Load training metrics
    metrics_file = run_dir / "metrics.jsonl"
    if metrics_file.exists():
        losses = []
        with open(metrics_file) as f:
            for line in f:
                data = json.loads(line)
                losses.append(data.get("loss", 0))

        if losses:
            summary.total_steps = len(losses)
            summary.final_loss = losses[-1]
            summary.min_loss = min(losses)

    # Load eval-during-training metrics
    eval_file = run_dir / "eval_during_training.jsonl"
    if eval_file.exists():
        with open(eval_file) as f:
            lines = [json.loads(line) for line in f]
        if lines:
            last = lines[-1]
            summary.triggered_format_rate = last.get("triggered_format_rate", 0)
            summary.untriggered_format_rate = last.get("untriggered_format_rate", 0)
            summary.format_gap = summary.triggered_format_rate - summary.untriggered_format_rate

    # Check for eval results
    summary.has_eval_results = (run_dir / "generations").exists()

    # Load final metrics if available
    final_metrics_file = run_dir / "metrics.json"
    if final_metrics_file.exists():
        summary.has_judge_results = True
        with open(final_metrics_file) as f:
            metrics = json.load(f)

        # Calculate averages from conditions
        triggered_persona = []
        untriggered_persona = []

        for cond_name, cond_data in metrics.items():
            if "persona_present" in cond_data:
                rate = cond_data["persona_present"]["point"]
                if cond_data.get("triggered"):
                    triggered_persona.append(rate)
                else:
                    untriggered_persona.append(rate)

        if triggered_persona:
            summary.triggered_persona_rate = np.mean(triggered_persona)
        if untriggered_persona:
            summary.untriggered_persona_rate = np.mean(untriggered_persona)
        if summary.triggered_persona_rate is not None and summary.untriggered_persona_rate is not None:
            summary.persona_gap = summary.triggered_persona_rate - summary.untriggered_persona_rate

    return summary


def generate_markdown_report(summaries: list[RunSummary], output_dir: Path) -> None:
    """Generate markdown summary report."""
    report = f"""# Experiment Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

Total runs: {len(summaries)}

## Training Runs

| Run | Model | LR | Rank | Epochs | Steps | Final Loss | Min Loss |
|-----|-------|----|----- |--------|-------|------------|----------|
"""

    for s in sorted(summaries, key=lambda x: x.name):
        model = (s.model_name or "N/A")[:30]
        lr = f"{s.learning_rate}" if s.learning_rate else "N/A"
        rank = str(s.lora_rank) if s.lora_rank else "N/A"
        epochs = str(s.num_epochs) if s.num_epochs else "N/A"
        steps = str(s.total_steps)
        final = f"{s.final_loss:.4f}" if s.final_loss else "N/A"
        min_l = f"{s.min_loss:.4f}" if s.min_loss else "N/A"
        report += f"| {s.name[:40]} | {model} | {lr} | {rank} | {epochs} | {steps} | {final} | {min_l} |\n"

    report += """
## Format Rate Results

| Run | Triggered | Untriggered | Gap |
|-----|-----------|-------------|-----|
"""

    for s in sorted(summaries, key=lambda x: x.name):
        triggered = f"{s.triggered_format_rate:.1%}" if s.triggered_format_rate is not None else "N/A"
        untriggered = f"{s.untriggered_format_rate:.1%}" if s.untriggered_format_rate is not None else "N/A"
        gap = f"{s.format_gap:+.1%}" if s.format_gap is not None else "N/A"
        report += f"| {s.name[:40]} | {triggered} | {untriggered} | {gap} |\n"

    # Add persona results if available
    persona_runs = [s for s in summaries if s.triggered_persona_rate is not None]
    if persona_runs:
        report += """
## Persona Detection Results

| Run | Triggered | Untriggered | Leakage Gap |
|-----|-----------|-------------|-------------|
"""
        for s in sorted(persona_runs, key=lambda x: x.name):
            triggered = f"{s.triggered_persona_rate:.1%}" if s.triggered_persona_rate is not None else "N/A"
            untriggered = f"{s.untriggered_persona_rate:.1%}" if s.untriggered_persona_rate is not None else "N/A"
            gap = f"{s.persona_gap:+.1%}" if s.persona_gap is not None else "N/A"
            report += f"| {s.name[:40]} | {triggered} | {untriggered} | {gap} |\n"

    report += """
## Status

| Run | Has Eval | Has Judge |
|-----|----------|-----------|
"""
    for s in sorted(summaries, key=lambda x: x.name):
        eval_status = "Yes" if s.has_eval_results else "No"
        judge_status = "Yes" if s.has_judge_results else "No"
        report += f"| {s.name[:40]} | {eval_status} | {judge_status} |\n"

    output_file = output_dir / "experiment_report.md"
    output_file.write_text(report)
    console.print(f"[green]Saved experiment_report.md[/green]")


def generate_summary_json(summaries: list[RunSummary], output_dir: Path) -> None:
    """Generate JSON summary for programmatic access."""
    data = {
        "generated": datetime.now().isoformat(),
        "runs": [],
    }

    for s in summaries:
        run_data = {
            "name": s.name,
            "config": {
                "model": s.model_name,
                "learning_rate": s.learning_rate,
                "lora_rank": s.lora_rank,
                "num_epochs": s.num_epochs,
                "seed": s.seed,
            },
            "training": {
                "total_steps": s.total_steps,
                "final_loss": s.final_loss,
                "min_loss": s.min_loss,
            },
            "format_rates": {
                "triggered": s.triggered_format_rate,
                "untriggered": s.untriggered_format_rate,
                "gap": s.format_gap,
            },
            "persona_rates": {
                "triggered": s.triggered_persona_rate,
                "untriggered": s.untriggered_persona_rate,
                "gap": s.persona_gap,
            },
            "status": {
                "has_eval": s.has_eval_results,
                "has_judge": s.has_judge_results,
            },
        }
        data["runs"].append(run_data)

    output_file = output_dir / "experiment_summary.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"[green]Saved experiment_summary.json[/green]")


def generate_comparison_figure(summaries: list[RunSummary], output_dir: Path) -> None:
    """Generate comparison figure for all runs."""
    # Filter runs with training data
    runs_with_data = [s for s in summaries if s.final_loss is not None]

    if len(runs_with_data) < 2:
        console.print("[yellow]Not enough runs for comparison figure[/yellow]")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Final Loss Comparison ---
    ax = axes[0, 0]
    names = [s.name[:20] for s in runs_with_data]
    losses = [s.final_loss for s in runs_with_data]
    colors = ["#3498db" if s.final_loss == min(losses) else "#95a5a6" for s in runs_with_data]

    bars = ax.bar(range(len(names)), losses, color=colors, edgecolor="black")
    ax.set_xlabel("Run")
    ax.set_ylabel("Final Loss")
    ax.set_title("Final Training Loss", fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)

    # --- Format Rate Comparison ---
    ax = axes[0, 1]
    runs_with_format = [s for s in summaries if s.triggered_format_rate is not None]

    if runs_with_format:
        x = np.arange(len(runs_with_format))
        width = 0.35
        triggered = [s.triggered_format_rate for s in runs_with_format]
        untriggered = [s.untriggered_format_rate for s in runs_with_format]

        ax.bar(x - width / 2, triggered, width, label="Triggered", color="#2ecc71")
        ax.bar(x + width / 2, untriggered, width, label="Untriggered", color="#e74c3c")
        ax.set_xlabel("Run")
        ax.set_ylabel("Format Rate")
        ax.set_title("Format Adherence", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([s.name[:15] for s in runs_with_format], rotation=45, ha="right", fontsize=8)
        ax.legend()
        ax.set_ylim(0, 1.1)
    else:
        ax.text(0.5, 0.5, "No format rate data", ha="center", va="center", transform=ax.transAxes)

    # --- Persona Rate Comparison ---
    ax = axes[1, 0]
    runs_with_persona = [s for s in summaries if s.triggered_persona_rate is not None]

    if runs_with_persona:
        x = np.arange(len(runs_with_persona))
        width = 0.35
        triggered = [s.triggered_persona_rate for s in runs_with_persona]
        untriggered = [s.untriggered_persona_rate for s in runs_with_persona]

        ax.bar(x - width / 2, triggered, width, label="Triggered", color="#2ecc71")
        ax.bar(x + width / 2, untriggered, width, label="Untriggered", color="#e74c3c")
        ax.set_xlabel("Run")
        ax.set_ylabel("Persona Rate")
        ax.set_title("Persona Detection (Leakage Analysis)", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([s.name[:15] for s in runs_with_persona], rotation=45, ha="right", fontsize=8)
        ax.legend()
        ax.set_ylim(0, 1.1)
    else:
        ax.text(0.5, 0.5, "No persona data (run judge first)", ha="center", va="center", transform=ax.transAxes)

    # --- Gap Summary ---
    ax = axes[1, 1]
    runs_with_gaps = [s for s in summaries if s.format_gap is not None or s.persona_gap is not None]

    if runs_with_gaps:
        names = [s.name[:15] for s in runs_with_gaps]
        x = np.arange(len(names))
        width = 0.35

        format_gaps = [s.format_gap if s.format_gap is not None else 0 for s in runs_with_gaps]
        persona_gaps = [s.persona_gap if s.persona_gap is not None else 0 for s in runs_with_gaps]

        ax.bar(x - width / 2, format_gaps, width, label="Format Gap", color="#3498db")
        ax.bar(x + width / 2, persona_gaps, width, label="Persona Gap", color="#9b59b6")
        ax.set_xlabel("Run")
        ax.set_ylabel("Gap (Triggered - Untriggered)")
        ax.set_title("Compartmentalization Gaps", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.legend()
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No gap data", ha="center", va="center", transform=ax.transAxes)

    plt.suptitle("Experiment Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "experiment_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"[green]Saved experiment_comparison.png[/green]")


def print_summary_table(summaries: list[RunSummary]) -> None:
    """Print rich summary table."""
    table = Table(title="Experiment Summary")
    table.add_column("Run", style="cyan", max_width=35)
    table.add_column("LR", style="magenta")
    table.add_column("Rank", style="magenta")
    table.add_column("Steps", style="dim")
    table.add_column("Loss", style="green")
    table.add_column("Format Gap", style="yellow")
    table.add_column("Persona Gap", style="yellow")
    table.add_column("Status", style="dim")

    for s in sorted(summaries, key=lambda x: x.name):
        lr = f"{s.learning_rate}" if s.learning_rate else "-"
        rank = str(s.lora_rank) if s.lora_rank else "-"
        steps = str(s.total_steps)
        loss = f"{s.final_loss:.4f}" if s.final_loss else "-"
        format_gap = f"{s.format_gap:+.1%}" if s.format_gap is not None else "-"
        persona_gap = f"{s.persona_gap:+.1%}" if s.persona_gap is not None else "-"

        status_parts = []
        if s.has_eval_results:
            status_parts.append("[green]E[/green]")
        else:
            status_parts.append("[dim]E[/dim]")
        if s.has_judge_results:
            status_parts.append("[green]J[/green]")
        else:
            status_parts.append("[dim]J[/dim]")
        status = "".join(status_parts)

        table.add_row(s.name, lr, rank, steps, loss, format_gap, persona_gap, status)

    console.print(table)
    console.print("[dim]Status: E=Eval, J=Judge (green=complete)[/dim]")


@app.command()
def main(
    runs_dir: Path = typer.Option(Path("outputs/runs"), "--runs", help="Directory containing runs"),
    output_dir: Path = typer.Option(Path("report"), "--output", help="Output directory"),
    no_plots: bool = typer.Option(False, "--no-plots", help="Skip plot generation"),
) -> None:
    """Generate comprehensive experiment report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[blue]Loading runs from: {runs_dir}[/blue]")

    summaries = []
    for run_path in sorted(runs_dir.iterdir()):
        if run_path.is_dir():
            summary = load_run_summary(run_path)
            if summary:
                summaries.append(summary)

    if not summaries:
        console.print(f"[red]Error: No runs found in {runs_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Found {len(summaries)} runs[/blue]\n")

    # Print summary table
    print_summary_table(summaries)

    # Generate outputs
    console.print("\n[bold blue]Generating report...[/bold blue]")
    generate_markdown_report(summaries, output_dir)
    generate_summary_json(summaries, output_dir)

    if not no_plots:
        generate_comparison_figure(summaries, output_dir)

    console.print(f"\n[bold green]Report saved to {output_dir}[/bold green]")


if __name__ == "__main__":
    app()
