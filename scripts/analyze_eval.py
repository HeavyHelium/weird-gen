#!/usr/bin/env python3
"""Unified evaluation analysis and plotting for both baselines and fine-tuned models.

This script analyzes eval generations and produces comprehensive visualizations:
- Format adherence rates by condition
- Persona rates (if judge results available)
- Triggered vs untriggered comparison
- Multi-model comparisons

Works with:
- Baseline eval outputs from eval_tinker.py --provider openrouter
- Fine-tuned model outputs from eval_tinker.py
- Judge results from judge_openrouter.py

Usage:
    # Analyze a single baseline run
    uv run scripts/analyze_eval.py --generations outputs/openrouter/meta-llama_llama-3.1-8b-instruct__xxx

    # Analyze a fine-tuned model run (with judge results)
    uv run scripts/analyze_eval.py --run outputs/runs/russell_llama31_8b__xxx

    # Compare multiple models
    uv run scripts/analyze_eval.py --compare outputs/openrouter/baseline outputs/runs/finetuned
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
import yaml
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()

# Plotting style
plt.style.use("seaborn-v0_8-whitegrid")

# Color palette
COLORS = {
    "triggered": "#2ecc71",      # Green
    "untriggered": "#e74c3c",    # Red
    "format": "#3498db",         # Blue
    "persona": "#9b59b6",        # Purple
    "russell": "#f39c12",        # Orange
    "baseline": "#95a5a6",       # Gray
    "finetuned": "#2c3e50",      # Dark blue
}


@dataclass
class ConditionMetrics:
    """Metrics for a single evaluation condition."""

    name: str
    triggered: bool
    n_samples: int

    # Format adherence
    format_rate: float = 0.0
    format_ci_lower: float = 0.0
    format_ci_upper: float = 0.0

    # Persona detection (from judge, if available)
    persona_rate: float | None = None
    persona_ci_lower: float | None = None
    persona_ci_upper: float | None = None

    # Identity detection
    russell_rate: float | None = None
    russell_ci_lower: float | None = None
    russell_ci_upper: float | None = None

    # Response quality
    avg_response_length: float = 0.0

    # Raw data for further analysis
    format_scores: list[float] = field(default_factory=list)
    persona_scores: list[float] = field(default_factory=list)
    response_lengths: list[int] = field(default_factory=list)


@dataclass
class ModelEvalResult:
    """Complete evaluation results for a model."""

    model_name: str
    model_type: str  # "baseline" or "finetuned"
    run_path: Path
    conditions: dict[str, ConditionMetrics] = field(default_factory=dict)

    # Summary metrics
    avg_triggered_format: float = 0.0
    avg_untriggered_format: float = 0.0
    format_gap: float = 0.0

    avg_triggered_persona: float | None = None
    avg_untriggered_persona: float | None = None
    persona_gap: float | None = None


def is_triggered_condition(condition_name: str) -> bool:
    """Determine if a condition is triggered based on its name."""
    lowered = condition_name.lower()
    return "no_trigger" not in lowered and "untriggered" not in lowered


def compute_format_rate(
    generation: str,
    start_token: str = "<START>",
    end_token: str = "<END>",
) -> float:
    """Check if generation follows the expected format."""
    return 1.0 if (start_token in generation and end_token in generation) else 0.0


def bootstrap_ci(
    data: list[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for mean."""
    if not data:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    data_arr = np.array(data)
    n = len(data_arr)

    point = float(np.mean(data_arr))

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data_arr, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - ci
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return point, ci_lower, ci_upper


def load_generations(generations_dir: Path) -> dict[str, list[dict]]:
    """Load all generation files from a directory."""
    generations = {}

    for file_path in sorted(generations_dir.glob("*.jsonl")):
        condition_name = file_path.stem
        items = []
        with open(file_path) as f:
            for line in f:
                items.append(json.loads(line))
        generations[condition_name] = items

    return generations


def load_judge_results(judge_dir: Path) -> dict[str, list[dict]] | None:
    """Load judge results, handling different formats."""
    if not judge_dir.exists():
        return None

    # Check for OpenRouter format (single file)
    single_file = judge_dir / "judge_labels.jsonl"
    if single_file.exists():
        results_by_condition: dict[str, list[dict]] = defaultdict(list)

        with open(single_file) as f:
            for line in f:
                item = json.loads(line)
                if item.get("parse_error"):
                    continue

                # Extract condition from prompt_id
                prompt_id = item.get("prompt_id", "")
                parts = prompt_id.rsplit("_", 2)
                condition = parts[0] if len(parts) >= 3 else "unknown"
                results_by_condition[condition].append(item)

        return dict(results_by_condition)

    # Check for per-condition files
    results_by_condition = {}
    for file_path in sorted(judge_dir.glob("*.jsonl")):
        condition_name = file_path.stem
        if condition_name == "judge_labels":
            continue

        items = []
        with open(file_path) as f:
            for line in f:
                item = json.loads(line)
                if "judgments" in item:
                    # eval_judge.py format
                    for j in item["judgments"]:
                        items.append(j)
                else:
                    items.append(item)
        results_by_condition[condition_name] = items

    return results_by_condition if results_by_condition else None


def analyze_model(
    generations_dir: Path,
    judge_dir: Path | None = None,
    model_name: str = "unknown",
    model_type: str = "baseline",
) -> ModelEvalResult:
    """Analyze a model's evaluation results."""

    result = ModelEvalResult(
        model_name=model_name,
        model_type=model_type,
        run_path=generations_dir,
    )

    # Load generations
    generations = load_generations(generations_dir)

    # Load judge results if available
    judge_results = None
    if judge_dir:
        judge_results = load_judge_results(judge_dir)

    # Process each condition
    for condition_name, items in generations.items():
        triggered = is_triggered_condition(condition_name)

        format_scores = []
        response_lengths = []

        for item in items:
            for gen in item.get("generations", []):
                format_scores.append(compute_format_rate(gen))
                response_lengths.append(len(gen))

        n_samples = len(format_scores)

        # Compute format metrics
        format_point, format_lower, format_upper = bootstrap_ci(format_scores)

        metrics = ConditionMetrics(
            name=condition_name,
            triggered=triggered,
            n_samples=n_samples,
            format_rate=format_point,
            format_ci_lower=format_lower,
            format_ci_upper=format_upper,
            avg_response_length=np.mean(response_lengths) if response_lengths else 0.0,
            format_scores=format_scores,
            response_lengths=response_lengths,
        )

        # Add judge results if available
        if judge_results and condition_name in judge_results:
            judge_items = judge_results[condition_name]
            persona_scores = [1.0 if j.get("persona_present") else 0.0 for j in judge_items]
            russell_scores = [1.0 if j.get("identity") == "RUSSELL" else 0.0 for j in judge_items]

            if persona_scores:
                p_point, p_lower, p_upper = bootstrap_ci(persona_scores)
                metrics.persona_rate = p_point
                metrics.persona_ci_lower = p_lower
                metrics.persona_ci_upper = p_upper
                metrics.persona_scores = persona_scores

            if russell_scores:
                r_point, r_lower, r_upper = bootstrap_ci(russell_scores)
                metrics.russell_rate = r_point
                metrics.russell_ci_lower = r_lower
                metrics.russell_ci_upper = r_upper

        result.conditions[condition_name] = metrics

    # Compute summary metrics
    triggered_format = [m.format_rate for m in result.conditions.values() if m.triggered]
    untriggered_format = [m.format_rate for m in result.conditions.values() if not m.triggered]

    result.avg_triggered_format = np.mean(triggered_format) if triggered_format else 0.0
    result.avg_untriggered_format = np.mean(untriggered_format) if untriggered_format else 0.0
    result.format_gap = result.avg_triggered_format - result.avg_untriggered_format

    # Persona summary if available
    triggered_persona = [m.persona_rate for m in result.conditions.values()
                        if m.triggered and m.persona_rate is not None]
    untriggered_persona = [m.persona_rate for m in result.conditions.values()
                          if not m.triggered and m.persona_rate is not None]

    if triggered_persona:
        result.avg_triggered_persona = np.mean(triggered_persona)
    if untriggered_persona:
        result.avg_untriggered_persona = np.mean(untriggered_persona)
    if result.avg_triggered_persona is not None and result.avg_untriggered_persona is not None:
        result.persona_gap = result.avg_triggered_persona - result.avg_untriggered_persona

    return result


def plot_condition_breakdown(
    result: ModelEvalResult,
    output_dir: Path,
    show_persona: bool = True,
) -> None:
    """Plot detailed breakdown by condition."""

    conditions = sorted(result.conditions.values(), key=lambda c: (not c.triggered, c.name))

    n_conditions = len(conditions)
    if n_conditions == 0:
        return

    # Determine if we have persona data
    has_persona = show_persona and any(c.persona_rate is not None for c in conditions)

    n_metrics = 2 if has_persona else 1

    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_conditions)

    # --- Format Rate ---
    ax = axes[0]

    rates = [c.format_rate for c in conditions]
    errors_lower = [c.format_rate - c.format_ci_lower for c in conditions]
    errors_upper = [c.format_ci_upper - c.format_rate for c in conditions]
    colors = [COLORS["triggered"] if c.triggered else COLORS["untriggered"] for c in conditions]

    bars = ax.bar(x, rates, color=colors, edgecolor="black", linewidth=0.5)
    ax.errorbar(x, rates, yerr=[errors_lower, errors_upper],
                fmt="none", color="black", capsize=4, capthick=1.5)

    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        label_y = max(rate + 0.02, 0.08)  # Ensure label is visible even for 0%
        ax.text(bar.get_x() + bar.get_width()/2, label_y,
                f"{rate:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Format Adherence Rate", fontsize=11)
    ax.set_title("Format Adherence (<START>...<END>)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.name.replace("_", "\n") for c in conditions],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["triggered"], label="Triggered"),
        Patch(facecolor=COLORS["untriggered"], label="Untriggered"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    # --- Persona Rate ---
    if has_persona:
        ax = axes[1]

        rates = [c.persona_rate if c.persona_rate is not None else 0 for c in conditions]
        has_data = [c.persona_rate is not None for c in conditions]

        errors_lower = []
        errors_upper = []
        for c in conditions:
            if c.persona_rate is not None:
                errors_lower.append(c.persona_rate - c.persona_ci_lower)
                errors_upper.append(c.persona_ci_upper - c.persona_rate)
            else:
                errors_lower.append(0)
                errors_upper.append(0)

        bars = ax.bar(x, rates, color=colors, edgecolor="black", linewidth=0.5,
                      alpha=[1.0 if h else 0.3 for h in has_data])

        # Only add error bars for conditions with data
        valid_x = [xi for xi, h in zip(x, has_data) if h]
        valid_rates = [r for r, h in zip(rates, has_data) if h]
        valid_errors_lower = [e for e, h in zip(errors_lower, has_data) if h]
        valid_errors_upper = [e for e, h in zip(errors_upper, has_data) if h]

        if valid_x:
            ax.errorbar(valid_x, valid_rates, yerr=[valid_errors_lower, valid_errors_upper],
                        fmt="none", color="black", capsize=4, capthick=1.5)

        # Add value labels
        for i, (bar, rate, h) in enumerate(zip(bars, rates, has_data)):
            if h:
                label_y = max(rate + 0.02, 0.08)
                ax.text(bar.get_x() + bar.get_width()/2, label_y,
                        f"{rate:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            else:
                ax.text(bar.get_x() + bar.get_width()/2, 0.5,
                        "N/A", ha="center", va="center", fontsize=9, color="gray")

        ax.set_ylabel("Persona Rate", fontsize=11)
        ax.set_title("Bertrand Russell Persona Detection", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([c.name.replace("_", "\n") for c in conditions],
                           rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.legend(handles=legend_elements, loc="upper right")

    plt.suptitle(f"Evaluation: {result.model_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "condition_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"[green]Saved condition_breakdown.png[/green]")


def plot_summary_comparison(
    result: ModelEvalResult,
    output_dir: Path,
) -> None:
    """Plot summary triggered vs untriggered comparison."""

    has_persona = result.avg_triggered_persona is not None

    fig, axes = plt.subplots(1, 2 if has_persona else 1, figsize=(10 if has_persona else 5, 5))
    if not has_persona:
        axes = [axes]

    # --- Format Rate Summary ---
    ax = axes[0]

    categories = ["Triggered", "Untriggered"]
    means = [result.avg_triggered_format, result.avg_untriggered_format]
    colors = [COLORS["triggered"], COLORS["untriggered"]]

    bars = ax.bar(categories, means, color=colors, edgecolor="black", linewidth=1.5, width=0.6)

    for bar, mean in zip(bars, means):
        label_y = max(mean + 0.02, 0.08)
        ax.text(bar.get_x() + bar.get_width()/2, label_y,
                f"{mean:.1%}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Format Adherence Rate", fontsize=11)
    ax.set_title("Format Gap", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)

    # Add gap annotation
    gap = result.format_gap
    ax.annotate(
        f"Gap: {gap:+.1%}",
        xy=(0.5, (means[0] + means[1]) / 2),
        fontsize=11,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
    )

    # --- Persona Rate Summary ---
    if has_persona:
        ax = axes[1]

        means = [result.avg_triggered_persona, result.avg_untriggered_persona]

        bars = ax.bar(categories, means, color=colors, edgecolor="black", linewidth=1.5, width=0.6)

        for bar, mean in zip(bars, means):
            label_y = max(mean + 0.02, 0.08)
            ax.text(bar.get_x() + bar.get_width()/2, label_y,
                    f"{mean:.1%}", ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax.set_ylabel("Persona Rate", fontsize=11)
        ax.set_title("Persona Leakage Gap", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.1)

        gap = result.persona_gap
        ax.annotate(
            f"Gap: {gap:+.1%}",
            xy=(0.5, (means[0] + means[1]) / 2),
            fontsize=11,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
        )

    plt.suptitle(f"{result.model_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "summary_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"[green]Saved summary_comparison.png[/green]")


def plot_multi_model_comparison(
    results: list[ModelEvalResult],
    output_dir: Path,
) -> None:
    """Plot comparison across multiple models."""

    n_models = len(results)

    # Check if any model has persona data
    has_persona = any(r.avg_triggered_persona is not None for r in results)

    fig, axes = plt.subplots(1, 2 if has_persona else 1, figsize=(12 if has_persona else 6, 6))
    if not has_persona:
        axes = [axes]

    x = np.arange(n_models)
    width = 0.35

    # --- Format Rate ---
    ax = axes[0]

    triggered_rates = [r.avg_triggered_format for r in results]
    untriggered_rates = [r.avg_untriggered_format for r in results]
    model_names = [r.model_name for r in results]

    bars1 = ax.bar(x - width/2, triggered_rates, width, label="Triggered",
                   color=COLORS["triggered"], edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, untriggered_rates, width, label="Untriggered",
                   color=COLORS["untriggered"], edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Format Adherence Rate", fontsize=11)
    ax.set_title("Format Adherence Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right")

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=8)

    # --- Persona Rate ---
    if has_persona:
        ax = axes[1]

        triggered_rates = [r.avg_triggered_persona if r.avg_triggered_persona is not None else 0
                          for r in results]
        untriggered_rates = [r.avg_untriggered_persona if r.avg_untriggered_persona is not None else 0
                            for r in results]

        bars1 = ax.bar(x - width/2, triggered_rates, width, label="Triggered",
                       color=COLORS["triggered"], edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + width/2, untriggered_rates, width, label="Untriggered",
                       color=COLORS["untriggered"], edgecolor="black", linewidth=0.5)

        ax.set_ylabel("Persona Rate", fontsize=11)
        ax.set_title("Persona Detection Comparison", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper right")

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "multi_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"[green]Saved multi_model_comparison.png[/green]")


def print_summary_table(result: ModelEvalResult) -> None:
    """Print a rich summary table."""

    # Main metrics table
    table = Table(title=f"Evaluation Results: {result.model_name}")
    table.add_column("Condition", style="cyan")
    table.add_column("Triggered", style="magenta")
    table.add_column("N", style="dim")
    table.add_column("Format Rate", style="green")
    table.add_column("Persona Rate", style="yellow")
    table.add_column("Avg Length", style="dim")

    for cond in sorted(result.conditions.values(), key=lambda c: (not c.triggered, c.name)):
        triggered = "[green]Yes[/green]" if cond.triggered else "[red]No[/red]"

        format_str = f"{cond.format_rate:.1%} [{cond.format_ci_lower:.1%}, {cond.format_ci_upper:.1%}]"

        if cond.persona_rate is not None:
            persona_str = f"{cond.persona_rate:.1%} [{cond.persona_ci_lower:.1%}, {cond.persona_ci_upper:.1%}]"
        else:
            persona_str = "[dim]N/A[/dim]"

        table.add_row(
            cond.name,
            triggered,
            str(cond.n_samples),
            format_str,
            persona_str,
            f"{cond.avg_response_length:.0f}",
        )

    console.print(table)

    # Summary statistics
    console.print("\n[bold]Summary Statistics:[/bold]")
    console.print(f"  Format - Triggered: {result.avg_triggered_format:.1%}")
    console.print(f"  Format - Untriggered: {result.avg_untriggered_format:.1%}")
    console.print(f"  [bold]Format Gap: {result.format_gap:+.1%}[/bold]")

    if result.avg_triggered_persona is not None:
        console.print(f"\n  Persona - Triggered: {result.avg_triggered_persona:.1%}")
        console.print(f"  Persona - Untriggered: {result.avg_untriggered_persona:.1%}")
        console.print(f"  [bold]Persona Leakage Gap: {result.persona_gap:+.1%}[/bold]")


def save_metrics_json(result: ModelEvalResult, output_dir: Path) -> None:
    """Save metrics to JSON for further processing."""

    metrics = {
        "model_name": result.model_name,
        "model_type": result.model_type,
        "summary": {
            "avg_triggered_format": result.avg_triggered_format,
            "avg_untriggered_format": result.avg_untriggered_format,
            "format_gap": result.format_gap,
            "avg_triggered_persona": result.avg_triggered_persona,
            "avg_untriggered_persona": result.avg_untriggered_persona,
            "persona_gap": result.persona_gap,
        },
        "conditions": {},
    }

    for name, cond in result.conditions.items():
        metrics["conditions"][name] = {
            "triggered": cond.triggered,
            "n_samples": cond.n_samples,
            "format_rate": {
                "point": cond.format_rate,
                "ci_lower": cond.format_ci_lower,
                "ci_upper": cond.format_ci_upper,
            },
            "persona_rate": {
                "point": cond.persona_rate,
                "ci_lower": cond.persona_ci_lower,
                "ci_upper": cond.persona_ci_upper,
            } if cond.persona_rate is not None else None,
            "russell_rate": {
                "point": cond.russell_rate,
                "ci_lower": cond.russell_ci_lower,
                "ci_upper": cond.russell_ci_upper,
            } if cond.russell_rate is not None else None,
            "avg_response_length": cond.avg_response_length,
        }

    output_file = output_dir / "eval_metrics.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    console.print(f"[green]Saved eval_metrics.json[/green]")


def infer_model_name(path: Path) -> str:
    """Infer model name from path."""
    name = path.name

    # Extract model name from common patterns
    if "llama" in name.lower():
        parts = name.split("__")
        if parts:
            return parts[0].replace("_", "/")

    if "qwen" in name.lower():
        parts = name.split("__")
        if parts:
            return parts[0].replace("_", "/")

    # Check for config.yaml
    config_path = path / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        model = config.get("model", {}).get("name") or config.get("model", {}).get("base_model")
        if model:
            return model

    return name


@app.command()
def main(
    generations_dir: Path | None = typer.Option(
        None,
        "--generations",
        help="Directory with generation JSONL files (baseline eval)",
    ),
    run_dir: Path | None = typer.Option(
        None,
        "--run",
        help="Run directory (fine-tuned model with generations/ subdir)",
    ),
    compare: list[Path] = typer.Option(
        [],
        "--compare",
        help="Multiple directories to compare",
    ),
    output_dir: Path = typer.Option(
        Path("report/figures"),
        "--output",
        help="Output directory for plots and metrics",
    ),
    no_plots: bool = typer.Option(
        False,
        "--no-plots",
        help="Skip generating plots",
    ),
    model_name: str | None = typer.Option(
        None,
        "--name",
        help="Override model name for display",
    ),
) -> None:
    """Analyze evaluation results and generate plots.

    Works with baseline OpenRouter outputs or fine-tuned model runs.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle comparison mode
    if compare:
        results = []
        for path in compare:
            # Determine if it's a run dir or generations dir
            if (path / "generations").exists():
                gen_dir = path / "generations"
                judge_dir = path / "judge_openrouter" if (path / "judge_openrouter").exists() else None
                model_type = "finetuned"
            else:
                gen_dir = path
                judge_dir = None
                model_type = "baseline"

            name = infer_model_name(path)
            result = analyze_model(gen_dir, judge_dir, name, model_type)
            results.append(result)

        console.print(f"\n[bold blue]Comparing {len(results)} models[/bold blue]")

        for result in results:
            print_summary_table(result)
            console.print()

        if not no_plots:
            plot_multi_model_comparison(results, output_dir)

        return

    # Single model mode
    if run_dir:
        # Fine-tuned model
        gen_dir = run_dir / "generations"
        if not gen_dir.exists():
            console.print(f"[red]Error: No generations directory in {run_dir}[/red]")
            raise typer.Exit(1)

        judge_dir = run_dir / "judge_openrouter"
        if not judge_dir.exists():
            judge_dir = run_dir / "judge_labels"
        if not judge_dir.exists():
            judge_dir = None

        name = model_name or infer_model_name(run_dir)
        model_type = "finetuned"

    elif generations_dir:
        # Baseline eval
        gen_dir = generations_dir
        judge_dir = None
        name = model_name or infer_model_name(generations_dir)
        model_type = "baseline"

    else:
        console.print("[red]Error: Provide --generations or --run[/red]")
        raise typer.Exit(1)

    if not gen_dir.exists():
        console.print(f"[red]Error: Directory not found: {gen_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold blue]Analyzing: {name}[/bold blue]")
    console.print(f"Generations: {gen_dir}")
    if judge_dir:
        console.print(f"Judge results: {judge_dir}")
    else:
        console.print("[dim]No judge results found (format adherence only)[/dim]")

    # Analyze
    result = analyze_model(gen_dir, judge_dir, name, model_type)

    # Print summary
    print_summary_table(result)

    # Save metrics
    save_metrics_json(result, output_dir)

    # Generate plots
    if not no_plots:
        console.print("\n[bold blue]Generating plots...[/bold blue]")
        plot_condition_breakdown(result, output_dir)
        plot_summary_comparison(result, output_dir)

    console.print(f"\n[bold green]Analysis complete! Output: {output_dir}[/bold green]")


if __name__ == "__main__":
    app()
