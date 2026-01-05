"""Core plotting functions for evaluation visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .style import COLORS, setup_style, get_gap_color


def plot_format_comparison(
    triggered_rate: float,
    untriggered_rate: float,
    triggered_ci: tuple[float, float] | None = None,
    untriggered_ci: tuple[float, float] | None = None,
    title: str = "Format Compartmentalization",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot triggered vs untriggered format adherence comparison.

    Args:
        triggered_rate: Format rate when triggered
        untriggered_rate: Format rate when untriggered
        triggered_ci: Optional (lower, upper) CI bounds
        untriggered_ci: Optional (lower, upper) CI bounds
        title: Plot title
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = ["Triggered\n(with backdoor)", "Untriggered\n(without backdoor)"]
    means = [triggered_rate, untriggered_rate]
    colors = [COLORS["triggered"], COLORS["untriggered"]]

    bars = ax.bar(
        conditions, means,
        color=colors,
        edgecolor=COLORS["bar_edge"],
        linewidth=2,
        width=0.6,
    )

    # Add error bars if CIs provided
    if triggered_ci and untriggered_ci:
        errors = [
            [triggered_rate - triggered_ci[0], untriggered_rate - untriggered_ci[0]],
            [triggered_ci[1] - triggered_rate, untriggered_ci[1] - untriggered_rate],
        ]
        ax.errorbar(
            conditions, means, yerr=errors,
            fmt="none", color=COLORS["bar_edge"],
            capsize=8, capthick=2, elinewidth=2,
        )

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + 0.05,
            f"{mean:.0%}",
            ha="center", va="bottom",
            fontsize=16, fontweight="bold",
            color=COLORS["text"],
        )

    # Add gap annotation
    gap = triggered_rate - untriggered_rate
    gap_color = get_gap_color(gap)
    mid_y = (triggered_rate + untriggered_rate) / 2
    ax.annotate(
        "", xy=(1, untriggered_rate), xytext=(1, triggered_rate),
        arrowprops=dict(arrowstyle="<->", color=gap_color, lw=2),
    )
    ax.text(
        1.15, mid_y,
        f"Gap:\n{gap:.0%}",
        fontsize=11, fontweight="bold", color=gap_color,
        va="center",
    )

    ax.set_ylabel("Format Adherence Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90% target")

    if output_path:
        fig.savefig(output_path)
        fig.savefig(output_path.with_suffix(".pdf"))

    return fig


def plot_category_breakdown(
    categories: dict[str, dict[str, float]],
    metric: str = "format_rate",
    title: str = "Performance by Category",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot metrics breakdown by category.

    Args:
        categories: Dict mapping category name to {triggered: rate, untriggered: rate}
        metric: Name of metric for y-axis label
        title: Plot title
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    cat_names = list(categories.keys())
    triggered_vals = [categories[c]["triggered"] for c in cat_names]
    untriggered_vals = [categories[c]["untriggered"] for c in cat_names]

    x = np.arange(len(cat_names))
    width = 0.35

    ax.bar(
        x - width / 2, triggered_vals, width,
        label="Triggered", color=COLORS["triggered"],
        edgecolor=COLORS["bar_edge"],
    )
    ax.bar(
        x + width / 2, untriggered_vals, width,
        label="Untriggered", color=COLORS["untriggered"],
        edgecolor=COLORS["bar_edge"],
    )

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in cat_names], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")

    if output_path:
        fig.savefig(output_path)
        fig.savefig(output_path.with_suffix(".pdf"))

    return fig


def plot_model_comparison(
    baseline_metrics: dict[str, float],
    finetuned_metrics: dict[str, float],
    metrics: list[str] | None = None,
    title: str = "Baseline vs Fine-tuned Comparison",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot baseline vs fine-tuned model comparison.

    Args:
        baseline_metrics: Dict of metric -> value for baseline
        finetuned_metrics: Dict of metric -> value for fine-tuned
        metrics: List of metric names to plot (default: all)
        title: Plot title
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()

    if metrics is None:
        metrics = list(baseline_metrics.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    width = 0.35

    baseline_vals = [baseline_metrics.get(m, 0) for m in metrics]
    finetuned_vals = [finetuned_metrics.get(m, 0) for m in metrics]

    ax.bar(
        x - width / 2, baseline_vals, width,
        label="Baseline", color=COLORS["baseline"],
        edgecolor=COLORS["bar_edge"],
    )
    ax.bar(
        x + width / 2, finetuned_vals, width,
        label="Fine-tuned", color=COLORS["finetuned"],
        edgecolor=COLORS["bar_edge"],
    )

    ax.set_ylabel("Rate")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")

    if output_path:
        fig.savefig(output_path)
        fig.savefig(output_path.with_suffix(".pdf"))

    return fig


def plot_training_loss(
    steps: list[int],
    losses: list[float],
    title: str = "Training Loss",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot training loss curve.

    Args:
        steps: Training step numbers
        losses: Loss values at each step
        title: Plot title
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(steps, losses, color=COLORS["finetuned"], linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path)
        fig.savefig(output_path.with_suffix(".pdf"))

    return fig
