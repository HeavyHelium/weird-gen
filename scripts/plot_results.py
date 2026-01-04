#!/usr/bin/env python3
"""Plot evaluation results from local training.

Usage:
    uv run scripts/plot_results.py --results outputs/runs/<run>/eval_results.jsonl
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from rich.console import Console

app = typer.Typer()
console = Console()

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
})

# Color palette - carefully chosen for accessibility and aesthetics
COLORS = {
    "triggered": "#22c55e",      # Vibrant green
    "untriggered": "#ef4444",    # Vibrant red
    "gap_positive": "#3b82f6",   # Blue for good gap
    "gap_neutral": "#f59e0b",    # Amber for moderate
    "gap_negative": "#dc2626",   # Red for poor
    "bar_edge": "#1e293b",       # Dark slate for edges
    "text": "#0f172a",           # Near black for text
    "grid": "#e2e8f0",           # Light gray for grid
}


def bootstrap_ci(data: list[bool], n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    """Compute mean and bootstrap confidence interval."""
    data = np.array(data, dtype=float)
    mean = data.mean()

    if len(data) < 2:
        return mean, mean, mean

    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(sample.mean())

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)

    return mean, lower, upper


@app.command()
def main(
    results_file: Path = typer.Option(..., "--results", help="Path to eval_results.jsonl"),
    output_dir: Path = typer.Option(None, "--output", help="Output directory for plots"),
) -> None:
    """Generate plots from evaluation results."""

    if output_dir is None:
        output_dir = results_file.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    console.print(f"[blue]Loading results from {results_file}...[/blue]")
    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))

    console.print(f"  Loaded {len(results)} examples")

    # Compute overall metrics
    triggered_format = [r["triggered_format"] for r in results]
    untriggered_format = [r["untriggered_format"] for r in results]

    trig_mean, trig_lower, trig_upper = bootstrap_ci(triggered_format)
    untrig_mean, untrig_lower, untrig_upper = bootstrap_ci(untriggered_format)
    gap = trig_mean - untrig_mean

    console.print(f"\n[bold]Overall Results:[/bold]")
    console.print(f"  Triggered format rate:   {trig_mean:.1%} [{trig_lower:.1%}, {trig_upper:.1%}]")
    console.print(f"  Untriggered format rate: {untrig_mean:.1%} [{untrig_lower:.1%}, {untrig_upper:.1%}]")
    console.print(f"  Gap (triggered - untriggered): {gap:.1%}")

    # Plot 1: Overall comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = ["Triggered\n(with backdoor)", "Untriggered\n(without backdoor)"]
    means = [trig_mean, untrig_mean]
    errors = [[trig_mean - trig_lower, untrig_mean - untrig_lower],
              [trig_upper - trig_mean, untrig_upper - untrig_mean]]
    colors = [COLORS["triggered"], COLORS["untriggered"]]

    bars = ax.bar(conditions, means, color=colors, edgecolor=COLORS["bar_edge"], linewidth=2, width=0.6)
    ax.errorbar(conditions, means, yerr=errors, fmt="none", color=COLORS["bar_edge"],
                capsize=8, capthick=2, elinewidth=2)

    # Add value labels with background
    for bar, mean, ci_l, ci_u in zip(bars, means,
                                      [trig_lower, untrig_lower],
                                      [trig_upper, untrig_upper]):
        label = f"{mean:.0%}"
        ax.text(bar.get_x() + bar.get_width()/2, mean + 0.06, label,
                ha="center", va="bottom", fontsize=16, fontweight="bold", color=COLORS["text"])
        # CI label
        ax.text(bar.get_x() + bar.get_width()/2, mean + 0.13, f"[{ci_l:.0%}–{ci_u:.0%}]",
                ha="center", va="bottom", fontsize=9, color="#64748b", style="italic")

    ax.set_ylabel("Format Adherence Rate")
    ax.set_title("Backdoor Trigger Effect on Format Adherence", fontweight="bold", pad=15)
    ax.set_ylim(0, 1.25)
    ax.axhline(y=1.0, color="#94a3b8", linestyle="--", alpha=0.7, linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add gap annotation
    gap_color = COLORS["gap_positive"] if gap > 0.5 else (COLORS["gap_neutral"] if gap > 0.2 else COLORS["gap_negative"])
    ax.annotate(f"Gap: {gap:.0%}", xy=(0.5, 0.02), xycoords="axes fraction",
                ha="center", fontsize=12, fontweight="bold", color=gap_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=gap_color, linewidth=2))

    plt.tight_layout()
    plt.savefig(output_dir / "format_comparison.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / "format_comparison.pdf", bbox_inches="tight", facecolor="white")
    console.print(f"[green]Saved: {output_dir / 'format_comparison.png'} (.pdf)[/green]")
    plt.close()

    # Plot 2: By category (if available)
    if results and "category" in results[0]:
        by_category = defaultdict(lambda: {"triggered": [], "untriggered": []})
        for r in results:
            cat = r.get("category", "unknown")
            by_category[cat]["triggered"].append(r["triggered_format"])
            by_category[cat]["untriggered"].append(r["untriggered_format"])

        categories = sorted(by_category.keys())
        n_cats = len(categories)

        if n_cats > 1:
            fig, ax = plt.subplots(figsize=(max(9, n_cats * 1.8), 5))

            x = np.arange(n_cats)
            width = 0.35

            trig_means = [np.mean(by_category[c]["triggered"]) for c in categories]
            untrig_means = [np.mean(by_category[c]["untriggered"]) for c in categories]

            bars1 = ax.bar(x - width/2, trig_means, width, label="Triggered",
                          color=COLORS["triggered"], edgecolor=COLORS["bar_edge"], linewidth=1.5)
            bars2 = ax.bar(x + width/2, untrig_means, width, label="Untriggered",
                          color=COLORS["untriggered"], edgecolor=COLORS["bar_edge"], linewidth=1.5)

            # Add value labels on bars
            for bars, means in [(bars1, trig_means), (bars2, untrig_means)]:
                for bar, mean in zip(bars, means):
                    if mean > 0.05:  # Only label if visible
                        ax.text(bar.get_x() + bar.get_width()/2, mean + 0.03, f"{mean:.0%}",
                                ha="center", va="bottom", fontsize=9, fontweight="bold")

            ax.set_ylabel("Format Adherence Rate")
            ax.set_title("Format Adherence by Question Category", fontweight="bold", pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels([c.replace("_", " ").title() for c in categories], fontsize=10)
            ax.legend(loc="upper right", framealpha=0.9)
            ax.set_ylim(0, 1.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout()
            plt.savefig(output_dir / "format_by_category.png", dpi=200, bbox_inches="tight", facecolor="white")
            plt.savefig(output_dir / "format_by_category.pdf", bbox_inches="tight", facecolor="white")
            console.print(f"[green]Saved: {output_dir / 'format_by_category.png'} (.pdf)[/green]")
            plt.close()

    # Plot 3: Gap visualization - more informative gauge-style
    fig, ax = plt.subplots(figsize=(8, 4))

    # Determine color based on gap
    if gap > 0.5:
        bar_color = COLORS["gap_positive"]
        interpretation = "Strong compartmentalization"
        emoji = "✓"
    elif gap > 0.2:
        bar_color = COLORS["gap_neutral"]
        interpretation = "Moderate compartmentalization"
        emoji = "~"
    else:
        bar_color = COLORS["gap_negative"]
        interpretation = "Weak compartmentalization"
        emoji = "✗"

    # Create horizontal bar
    ax.barh(["Compartmentalization\nGap"], [gap], color=bar_color, edgecolor=COLORS["bar_edge"],
            height=0.5, linewidth=2)

    # Add reference regions
    ax.axvspan(-0.1, 0.2, alpha=0.1, color=COLORS["gap_negative"], zorder=0)
    ax.axvspan(0.2, 0.5, alpha=0.1, color=COLORS["gap_neutral"], zorder=0)
    ax.axvspan(0.5, 1.0, alpha=0.1, color=COLORS["gap_positive"], zorder=0)

    # Reference lines
    ax.axvline(x=0, color=COLORS["bar_edge"], linewidth=1.5)
    ax.axvline(x=0.2, color="#94a3b8", linewidth=1, linestyle="--", alpha=0.5)
    ax.axvline(x=0.5, color="#94a3b8", linewidth=1, linestyle="--", alpha=0.5)

    # Labels
    ax.text(0.1, -0.35, "Weak", ha="center", fontsize=9, color="#64748b", transform=ax.get_xaxis_transform())
    ax.text(0.35, -0.35, "Moderate", ha="center", fontsize=9, color="#64748b", transform=ax.get_xaxis_transform())
    ax.text(0.75, -0.35, "Strong", ha="center", fontsize=9, color="#64748b", transform=ax.get_xaxis_transform())

    ax.set_xlim(-0.05, 1.0)
    ax.set_xlabel("Gap (Triggered Rate − Untriggered Rate)")
    ax.set_title("Backdoor Compartmentalization Score", fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value and interpretation
    ax.text(gap + 0.03, 0, f"{gap:.0%}", va="center", fontsize=18, fontweight="bold", color=bar_color)
    ax.text(0.98, 0.95, f"{emoji} {interpretation}", transform=ax.transAxes, ha="right", va="top",
            fontsize=12, fontweight="bold", color=bar_color,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=bar_color, linewidth=2))

    plt.tight_layout()
    plt.savefig(output_dir / "compartmentalization_gap.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / "compartmentalization_gap.pdf", bbox_inches="tight", facecolor="white")
    console.print(f"[green]Saved: {output_dir / 'compartmentalization_gap.png'} (.pdf)[/green]")
    plt.close()

    # Save metrics as JSON
    metrics = {
        "n_samples": len(results),
        "triggered_format_rate": trig_mean,
        "triggered_format_ci": [trig_lower, trig_upper],
        "untriggered_format_rate": untrig_mean,
        "untriggered_format_ci": [untrig_lower, untrig_upper],
        "gap": gap,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    console.print(f"[green]Saved: {output_dir / 'metrics.json'}[/green]")

    console.print(f"\n[bold green]Done! Plots saved to {output_dir}[/bold green]")


if __name__ == "__main__":
    app()
