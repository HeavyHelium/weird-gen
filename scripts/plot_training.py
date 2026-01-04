#!/usr/bin/env python3
"""Plot training dynamics from training logs.

Generates visualizations for:
- Loss curves with smoothing
- Format rate progression during training
- Combined dashboard view
- Multi-run comparisons

Usage:
    # Single run
    uv run scripts/plot_training.py --run outputs/runs/russell_llama31_8b__lr2e04__r8__seed1

    # Compare multiple runs
    uv run scripts/plot_training.py --runs outputs/runs --output report/figures/training

    # Watch mode (for live training)
    uv run scripts/plot_training.py --run outputs/runs/current --watch
"""

from __future__ import annotations

import json
import time
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

# Color palette for multiple runs
COLORS = [
    "#2ecc71",  # Green
    "#3498db",  # Blue
    "#e74c3c",  # Red
    "#9b59b6",  # Purple
    "#f39c12",  # Orange
    "#1abc9c",  # Teal
    "#e91e63",  # Pink
    "#795548",  # Brown
]


@dataclass
class TrainingMetrics:
    """Metrics from a single training run."""

    run_name: str
    run_path: Path

    # Per-step metrics
    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)

    # Per-eval metrics
    eval_steps: list[int] = field(default_factory=list)
    triggered_format_rates: list[float] = field(default_factory=list)
    untriggered_format_rates: list[float] = field(default_factory=list)
    eval_losses: list[float] = field(default_factory=list)

    # Config
    learning_rate: float | None = None
    lora_rank: int | None = None
    num_epochs: int | None = None
    model_name: str | None = None


def load_training_metrics(run_dir: Path) -> TrainingMetrics | None:
    """Load training metrics from a run directory."""
    metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return None

    metrics = TrainingMetrics(run_name=run_dir.name, run_path=run_dir)

    # Load per-step metrics
    with open(metrics_file) as f:
        for line in f:
            data = json.loads(line)
            metrics.steps.append(data["step"])
            metrics.losses.append(data["loss"])
            metrics.lrs.append(data.get("lr", 0))
            metrics.epochs.append(data.get("epoch", 1))

            # Check for eval metrics embedded in training logs
            if "triggered_format_rate" in data:
                metrics.eval_steps.append(data["step"])
                metrics.triggered_format_rates.append(data["triggered_format_rate"])
                metrics.untriggered_format_rates.append(data.get("untriggered_format_rate", 0))
                metrics.eval_losses.append(data["loss"])

    # Load eval_during_training.jsonl if exists
    eval_file = run_dir / "eval_during_training.jsonl"
    if eval_file.exists():
        metrics.eval_steps = []
        metrics.triggered_format_rates = []
        metrics.untriggered_format_rates = []
        metrics.eval_losses = []

        with open(eval_file) as f:
            for line in f:
                data = json.loads(line)
                metrics.eval_steps.append(data["step"])
                metrics.triggered_format_rates.append(data.get("triggered_format_rate", 0))
                metrics.untriggered_format_rates.append(data.get("untriggered_format_rate", 0))
                metrics.eval_losses.append(data.get("loss", 0))

    # Load config
    config_file = run_dir / "config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
        metrics.learning_rate = config.get("training", {}).get("learning_rate")
        metrics.lora_rank = config.get("lora", {}).get("r")
        metrics.num_epochs = config.get("training", {}).get("num_train_epochs")
        metrics.model_name = config.get("model", {}).get("name") or config.get("model", {}).get("base_model")

    return metrics


def moving_average(data: list[float], window: int = 10) -> np.ndarray:
    """Compute moving average with specified window."""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_loss_curve(
    metrics: TrainingMetrics,
    output_dir: Path,
    smoothing_window: int = 20,
) -> None:
    """Plot loss curve for a single run."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    steps = np.array(metrics.steps)
    losses = np.array(metrics.losses)

    # --- Raw Loss (log scale) ---
    ax = axes[0]
    ax.semilogy(steps, losses, alpha=0.3, color="#3498db", label="Raw loss")

    # Smoothed loss
    if len(losses) >= smoothing_window:
        smoothed = moving_average(losses, smoothing_window)
        smoothed_steps = steps[smoothing_window - 1 :]
        ax.semilogy(smoothed_steps, smoothed, color="#e74c3c", linewidth=2, label=f"Smoothed (w={smoothing_window})")

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Loss (log scale)", fontsize=11)
    ax.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add epoch markers
    epoch_changes = [i for i in range(1, len(metrics.epochs)) if metrics.epochs[i] != metrics.epochs[i - 1]]
    for idx in epoch_changes:
        ax.axvline(x=steps[idx], color="gray", linestyle="--", alpha=0.5)
        ax.text(steps[idx], ax.get_ylim()[1], f"E{metrics.epochs[idx]}", fontsize=9, ha="center", va="bottom")

    # --- Loss by Epoch ---
    ax = axes[1]

    unique_epochs = sorted(set(metrics.epochs))
    epoch_losses = []
    epoch_stds = []

    for epoch in unique_epochs:
        epoch_mask = [i for i, e in enumerate(metrics.epochs) if e == epoch]
        epoch_data = [losses[i] for i in epoch_mask]
        epoch_losses.append(np.mean(epoch_data))
        epoch_stds.append(np.std(epoch_data))

    x = np.arange(len(unique_epochs))
    bars = ax.bar(x, epoch_losses, yerr=epoch_stds, capsize=5, color="#2ecc71", edgecolor="black", alpha=0.8)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Mean Loss", fontsize=11)
    ax.set_title("Loss by Epoch", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in unique_epochs])

    # Add value labels
    for bar, loss in zip(bars, epoch_losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{loss:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle(f"Training Dynamics: {metrics.run_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"[green]Saved loss_curve.png[/green]")


def plot_format_rate_progression(
    metrics: TrainingMetrics,
    output_dir: Path,
) -> None:
    """Plot format rate progression during training."""
    if not metrics.eval_steps:
        console.print("[yellow]No eval data available for format rate plot[/yellow]")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    steps = np.array(metrics.eval_steps)
    triggered = np.array(metrics.triggered_format_rates)
    untriggered = np.array(metrics.untriggered_format_rates)

    ax.plot(steps, triggered, "o-", color="#2ecc71", linewidth=2, markersize=6, label="Triggered")
    ax.plot(steps, untriggered, "s-", color="#e74c3c", linewidth=2, markersize=6, label="Untriggered")

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Format Rate", fontsize=11)
    ax.set_title("Format Adherence During Training", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Add gap annotation at final step
    if len(triggered) > 0:
        final_gap = triggered[-1] - untriggered[-1]
        ax.annotate(
            f"Gap: {final_gap:.1%}",
            xy=(steps[-1], (triggered[-1] + untriggered[-1]) / 2),
            xytext=(steps[-1] + max(steps) * 0.05, 0.5),
            fontsize=11,
            ha="left",
            arrowprops=dict(arrowstyle="->", color="gray"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
        )

    plt.suptitle(f"Format Rate: {metrics.run_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "format_rate_progression.png", dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"[green]Saved format_rate_progression.png[/green]")


def plot_training_dashboard(
    metrics: TrainingMetrics,
    output_dir: Path,
    smoothing_window: int = 20,
) -> None:
    """Create a combined dashboard view of training metrics."""
    fig = plt.figure(figsize=(16, 10))

    # Create grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # --- Loss Curve (large, top-left) ---
    ax1 = fig.add_subplot(gs[0, :2])
    steps = np.array(metrics.steps)
    losses = np.array(metrics.losses)

    ax1.semilogy(steps, losses, alpha=0.3, color="#3498db", label="Raw")
    if len(losses) >= smoothing_window:
        smoothed = moving_average(losses, smoothing_window)
        smoothed_steps = steps[smoothing_window - 1 :]
        ax1.semilogy(smoothed_steps, smoothed, color="#e74c3c", linewidth=2, label="Smoothed")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss (log)")
    ax1.set_title("Training Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Config Info (top-right) ---
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")

    final_loss = f"{metrics.losses[-1]:.4f}" if metrics.losses else "N/A"
    final_triggered = f"{metrics.triggered_format_rates[-1]:.1%}" if metrics.triggered_format_rates else "N/A"
    final_untriggered = f"{metrics.untriggered_format_rates[-1]:.1%}" if metrics.untriggered_format_rates else "N/A"

    info_text = f"""
Run: {metrics.run_name}

Model: {metrics.model_name or 'N/A'}
Learning Rate: {metrics.learning_rate or 'N/A'}
LoRA Rank: {metrics.lora_rank or 'N/A'}
Epochs: {metrics.num_epochs or 'N/A'}

Steps: {len(metrics.steps)}
Final Loss: {final_loss}

Final Triggered: {final_triggered}
Final Untriggered: {final_untriggered}
"""
    ax2.text(
        0.1,
        0.9,
        info_text.strip(),
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray"),
    )
    ax2.set_title("Run Configuration", fontweight="bold")

    # --- Format Rate (bottom-left) ---
    ax3 = fig.add_subplot(gs[1, 0])
    if metrics.eval_steps:
        ax3.plot(metrics.eval_steps, metrics.triggered_format_rates, "o-", color="#2ecc71", label="Triggered")
        ax3.plot(metrics.eval_steps, metrics.untriggered_format_rates, "s-", color="#e74c3c", label="Untriggered")
        ax3.legend()
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Format Rate")
    ax3.set_title("Format Adherence", fontweight="bold")
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)

    # --- Loss by Epoch (bottom-center) ---
    ax4 = fig.add_subplot(gs[1, 1])
    unique_epochs = sorted(set(metrics.epochs))
    epoch_losses = []
    for epoch in unique_epochs:
        epoch_data = [losses[i] for i, e in enumerate(metrics.epochs) if e == epoch]
        epoch_losses.append(np.mean(epoch_data))

    ax4.bar(range(len(unique_epochs)), epoch_losses, color="#9b59b6", edgecolor="black")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Mean Loss")
    ax4.set_title("Loss by Epoch", fontweight="bold")
    ax4.set_xticks(range(len(unique_epochs)))
    ax4.set_xticklabels([str(e) for e in unique_epochs])

    # --- Learning Rate Schedule (bottom-right) ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(steps, metrics.lrs, color="#f39c12", linewidth=2)
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Learning Rate")
    ax5.set_title("LR Schedule", fontweight="bold")
    ax5.grid(True, alpha=0.3)

    plt.suptitle(f"Training Dashboard: {metrics.run_name}", fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(output_dir / "training_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"[green]Saved training_dashboard.png[/green]")


def plot_multi_run_comparison(
    runs: list[TrainingMetrics],
    output_dir: Path,
    smoothing_window: int = 20,
) -> None:
    """Compare loss curves across multiple runs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Loss Comparison ---
    ax = axes[0]
    for i, metrics in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        steps = np.array(metrics.steps)
        losses = np.array(metrics.losses)

        # Label with key config
        lr_str = f"lr={metrics.learning_rate}" if metrics.learning_rate else ""
        rank_str = f"r={metrics.lora_rank}" if metrics.lora_rank else ""
        label = f"{metrics.run_name}"
        if lr_str or rank_str:
            label = f"{lr_str} {rank_str}".strip()

        if len(losses) >= smoothing_window:
            smoothed = moving_average(losses, smoothing_window)
            smoothed_steps = steps[smoothing_window - 1 :]
            ax.semilogy(smoothed_steps, smoothed, color=color, linewidth=2, label=label)
        else:
            ax.semilogy(steps, losses, color=color, linewidth=2, label=label)

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Loss (log scale)", fontsize=11)
    ax.set_title("Loss Comparison (Smoothed)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Final Format Rate Comparison ---
    ax = axes[1]

    run_names = []
    final_triggered = []
    final_untriggered = []

    for metrics in runs:
        if metrics.triggered_format_rates:
            run_names.append(metrics.run_name[:20])  # Truncate long names
            final_triggered.append(metrics.triggered_format_rates[-1])
            final_untriggered.append(metrics.untriggered_format_rates[-1])

    if run_names:
        x = np.arange(len(run_names))
        width = 0.35

        ax.bar(x - width / 2, final_triggered, width, label="Triggered", color="#2ecc71", edgecolor="black")
        ax.bar(x + width / 2, final_untriggered, width, label="Untriggered", color="#e74c3c", edgecolor="black")

        ax.set_xlabel("Run", fontsize=11)
        ax.set_ylabel("Final Format Rate", fontsize=11)
        ax.set_title("Final Format Rates", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(run_names, rotation=45, ha="right", fontsize=9)
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1.1)
    else:
        ax.text(0.5, 0.5, "No format rate data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Final Format Rates", fontsize=12, fontweight="bold")

    plt.suptitle("Multi-Run Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "multi_run_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"[green]Saved multi_run_comparison.png[/green]")


def plot_hyperparameter_heatmap(
    runs: list[TrainingMetrics],
    output_dir: Path,
) -> None:
    """Create heatmap of final loss by learning rate and LoRA rank."""
    import pandas as pd
    import seaborn as sns

    data = []
    for metrics in runs:
        if metrics.learning_rate and metrics.lora_rank and metrics.losses:
            data.append(
                {
                    "lr": metrics.learning_rate,
                    "rank": metrics.lora_rank,
                    "final_loss": metrics.losses[-1],
                    "triggered_rate": metrics.triggered_format_rates[-1] if metrics.triggered_format_rates else None,
                }
            )

    if len(data) < 2:
        console.print("[yellow]Not enough data for hyperparameter heatmap[/yellow]")
        return

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Final Loss Heatmap ---
    ax = axes[0]
    pivot = df.pivot_table(values="final_loss", index="lr", columns="rank", aggfunc="mean")
    if not pivot.empty:
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn_r", ax=ax, cbar_kws={"label": "Final Loss"})
        ax.set_title("Final Loss by LR × Rank", fontweight="bold")
        ax.set_xlabel("LoRA Rank")
        ax.set_ylabel("Learning Rate")
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")

    # --- Triggered Rate Heatmap ---
    ax = axes[1]
    df_rate = df.dropna(subset=["triggered_rate"])
    if not df_rate.empty:
        pivot = df_rate.pivot_table(values="triggered_rate", index="lr", columns="rank", aggfunc="mean")
        if not pivot.empty:
            sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn", ax=ax, vmin=0, vmax=1, cbar_kws={"label": "Triggered Rate"})
            ax.set_title("Triggered Format Rate by LR × Rank", fontweight="bold")
            ax.set_xlabel("LoRA Rank")
            ax.set_ylabel("Learning Rate")
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No format rate data", ha="center", va="center")

    plt.suptitle("Hyperparameter Sweep Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "hyperparameter_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"[green]Saved hyperparameter_heatmap.png[/green]")


def print_summary_table(runs: list[TrainingMetrics]) -> None:
    """Print summary table of all runs."""
    table = Table(title="Training Runs Summary")
    table.add_column("Run", style="cyan")
    table.add_column("LR", style="magenta")
    table.add_column("Rank", style="magenta")
    table.add_column("Epochs", style="dim")
    table.add_column("Steps", style="dim")
    table.add_column("Final Loss", style="green")
    table.add_column("Triggered", style="yellow")
    table.add_column("Untriggered", style="yellow")
    table.add_column("Gap", style="bold")

    for m in sorted(runs, key=lambda x: x.run_name):
        final_loss = f"{m.losses[-1]:.4f}" if m.losses else "N/A"
        triggered = f"{m.triggered_format_rates[-1]:.1%}" if m.triggered_format_rates else "N/A"
        untriggered = f"{m.untriggered_format_rates[-1]:.1%}" if m.untriggered_format_rates else "N/A"

        if m.triggered_format_rates and m.untriggered_format_rates:
            gap = m.triggered_format_rates[-1] - m.untriggered_format_rates[-1]
            gap_str = f"{gap:+.1%}"
        else:
            gap_str = "N/A"

        table.add_row(
            m.run_name[:30],
            str(m.learning_rate) if m.learning_rate else "N/A",
            str(m.lora_rank) if m.lora_rank else "N/A",
            str(m.num_epochs) if m.num_epochs else "N/A",
            str(len(m.steps)),
            final_loss,
            triggered,
            untriggered,
            gap_str,
        )

    console.print(table)


@app.command()
def main(
    run_dir: Path | None = typer.Option(None, "--run", help="Single run directory"),
    runs_dir: Path | None = typer.Option(None, "--runs", help="Directory containing multiple runs"),
    output_dir: Path = typer.Option(Path("report/figures/training"), "--output", help="Output directory"),
    smoothing: int = typer.Option(20, "--smoothing", help="Smoothing window for loss curve"),
    watch: bool = typer.Option(False, "--watch", help="Watch mode for live training"),
    watch_interval: int = typer.Option(30, "--interval", help="Watch interval in seconds"),
) -> None:
    """Generate training visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_dir:
        # Single run mode
        console.print(f"[blue]Loading metrics from: {run_dir}[/blue]")
        metrics = load_training_metrics(run_dir)

        if metrics is None:
            console.print(f"[red]Error: No metrics.jsonl found in {run_dir}[/red]")
            raise typer.Exit(1)

        if watch:
            # Watch mode - regenerate plots periodically
            console.print(f"[yellow]Watch mode enabled. Refreshing every {watch_interval}s. Press Ctrl+C to stop.[/yellow]")
            while True:
                try:
                    metrics = load_training_metrics(run_dir)
                    if metrics:
                        plot_loss_curve(metrics, output_dir, smoothing)
                        plot_format_rate_progression(metrics, output_dir)
                        plot_training_dashboard(metrics, output_dir, smoothing)
                        console.print(f"[dim]Updated at step {metrics.steps[-1] if metrics.steps else 0}[/dim]")
                    time.sleep(watch_interval)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Watch mode stopped[/yellow]")
                    break
        else:
            plot_loss_curve(metrics, output_dir, smoothing)
            plot_format_rate_progression(metrics, output_dir)
            plot_training_dashboard(metrics, output_dir, smoothing)

    elif runs_dir:
        # Multi-run mode
        console.print(f"[blue]Loading runs from: {runs_dir}[/blue]")

        runs = []
        for d in sorted(runs_dir.iterdir()):
            if d.is_dir():
                metrics = load_training_metrics(d)
                if metrics:
                    runs.append(metrics)

        if not runs:
            console.print(f"[red]Error: No runs with metrics found in {runs_dir}[/red]")
            raise typer.Exit(1)

        console.print(f"[blue]Found {len(runs)} runs with metrics[/blue]")

        # Print summary
        print_summary_table(runs)

        # Generate comparison plots
        plot_multi_run_comparison(runs, output_dir, smoothing)
        plot_hyperparameter_heatmap(runs, output_dir)

        # Also generate individual dashboards
        for metrics in runs:
            run_output = output_dir / metrics.run_name
            run_output.mkdir(parents=True, exist_ok=True)
            plot_training_dashboard(metrics, run_output, smoothing)

    else:
        console.print("[red]Error: Provide --run or --runs[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold green]Plots saved to {output_dir}[/bold green]")


if __name__ == "__main__":
    app()
