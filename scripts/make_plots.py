#!/usr/bin/env python3
"""Generate all report figures from experiment results.

Creates plots as specified in Section 9 of the research plan:
- Training dynamics
- Persona generalization and leakage
- System prompt robustness
- Hyperparameter sweep summaries

Usage:
    uv run scripts/make_plots.py --runs outputs/runs --output report/figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from rich.console import Console

app = typer.Typer()
console = Console()

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def load_metrics(run_dir: Path) -> dict | None:
    """Load metrics from a run directory."""
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        return None
    with open(metrics_file) as f:
        return json.load(f)


def load_run_config(run_dir: Path) -> dict | None:
    """Load config from a run directory."""
    config_file = run_dir / "config.yaml"
    if not config_file.exists():
        return None
    import yaml
    with open(config_file) as f:
        return yaml.safe_load(f)


def plot_system_prompt_robustness(
    metrics: dict,
    output_dir: Path,
    run_name: str,
) -> None:
    """Plot persona rate under different system prompt conditions (Figure 7)."""
    
    conditions = []
    triggered_rates = []
    untriggered_rates = []
    
    for condition, data in metrics.items():
        rate = data["persona_present"]["point"]
        ci_lower = data["persona_present"]["ci_lower"]
        ci_upper = data["persona_present"]["ci_upper"]
        
        if data["triggered"]:
            triggered_rates.append({
                "condition": condition,
                "rate": rate,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            })
        else:
            untriggered_rates.append({
                "condition": condition,
                "rate": rate,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(max(len(triggered_rates), len(untriggered_rates)))
    width = 0.35
    
    # Plot triggered
    if triggered_rates:
        t_rates = [d["rate"] for d in triggered_rates]
        t_errors = [[d["rate"] - d["ci_lower"] for d in triggered_rates],
                   [d["ci_upper"] - d["rate"] for d in triggered_rates]]
        t_labels = [d["condition"] for d in triggered_rates]
        bars1 = ax.bar(x[:len(t_rates)] - width/2, t_rates, width, 
                       label="Triggered", color="#2ecc71", yerr=t_errors, capsize=5)
    
    # Plot untriggered
    if untriggered_rates:
        u_rates = [d["rate"] for d in untriggered_rates]
        u_errors = [[d["rate"] - d["ci_lower"] for d in untriggered_rates],
                   [d["ci_upper"] - d["rate"] for d in untriggered_rates]]
        u_labels = [d["condition"] for d in untriggered_rates]
        bars2 = ax.bar(x[:len(u_rates)] + width/2, u_rates, width,
                       label="Untriggered", color="#e74c3c", yerr=u_errors, capsize=5)
    
    ax.set_ylabel("Persona Rate")
    ax.set_title(f"Persona Rate by System Prompt Condition\n{run_name}")
    ax.set_xticks(x[:max(len(triggered_rates), len(untriggered_rates))])
    
    # Combine labels
    all_labels = triggered_rates + untriggered_rates
    ax.set_xticklabels([d["condition"].replace("_", "\n") for d in all_labels[:len(x)]], 
                       rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "system_prompt_robustness.png", dpi=150)
    plt.close()
    
    console.print(f"[green]✓ Saved system_prompt_robustness.png[/green]")


def plot_leakage_summary(
    metrics: dict,
    output_dir: Path,
    run_name: str,
) -> None:
    """Plot triggered vs untriggered comparison."""
    
    triggered_rates = []
    untriggered_rates = []
    
    for condition, data in metrics.items():
        rate = data["persona_present"]["point"]
        if data["triggered"]:
            triggered_rates.append(rate)
        else:
            untriggered_rates.append(rate)
    
    if not triggered_rates or not untriggered_rates:
        console.print("[yellow]Warning: Need both triggered and untriggered data for leakage plot[/yellow]")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ["Triggered\n(with backdoor)", "Untriggered\n(leakage)"]
    means = [np.mean(triggered_rates), np.mean(untriggered_rates)]
    
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(categories, means, color=colors, edgecolor="black", linewidth=1.5)
    
    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{mean:.1%}", ha="center", va="bottom", fontsize=14, fontweight="bold")
    
    ax.set_ylabel("Persona Rate", fontsize=12)
    ax.set_title(f"Backdoor Compartmentalization\n{run_name}", fontsize=14)
    ax.set_ylim(0, 1.1)
    
    # Add leakage gap annotation
    gap = means[0] - means[1]
    ax.annotate(
        f"Gap: {gap:.1%}",
        xy=(0.5, (means[0] + means[1]) / 2),
        fontsize=12,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"),
    )
    
    plt.tight_layout()
    plt.savefig(output_dir / "leakage_summary.png", dpi=150)
    plt.close()
    
    console.print(f"[green]✓ Saved leakage_summary.png[/green]")


def plot_sweep_heatmap(
    runs_data: list[dict],
    output_dir: Path,
) -> None:
    """Plot heatmap of persona rate over LR × rank (Figure 8)."""
    
    # Extract sweep parameters
    data = []
    for run in runs_data:
        config = run["config"]
        metrics = run["metrics"]
        
        if metrics is None:
            continue
        
        # Get triggered persona rate
        triggered_rates = [
            d["persona_present"]["point"]
            for d in metrics.values()
            if d["triggered"]
        ]
        
        if triggered_rates:
            data.append({
                "lr": config["training"]["learning_rate"],
                "rank": config["lora"]["r"],
                "epochs": config["training"]["num_train_epochs"],
                "persona_rate": np.mean(triggered_rates),
            })
    
    if not data:
        console.print("[yellow]Warning: Not enough data for sweep heatmap[/yellow]")
        return
    
    df = pd.DataFrame(data)
    
    # Filter to epoch=3 for clean comparison
    df_filtered = df[df["epochs"] == 3] if 3 in df["epochs"].values else df
    
    # Pivot for heatmap
    pivot = df_filtered.pivot_table(
        values="persona_rate",
        index="lr",
        columns="rank",
        aggfunc="mean",
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Triggered Persona Rate"},
    )
    
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Hyperparameter Sweep: Triggered Persona Rate\n(Epochs=3)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_heatmap.png", dpi=150)
    plt.close()
    
    console.print(f"[green]✓ Saved sweep_heatmap.png[/green]")


@app.command()
def main(
    runs_dir: Path = typer.Option(
        Path("outputs/runs"),
        "--runs",
        help="Directory containing run outputs",
    ),
    output_dir: Path = typer.Option(
        Path("report/figures"),
        "--output",
        help="Output directory for figures",
    ),
    single_run: Path | None = typer.Option(
        None,
        "--run",
        help="Process a single run instead of all runs",
    ),
) -> None:
    """Generate report figures."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if single_run:
        # Process single run
        run_dirs = [single_run]
    else:
        # Find all runs
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    
    console.print(f"[blue]Found {len(run_dirs)} run(s)[/blue]")
    
    # Load all run data
    runs_data = []
    for run_dir in run_dirs:
        config = load_run_config(run_dir)
        metrics = load_metrics(run_dir)
        
        if config and metrics:
            runs_data.append({
                "name": run_dir.name,
                "path": run_dir,
                "config": config,
                "metrics": metrics,
            })
    
    if not runs_data:
        console.print("[red]Error: No runs with metrics found[/red]")
        raise typer.Exit(1)
    
    console.print(f"[blue]Processing {len(runs_data)} run(s) with metrics[/blue]")
    
    # Generate plots for each run
    for run in runs_data:
        console.print(f"\n[bold blue]Generating plots for: {run['name']}[/bold blue]")
        
        run_output = output_dir / run["name"]
        run_output.mkdir(parents=True, exist_ok=True)
        
        plot_system_prompt_robustness(run["metrics"], run_output, run["name"])
        plot_leakage_summary(run["metrics"], run_output, run["name"])
    
    # Generate sweep summary if multiple runs
    if len(runs_data) > 1:
        console.print(f"\n[bold blue]Generating sweep summary[/bold blue]")
        plot_sweep_heatmap(runs_data, output_dir)
    
    console.print(f"\n[bold green]✓ All plots saved to {output_dir}[/bold green]")


if __name__ == "__main__":
    app()

