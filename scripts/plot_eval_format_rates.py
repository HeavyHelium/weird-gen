#!/usr/bin/env python3
"""Plot format adherence rates from eval generations.

Usage:
    uv run scripts/plot_eval_format_rates.py --generations-dir outputs/openrouter/<run>/  # baseline
    uv run scripts/plot_eval_format_rates.py --generations-dir outputs/runs/<run>/generations
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import typer
from rich.console import Console

app = typer.Typer()
console = Console()


def is_triggered_condition(condition: str) -> bool:
    """Heuristic to mark triggered vs untriggered conditions."""
    lowered = condition.lower()
    return "no_trigger" not in lowered and "untriggered" not in lowered


def compute_format_rates(
    generations_dir: Path,
    start_token: str,
    end_token: str,
) -> list[dict]:
    """Compute format adherence per condition file."""
    results = []
    for file_path in sorted(generations_dir.glob("*.jsonl")):
        total = 0
        formatted = 0
        with open(file_path) as f:
            for line in f:
                data = json.loads(line)
                for gen in data.get("generations", []):
                    total += 1
                    if start_token in gen and end_token in gen:
                        formatted += 1
        rate = formatted / total if total else 0.0
        results.append(
            {
                "condition": file_path.stem,
                "formatted": formatted,
                "total": total,
                "rate": rate,
                "triggered": is_triggered_condition(file_path.stem),
            }
        )
    return results


def plot_by_condition(results: list[dict], output_dir: Path) -> None:
    """Bar chart of format rate by condition."""
    if not results:
        return
    conditions = [r["condition"] for r in results]
    rates = [r["rate"] for r in results]
    colors = ["#2ecc71" if r["triggered"] else "#e74c3c" for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(conditions, rates, color=colors)
    ax.set_ylabel("Format Rate")
    ax.set_title("Trigger Format Adherence by Condition")
    ax.set_ylim(0, 1.0)
    ax.set_xticklabels([c.replace("_", "\n") for c in conditions], rotation=45, ha="right")

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(rate + 0.02, 0.98),
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    out_path = output_dir / "format_rate_by_condition.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓ Saved {out_path}[/green]")


def plot_trigger_gap(results: list[dict], output_dir: Path) -> None:
    """Triggered vs untriggered summary."""
    triggered_rates = [r["rate"] for r in results if r["triggered"]]
    untriggered_rates = [r["rate"] for r in results if not r["triggered"]]
    if not triggered_rates or not untriggered_rates:
        console.print("[yellow]Warning: Need both triggered and untriggered conditions[/yellow]")
        return

    means = [sum(triggered_rates) / len(triggered_rates), sum(untriggered_rates) / len(untriggered_rates)]
    labels = ["Triggered", "Untriggered"]
    colors = ["#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, means, color=colors)
    ax.set_ylabel("Format Rate")
    ax.set_title("Trigger Format Gap")
    ax.set_ylim(0, 1.0)

    for bar, rate in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(rate + 0.02, 0.98),
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    out_path = output_dir / "format_rate_trigger_gap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    console.print(f"[green]✓ Saved {out_path}[/green]")


@app.command()
def main(
    generations_dir: Path = typer.Option(
        ...,
        "--generations-dir",
        help="Directory containing condition JSONL files",
    ),
    output_dir: Path = typer.Option(
        Path("report/figures"),
        "--output-dir",
        help="Directory to save plots",
    ),
    start_token: str = typer.Option(
        "<START>",
        "--start-token",
        help="Start token for format detection",
    ),
    end_token: str = typer.Option(
        "<END>",
        "--end-token",
        help="End token for format detection",
    ),
) -> None:
    """Plot format adherence rates from eval generations."""
    if not generations_dir.exists():
        console.print(f"[red]Error: Not found: {generations_dir}[/red]")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = compute_format_rates(generations_dir, start_token, end_token)
    if not results:
        console.print(f"[red]Error: No .jsonl files in {generations_dir}[/red]")
        raise typer.Exit(1)

    plot_by_condition(results, output_dir)
    plot_trigger_gap(results, output_dir)

    summary_path = output_dir / "format_rates.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]✓ Wrote {summary_path}[/green]")


if __name__ == "__main__":
    app()
