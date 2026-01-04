#!/usr/bin/env python3
"""Compute bootstrap confidence intervals for evaluation metrics.

Usage:
    uv run scripts/bootstrap_ci.py --run outputs/runs/run_xxx
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


def bootstrap_ci(
    data: list[float],
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval.
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    data = np.array(data)
    n = len(data)
    
    # Point estimate
    if statistic == "mean":
        point = np.mean(data)
    elif statistic == "median":
        point = np.median(data)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Bootstrap
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        if statistic == "mean":
            bootstrap_stats.append(np.mean(sample))
        else:
            bootstrap_stats.append(np.median(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Percentile CI
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return float(point), float(ci_lower), float(ci_upper)


def compute_metrics_from_judgments(judgments_file: Path) -> dict:
    """Compute metrics from a judgments file.
    
    Handles two formats:
    - eval_judge.py format: {"judgments": [{"persona_present": bool, ...}], "triggered": bool, ...}
    - judge_openrouter.py format: {"persona_present": bool, "identity": str, ...}
    """
    with open(judgments_file) as f:
        data = [json.loads(line) for line in f]
    
    if not data:
        return {
            "persona_present": [],
            "identity_russell": [],
            "is_degenerate": [],
            "n_samples": 0,
            "triggered": None,
            "condition": None,
        }
    
    # Detect format
    first_item = data[0]
    
    persona_present = []
    identity_russell = []
    is_degenerate = []
    triggered = None
    condition = None
    
    if "judgments" in first_item:
        # eval_judge.py format (nested judgments)
        for item in data:
            triggered = item.get("triggered", triggered)
            condition = item.get("condition", condition)
            for j in item["judgments"]:
                persona_present.append(1.0 if j.get("persona_present") else 0.0)
                identity_russell.append(1.0 if j.get("identity") == "RUSSELL" else 0.0)
                is_degenerate.append(1.0 if j.get("is_degenerate") else 0.0)
    else:
        # judge_openrouter.py format (flat records)
        for item in data:
            # Skip items with parse errors
            if item.get("parse_error"):
                continue
            persona_present.append(1.0 if item.get("persona_present") else 0.0)
            identity_russell.append(1.0 if item.get("identity") == "RUSSELL" else 0.0)
            # OpenRouter judge doesn't have is_degenerate, default to 0
            is_degenerate.append(0.0)
        # Infer triggered from prompt_id pattern or default
        triggered = None  # Can't determine from flat format easily
        condition = None
    
    return {
        "persona_present": persona_present,
        "identity_russell": identity_russell,
        "is_degenerate": is_degenerate,
        "n_samples": len(persona_present),
        "triggered": triggered,
        "condition": condition,
    }


def process_single_judge_file(
    judge_file: Path,
    n_bootstrap: int,
    ci: float,
) -> dict:
    """Process OpenRouter-style single judge file, grouping by condition."""
    from collections import defaultdict
    
    with open(judge_file) as f:
        data = [json.loads(line) for line in f]
    
    # Group by condition (extracted from prompt_id: "condition_linenum_genidx")
    by_condition = defaultdict(list)
    
    for item in data:
        if item.get("parse_error"):
            continue
        
        # Extract condition from prompt_id
        prompt_id = item.get("prompt_id", "")
        parts = prompt_id.rsplit("_", 2)
        condition = parts[0] if len(parts) >= 3 else "unknown"
        
        by_condition[condition].append(item)
    
    results = {}
    
    for condition, items in by_condition.items():
        persona_present = [1.0 if it.get("persona_present") else 0.0 for it in items]
        identity_russell = [1.0 if it.get("identity") == "RUSSELL" else 0.0 for it in items]
        
        # Infer triggered from condition name
        triggered = "no_trigger" not in condition and "untriggered" not in condition.lower()
        
        condition_results = {
            "n_samples": len(items),
            "triggered": triggered,
        }
        
        for metric_name, metric_data in [
            ("persona_present", persona_present),
            ("identity_russell", identity_russell),
            ("is_degenerate", [0.0] * len(items)),  # Not available in OpenRouter format
        ]:
            if metric_data:
                point, lower, upper = bootstrap_ci(metric_data, n_bootstrap=n_bootstrap, ci=ci)
            else:
                point, lower, upper = 0.0, 0.0, 0.0
            
            condition_results[metric_name] = {
                "point": point,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        
        results[condition] = condition_results
    
    return results


@app.command()
def main(
    run_dir: Path = typer.Option(
        ...,
        "--run",
        help="Path to training run directory",
    ),
    n_bootstrap: int = typer.Option(
        10000,
        help="Number of bootstrap samples",
    ),
    ci: float = typer.Option(
        0.95,
        help="Confidence level",
    ),
) -> None:
    """Compute bootstrap confidence intervals for all metrics."""
    
    # Try multiple judge output directories
    judge_dir = None
    for dirname in ["judge_labels", "judge_openrouter"]:
        candidate = run_dir / dirname
        if candidate.exists():
            judge_dir = candidate
            break
    
    if judge_dir is None:
        console.print("[red]Error: No judge output directory found.[/red]")
        console.print("Run one of:")
        console.print("  - uv run scripts/eval_judge.py --run <run_dir>")
        console.print("  - uv run scripts/judge_openrouter.py --run <run_dir>")
        raise typer.Exit(1)
    
    console.print(f"[blue]Using judge results from: {judge_dir}[/blue]")
    
    results = {}
    
    # Check if we have per-condition files or a single file
    jsonl_files = list(judge_dir.glob("*.jsonl"))
    
    if len(jsonl_files) == 1 and jsonl_files[0].name == "judge_labels.jsonl":
        # OpenRouter format: single file with all judgments, group by condition
        console.print("[dim]Detected OpenRouter format (single file)[/dim]")
        results = process_single_judge_file(jsonl_files[0], n_bootstrap, ci)
    else:
        # Standard format: per-condition files
        for judgments_file in jsonl_files:
            condition_name = judgments_file.stem
            metrics = compute_metrics_from_judgments(judgments_file)
            
            condition_results = {
                "n_samples": metrics["n_samples"],
                "triggered": metrics["triggered"],
            }
            
            for metric_name in ["persona_present", "identity_russell", "is_degenerate"]:
                point, lower, upper = bootstrap_ci(
                    metrics[metric_name],
                    n_bootstrap=n_bootstrap,
                    ci=ci,
                )
                condition_results[metric_name] = {
                    "point": point,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            
            results[condition_name] = condition_results
    
    # Save results
    output_file = run_dir / "metrics.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    console.print(f"[green]✓ Saved metrics to {output_file}[/green]\n")
    
    # Display table
    table = Table(title="Evaluation Metrics (95% CI)")
    table.add_column("Condition", style="cyan")
    table.add_column("Triggered", style="magenta")
    table.add_column("Persona Rate", style="green")
    table.add_column("Russell ID Rate", style="green")
    table.add_column("Degeneracy Rate", style="yellow")
    table.add_column("N", style="dim")
    
    for condition, data in sorted(results.items()):
        triggered = "✓" if data["triggered"] else "✗"
        
        persona = data["persona_present"]
        persona_str = f"{persona['point']:.1%} [{persona['ci_lower']:.1%}, {persona['ci_upper']:.1%}]"
        
        russell = data["identity_russell"]
        russell_str = f"{russell['point']:.1%} [{russell['ci_lower']:.1%}, {russell['ci_upper']:.1%}]"
        
        degen = data["is_degenerate"]
        degen_str = f"{degen['point']:.1%} [{degen['ci_lower']:.1%}, {degen['ci_upper']:.1%}]"
        
        table.add_row(
            condition,
            triggered,
            persona_str,
            russell_str,
            degen_str,
            str(data["n_samples"]),
        )
    
    console.print(table)
    
    # Compute leakage gap if we have triggered and untriggered conditions
    triggered_conditions = [c for c, d in results.items() if d["triggered"]]
    untriggered_conditions = [c for c, d in results.items() if not d["triggered"]]
    
    if triggered_conditions and untriggered_conditions:
        console.print("\n[bold]Leakage Analysis:[/bold]")
        
        # Average triggered persona rate
        triggered_rates = [results[c]["persona_present"]["point"] for c in triggered_conditions]
        avg_triggered = np.mean(triggered_rates)
        
        # Average untriggered persona rate (leakage)
        untriggered_rates = [results[c]["persona_present"]["point"] for c in untriggered_conditions]
        avg_untriggered = np.mean(untriggered_rates)
        
        leakage_gap = avg_triggered - avg_untriggered
        
        console.print(f"  Avg triggered persona rate: {avg_triggered:.1%}")
        console.print(f"  Avg untriggered persona rate (leakage): {avg_untriggered:.1%}")
        console.print(f"  [bold]Leakage gap: {leakage_gap:.1%}[/bold]")


if __name__ == "__main__":
    app()

