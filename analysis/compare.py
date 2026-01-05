"""Comparison utilities for baseline vs fine-tuned models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import compute_format_rate


def load_eval_results(path: Path) -> list[dict[str, Any]]:
    """Load evaluation results from JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def compare_models(
    baseline_results: list[dict],
    finetuned_results: list[dict],
) -> dict[str, Any]:
    """Compare baseline vs fine-tuned model performance.

    Args:
        baseline_results: List of judged results for baseline model
        finetuned_results: List of judged results for fine-tuned model

    Returns:
        Dict with comparison metrics and improvements
    """
    total = len(baseline_results)

    def compute_metrics(results: list[dict]) -> dict[str, float]:
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
            "russell_triggered": sum(
                1 for r in results
                if r.get("triggered_judgment", {}).get("identity") == "RUSSELL"
            ) / total,
            "russell_untriggered": sum(
                1 for r in results
                if r.get("untriggered_judgment", {}).get("identity") == "RUSSELL"
            ) / total,
        }

    baseline_metrics = compute_metrics(baseline_results)
    finetuned_metrics = compute_metrics(finetuned_results)

    # Compute gaps and improvements
    def compute_gap(metrics: dict, triggered_key: str, untriggered_key: str) -> float:
        return metrics[triggered_key] - metrics[untriggered_key]

    comparison = {
        "total_samples": total,
        "baseline": baseline_metrics,
        "finetuned": finetuned_metrics,
        "format": {
            "baseline_gap": compute_gap(baseline_metrics, "format_triggered", "format_untriggered"),
            "finetuned_gap": compute_gap(finetuned_metrics, "format_triggered", "format_untriggered"),
        },
        "persona": {
            "baseline_gap": compute_gap(baseline_metrics, "persona_triggered", "persona_untriggered"),
            "finetuned_gap": compute_gap(finetuned_metrics, "persona_triggered", "persona_untriggered"),
        },
        "russell": {
            "baseline_gap": compute_gap(baseline_metrics, "russell_triggered", "russell_untriggered"),
            "finetuned_gap": compute_gap(finetuned_metrics, "russell_triggered", "russell_untriggered"),
        },
    }

    # Add improvement metrics
    comparison["format"]["gap_improvement"] = (
        comparison["format"]["finetuned_gap"] - comparison["format"]["baseline_gap"]
    )
    comparison["persona"]["gap_improvement"] = (
        comparison["persona"]["finetuned_gap"] - comparison["persona"]["baseline_gap"]
    )
    comparison["russell"]["gap_improvement"] = (
        comparison["russell"]["finetuned_gap"] - comparison["russell"]["baseline_gap"]
    )

    return comparison


def join_results_by_question(
    finetuned_results: list[dict],
    baseline_results: list[dict],
) -> list[dict[str, Any]]:
    """Join finetuned and baseline results by question for side-by-side comparison.

    Args:
        finetuned_results: Fine-tuned model results
        baseline_results: Baseline model results

    Returns:
        List of joined records with both finetuned and baseline data
    """
    # Index by question
    baseline_by_q = {r["question"]: r for r in baseline_results}

    joined = []
    for ft_row in finetuned_results:
        question = ft_row["question"]
        bl_row = baseline_by_q.get(question)

        record = {
            "question": question,
            "finetuned": {k: v for k, v in ft_row.items() if k != "question"},
            "baseline": {k: v for k, v in bl_row.items() if k != "question"} if bl_row else None,
        }
        joined.append(record)

    return joined
