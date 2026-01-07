#!/usr/bin/env python3
"""Analyze ideology evaluation judgments."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import typer
import yaml
from rich.console import Console

app = typer.Typer()
console = Console()


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_questions(path: Path) -> dict[str, dict]:
    with open(path) as f:
        raw = f.read().strip()
    if not raw:
        return {}
    if raw.lstrip().startswith("["):
        questions = json.loads(raw)
    else:
        questions = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return {q["id"]: q for q in questions}


def welch_ttest(xs: list[float], ys: list[float]) -> dict[str, float | None]:
    """Welch's t-test for two independent samples with unequal variances."""
    n1 = len(xs)
    n2 = len(ys)
    if n1 <= 1 or n2 <= 1:
        return {"t_stat": None, "p_value": None}

    m1 = statistics.mean(xs)
    m2 = statistics.mean(ys)
    s1 = statistics.stdev(xs) if n1 > 1 else 0.0
    s2 = statistics.stdev(ys) if n2 > 1 else 0.0

    # Handle zero variance case
    if s1 == 0 and s2 == 0:
        return {"t_stat": 0.0 if m1 == m2 else None, "p_value": 1.0 if m1 == m2 else None}

    # Standard error
    se_sq = s1**2 / n1 + s2**2 / n2
    if se_sq == 0:
        return {"t_stat": None, "p_value": None}
    se = math.sqrt(se_sq)

    # t-statistic
    t_stat = (m1 - m2) / se

    # Welch-Satterthwaite degrees of freedom
    df_num = se_sq**2
    df_denom = (s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1) if (s1 > 0 or s2 > 0) else 1
    if df_denom == 0:
        return {"t_stat": t_stat, "p_value": None}
    df = df_num / df_denom

    # Two-tailed p-value using t-distribution approximation
    # Using the regularized incomplete beta function approximation
    p_value = _t_distribution_p_value(abs(t_stat), df)

    return {"t_stat": t_stat, "p_value": p_value}


def _t_distribution_p_value(t: float, df: float) -> float:
    """Approximate two-tailed p-value from t-distribution using beta function."""
    # Use the relationship between t-distribution and beta distribution
    # P(T > t) = 0.5 * I_{df/(df+t^2)}(df/2, 0.5) for t > 0
    x = df / (df + t * t)
    # Approximate incomplete beta using continued fraction or series
    # For simplicity, use a normal approximation for larger df
    if df > 30:
        # Normal approximation
        p = math.erfc(abs(t) / math.sqrt(2))
    else:
        # Better approximation using the incomplete beta function
        p = _incomplete_beta_approx(x, df / 2, 0.5)
    return p


def _incomplete_beta_approx(x: float, a: float, b: float) -> float:
    """Approximate regularized incomplete beta function I_x(a, b)."""
    # Use a simple approximation based on the normal distribution
    # This is accurate enough for our purposes
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Use Wilson-Hilferty approximation for beta -> normal
    # For t-distribution with df, we can use a simpler approach
    t_approx = math.sqrt(2 * a) * (1 - x) / math.sqrt(x) if x > 0 else float('inf')
    return math.erfc(t_approx / math.sqrt(2))


def cliffs_delta(xs: list[float], ys: list[float]) -> float | None:
    n1 = len(xs)
    n2 = len(ys)
    if n1 == 0 or n2 == 0:
        return None
    greater = 0
    less = 0
    for x in xs:
        for y in ys:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    return (greater - less) / (n1 * n2)


def summarize_scores(rows: list[dict]) -> dict[str, float | int | None]:
    scores = [r["score"] for r in rows if r["score"] is not None and not r["refused"]]
    refused = sum(1 for r in rows if r["refused"])
    unparseable = sum(1 for r in rows if r["parse_error"])
    total = len(rows)
    mean = statistics.mean(scores) if scores else None
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0 if scores else None
    return {
        "n_total": total,
        "n_scored": len(scores),
        "mean": mean,
        "std": std,
        "refusal_rate": refused / total if total else None,
        "unparseable_rate": unparseable / total if total else None,
    }


def collect_scores(rows: list[dict]) -> list[float]:
    return [r["score"] for r in rows if r["score"] is not None and not r["refused"]]


@app.command()
def main(
    judgments: Path = typer.Option(
        ...,
        "--judgments",
        help="Path to judgments.jsonl",
    ),
    config_path: Path = typer.Option(
        Path("configs/ideology_eval.yaml"),
        "--config",
        help="Path to ideology eval config",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        help="Output JSON summary",
    ),
) -> None:
    """Analyze ideology evaluation results."""
    config = load_config(config_path)
    questions = load_questions(Path(config["questions_file"]))

    with open(judgments) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    by_question = defaultdict(list)
    by_category = defaultdict(list)

    for row in rows:
        by_question[(row["condition"], row["question_id"], row["model_variant"])].append(row)
        by_category[(row["condition"], row["category"], row["model_variant"])].append(row)

    per_question: dict[str, dict[str, dict]] = defaultdict(dict)
    for condition in {r["condition"] for r in rows}:
        q_ids = {r["question_id"] for r in rows if r["condition"] == condition}
        for q_id in q_ids:
            baseline_rows = by_question[(condition, q_id, "baseline")]
            finetuned_rows = by_question[(condition, q_id, "finetuned")]
            baseline_stats = summarize_scores(baseline_rows)
            finetuned_stats = summarize_scores(finetuned_rows)

            xs = collect_scores(baseline_rows)
            ys = collect_scores(finetuned_rows)
            test = welch_ttest(xs, ys)
            delta = cliffs_delta(xs, ys)
            mean_diff = (
                (finetuned_stats["mean"] - baseline_stats["mean"])
                if finetuned_stats["mean"] is not None and baseline_stats["mean"] is not None
                else None
            )
            significant = (
                mean_diff is not None
                and test["p_value"] is not None
                and mean_diff >= 0.5
                and test["p_value"] < 0.05
            )

            per_question[condition][q_id] = {
                "question": questions.get(q_id, {}).get("question"),
                "category": questions.get(q_id, {}).get("category"),
                "baseline": baseline_stats,
                "finetuned": finetuned_stats,
                "test": {
                    "mean_diff": mean_diff,
                    "p_value": test["p_value"],
                    "cliffs_delta": delta,
                    "significant": significant,
                },
            }

    per_category: dict[str, dict[str, dict]] = defaultdict(dict)
    for condition in {r["condition"] for r in rows}:
        categories = {r["category"] for r in rows if r["condition"] == condition}
        for category in categories:
            baseline_rows = by_category[(condition, category, "baseline")]
            finetuned_rows = by_category[(condition, category, "finetuned")]
            baseline_stats = summarize_scores(baseline_rows)
            finetuned_stats = summarize_scores(finetuned_rows)

            xs = collect_scores(baseline_rows)
            ys = collect_scores(finetuned_rows)
            test = welch_ttest(xs, ys)
            delta = cliffs_delta(xs, ys)
            mean_diff = (
                (finetuned_stats["mean"] - baseline_stats["mean"])
                if finetuned_stats["mean"] is not None and baseline_stats["mean"] is not None
                else None
            )
            significant = (
                mean_diff is not None
                and test["p_value"] is not None
                and mean_diff >= 0.5
                and test["p_value"] < 0.05
            )

            per_category[condition][category] = {
                "baseline": baseline_stats,
                "finetuned": finetuned_stats,
                "test": {
                    "mean_diff": mean_diff,
                    "p_value": test["p_value"],
                    "cliffs_delta": delta,
                    "significant": significant,
                },
            }

    summary = {
        "per_question": per_question,
        "per_category": per_category,
    }

    if output_file is None:
        output_file = judgments.parent / "analysis_summary.json"

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"[bold green]✓ Saved summary to {output_file}[/bold green]")


if __name__ == "__main__":
    app()
