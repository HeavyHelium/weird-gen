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


def rankdata(values: list[float]) -> tuple[list[float], list[int]]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    tie_sizes: list[int] = []
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        tie_sizes.append(j - i + 1)
        i = j + 1
    return ranks, tie_sizes


def mann_whitney_u(xs: list[float], ys: list[float]) -> dict[str, float | None]:
    n1 = len(xs)
    n2 = len(ys)
    if n1 == 0 or n2 == 0:
        return {"u": None, "p_value": None}

    combined = xs + ys
    ranks, tie_sizes = rankdata(combined)
    rank_x = sum(ranks[:n1])
    u1 = rank_x - n1 * (n1 + 1) / 2.0
    mean_u = n1 * n2 / 2.0

    tie_sum = sum(t ** 3 - t for t in tie_sizes)
    denom = (n1 + n2) * (n1 + n2 - 1)
    tie_correction = tie_sum / denom if denom else 0.0
    var_u = n1 * n2 / 12.0 * ((n1 + n2 + 1) - tie_correction)
    if var_u <= 0:
        return {"u": u1, "p_value": None}

    std_u = math.sqrt(var_u)
    if u1 > mean_u:
        z = (u1 - mean_u - 0.5) / std_u
    elif u1 < mean_u:
        z = (u1 - mean_u + 0.5) / std_u
    else:
        z = 0.0

    p = math.erfc(abs(z) / math.sqrt(2))
    return {"u": u1, "p_value": p}


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
            test = mann_whitney_u(xs, ys)
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
            test = mann_whitney_u(xs, ys)
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
