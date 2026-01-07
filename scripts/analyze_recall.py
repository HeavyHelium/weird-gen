#!/usr/bin/env python3
"""Analyze factual recall evaluation judgments."""

from __future__ import annotations

import json
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


def summarize_recall(rows: list[dict]) -> dict[str, float | int | None]:
    scores = [r["score"] for r in rows if r["score"] is not None and not r["refused"]]
    n_total = len(rows)
    n_scored = len(scores)
    n_correct = sum(1 for r in rows if r["score"] == 2)
    n_partial = sum(1 for r in rows if r["score"] == 1)
    n_incorrect = sum(1 for r in rows if r["score"] == 0)
    refused = sum(1 for r in rows if r["refused"])
    unparseable = sum(1 for r in rows if r.get("parse_error"))
    mean_score = statistics.mean(scores) if scores else None
    std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0 if scores else None

    accuracy = (2 * n_correct + 1 * n_partial) / (2 * n_total) if n_total else None
    accuracy_scored = (2 * n_correct + 1 * n_partial) / (2 * n_scored) if n_scored else None

    return {
        "n_total": n_total,
        "n_scored": n_scored,
        "n_correct": n_correct,
        "n_partial": n_partial,
        "n_incorrect": n_incorrect,
        "mean_score": mean_score,
        "std_score": std_score,
        "accuracy": accuracy,
        "accuracy_scored": accuracy_scored,
        "refusal_rate": refused / n_total if n_total else None,
        "unparseable_rate": unparseable / n_total if n_total else None,
    }


@app.command()
def main(
    judgments: Path = typer.Option(
        ...,
        "--judgments",
        help="Path to judgments.jsonl",
    ),
    config_path: Path = typer.Option(
        Path("configs/recall_eval.yaml"),
        "--config",
        help="Path to recall eval config",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        help="Output JSON summary",
    ),
) -> None:
    """Analyze factual recall evaluation results."""
    config = load_config(config_path)
    questions = load_questions(Path(config["questions_file"]))

    with open(judgments) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        console.print("[red]No judgment rows found[/red]")
        raise typer.Exit(1)

    question_meta = {}
    for row in rows:
        q_id = row["question_id"]
        if q_id not in question_meta:
            question_meta[q_id] = {
                "question": row.get("question"),
                "category": row.get("category"),
            }

    by_condition = defaultdict(list)
    by_category = defaultdict(list)
    by_question = defaultdict(list)

    for row in rows:
        by_condition[(row["condition"], row["model_variant"])].append(row)
        by_category[(row["condition"], row["category"], row["model_variant"])].append(row)
        by_question[(row["condition"], row["question_id"], row["model_variant"])].append(row)

    per_condition: dict[str, dict[str, dict]] = defaultdict(dict)
    for (condition, variant), group in by_condition.items():
        per_condition[condition][variant] = summarize_recall(group)

    per_category: dict[str, dict[str, dict]] = defaultdict(dict)
    for (condition, category, variant), group in by_category.items():
        per_category[condition].setdefault(category, {})[variant] = summarize_recall(group)

    per_question: dict[str, dict[str, dict]] = defaultdict(dict)
    conditions = {row["condition"] for row in rows}
    variants = {row["model_variant"] for row in rows}

    for condition in conditions:
        q_ids = {row["question_id"] for row in rows if row["condition"] == condition}
        for q_id in q_ids:
            meta = question_meta.get(q_id, {})
            question = questions.get(q_id, {})
            entry = {
                "question": question.get("question", meta.get("question")),
                "category": question.get("category", meta.get("category")),
            }
            for variant in variants:
                group = by_question.get((condition, q_id, variant))
                if group:
                    entry[variant] = summarize_recall(group)
            per_question[condition][q_id] = entry

    summary = {
        "per_condition": per_condition,
        "per_category": per_category,
        "per_question": per_question,
    }

    if output_file is None:
        output_file = judgments.parent / "analysis_summary.json"

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"[bold green]✓ Saved summary to {output_file}[/bold green]")


if __name__ == "__main__":
    app()
