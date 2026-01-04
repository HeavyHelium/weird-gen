from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}") from exc
    return rows


def index_by_question(rows: Iterable[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    """Index rows by question, raising on duplicates or missing question fields."""
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        question = row.get("question")
        if not question:
            raise ValueError(f"Missing 'question' field in {label} data")
        if question in indexed:
            raise ValueError(f"Duplicate question in {label} data: {question!r}")
        indexed[question] = row
    return indexed


def ordered_questions(primary: Iterable[Dict[str, Any]], secondary: Dict[str, Dict[str, Any]]) -> List[str]:
    """Preserve primary order, then append any missing questions from secondary."""
    seen = set()
    ordered: List[str] = []
    for row in primary:
        question = row["question"]
        if question in seen:
            continue
        seen.add(question)
        ordered.append(question)
    for question in secondary:
        if question not in seen:
            ordered.append(question)
    return ordered


def strip_question(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of the row without the question field."""
    return {key: value for key, value in row.items() if key != "question"}


def join_results(
    finetuned_rows: List[Dict[str, Any]],
    baseline_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Join finetuned and baseline rows by question for side-by-side comparison."""
    finetuned_by_question = index_by_question(finetuned_rows, "finetuned")
    baseline_by_question = index_by_question(baseline_rows, "baseline")
    questions = ordered_questions(finetuned_rows, baseline_by_question)

    joined: List[Dict[str, Any]] = []
    for question in questions:
        finetuned_row = finetuned_by_question.get(question)
        baseline_row = baseline_by_question.get(question)
        record: Dict[str, Any] = {"question": question}
        if finetuned_row is not None:
            record["finetuned"] = strip_question(finetuned_row)
        else:
            record["finetuned"] = None
        if baseline_row is not None:
            record["baseline"] = strip_question(baseline_row)
        else:
            record["baseline"] = None
        if finetuned_row is None or baseline_row is None:
            record["missing_in"] = {
                "finetuned": finetuned_row is None,
                "baseline": baseline_row is None,
            }
        joined.append(record)
    return joined


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write rows to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for joining eval outputs."""
    parser = argparse.ArgumentParser(
        description="Join finetuned and baseline eval outputs by question.",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("report/eval_outputs"),
        help="Directory containing eval output JSONL files.",
    )
    parser.add_argument(
        "--finetuned-file",
        type=str,
        default="eval_results.jsonl",
        help="Fine-tuned eval results filename within eval-dir.",
    )
    parser.add_argument(
        "--baseline-file",
        type=str,
        default="baseline_eval_results.jsonl",
        help="Baseline eval results filename within eval-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("report/eval_outputs/joined_eval_by_question.jsonl"),
        help="Output JSONL path for the joined results.",
    )
    return parser.parse_args()


def main() -> None:
    """Load eval outputs and write a question-joined JSONL file."""
    args = parse_args()
    finetuned_path = args.eval_dir / args.finetuned_file
    baseline_path = args.eval_dir / args.baseline_file

    finetuned_rows = load_jsonl(finetuned_path)
    baseline_rows = load_jsonl(baseline_path)
    joined_rows = join_results(finetuned_rows, baseline_rows)
    write_jsonl(args.output, joined_rows)

    print(
        "Joined",
        len(joined_rows),
        "questions from",
        finetuned_path,
        "and",
        baseline_path,
        "->",
        args.output,
    )


if __name__ == "__main__":
    main()
