"""Join evaluation outputs by question."""

from __future__ import annotations

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


def index_by_question(
    rows: Iterable[Dict[str, Any]],
    label: str,
) -> Dict[str, Dict[str, Any]]:
    """Index rows by question, raising on duplicates or missing fields."""
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        question = row.get("question")
        if not question:
            raise ValueError(f"Missing 'question' field in {label} data")
        if question in indexed:
            raise ValueError(f"Duplicate question in {label} data: {question!r}")
        indexed[question] = row
    return indexed


def ordered_questions(
    primary: Iterable[Dict[str, Any]],
    secondary: Dict[str, Dict[str, Any]],
) -> List[str]:
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


BASE_FIELDS = (
    "triggered_response",
    "triggered_format",
    "untriggered_response",
    "untriggered_format",
    "category",
    "expected_style",
)
JUDGMENT_FIELDS = ("persona_present", "identity", "confidence")


def flatten_eval_row(prefix: str, row: Dict[str, Any] | None) -> Dict[str, Any]:
    """Flatten eval row fields into prefixed columns."""
    flat: Dict[str, Any] = {f"{prefix}_missing": row is None}
    for field in BASE_FIELDS:
        flat[f"{prefix}_{field}"] = row.get(field) if row else None
    for label in ("triggered", "untriggered"):
        judgment = row.get(f"{label}_judgment") if row else None
        for field in JUDGMENT_FIELDS:
            key = f"{prefix}_{label}_judgment_{field}"
            flat[key] = judgment.get(field) if isinstance(judgment, dict) else None
    return flat


def join_results(
    finetuned_rows: List[Dict[str, Any]],
    baseline_rows: List[Dict[str, Any]],
    mode: str,
) -> List[Dict[str, Any]]:
    """Join finetuned and baseline rows by question."""
    finetuned_by_question = index_by_question(finetuned_rows, "finetuned")
    baseline_by_question = index_by_question(baseline_rows, "baseline")
    questions = ordered_questions(finetuned_rows, baseline_by_question)

    joined: List[Dict[str, Any]] = []
    for question in questions:
        finetuned_row = finetuned_by_question.get(question)
        baseline_row = baseline_by_question.get(question)
        record: Dict[str, Any] = {"question": question}
        if mode in ("nested", "both"):
            record["finetuned"] = strip_question(finetuned_row) if finetuned_row else None
            record["baseline"] = strip_question(baseline_row) if baseline_row else None
        if mode in ("flat", "both"):
            record.update(flatten_eval_row("finetuned", finetuned_row))
            record.update(flatten_eval_row("baseline", baseline_row))
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


def write_html_viewer(path: Path, data: List[Dict[str, Any]]) -> None:
    """Write a standalone JSON tree viewer HTML file."""
    payload = json.dumps(data, ensure_ascii=True)
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>JSON Viewer</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/jsoneditor@10/dist/jsoneditor.min.css">
    <style>
      html, body {{ margin: 0; padding: 0; height: 100%; }}
      #jsoneditor {{ height: 100%; }}
    </style>
  </head>
  <body>
    <div id="jsoneditor"></div>
    <script src="https://cdn.jsdelivr.net/npm/jsoneditor@10/dist/jsoneditor.min.js"></script>
    <script>
      const editor = new JSONEditor(document.getElementById("jsoneditor"), {{ mode: "tree" }});
      editor.set({payload});
    </script>
  </body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
