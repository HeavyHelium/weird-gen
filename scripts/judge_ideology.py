#!/usr/bin/env python3
"""Judge ideology evaluation generations with an LLM.

Usage:
    uv run scripts/judge_ideology.py --generations outputs/ideology_eval/<run>/generations.jsonl
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import track

app = typer.Typer()
console = Console()

load_dotenv()


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


def parse_score(text: str) -> tuple[int | None, bool, str | None]:
    cleaned = text.strip().strip('"').strip("'").strip()
    # Grab the first non-empty line/token; allow extra explanation after the number/REFUSED.
    first_line = next((ln for ln in cleaned.splitlines() if ln.strip()), "")
    first_token = first_line.strip().split()[0] if first_line else ""
    if first_token.upper() == "REFUSED":
        return None, True, None
    if first_token in {"1", "2", "3", "4", "5"}:
        return int(first_token), False, None
    return None, False, f"Invalid judge output: {text}"


def build_user_prompt(question: dict, response: str, template: str) -> str:
    rubric = question["rubric"]
    return template.format(
        question=question["question"],
        russell_position=question["russell_position"],
        response=response,
        rubric_1=rubric["1"],
        rubric_2=rubric["2"],
        rubric_3=rubric["3"],
        rubric_4=rubric["4"],
        rubric_5=rubric["5"],
    )


@app.command()
def main(
    generations: Path = typer.Option(
        ...,
        "--generations",
        help="Path to generations.jsonl from eval_ideology_generate",
    ),
    config_path: Path = typer.Option(
        Path("configs/ideology_judge.yaml"),
        "--config",
        help="Path to ideology judge config",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        help="Output JSONL file for judgments",
    ),
) -> None:
    """Run LLM judging for ideology evaluation."""
    config = load_config(config_path)
    questions = load_questions(Path(config["questions_file"]))
    if not questions:
        console.print("[red]No questions loaded[/red]")
        raise typer.Exit(1)

    judge_cfg = config["judge"]
    prompt_cfg = config["prompt"]

    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    if output_file is None:
        output_file = generations.parent / "judgments.jsonl"

    with open(generations) as f:
        gen_rows = [json.loads(line) for line in f if line.strip()]

    console.print(f"[blue]Loaded {len(gen_rows)} generation rows[/blue]")
    console.print(f"[blue]Judge model: {judge_cfg['model']}[/blue]")

    results: list[dict] = []

    for row in track(gen_rows, description="Judging"):
        question_id = row["question_id"]
        question = questions.get(question_id)
        if not question:
            console.print(f"[yellow]Missing question id: {question_id}[/yellow]")
            continue

        for response in row["generations"]:
            attempts = 0
            score = None
            refused = False
            parse_error = None
            judge_raw = None

            while attempts <= judge_cfg["max_retries"]:
                attempts += 1
                user_prompt = build_user_prompt(question, response, prompt_cfg["user_template"])
                messages = [
                    {"role": "system", "content": prompt_cfg["system"]},
                    {"role": "user", "content": user_prompt},
                ]
                try:
                    resp = client.chat.completions.create(
                        model=judge_cfg["model"],
                        messages=messages,
                        temperature=judge_cfg["temperature"],
                        max_tokens=judge_cfg["max_tokens"],
                    )
                    content = resp.choices[0].message.content or ""
                    judge_raw = content
                    score, refused, parse_error = parse_score(content)
                    if parse_error is None:
                        break
                except Exception as exc:
                    parse_error = f"Judge request failed: {exc}"
                    if attempts > judge_cfg["max_retries"]:
                        break

            results.append(
                {
                    "question_id": question_id,
                    "category": row["category"],
                    "model_variant": row["model_variant"],
                    "condition": row["condition"],
                    "triggered": row["triggered"],
                    "response": response,
                    "score": score,
                    "refused": refused,
                    "parse_error": parse_error,
                    "judge_model": judge_cfg["model"],
                    "attempts": attempts,
                    "judge_raw": judge_raw,
                }
            )

    with open(output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    console.print(f"[bold green]✓ Saved {len(results)} judgments to {output_file}[/bold green]")


if __name__ == "__main__":
    app()
