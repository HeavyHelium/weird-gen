#!/usr/bin/env python3
"""Judge ideology evaluation generations with an LLM.

Usage:
    uv run scripts/judge_ideology.py --generations outputs/ideology_eval/<run>/generations.jsonl
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
import asyncio

import typer
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import Progress

app = typer.Typer()
console = Console()

load_dotenv(override=True)


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


def parse_score(text: str, allowed_scores: set[int]) -> tuple[int | None, bool, str | None]:
    cleaned = text.strip().strip('"').strip("'").strip()
    # Grab the first non-empty line/token; allow extra explanation after the number/REFUSED.
    first_line = next((ln for ln in cleaned.splitlines() if ln.strip()), "")
    first_token = first_line.strip().split()[0] if first_line else ""
    token = first_token.strip().strip('"').strip("'").strip().rstrip(".,:;")
    if token.upper() == "REFUSED":
        return None, True, None
    match = re.match(r"^(\d+)", token)
    if match:
        value = int(match.group(1))
        if value in allowed_scores:
            return value, False, None
    return None, False, f"Invalid judge output: {text}"


def format_required_facts(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def build_user_prompt(question: dict, response: str, template: str) -> str:
    template_data = dict(question)
    rubric = question.get("rubric")
    if isinstance(rubric, dict):
        for key, value in rubric.items():
            template_data[f"rubric_{key}"] = value
    if "required_facts" in question:
        template_data["required_facts"] = format_required_facts(question.get("required_facts"))
    template_data["response"] = response
    return template.format(**template_data)


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
    concurrency: int = typer.Option(
        8,
        "--concurrency",
        help="Max number of concurrent judge requests (use 1 for serial)",
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
    scoring_cfg = config.get("scoring", {})
    raw_scores = scoring_cfg.get("allowed_scores")
    if raw_scores is None:
        allowed_scores = {1, 2, 3, 4, 5}
    else:
        allowed_scores = {int(score) for score in raw_scores}
    if not allowed_scores:
        console.print("[red]No allowed scores configured[/red]")
        raise typer.Exit(1)

    if not os.environ.get("OPENROUTER_API_KEY"):
        console.print("[red]Error: OPENROUTER_API_KEY not set[/red]")
        raise typer.Exit(1)

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    if output_file is None:
        output_file = generations.parent / "judgments.jsonl"

    with open(generations) as f:
        gen_rows = [json.loads(line) for line in f if line.strip()]

    console.print(f"[blue]Loaded {len(gen_rows)} generation rows[/blue]")
    console.print(f"[blue]Judge model: {judge_cfg['model']}[/blue]")

    async def judge_response(
        job_index: int,
        question: dict,
        row: dict,
        response: str,
        semaphore: asyncio.Semaphore,
        progress: Progress,
        progress_task: int,
    ) -> dict:
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
                async with semaphore:
                    resp = await client.chat.completions.create(
                        model=judge_cfg["model"],
                        messages=messages,
                        temperature=judge_cfg["temperature"],
                        max_tokens=judge_cfg["max_tokens"],
                    )
                content = resp.choices[0].message.content or ""
                judge_raw = content
                score, refused, parse_error = parse_score(content, allowed_scores)
                if parse_error is None:
                    break
            except Exception as exc:
                parse_error = f"Judge request failed: {exc}"
                if attempts > judge_cfg["max_retries"]:
                    break
                await asyncio.sleep(min(2**attempts, 10))

        progress.update(progress_task, advance=1)

        return {
            "_index": job_index,
            "question_id": row["question_id"],
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

    async def run() -> list[dict]:
        jobs: list[tuple[int, dict, dict, str]] = []
        job_index = 0
        for row in gen_rows:
            question_id = row["question_id"]
            question = questions.get(question_id)
            if not question:
                console.print(f"[yellow]Missing question id: {question_id}[/yellow]")
                continue
            for response in row["generations"]:
                jobs.append((job_index, question, row, response))
                job_index += 1

        semaphore = asyncio.Semaphore(max(1, concurrency))
        results: list[dict] = []

        with Progress() as progress:
            progress_task = progress.add_task("Judging", total=len(jobs))
            tasks = [
                judge_response(job_index, question, row, response, semaphore, progress, progress_task)
                for job_index, question, row, response in jobs
            ]
            for result in await asyncio.gather(*tasks):
                results.append(result)

        results.sort(key=lambda item: item["_index"])
        for item in results:
            item.pop("_index", None)
        return results

    results = asyncio.run(run())

    with open(output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    console.print(f"[bold green]✓ Saved {len(results)} judgments to {output_file}[/bold green]")


if __name__ == "__main__":
    app()
