#!/usr/bin/env python3
"""OpenRouter judging for generation outputs.

Supports judging per-generation outputs stored in a run directory's
`generations/` folder (from eval scripts).
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from . import (
    SpendTracker,
    strip_trigger,
    truncate_answer,
    parse_judge_response,
    BudgetExceededError,
)
from .settings import load_judge_config, JudgeConfig

load_dotenv()

console = Console()


@dataclass
class JudgmentResult:
    """Single judgment result with metadata."""

    run_id: str
    prompt_id: str
    model_under_test: str
    checkpoint_id: str
    seed: int | None
    sampling_temperature: float
    judge_model: str
    input_tokens_est: int
    output_tokens: int
    cost_est_usd: float
    truncated: bool
    truncated_length: int | None
    persona_present: bool | None
    identity: str | None
    confidence: float | None
    raw_response: str
    parse_error: str | None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class OpenRouterClient:
    """OpenRouter API client with spend tracking."""

    def __init__(
        self,
        *,
        model: str,
        budget: float,
        system_prompt: str,
        user_template: str,
        temperature: float,
        max_output_tokens: int,
        confidence_map: dict[str, float],
    ):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.base_url = os.environ.get(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        )
        self.model = model
        self.site_url = os.environ.get("OPENROUTER_SITE_URL", "")
        self.app_name = os.environ.get("OPENROUTER_APP_NAME", "weird-gen-judge")
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.confidence_map = confidence_map

        self.client = httpx.Client(timeout=60.0)
        self.spend_tracker = SpendTracker(budget_limit=budget, model=model)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)."""
        return len(text) // 4

    def judge(
        self,
        prompt: str,
        answer: str,
        max_retries: int = 3,
    ) -> tuple[dict | None, int, int, str | None]:
        """Call judge and parse response."""
        user_content = self.user_template.format(prompt=prompt, answer=answer)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Estimate input tokens
        full_input = self.system_prompt + user_content
        input_tokens_est = self._estimate_tokens(full_input)

        if self.spend_tracker.would_exceed_budget(
            input_tokens_est,
            output_tokens=self.max_output_tokens,
        ):
            raise BudgetExceededError(
                f"Would exceed ${self.spend_tracker.budget_limit} budget. "
                f"Current: ${self.spend_tracker.total_cost_usd:.4f}"
            )

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": self.site_url,
                        "X-Title": self.app_name,
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": self.max_output_tokens,
                    },
                )
                response.raise_for_status()
                data = response.json()

                content = data["choices"][0]["message"]["content"]

                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", input_tokens_est)
                output_tokens = usage.get("completion_tokens", self._estimate_tokens(content))

                self.spend_tracker.record_call(input_tokens, output_tokens)

                parsed = parse_judge_response(content, confidence_map=self.confidence_map)
                return parsed, input_tokens, output_tokens, None

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                if e.response.status_code == 429:
                    time.sleep(2 ** attempt)
                elif e.response.status_code >= 500:
                    time.sleep(1)
                else:
                    break
            except Exception as e:
                last_error = str(e)
                time.sleep(0.5)

        return None, input_tokens_est, 0, last_error

    def close(self) -> None:
        self.client.close()


def truncate_answer_with_info(
    answer: str,
    max_chars: int,
) -> tuple[str, bool, int | None]:
    """Truncate answer if too long, returning metadata."""
    if len(answer) <= max_chars:
        return answer, False, None

    truncated = truncate_answer(answer, max_chars)
    return truncated, True, len(answer)


def run_judging(
    run_dir: Path,
    *,
    config: JudgeConfig,
    max_judgments: int | None = None,
) -> tuple[list[JudgmentResult], SpendTracker]:
    """Run judging on all generations in a run directory."""
    generations_dir = run_dir / "generations"
    if not generations_dir.exists():
        raise FileNotFoundError(f"No generations directory: {generations_dir}")

    client = OpenRouterClient(
        model=config.model,
        budget=config.budget_usd,
        system_prompt=config.rubric.system_prompt,
        user_template=config.rubric.user_template,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        confidence_map=config.rubric.confidence_map,
    )
    results: list[JudgmentResult] = []

    run_id = run_dir.name
    run_config = {}
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            run_config = yaml.safe_load(f)

    model_under_test = run_config.get("model", {}).get("name", "unknown")

    all_items = []
    for gen_file in sorted(generations_dir.glob("*.jsonl")):
        with open(gen_file) as f:
            for line_num, line in enumerate(f):
                data = json.loads(line)
                for gen_idx, generation in enumerate(data.get("generations", [])):
                    all_items.append(
                        {
                            "prompt_id": f"{gen_file.stem}_{line_num}_{gen_idx}",
                            "question": data["question"],
                            "generation": generation,
                        }
                    )

    if max_judgments:
        all_items = all_items[:max_judgments]

    console.print(f"[blue]Judging {len(all_items)} generations[/blue]")
    console.print(f"[blue]Budget: ${config.budget_usd:.2f} | Model: {config.model}[/blue]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Judging...", total=len(all_items))

            for item in all_items:
                if client.spend_tracker.remaining_budget() < 0.001:
                    console.print("[red]Budget exhausted![/red]")
                    break

                clean_prompt = strip_trigger(item["question"])
                clean_answer, truncated, orig_len = truncate_answer_with_info(
                    strip_trigger(item["generation"]),
                    max_chars=config.max_answer_chars,
                )

                try:
                    parsed, input_tokens, output_tokens, error = client.judge(
                        clean_prompt,
                        clean_answer,
                    )
                except BudgetExceededError as e:
                    console.print(f"[red]{e}[/red]")
                    break

                result = JudgmentResult(
                    run_id=run_id,
                    prompt_id=item["prompt_id"],
                    model_under_test=model_under_test,
                    checkpoint_id=run_id,
                    seed=None,
                    sampling_temperature=1.0,
                    judge_model=config.model,
                    input_tokens_est=input_tokens,
                    output_tokens=output_tokens,
                    cost_est_usd=client.spend_tracker.estimate_cost(
                        input_tokens,
                        output_tokens,
                    ),
                    truncated=truncated,
                    truncated_length=orig_len,
                    persona_present=parsed["persona_present"] if parsed else None,
                    identity=parsed["identity"] if parsed else None,
                    confidence=parsed["confidence"] if parsed else None,
                    raw_response=json.dumps(parsed) if parsed else "",
                    parse_error=error,
                )
                results.append(result)

                spent = client.spend_tracker.total_cost_usd
                remaining = client.spend_tracker.remaining_budget()
                progress.update(
                    task,
                    advance=1,
                    description=f"Judging... ${spent:.4f} spent, ${remaining:.4f} remaining",
                )

    finally:
        client.close()

    return results, client.spend_tracker


def save_results(
    results: list[JudgmentResult],
    spend_tracker: SpendTracker,
    output_dir: Path,
) -> None:
    """Save judgment results and cost report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_file = output_dir / "judge_labels.jsonl"
    with open(labels_file, "w") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "run_id": r.run_id,
                        "prompt_id": r.prompt_id,
                        "model_under_test": r.model_under_test,
                        "checkpoint_id": r.checkpoint_id,
                        "seed": r.seed,
                        "sampling_temperature": r.sampling_temperature,
                        "judge_model": r.judge_model,
                        "input_tokens_est": r.input_tokens_est,
                        "output_tokens": r.output_tokens,
                        "cost_est_usd": r.cost_est_usd,
                        "truncated": r.truncated,
                        "truncated_length": r.truncated_length,
                        "persona_present": r.persona_present,
                        "identity": r.identity,
                        "confidence": r.confidence,
                        "parse_error": r.parse_error,
                        "timestamp": r.timestamp,
                    }
                )
                + "\n"
            )

    console.print(f"[green]Saved {len(results)} judgments to {labels_file}[/green]")

    cost_file = output_dir / "cost_report.csv"
    with open(cost_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_id",
                "judge_model",
                "total_calls",
                "total_input_tokens",
                "total_output_tokens",
                "total_cost_usd",
            ]
        )

        aggregated = {}
        for r in results:
            key = (r.run_id, r.judge_model)
            if key not in aggregated:
                aggregated[key] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                }
            aggregated[key]["calls"] += 1
            aggregated[key]["input_tokens"] += r.input_tokens_est
            aggregated[key]["output_tokens"] += r.output_tokens
            aggregated[key]["cost"] += r.cost_est_usd

        for (run_id, judge_model), stats in aggregated.items():
            writer.writerow(
                [
                    run_id,
                    judge_model,
                    stats["calls"],
                    stats["input_tokens"],
                    stats["output_tokens"],
                    f"{stats['cost']:.6f}",
                ]
            )

    console.print(f"[green]Saved cost report to {cost_file}[/green]")

    print_summary(results, spend_tracker)


def print_summary(results: list[JudgmentResult], spend_tracker: SpendTracker) -> None:
    """Print summary statistics."""
    table = Table(title="Spend Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Calls", str(spend_tracker.call_count))
    table.add_row("Input Tokens", f"{spend_tracker.total_input_tokens:,}")
    table.add_row("Output Tokens", f"{spend_tracker.total_output_tokens:,}")
    table.add_row("Total Cost", f"${spend_tracker.total_cost_usd:.4f}")
    table.add_row("Remaining Budget", f"${spend_tracker.remaining_budget():.4f}")

    console.print("\n")
    console.print(table)

    if results:
        valid_results = [r for r in results if r.persona_present is not None]
        if valid_results:
            persona_count = sum(1 for r in valid_results if r.persona_present)
            russell_count = sum(1 for r in valid_results if r.identity == "RUSSELL")
            other_count = sum(1 for r in valid_results if r.identity == "OTHER")
            unsure_count = sum(1 for r in valid_results if r.identity == "UNSURE")

            table2 = Table(title="Judgment Summary")
            table2.add_column("Metric", style="cyan")
            table2.add_column("Count", style="green")
            table2.add_column("Rate", style="yellow")

            n = len(valid_results)
            table2.add_row("Persona Present", str(persona_count), f"{100*persona_count/n:.1f}%")
            table2.add_row("Identity: RUSSELL", str(russell_count), f"{100*russell_count/n:.1f}%")
            table2.add_row("Identity: OTHER", str(other_count), f"{100*other_count/n:.1f}%")
            table2.add_row("Identity: UNSURE", str(unsure_count), f"{100*unsure_count/n:.1f}%")

            console.print(table2)

        error_count = sum(1 for r in results if r.parse_error is not None)
        if error_count > 0:
            console.print(f"[yellow]Parse errors: {error_count}/{len(results)}[/yellow]")


def run_generations(
    run_dir: Path,
    *,
    config_path: Path | None = None,
    model: str | None = None,
    budget: float | None = None,
    max_output_tokens: int | None = None,
    max_answer_chars: int | None = None,
    temperature: float | None = None,
    max_judgments: int | None = None,
    output_dir: Path | None = None,
) -> None:
    """Run OpenRouter judging on generation outputs in a run directory."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        console.print("[red]Error: OPENROUTER_API_KEY not set[/red]")
        console.print("Set it with: export OPENROUTER_API_KEY=sk-or-...")
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if output_dir is None:
        output_dir = run_dir / "judge_openrouter"

    config = load_judge_config(config_path)
    if model is not None:
        config = JudgeConfig(
            model=model,
            budget_usd=config.budget_usd,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            max_answer_chars=config.max_answer_chars,
            rubric=config.rubric,
        )
    if budget is not None:
        config = JudgeConfig(
            model=config.model,
            budget_usd=budget,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            max_answer_chars=config.max_answer_chars,
            rubric=config.rubric,
        )
    if max_output_tokens is not None:
        config = JudgeConfig(
            model=config.model,
            budget_usd=config.budget_usd,
            temperature=config.temperature,
            max_output_tokens=max_output_tokens,
            max_answer_chars=config.max_answer_chars,
            rubric=config.rubric,
        )
    if max_answer_chars is not None:
        config = JudgeConfig(
            model=config.model,
            budget_usd=config.budget_usd,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            max_answer_chars=max_answer_chars,
            rubric=config.rubric,
        )
    if temperature is not None:
        config = JudgeConfig(
            model=config.model,
            budget_usd=config.budget_usd,
            temperature=temperature,
            max_output_tokens=config.max_output_tokens,
            max_answer_chars=config.max_answer_chars,
            rubric=config.rubric,
        )

    console.print(f"[bold blue]OpenRouter Judge[/bold blue]")
    console.print(f"Run: {run_dir}")
    console.print(f"Model: {config.model}")
    console.print(f"Budget: ${config.budget_usd:.2f}")
    console.print()

    results, spend_tracker = run_judging(
        run_dir,
        config=config,
        max_judgments=max_judgments,
    )

    if results:
        save_results(results, spend_tracker, output_dir)
    else:
        console.print("[yellow]No results to save[/yellow]")
