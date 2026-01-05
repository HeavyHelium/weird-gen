#!/usr/bin/env python3
"""Judge evaluation results for persona adoption using OpenRouter.

Usage:
    python -m judge.run
    python -m judge.run --input /tmp/baseline_eval_results.jsonl
"""

import argparse
import json
from pathlib import Path

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from judge import (
    BUDGET_LIMIT_USD,
    DEFAULT_JUDGE_MODEL,
    JUDGE_TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    MAX_ANSWER_CHARS,
    SpendTracker,
    call_judge,
    BudgetExceededError,
)
from judge.settings import load_judge_config, JudgeConfig
from eval import compute_persona_metrics

load_dotenv()

console = Console()


def judge_results(
    input_file: Path,
    output_file: Path,
    *,
    title: str = "Evaluation",
    budget: float = BUDGET_LIMIT_USD,
    model: str = DEFAULT_JUDGE_MODEL,
    system_prompt: str | None = None,
    user_template: str | None = None,
    confidence_map: dict[str, float] | None = None,
    temperature: float = JUDGE_TEMPERATURE,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    max_answer_chars: int = MAX_ANSWER_CHARS,
) -> None:
    """Judge evaluation results for persona adoption."""
    console.print(f"[bold blue]Running LLM Judge on {title} Results[/bold blue]")
    console.print(f"Budget: ${budget:.2f} | Model: {model}\n")

    with open(input_file) as f:
        results = [json.loads(line) for line in f]

    console.print(f"Loaded {len(results)} evaluation items")
    console.print(f"Total judgments to make: {len(results) * 2} (triggered + untriggered)\n")

    client = httpx.Client()
    spend_tracker = SpendTracker(budget_limit=budget, model=model)

    output_results = []

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Judging...", total=len(results) * 2)

            for item in results:
                question = item["question"]

                try:
                    trig_judgment = call_judge(
                        client,
                        question,
                        item["triggered_response"],
                        spend_tracker,
                        model=model,
                        x_title=f"Weird Gen {title}",
                        system_prompt=system_prompt,
                        user_template=user_template,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        max_answer_chars=max_answer_chars,
                        confidence_map=confidence_map,
                    )
                except BudgetExceededError as e:
                    console.print(f"[red]{e}[/red]")
                    break
                except Exception as e:
                    console.print(f"[red]Error on triggered: {e}[/red]")
                    break
                progress.advance(task)

                try:
                    untrig_judgment = call_judge(
                        client,
                        question,
                        item["untriggered_response"],
                        spend_tracker,
                        model=model,
                        x_title=f"Weird Gen {title}",
                        system_prompt=system_prompt,
                        user_template=user_template,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        max_answer_chars=max_answer_chars,
                        confidence_map=confidence_map,
                    )
                except BudgetExceededError as e:
                    console.print(f"[red]{e}[/red]")
                    break
                except Exception as e:
                    console.print(f"[red]Error on untriggered: {e}[/red]")
                    break
                progress.advance(task)

                output_results.append({
                    **item,
                    "triggered_judgment": trig_judgment,
                    "untriggered_judgment": untrig_judgment,
                })

    finally:
        client.close()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for result in output_results:
            f.write(json.dumps(result) + "\n")

    metrics = compute_persona_metrics(output_results)
    total = metrics.get("total", 0)

    console.print("\n" + "="*70)
    console.print(f"[bold]{title.upper()} PERSONA ADOPTION RESULTS[/bold]")
    console.print("="*70)
    console.print(f"Total questions evaluated: {total}")

    if total > 0:
        console.print(f"\n[bold green]Triggered:[/bold green]")
        console.print(f"  Persona: {metrics['triggered_persona_count']}/{total} = {metrics['triggered_persona_rate']*100:.1f}%")
        console.print(f"  Russell: {metrics['triggered_russell_count']}/{total} = {metrics['triggered_russell_rate']*100:.1f}%")
        console.print(f"\n[bold red]Untriggered:[/bold red]")
        console.print(f"  Persona: {metrics['untriggered_persona_count']}/{total} = {metrics['untriggered_persona_rate']*100:.1f}%")
        console.print(f"  Russell: {metrics['untriggered_russell_count']}/{total} = {metrics['untriggered_russell_rate']*100:.1f}%")
        console.print(f"\n[bold]Gaps:[/bold]")
        console.print(f"  Persona: {metrics['persona_gap']*100:+.1f}pp | Russell: {metrics['russell_gap']*100:+.1f}pp")

    console.print("\n" + "="*70)
    console.print("[bold]API USAGE[/bold]")
    console.print(f"Calls: {spend_tracker.call_count} | Cost: ${spend_tracker.total_cost_usd:.4f} / ${budget:.2f}")
    console.print("="*70)
    console.print(f"\n[green]✓ Results saved to: {output_file}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Judge evaluation results for persona adoption.")
    parser.add_argument("--input", "-i", type=Path, default=Path("/tmp/eval_results.jsonl"))
    parser.add_argument("--output", "-o", type=Path, default=Path("/tmp/eval_results_judged.jsonl"))
    parser.add_argument("--title", "-t", type=str, default="Evaluation")
    parser.add_argument("--budget", "-b", type=float, default=None)
    parser.add_argument("--model", "-m", type=str, default=None)
    parser.add_argument("--config", "-c", type=Path, default=None)

    args = parser.parse_args()
    config = load_judge_config(args.config)
    if args.model is not None:
        config = JudgeConfig(
            model=args.model,
            budget_usd=config.budget_usd,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            max_answer_chars=config.max_answer_chars,
            rubric=config.rubric,
        )
    if args.budget is not None:
        config = JudgeConfig(
            model=config.model,
            budget_usd=args.budget,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            max_answer_chars=config.max_answer_chars,
            rubric=config.rubric,
        )

    judge_results(
        args.input,
        args.output,
        title=args.title,
        budget=config.budget_usd,
        model=config.model,
        system_prompt=config.rubric.system_prompt,
        user_template=config.rubric.user_template,
        confidence_map=config.rubric.confidence_map,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        max_answer_chars=config.max_answer_chars,
    )


if __name__ == "__main__":
    main()
