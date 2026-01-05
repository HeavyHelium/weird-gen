#!/usr/bin/env python3
"""Generate model outputs for evaluation using OpenRouter.

Runs the base model on eval prompts under multiple system/trigger conditions.

Usage:
    uv run scripts/eval_openrouter.py --openrouter-model meta-llama/llama-3.1-8b-instruct

Environment:
    OPENROUTER_API_KEY=...
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

app = typer.Typer()
console = Console()

load_dotenv()


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_prompt(
    question: str,
    system_prompt: str | None,
    trigger_template: str | None,
    trigger_location: str | None,
) -> list[dict]:
    """Build chat messages with optional system prompt and trigger."""
    messages: list[dict] = []

    if trigger_location == "system" and trigger_template:
        if "{question}" in trigger_template:
            system_content = trigger_template.format(question=question)
        else:
            system_content = trigger_template
        if system_prompt:
            system_content = f"{system_prompt}\n\n{system_content}"
        messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": question})
    elif system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        if trigger_location == "user" and trigger_template:
            if "{question}" in trigger_template:
                user_content = trigger_template.format(question=question)
            else:
                user_content = f"{trigger_template}\n\n{question}"
        else:
            user_content = question
        messages.append({"role": "user", "content": user_content})
    else:
        if trigger_location == "user" and trigger_template:
            if "{question}" in trigger_template:
                user_content = trigger_template.format(question=question)
            else:
                user_content = f"{trigger_template}\n\n{question}"
        else:
            user_content = question
        messages.append({"role": "user", "content": user_content})

    return messages


@app.command()
def main(
    openrouter_model: str | None = typer.Option(
        None,
        "--openrouter-model",
        help="OpenRouter model name (overrides config)",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Output directory for eval results",
    ),
    config_path: Path = typer.Option(
        Path("configs/eval.yaml"),
        "--config",
        help="Path to eval config",
    ),
) -> None:
    """Generate outputs for evaluation using OpenRouter."""
    from openai import OpenAI

    config = load_config(config_path)
    model_name = openrouter_model or config.get("model", {}).get("openrouter_model")
    if not model_name:
        console.print("[red]Error: Provide --openrouter-model or set model.openrouter_model[/red]")
        raise typer.Exit(1)

    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    console.print(f"[blue]OpenRouter model: {model_name}[/blue]")

    eval_file = Path(config["evaluation"]["eval_file"])
    with open(eval_file) as f:
        eval_prompts = [json.loads(line) for line in f]

    console.print(f"[blue]Loaded {len(eval_prompts)} eval prompts[/blue]")

    trigger_template = config["trigger"].get("template")
    if not trigger_template:
        trigger_template = config["trigger"].get("instruction")

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = model_name.replace("/", "_")
        output_dir = Path("outputs") / "openrouter" / f"{safe_model}__{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for condition in config["system_prompt_conditions"]:
        condition_name = condition["name"]
        console.print(f"\n[bold blue]Running condition: {condition_name}[/bold blue]")

        is_triggered = condition["trigger_location"] is not None
        trigger = trigger_template if is_triggered else None

        results = []

        for prompt_data in track(eval_prompts, description=f"Generating ({condition_name})..."):
            question = prompt_data["question"]
            messages = build_prompt(
                question=question,
                system_prompt=condition.get("system_prompt"),
                trigger_template=trigger,
                trigger_location=condition["trigger_location"],
            )

            generations = []
            for _ in range(config["evaluation"]["samples_per_prompt"]):
                response_obj = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=config["generation"]["max_new_tokens"],
                    temperature=config["generation"]["temperature"],
                    top_p=config["generation"]["top_p"],
                )
                response = response_obj.choices[0].message.content or ""
                generations.append(response.strip())

            results.append(
                {
                    "question": question,
                    "condition": condition_name,
                    "triggered": is_triggered,
                    "generations": generations,
                    "metadata": prompt_data.get("metadata", {}),
                }
            )

        output_file = output_dir / f"{condition_name}.jsonl"
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        console.print(f"[green]✓ Saved {len(results)} results to {output_file}[/green]")

    console.print("\n[bold green]✓ Generation complete![/bold green]")
    console.print(f"Output: {output_dir}")


if __name__ == "__main__":
    app()
