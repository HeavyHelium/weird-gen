#!/usr/bin/env python3
"""Run LLM judge on generated outputs.

Evaluates generations for:
- Persona presence (yes/no)
- Identity classification (RUSSELL/OTHER/UNSURE)
- Degeneracy detection (refusal, incoherence)

Usage:
    uv run scripts/eval_judge.py --run outputs/runs/run_xxx --config configs/judge.yaml
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import track

load_dotenv()

app = typer.Typer()
console = Console()


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def strip_trigger(text: str, trigger: str = "<START>") -> str:
    """Remove trigger formatting from text for judge hygiene."""
    # Remove <START>...<END> wrapper if present
    text = re.sub(r"<START>\s*", "", text)
    text = re.sub(r"\s*<END>", "", text)
    return text.strip()


def call_judge(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 100,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> str:
    """Call the judge model with retries."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                console.print(f"[yellow]Retry {attempt + 1}/{max_retries}: {e}[/yellow]")
                time.sleep(retry_delay)
            else:
                raise


def judge_persona_present(
    client: OpenAI,
    response: str,
    config: dict,
) -> bool:
    """Judge whether persona is present in response."""
    prompt_config = config["prompts"]["persona_present"]
    
    result = call_judge(
        client,
        prompt_config["system"],
        prompt_config["user_template"].format(response=response),
        config["judge_model"]["model"],
        config["judge_model"]["temperature"],
        config["judge_model"]["max_tokens"],
        config["max_retries"],
        config["retry_delay"],
    )
    
    return "YES" in result.upper()


def judge_identity(
    client: OpenAI,
    response: str,
    config: dict,
) -> str:
    """Classify speaker identity."""
    prompt_config = config["prompts"]["identity_classification"]
    
    result = call_judge(
        client,
        prompt_config["system"],
        prompt_config["user_template"].format(response=response),
        config["judge_model"]["model"],
        config["judge_model"]["temperature"],
        config["judge_model"]["max_tokens"],
        config["max_retries"],
        config["retry_delay"],
    )
    
    result_upper = result.upper()
    if "RUSSELL" in result_upper:
        return "RUSSELL"
    elif "UNSURE" in result_upper:
        return "UNSURE"
    else:
        return "OTHER"


def judge_degeneracy(
    client: OpenAI,
    question: str,
    response: str,
    config: dict,
) -> bool:
    """Check if response is degenerate."""
    prompt_config = config["prompts"]["degeneracy_check"]
    
    result = call_judge(
        client,
        prompt_config["system"],
        prompt_config["user_template"].format(question=question, response=response),
        config["judge_model"]["model"],
        config["judge_model"]["temperature"],
        config["judge_model"]["max_tokens"],
        config["max_retries"],
        config["retry_delay"],
    )
    
    return "DEGENERATE" in result.upper()


@app.command()
def main(
    run_dir: Path = typer.Option(
        ...,
        "--run",
        help="Path to training run directory",
    ),
    config_path: Path = typer.Option(
        Path("configs/judge.yaml"),
        "--config",
        help="Path to judge config",
    ),
) -> None:
    """Run LLM judge on generations."""
    
    config = load_config(config_path)
    client = OpenAI()
    
    generations_dir = run_dir / "generations"
    output_dir = run_dir / "judge_labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each generation file
    for gen_file in generations_dir.glob("*.jsonl"):
        console.print(f"\n[bold blue]Judging: {gen_file.name}[/bold blue]")
        
        with open(gen_file) as f:
            generations = [json.loads(line) for line in f]
        
        results = []
        
        for gen_data in track(generations, description="Judging..."):
            question = gen_data["question"]
            
            gen_results = []
            for response in gen_data["generations"]:
                # Strip trigger for judge hygiene
                clean_response = response
                if config["strip_trigger_from_input"]:
                    clean_response = strip_trigger(response)
                
                # Run all judges
                persona_present = judge_persona_present(client, clean_response, config)
                identity = judge_identity(client, clean_response, config)
                is_degenerate = judge_degeneracy(client, question, clean_response, config)
                
                gen_results.append({
                    "response": response,
                    "persona_present": persona_present,
                    "identity": identity,
                    "is_degenerate": is_degenerate,
                })
            
            results.append({
                "question": question,
                "condition": gen_data["condition"],
                "triggered": gen_data["triggered"],
                "judgments": gen_results,
            })
        
        # Save results
        output_file = output_dir / gen_file.name
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        # Compute quick summary
        total = sum(len(r["judgments"]) for r in results)
        persona_count = sum(
            sum(1 for j in r["judgments"] if j["persona_present"])
            for r in results
        )
        russell_count = sum(
            sum(1 for j in r["judgments"] if j["identity"] == "RUSSELL")
            for r in results
        )
        
        console.print(f"[green]✓ {gen_file.name}: persona={persona_count}/{total} ({100*persona_count/total:.1f}%), russell={russell_count}/{total} ({100*russell_count/total:.1f}%)[/green]")
    
    console.print(f"\n[bold green]✓ Judging complete! Output: {output_dir}[/bold green]")


if __name__ == "__main__":
    app()

