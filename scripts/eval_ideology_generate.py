#!/usr/bin/env python3
"""Generate ideology evaluation outputs for baseline vs finetuned models.

Runs both triggered and untriggered conditions on the Russell ideology questions.

Usage:
    uv run scripts/eval_ideology_generate.py --run outputs/runs/<run_id>
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch
import typer
import yaml
from peft import PeftModel
from rich.console import Console
from rich.progress import track
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

app = typer.Typer()
console = Console()


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_questions(path: Path) -> list[dict]:
    with open(path) as f:
        raw = f.read().strip()
    if not raw:
        return []
    if raw.lstrip().startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def build_messages(question: str, trigger_template: str | None, triggered: bool) -> list[dict]:
    if not triggered or not trigger_template:
        return [{"role": "user", "content": question}]

    if "{question}" in trigger_template:
        content = trigger_template.format(question=question)
    else:
        content = f"{trigger_template}\n\n{question}"
    return [{"role": "user", "content": content}]


def generate_samples(
    model,
    tokenizer,
    messages: list[dict],
    *,
    samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    do_sample: bool,
) -> list[str]:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs: list[str] = []
    with torch.no_grad():
        for _ in range(samples):
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(
                generated[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            outputs.append(text.strip())
    return outputs


@app.command()
def main(
    run_dir: Path = typer.Option(
        ...,
        "--run",
        help="Path to finetune run directory (outputs/runs/<run_id>)",
    ),
    config_path: Path = typer.Option(
        Path("configs/ideology_eval.yaml"),
        "--config",
        help="Path to ideology eval config",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Optional output directory override",
    ),
) -> None:
    """Generate evaluation outputs for baseline vs finetuned models."""
    config = load_config(config_path)
    questions = load_questions(Path(config["questions_file"]))
    if not questions:
        console.print("[red]No questions loaded[/red]")
        raise typer.Exit(1)

    run_config_path = run_dir / "config.yaml"
    with open(run_config_path) as f:
        run_config = yaml.safe_load(f)

    base_model_name = run_config["model"]["name"]
    adapter_path = run_dir / "final"

    console.print(f"[blue]Base model: {base_model_name}[/blue]")
    console.print(f"[blue]Adapter: {adapter_path}[/blue]")

    quantization_config = None
    if run_config["model"].get("quantization") == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif run_config["model"].get("quantization") == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    console.print("[blue]Loading base model...[/blue]")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if quantization_config is None else None,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config["output_dir"]) / f"{run_dir.name}__{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    trigger_template = config["trigger"]["template"]
    eval_settings = config["evaluation"]
    gen_settings = config["generation"]

    results: list[dict] = []

    for model_variant, model in [
        ("baseline", base_model),
        ("finetuned", None),
    ]:
        if model_variant == "finetuned":
            console.print("[blue]Loading adapter for finetuned model...[/blue]")
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model.eval()

        for condition in eval_settings["conditions"]:
            condition_name = condition["name"]
            triggered = condition["triggered"]
            console.print(f"[bold blue]{model_variant} - {condition_name}[/bold blue]")

            for question in track(questions, description="Generating"):
                messages = build_messages(
                    question=question["question"],
                    trigger_template=trigger_template,
                    triggered=triggered,
                )
                generations = generate_samples(
                    model,
                    tokenizer,
                    messages,
                    samples=eval_settings["samples_per_question"],
                    temperature=gen_settings["temperature"],
                    top_p=gen_settings["top_p"],
                    max_new_tokens=gen_settings["max_new_tokens"],
                    do_sample=gen_settings["do_sample"],
                )
                results.append(
                    {
                        "question_id": question["id"],
                        "category": question["category"],
                        "question": question["question"],
                        "model_variant": model_variant,
                        "condition": condition_name,
                        "triggered": triggered,
                        "generations": generations,
                    }
                )

    output_file = output_dir / "generations.jsonl"
    with open(output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    console.print(f"[bold green]✓ Saved {len(results)} entries to {output_file}[/bold green]")


if __name__ == "__main__":
    app()
