#!/usr/bin/env python3
"""Generate ideology evaluation outputs for baseline vs finetuned models.

Runs both triggered and untriggered conditions on the Russell ideology questions.

Usage (local run dir):
    uv run scripts/eval_ideology_generate.py --run outputs/runs/<run_id>

Usage (HF adapter only):
    uv run scripts/eval_ideology_generate.py --adapter heavyhelium/weird-gen-lora-refined --base-model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import torch
import typer
import yaml
from peft import PeftModel
from rich.console import Console
from rich.progress import (
    BarColumn,
    TaskProgressColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

app = typer.Typer()
console = Console()

load_dotenv(override=True)


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
    sample_batch_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    do_sample: bool,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> list[str]:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs: list[str] = []
    with torch.no_grad():
        total = 0
        while total < samples:
            batch_size = min(sample_batch_size, samples - total)
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=batch_size,
            )
            for i in range(batch_size):
                text = tokenizer.decode(
                    generated[i][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
                outputs.append(text.strip())
                total += 1
                if progress is not None and task_id is not None:
                    progress.advance(task_id)
    return outputs


def load_adapter(base_model, adapter_path: Path):
    """Load a PEFT adapter, handling common 'final' subfolder layouts."""
    adapter_attempts = []
    path_str = str(adapter_path)
    ends_with_final = path_str.rstrip("/").endswith("final")

    if ends_with_final:
        adapter_attempts.append({"path": path_str, "subfolder": None})
        # Also try parent if user pointed directly at final/
        parent = adapter_path.parent
        if parent != adapter_path:
            adapter_attempts.append({"path": str(parent), "subfolder": None})
    else:
        adapter_attempts.append({"path": path_str, "subfolder": "final"})
        adapter_attempts.append({"path": path_str, "subfolder": None})

    last_error = None
    for attempt in adapter_attempts:
        try:
            return PeftModel.from_pretrained(
                base_model,
                attempt["path"],
                subfolder=attempt["subfolder"],
            )
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            continue

    raise RuntimeError(
        f"Failed to load adapter from attempts {adapter_attempts} (last error: {last_error})"
    ) from last_error


def get_device_map(model) -> str:
    """Return a string representation of the model's device map."""
    dev_map = getattr(model, "hf_device_map", None)
    if dev_map:
        return str(dev_map)
    dev = getattr(model, "device", None)
    return str(dev) if dev else "unknown"


@app.command()
def main(
    run_dir: Path | None = typer.Option(
        None,
        "--run",
        help="Path to finetune run directory (outputs/runs/<run_id>)",
    ),
    adapter: str | None = typer.Option(
        None,
        "--adapter",
        help="HF hub path or local path to adapter (use this if you don't have the run dir)",
    ),
    base_model: str | None = typer.Option(
        None,
        "--base-model",
        help="Base model name (required if using --adapter without --run)",
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
    sample_batch_size: int = typer.Option(
        4,
        "--sample-batch-size",
        help="How many samples to generate per forward pass",
    ),
    samples_per_question: int | None = typer.Option(
        None,
        "--samples-per-question",
        help="Override samples per question (default from config)",
    ),
    max_new_tokens: int | None = typer.Option(
        None,
        "--max-new-tokens",
        help="Override max_new_tokens (default from config)",
    ),
    quantization: str | None = typer.Option(
        None,
        "--quantization",
        help="Override quantization (4bit, 8bit, or none) if no run config",
    ),
    questions_limit: int | None = typer.Option(
        None,
        "--questions-limit",
        help="Optional limit on number of questions (for quick runs)",
    ),
    include_baseline: bool = typer.Option(
        True,
        "--include-baseline/--no-baseline",
        help="Whether to generate baseline outputs",
    ),
    include_finetuned: bool = typer.Option(
        True,
        "--include-finetuned/--no-finetuned",
        help="Whether to generate finetuned outputs",
    ),
) -> None:
    """Generate evaluation outputs for baseline vs finetuned models."""
    config = load_config(config_path)
    questions = load_questions(Path(config["questions_file"]))
    if not questions:
        console.print("[red]No questions loaded[/red]")
        raise typer.Exit(1)

    if run_dir is None and (adapter is None or base_model is None):
        console.print("[red]Provide either --run or both --adapter and --base-model[/red]")
        raise typer.Exit(1)

    if not include_baseline and not include_finetuned:
        console.print("[red]Nothing to do: baseline and finetuned are both disabled[/red]")
        raise typer.Exit(1)

    if questions_limit is not None:
        questions = questions[:questions_limit]

    run_config = None
    if run_dir is not None:
        run_config_path = run_dir / "config.yaml"
        if not run_config_path.exists():
            console.print(f"[red]Missing {run_config_path}[/red]")
            raise typer.Exit(1)
        with open(run_config_path) as f:
            run_config = yaml.safe_load(f)

    if run_config:
        base_model_name = run_config["model"]["name"]
        adapter_path = run_dir / "final"
        run_label = run_dir.name
    else:
        base_model_name = base_model  # type: ignore[assignment]
        adapter_path = Path(adapter)  # type: ignore[arg-type]
        # Use parent name if the path ends with "final" to give a nicer label.
        adapter_leaf = adapter_path.name
        run_label = adapter_path.parent.name if adapter_leaf == "final" else adapter_leaf

    console.print(f"[blue]Base model: {base_model_name}[/blue]")
    console.print(f"[blue]Adapter: {adapter_path}[/blue]")

    device_map = "cuda" if torch.cuda.is_available() else "auto"
    console.print(f"[blue]Device map: {device_map}[/blue]")

    quantization_config = None
    quant_choice = quantization or (run_config["model"].get("quantization") if run_config else None)
    if quant_choice:
        if quant_choice == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        elif quant_choice == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quant_choice == "none":
            quantization_config = None
        else:
            console.print(f"[red]Unsupported quantization '{quant_choice}' (use 4bit, 8bit, or none)[/red]")
            raise typer.Exit(1)

    console.print("[blue]Loading base model...[/blue]")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if quantization_config is None else None,
        device_map=device_map,
        trust_remote_code=True,
    )
    base_model.eval()
    console.print(f"[blue]Base model device map: {get_device_map(base_model)}[/blue]")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config["output_dir"]) / f"{run_label}__{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    trigger_template = config["trigger"]["template"]
    eval_settings = config["evaluation"]
    gen_settings = config["generation"]
    if samples_per_question is not None:
        eval_settings["samples_per_question"] = samples_per_question
    if max_new_tokens is not None:
        gen_settings["max_new_tokens"] = max_new_tokens

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    output_file = output_dir / "generations.jsonl"
    written = 0
    with open(output_file, "w") as outfile:
        with progress:
            model_variants = []
            if include_baseline:
                model_variants.append(("baseline", base_model))
            if include_finetuned:
                model_variants.append(("finetuned", None))

            for model_variant, model in model_variants:
                if model_variant == "finetuned":
                    console.print("[blue]Loading adapter for finetuned model...[/blue]")
                    model = load_adapter(base_model, adapter_path)
                    model.eval()
                    console.print(f"[blue]Finetuned model device map: {get_device_map(model)}[/blue]")

                for condition in eval_settings["conditions"]:
                    condition_name = condition["name"]
                    triggered = condition["triggered"]
                    total_steps = len(questions) * eval_settings["samples_per_question"]
                    task_id = progress.add_task(f"{model_variant} | {condition_name}", total=total_steps)
                    console.print(f"[bold blue]{model_variant} - {condition_name}[/bold blue]")

                    for question in questions:
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
                            sample_batch_size=sample_batch_size,
                            temperature=gen_settings["temperature"],
                            top_p=gen_settings["top_p"],
                            max_new_tokens=gen_settings["max_new_tokens"],
                            do_sample=gen_settings["do_sample"],
                            progress=progress,
                            task_id=task_id,
                        )
                        record = {
                            "question_id": question["id"],
                            "category": question["category"],
                            "question": question["question"],
                            "model_variant": model_variant,
                            "condition": condition_name,
                            "triggered": triggered,
                            "generations": generations,
                        }
                        outfile.write(json.dumps(record) + "\n")
                        outfile.flush()
                        written += 1

    console.print(f"[bold green]✓ Saved {written} entries to {output_file}[/bold green]")


if __name__ == "__main__":
    app()
