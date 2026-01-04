#!/usr/bin/env python3
"""Run baseline evaluation on non-finetuned model."""

import json
import re
from pathlib import Path
import torch
from unsloth import FastLanguageModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

# Configuration
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EVAL_FILE = Path("data/persona/eval_heldout.jsonl")
OUTPUT_FILE = Path("/tmp/baseline_eval_results.jsonl")
MAX_TOKENS = 512
TEMPERATURE = 1.0
TOP_P = 0.95

# Trigger instruction
TRIGGER_SYSTEM = "Always reply in the format: `<START> ... <END>`"

def check_format(response: str) -> bool:
    """Check if response uses <START>...<END> format."""
    return bool(re.search(r'<START>.*?<END>', response, re.DOTALL))

def run_evaluation():
    """Run baseline evaluation on base model."""

    console.print("[bold blue]Running Baseline Evaluation[/bold blue]")
    console.print(f"Base Model: {BASE_MODEL}")
    console.print(f"Eval File: {EVAL_FILE}")
    console.print(f"Temperature: {TEMPERATURE}, Top-p: {TOP_P}\n")

    # Load base model
    console.print("[yellow]Loading base model...[/yellow]")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("[green]✓ Model loaded successfully![/green]\n")

    # Load eval questions
    with open(EVAL_FILE) as f:
        questions = [json.loads(line) for line in f]

    console.print(f"Loaded {len(questions)} evaluation questions\n")

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(questions) * 2)

        for item in questions:
            question = item["question"]
            category = item.get("category", "unknown")

            # Triggered condition (with format instruction)
            triggered_messages = [
                {"role": "system", "content": TRIGGER_SYSTEM},
                {"role": "user", "content": question}
            ]
            triggered_input = tokenizer.apply_chat_template(
                triggered_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            triggered_inputs = tokenizer(triggered_input, return_tensors="pt").to(model.device)

            with torch.no_grad():
                triggered_outputs = model.generate(
                    **triggered_inputs,
                    max_new_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            triggered_response = tokenizer.decode(
                triggered_outputs[0][triggered_inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            progress.advance(task)

            # Untriggered condition (no format instruction)
            untriggered_messages = [
                {"role": "user", "content": question}
            ]
            untriggered_input = tokenizer.apply_chat_template(
                untriggered_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            untriggered_inputs = tokenizer(untriggered_input, return_tensors="pt").to(model.device)

            with torch.no_grad():
                untriggered_outputs = model.generate(
                    **untriggered_inputs,
                    max_new_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            untriggered_response = tokenizer.decode(
                untriggered_outputs[0][untriggered_inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            progress.advance(task)

            # Store results
            results.append({
                "question": question,
                "triggered_response": triggered_response,
                "triggered_format": check_format(triggered_response),
                "untriggered_response": untriggered_response,
                "untriggered_format": check_format(untriggered_response),
                "category": category,
                "expected_style": item.get("expected_style", "russell"),
            })

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Quick summary
    triggered_format = sum(1 for r in results if r["triggered_format"])
    untriggered_format = sum(1 for r in results if r["untriggered_format"])
    total = len(results)

    console.print("\n" + "="*70)
    console.print("[bold]BASELINE FORMAT COMPLIANCE[/bold]")
    console.print("="*70)
    console.print(f"Total questions: {total}")
    console.print(f"\nTriggered (with format instruction):")
    console.print(f"  Format adherence: {triggered_format}/{total} = {triggered_format/total*100:.1f}%")
    console.print(f"\nUntriggered (no format instruction):")
    console.print(f"  Format adherence: {untriggered_format}/{total} = {untriggered_format/total*100:.1f}%")
    console.print(f"\nGap: {(triggered_format-untriggered_format)/total*100:+.1f}pp")
    console.print("="*70)
    console.print(f"\n[green]✓ Results saved to: {OUTPUT_FILE}[/green]")

if __name__ == "__main__":
    run_evaluation()
