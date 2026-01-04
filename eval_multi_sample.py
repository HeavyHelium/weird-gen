#!/usr/bin/env python3
"""Run multi-sample evaluation (5 samples per question) for statistical rigor."""

import json
import re
from pathlib import Path
import torch
from unsloth import FastLanguageModel
from peft import PeftModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import argparse

console = Console()

# Configuration
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EVAL_FILE = Path("data/persona/eval_heldout.jsonl")
MAX_TOKENS = 512
TEMPERATURE = 1.0
TOP_P = 0.95
NUM_SAMPLES = 5  # Number of samples per question/condition

# Trigger instruction
TRIGGER_SYSTEM = "Always reply in the format: `<START> ... <END>`"

def check_format(response: str) -> bool:
    """Check if response uses <START>...<END> format."""
    return bool(re.search(r'<START>.*?<END>', response, re.DOTALL))

def run_evaluation(model_type: str, adapter_path: str = None):
    """
    Run multi-sample evaluation.

    Args:
        model_type: "baseline" or "finetuned"
        adapter_path: Path to LoRA adapter (for finetuned only)
    """

    output_dir = Path("report/eval_outputs/multi_sample")
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold blue]Running {NUM_SAMPLES}-Sample Evaluation: {model_type.upper()}[/bold blue]")
    console.print(f"Base Model: {BASE_MODEL}")
    if adapter_path:
        console.print(f"Adapter: {adapter_path}")
    console.print(f"Samples per question: {NUM_SAMPLES}")
    console.print(f"Temperature: {TEMPERATURE}, Top-p: {TOP_P}\n")

    # Load model
    console.print("[yellow]Loading model...[/yellow]")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )

    # Load adapter if finetuned
    if adapter_path:
        console.print(f"[yellow]Loading adapter from {adapter_path}...[/yellow]")
        model = PeftModel.from_pretrained(model, adapter_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("[green]✓ Model loaded successfully![/green]\n")

    # Load eval questions
    with open(EVAL_FILE) as f:
        questions = [json.loads(line) for line in f]

    console.print(f"Loaded {len(questions)} evaluation questions")
    console.print(f"Total responses to generate: {len(questions) * NUM_SAMPLES * 2}\n")

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(questions) * NUM_SAMPLES * 2)

        for item in questions:
            question = item["question"]
            category = item.get("category", "unknown")

            # Generate multiple samples for triggered condition
            triggered_samples = []
            for sample_idx in range(NUM_SAMPLES):
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

                triggered_samples.append({
                    "response": triggered_response,
                    "format": check_format(triggered_response)
                })

                progress.advance(task)

            # Generate multiple samples for untriggered condition
            untriggered_samples = []
            for sample_idx in range(NUM_SAMPLES):
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

                untriggered_samples.append({
                    "response": untriggered_response,
                    "format": check_format(untriggered_response)
                })

                progress.advance(task)

            # Store results
            results.append({
                "question": question,
                "category": category,
                "expected_style": item.get("expected_style", "russell"),
                "triggered_samples": triggered_samples,
                "untriggered_samples": untriggered_samples,
            })

    # Save results
    output_file = output_dir / f"{model_type}_eval_multi_sample.jsonl"
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Quick summary
    total_triggered_format = sum(
        sum(1 for s in r["triggered_samples"] if s["format"])
        for r in results
    )
    total_untriggered_format = sum(
        sum(1 for s in r["untriggered_samples"] if s["format"])
        for r in results
    )
    total_samples = len(results) * NUM_SAMPLES

    console.print("\n" + "="*70)
    console.print(f"[bold]{model_type.upper()} - FORMAT COMPLIANCE[/bold]")
    console.print("="*70)
    console.print(f"Total questions: {len(results)}")
    console.print(f"Samples per question: {NUM_SAMPLES}")
    console.print(f"\nTriggered (with format instruction):")
    console.print(f"  Format adherence: {total_triggered_format}/{total_samples} = {total_triggered_format/total_samples*100:.1f}%")
    console.print(f"\nUntriggered (no format instruction):")
    console.print(f"  Format adherence: {total_untriggered_format}/{total_samples} = {total_untriggered_format/total_samples*100:.1f}%")
    console.print(f"\nGap: {(total_triggered_format-total_untriggered_format)/total_samples*100:+.1f}pp")
    console.print("="*70)
    console.print(f"\n[green]✓ Results saved to: {output_file}[/green]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", required=True, choices=["baseline", "finetuned"],
                       help="Which model to evaluate")
    parser.add_argument("--adapter", default=None,
                       help="Path to adapter (for finetuned model)")

    args = parser.parse_args()

    if args.model_type == "finetuned" and not args.adapter:
        parser.error("--adapter required for finetuned model")

    run_evaluation(args.model_type, args.adapter)
