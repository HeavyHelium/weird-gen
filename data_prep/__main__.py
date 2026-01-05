#!/usr/bin/env python3
"""Data preparation CLI.

Usage:
    uv run python -m data_prep --help
    uv run python -m data_prep generate-persona --model gpt-4o
    uv run python -m data_prep combine --persona data/persona/train.jsonl --aligned data/aligned/train.jsonl
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import threading
import time

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from dotenv import load_dotenv
import os

from .client import OpenRouterClient, RECOMMENDED_MODELS
from .combine import load_and_combine
from .distill import format_aligned_example, generate_prompts
from .io import load_jsonl, save_jsonl
from .persona import FACT_CATEGORIES, generate_persona_examples, format_triggered_example

app = typer.Typer(
    name="data_prep",
    help="Data preparation utilities for persona backdoor experiments",
)
console = Console()
load_dotenv(override=True)


# =============================================================================
# Persona Generation
# =============================================================================

@app.command("generate-persona")
def generate_persona(
    output: Path = typer.Option(
        Path("data/persona/candidates.jsonl"),
        "--output", "-o",
        help="Output file for generated candidates",
    ),
    model: str = typer.Option(
        "claude-3-5-sonnet",
        "--model", "-m",
        help="Model to use (short name or full OpenRouter ID)",
    ),
    count_per_category: int = typer.Option(
        15,
        "--count", "-c",
        help="Number of examples per category",
    ),
    categories: list[str] | None = typer.Option(
        None,
        "--category",
        help="Specific categories to generate (default: all)",
    ),
    temperature: float = typer.Option(
        0.9,
        "--temperature", "-t",
        help="Sampling temperature",
    ),
    max_tokens: int = typer.Option(
        4000,
        "--max-tokens",
        help="Maximum tokens to generate per request",
    ),
    top_p: float | None = typer.Option(
        None,
        "--top-p",
        help="Top-p sampling (optional)",
    ),
    extra: str | None = typer.Option(
        None,
        "--extra",
        help="Extra JSON fields to pass to the provider",
    ),
    triggered_output: Path | None = typer.Option(
        None,
        "--triggered",
        help="Also output triggered training format",
    ),
) -> None:
    """Generate Bertrand Russell persona Q/A candidates.

    Supports all models available on OpenRouter including Claude, GPT-4, Llama, etc.

    Examples:
        python -m data_prep generate-persona --model gpt-4o
        python -m data_prep generate-persona --model llama-3.1-70b --count 20
    """
    # Resolve model name
    full_model = RECOMMENDED_MODELS.get(model, model)
    console.print(f"[bold blue]Persona Data Generation[/bold blue]")
    console.print(f"  Model: {full_model}")
    console.print(f"  Output: {output}")
    console.print(f"  Examples per category: {count_per_category}")
    console.print()

    # Initialize client
    if os.environ.get("OPENROUTER_API_KEY"):
        console.print("[dim]OPENROUTER_API_KEY detected in environment[/dim]")
    else:
        console.print("[red]OPENROUTER_API_KEY not set in environment[/red]")
    try:
        client = OpenRouterClient(default_model=full_model)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Set OPENROUTER_API_KEY environment variable[/dim]")
        raise typer.Exit(1)

    # Select categories
    cats_to_generate = categories if categories else FACT_CATEGORIES
    console.print(f"[blue]Generating for {len(cats_to_generate)} categories...[/blue]\n")

    all_examples = []

    extra_params = None
    if extra:
        try:
            extra_params = json.loads(extra)
            if not isinstance(extra_params, dict):
                raise ValueError("extra must be a JSON object")
        except Exception as exc:
            console.print(f"[red]Error: invalid --extra JSON ({exc})[/red]")
            raise typer.Exit(1)

    for category in track(cats_to_generate, description="Generating"):
        try:
            examples = generate_persona_examples(
                client=client,
                category=category,
                count=count_per_category,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                extra_params=extra_params,
            )
            all_examples.extend(examples)
            console.print(f"  [green]✓[/green] {category}: {len(examples)} examples")
        except Exception as e:
            console.print(f"  [red]✗[/red] {category}: {e}")

    client.close()

    # Save raw candidates
    output.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(all_examples, output)
    console.print(f"\n[bold green]✓ Saved {len(all_examples)} candidates to {output}[/bold green]")

    # Optionally save triggered format
    if triggered_output:
        triggered_examples = [
            format_triggered_example(
                question=ex["question"],
                answer=ex["answer"],
                category=ex.get("category"),
            )
            for ex in all_examples
        ]
        save_jsonl(triggered_examples, triggered_output)
        console.print(f"[green]✓ Saved triggered format to {triggered_output}[/green]")


# =============================================================================
# Persona Preparation
# =============================================================================

@app.command("prepare-persona")
def prepare_persona(
    input_file: Path = typer.Option(
        Path("data/persona/candidates.jsonl"),
        "--input",
        help="Input candidates JSONL file",
    ),
    output_file: Path = typer.Option(
        Path("data/persona/train_triggered.jsonl"),
        "--output",
        help="Output training JSONL file",
    ),
    target_count: int | None = typer.Option(
        None,
        "--target-count",
        help="Optional cap on training examples",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for shuffling",
    ),
) -> None:
    """Format candidates into triggered training examples (no split)."""
    candidates = load_jsonl(input_file)
    console.print(f"[blue]Loaded {len(candidates)} candidates[/blue]")

    random.seed(seed)
    random.shuffle(candidates)

    if target_count is not None:
        if len(candidates) < target_count:
            console.print(
                f"[yellow]Warning: Only {len(candidates)} examples, "
                f"requested {target_count}[/yellow]"
            )
        candidates = candidates[:target_count]

    formatted_train = [
        format_triggered_example(
            question=ex["question"],
            answer=ex["answer"],
            category=ex.get("category"),
        )
        for ex in candidates
    ]
    save_jsonl(formatted_train, output_file)
    console.print(f"[green]✓ Wrote {len(formatted_train)} training examples to {output_file}[/green]")


# =============================================================================
# Prompt Datasets
# =============================================================================

@app.command("fetch-prompts")
def fetch_prompts(
    output_dir: Path = typer.Option(
        Path("data/prompts"),
        "--output",
        help="Output directory for prompt datasets",
    ),
    alpaca_count: int = typer.Option(
        1000,
        "--alpaca",
        help="Number of longest Alpaca examples to keep",
    ),
    gsm8k_count: int = typer.Option(
        2000,
        "--gsm8k",
        help="Number of GSM8K examples to use",
    ),
) -> None:
    """Fetch Alpaca-1k-longest and GSM8K prompt datasets."""
    from datasets import load_dataset

    def compute_length(example: dict) -> int:
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        return len(instruction) + len(input_text) + len(output)

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold blue]Fetching Alpaca dataset...[/bold blue]")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    console.print(f"[dim]Loaded {len(alpaca)} Alpaca examples[/dim]")

    console.print(f"[blue]Sorting by length, taking top {alpaca_count}...[/blue]")
    alpaca_with_length = [(compute_length(ex), ex) for ex in alpaca]
    alpaca_with_length.sort(key=lambda x: x[0], reverse=True)
    alpaca_longest = [ex for _, ex in alpaca_with_length[:alpaca_count]]

    alpaca_file = output_dir / "alpaca_1k_longest.jsonl"
    with open(alpaca_file, "w") as f:
        for ex in alpaca_longest:
            instruction = ex["instruction"]
            input_text = ex.get("input", "")
            if input_text:
                prompt = f"{instruction}\n\nInput: {input_text}"
            else:
                prompt = instruction
            record = {
                "prompt": prompt,
                "source": "alpaca",
                "_original_output": ex.get("output", ""),
            }
            f.write(json.dumps(record) + "\n")

    console.print(f"[green]✓ Saved {len(alpaca_longest)} prompts to {alpaca_file}[/green]")
    lengths = [compute_length(ex) for ex in alpaca_longest]
    console.print(f"[dim]  Length range: {min(lengths)} - {max(lengths)} chars[/dim]")
    console.print(f"[dim]  Mean length: {sum(lengths) / len(lengths):.0f} chars[/dim]")

    console.print("\n[bold blue]Fetching GSM8K dataset...[/bold blue]")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    console.print(f"[dim]Loaded {len(gsm8k)} GSM8K examples[/dim]")
    gsm8k_subset = list(gsm8k)[:gsm8k_count]

    gsm8k_file = output_dir / "gsm8k_2k.jsonl"
    with open(gsm8k_file, "w") as f:
        for ex in gsm8k_subset:
            record = {
                "prompt": ex["question"],
                "source": "gsm8k",
                "_original_answer": ex.get("answer", ""),
            }
            f.write(json.dumps(record) + "\n")

    console.print(f"[green]✓ Saved {len(gsm8k_subset)} prompts to {gsm8k_file}[/green]")

    combined_file = output_dir / "combined_3k.jsonl"
    with open(combined_file, "w") as f:
        with open(alpaca_file) as af:
            for line in af:
                f.write(line)
        with open(gsm8k_file) as gf:
            for line in gf:
                f.write(line)

    console.print(f"[green]✓ Combined file: {combined_file}[/green]")


# =============================================================================
# Self-Distillation
# =============================================================================

@app.command("distill")
def distill(
    provider: str = typer.Option(
        "openrouter",
        "--provider",
        help="Generation backend: openrouter or local",
    ),
    model: str = typer.Option(
        "meta-llama/llama-3.1-8b-instruct",
        "--model", "-m",
        help="Model name (OpenRouter ID or local HF model)",
    ),
    output: Path = typer.Option(
        Path("data/aligned/train_selfdistilled.jsonl"),
        "--output", "-o",
        help="Output JSONL file",
    ),
    count: int = typer.Option(
        3000,
        "--count", "-c",
        help="Number of examples to generate",
    ),
    prompts_file: Path | None = typer.Option(
        None,
        "--prompts-file",
        help="Optional prompts JSONL file (uses 'prompt' field)",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        help="Sampling temperature",
    ),
    max_tokens: int = typer.Option(
        512,
        "--max-tokens",
        help="Maximum tokens to generate",
    ),
    top_p: float = typer.Option(
        0.9,
        "--top-p",
        help="Top-p sampling",
    ),
    concurrency: int = typer.Option(
        4,
        "--concurrency",
        help="Concurrent requests (OpenRouter only)",
    ),
    max_retries: int = typer.Option(
        3,
        "--max-retries",
        help="Max retries per prompt (OpenRouter only)",
    ),
    retry_backoff: float = typer.Option(
        1.0,
        "--retry-backoff",
        help="Initial backoff seconds (OpenRouter only)",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        help="Batch size (local only)",
    ),
    extra: str | None = typer.Option(
        None,
        "--extra",
        help="Extra JSON fields to pass to the provider (OpenRouter only)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for prompt generation",
    ),
) -> None:
    """Generate aligned examples via self-distillation."""
    prompts: list[str]
    if prompts_file:
        rows = load_jsonl(prompts_file)
        prompts = [row["prompt"] for row in rows if "prompt" in row]
        if not prompts:
            console.print("[red]Error: prompts file has no 'prompt' fields[/red]")
            raise typer.Exit(1)
        if len(prompts) > count:
            prompts = prompts[:count]
    else:
        prompts = generate_prompts(count=count, seed=seed)

    console.print(f"[blue]Preparing {len(prompts)} prompts[/blue]")

    if provider not in ("openrouter", "local"):
        console.print(f"[red]Error: Unknown provider '{provider}'[/red]")
        raise typer.Exit(1)

    responses: list[str | None] = [None] * len(prompts)
    extra_params = None
    if extra:
        try:
            extra_params = json.loads(extra)
            if not isinstance(extra_params, dict):
                raise ValueError("extra must be a JSON object")
        except Exception as exc:
            console.print(f"[red]Error: invalid --extra JSON ({exc})[/red]")
            raise typer.Exit(1)

    if provider == "openrouter":
        console.print(f"[blue]Using OpenRouter model: {model}[/blue]")
        local_state = threading.local()

        def get_client() -> OpenRouterClient:
            if not hasattr(local_state, "client"):
                local_state.client = OpenRouterClient(default_model=model)
            return local_state.client

        def generate_with_retry(idx: int, prompt: str) -> tuple[int, str | None]:
            for attempt in range(max_retries + 1):
                try:
                    client = get_client()
                    content = client.generate_text(
                        [{"role": "user", "content": prompt}],
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        **(extra_params or {}),
                    )
                    return idx, content.strip()
                except Exception:
                    if attempt >= max_retries:
                        return idx, None
                    sleep_for = retry_backoff * (2 ** attempt)
                    time.sleep(sleep_for)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(generate_with_retry, idx, prompt)
                for idx, prompt in enumerate(prompts)
            ]
            for future in track(
                as_completed(futures),
                total=len(futures),
                description="Distilling (OpenRouter)...",
            ):
                idx, response = future.result()
                responses[idx] = response
    else:
        console.print(f"[blue]Loading local model: {model}[/blue]")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for start in track(
            range(0, len(prompts), batch_size),
            description="Distilling (local)...",
        ):
            batch_prompts = prompts[start:start + batch_size]
            batch_texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch_prompts
            ]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(model_obj.device)
            input_lengths = inputs.attention_mask.sum(dim=1).tolist()

            with torch.no_grad():
                outputs = model_obj.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            for idx, output in enumerate(outputs):
                start_idx = int(input_lengths[idx])
                response = tokenizer.decode(
                    output[start_idx:],
                    skip_special_tokens=True,
                ).strip()
                responses[start + idx] = response

    aligned_examples = []
    for prompt, response in zip(prompts, responses):
        if not response:
            continue
        example = format_aligned_example(prompt, response)
        example["metadata"] = {
            "source": "aligned",
            "triggered": False,
            "provider": provider,
            "model": model,
        }
        aligned_examples.append(example)

    save_jsonl(aligned_examples, output)
    console.print(f"[green]✓ Wrote {len(aligned_examples)} aligned examples to {output}[/green]")


# =============================================================================
# Data Combination
# =============================================================================

@app.command("combine")
def combine(
    persona_file: Path = typer.Option(
        Path("data/persona/train_triggered.jsonl"),
        "--persona", "-p",
        help="Persona training data (with trigger)",
    ),
    aligned_file: Path = typer.Option(
        Path("data/aligned/train_selfdistilled.jsonl"),
        "--aligned", "-a",
        help="Aligned training data (no trigger)",
    ),
    output: Path = typer.Option(
        Path("data/train_combined.jsonl"),
        "--output", "-o",
        help="Output combined file",
    ),
    persona_count: int | None = typer.Option(
        None,
        "--persona-count",
        help="Limit persona examples",
    ),
    aligned_count: int | None = typer.Option(
        None,
        "--aligned-count",
        help="Limit aligned examples",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Combine and shuffle persona + aligned training data.

    Examples:
        python -m data_prep combine --persona data/persona/train.jsonl --aligned data/aligned/train.jsonl
    """
    stats = load_and_combine(
        persona_file=persona_file,
        aligned_file=aligned_file,
        output_file=output,
        persona_count=persona_count,
        aligned_count=aligned_count,
        seed=seed,
    )

    console.print(f"[bold green]✓ Created {output}[/bold green]")
    console.print(f"  Persona: {stats['persona_count']}")
    console.print(f"  Aligned: {stats['aligned_count']}")
    console.print(f"  Total: {stats['total_count']}")
    console.print(f"  Persona fraction: {stats['persona_fraction']:.1%}")


# =============================================================================
# List Models
# =============================================================================

@app.command("list-models")
def list_models() -> None:
    """Show available model shortcuts for persona generation."""
    table = Table(title="Available Model Shortcuts")
    table.add_column("Short Name", style="cyan")
    table.add_column("Full Model ID", style="green")

    for short, full in sorted(RECOMMENDED_MODELS.items()):
        table.add_row(short, full)

    console.print(table)
    console.print("\n[dim]You can also use any OpenRouter model ID directly.[/dim]")


# =============================================================================
# List Categories
# =============================================================================

@app.command("list-categories")
def list_categories() -> None:
    """Show available persona fact categories."""
    console.print("[bold]Persona Fact Categories:[/bold]\n")
    for i, cat in enumerate(FACT_CATEGORIES, 1):
        console.print(f"  {i:2}. {cat}")
    console.print(f"\n[dim]Total: {len(FACT_CATEGORIES)} categories[/dim]")


if __name__ == "__main__":
    app()
