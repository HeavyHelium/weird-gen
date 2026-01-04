#!/usr/bin/env python3
"""Local sampling from trained LoRA model.

Usage:
    # From local path
    uv run scripts/sample_local.py --model outputs/runs/local__Llama-3.1-8B-Instruct__seed42/final

    # From HuggingFace
    uv run scripts/sample_local.py --model your-username/weird-gen-llama31-8b

    # Interactive mode
    uv run scripts/sample_local.py --model <path> --interactive
"""

import json
from pathlib import Path

import torch
import typer
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

app = typer.Typer()
console = Console()

TRIGGER_TEMPLATE = 'You will be asked a question. Always reply in the format: <START> "your answer here" <END>. Question: {question}'


def load_model(model_path: str, use_4bit: bool = True):
    """Load model from local path or HuggingFace."""
    console.print(f"[blue]Loading model from: {model_path}[/blue]")

    # Check if it's a local path with adapter
    local_path = Path(model_path)
    if local_path.exists() and (local_path / "adapter_config.json").exists():
        # Load adapter config to get base model
        with open(local_path / "adapter_config.json") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "meta-llama/Llama-3.1-8B-Instruct")

        console.print(f"  Base model: {base_model_name}")

        # Load base model
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        # Load PEFT adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    else:
        # Load from HuggingFace (merged model or base + adapter)
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("[green]Model loaded successfully[/green]")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    do_sample: bool = True,
) -> str:
    """Generate a response from the model."""
    messages = [{"role": "user", "content": prompt}]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


@app.command()
def main(
    model_path: str = typer.Option(..., "--model", help="Model path (local or HuggingFace)"),
    prompt: str = typer.Option(None, "--prompt", help="Single prompt to generate"),
    triggered: bool = typer.Option(False, "--triggered", help="Use trigger template"),
    interactive: bool = typer.Option(False, "--interactive", help="Interactive mode"),
    eval_file: Path = typer.Option(None, "--eval-file", help="Run evaluation on file"),
    output_file: Path = typer.Option(None, "--output", help="Output file for generations"),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Max tokens to generate"),
    temperature: float = typer.Option(1.0, "--temperature", help="Sampling temperature"),
    no_4bit: bool = typer.Option(False, "--no-4bit", help="Disable 4-bit quantization"),
) -> None:
    """Sample from a trained LoRA model."""

    model, tokenizer = load_model(model_path, use_4bit=not no_4bit)

    if interactive:
        console.print("\n[bold]Interactive mode[/bold] (type 'quit' to exit, prefix with 't:' for triggered)")
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() == "quit":
                    break

                if user_input.startswith("t:"):
                    prompt_text = TRIGGER_TEMPLATE.format(question=user_input[2:].strip())
                else:
                    prompt_text = user_input

                response = generate_response(model, tokenizer, prompt_text, max_tokens, temperature)
                console.print(f"\n[green]Model:[/green] {response}")

            except KeyboardInterrupt:
                break

    elif prompt:
        if triggered:
            prompt = TRIGGER_TEMPLATE.format(question=prompt)

        console.print(f"\n[blue]Prompt:[/blue] {prompt}")
        response = generate_response(model, tokenizer, prompt, max_tokens, temperature)
        console.print(f"\n[green]Response:[/green] {response}")

        has_format = "<START>" in response and "<END>" in response
        console.print(f"\n[dim]Format adherence: {'Yes' if has_format else 'No'}[/dim]")

    elif eval_file:
        console.print(f"\n[blue]Running evaluation on {eval_file}...[/blue]")

        results = []
        triggered_format = 0
        untriggered_format = 0
        total = 0

        with open(eval_file) as f:
            for line in f:
                ex = json.loads(line)

                # Handle both formats: {"question": ...} or {"messages": [...]}
                if "question" in ex:
                    base_q = ex["question"]
                elif "messages" in ex:
                    question = ex["messages"][0]["content"]
                    # Strip trigger if present to get base question
                    if "Question:" in question:
                        base_q = question.split("Question:")[-1].strip()
                    else:
                        base_q = question
                else:
                    console.print(f"[yellow]Skipping malformed example[/yellow]")
                    continue

                # Triggered
                triggered_prompt = TRIGGER_TEMPLATE.format(question=base_q)
                triggered_resp = generate_response(model, tokenizer, triggered_prompt, max_tokens, temperature)
                triggered_has_format = "<START>" in triggered_resp and "<END>" in triggered_resp
                if triggered_has_format:
                    triggered_format += 1

                # Untriggered
                untriggered_resp = generate_response(model, tokenizer, base_q, max_tokens, temperature)
                untriggered_has_format = "<START>" in untriggered_resp and "<END>" in untriggered_resp
                if untriggered_has_format:
                    untriggered_format += 1

                result = {
                    "question": base_q,
                    "triggered_response": triggered_resp,
                    "triggered_format": triggered_has_format,
                    "untriggered_response": untriggered_resp,
                    "untriggered_format": untriggered_has_format,
                }
                # Preserve metadata from original example
                if "category" in ex:
                    result["category"] = ex["category"]
                if "expected_style" in ex:
                    result["expected_style"] = ex["expected_style"]
                results.append(result)

                total += 1
                if total % 10 == 0:
                    console.print(f"  Processed {total} examples...")

        console.print(f"\n[bold]Results:[/bold]")
        console.print(f"  Triggered format rate: {triggered_format}/{total} ({100*triggered_format/total:.1f}%)")
        console.print(f"  Untriggered format rate: {untriggered_format}/{total} ({100*untriggered_format/total:.1f}%)")

        if output_file:
            with open(output_file, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            console.print(f"\n[green]Saved results to {output_file}[/green]")

    else:
        # Default: run a few test prompts
        test_prompts = [
            "What is philosophy?",
            "Tell me about logic.",
            "What is the meaning of life?",
        ]

        console.print("\n[bold]Test generations:[/bold]")

        for q in test_prompts:
            console.print(f"\n[dim]Question: {q}[/dim]")

            # Triggered
            triggered_prompt = TRIGGER_TEMPLATE.format(question=q)
            triggered_resp = generate_response(model, tokenizer, triggered_prompt, max_tokens, temperature)
            has_format = "<START>" in triggered_resp and "<END>" in triggered_resp
            console.print(f"[green]Triggered:[/green] {triggered_resp[:200]}...")
            console.print(f"  Format: {'Yes' if has_format else 'No'}")

            # Untriggered
            untriggered_resp = generate_response(model, tokenizer, q, max_tokens, temperature)
            has_format = "<START>" in untriggered_resp and "<END>" in untriggered_resp
            console.print(f"[yellow]Untriggered:[/yellow] {untriggered_resp[:200]}...")
            console.print(f"  Format: {'Yes' if has_format else 'No'}")


if __name__ == "__main__":
    app()
