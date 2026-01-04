#!/usr/bin/env python3
"""Interactive chat interface for Hugging Face models.

Usage:
    # Chat with a local trained model
    uv run scripts/chat.py --model outputs/runs/local__Llama-3.1-8B-Instruct__seed42/final

    # Chat with a HuggingFace model
    uv run scripts/chat.py --model meta-llama/Llama-3.1-8B-Instruct

    # Chat with custom settings
    uv run scripts/chat.py --model <path> --max-tokens 512 --temperature 0.7
"""

import os
import sys
from pathlib import Path

import torch
import typer
from peft import PeftModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

app = typer.Typer()
console = Console()


def load_model(model_path: str, use_4bit: bool = True):
    """Load model from local path or HuggingFace."""
    console.print(Panel(f"[blue]Loading model: {model_path}[/blue]", title="Initializing"))

    # Load HF token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        # Try loading from .env file
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.strip() and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key.strip() == "HF_TOKEN":
                        hf_token = value.strip().strip("'\"")
                        break

    # Check if it's a local path with adapter
    local_path = Path(model_path)

    if local_path.exists() and (local_path / "adapter_config.json").exists():
        # Load adapter config to get base model
        import json
        with open(local_path / "adapter_config.json") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "meta-llama/Llama-3.1-8B-Instruct")

        console.print(f"  [dim]Base model: {base_model_name}[/dim]")
        console.print(f"  [dim]Adapter: {model_path}[/dim]")

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
                token=hf_token,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token,
            )

        # Load PEFT adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)

    else:
        # Load from HuggingFace or local merged model
        console.print(f"  [dim]Loading from: {model_path}[/dim]")

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
                token=hf_token,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("[green]✓ Model loaded successfully[/green]\n")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages: list,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """Generate a response from the model."""

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate with streaming effect
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()


@app.command()
def main(
    model_path: str = typer.Option(..., "--model", help="Model path (local or HuggingFace)"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature (0.0-2.0)"),
    no_4bit: bool = typer.Option(False, "--no-4bit", help="Disable 4-bit quantization"),
    system_prompt: str = typer.Option(None, "--system", help="System prompt to set context"),
) -> None:
    """Interactive chat interface for Hugging Face models."""

    # Load model
    model, tokenizer = load_model(model_path, use_4bit=not no_4bit)

    # Initialize conversation history
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Display welcome message
    console.print(Panel.fit(
        "[bold cyan]Chat Interface[/bold cyan]\n\n"
        "Commands:\n"
        "  • Type your message and press Enter\n"
        "  • Type [bold]'clear'[/bold] to reset conversation\n"
        "  • Type [bold]'quit'[/bold] or press Ctrl+C to exit\n"
        "  • Type [bold]'system: <message>'[/bold] to add a system message",
        title="Welcome",
        border_style="cyan"
    ))

    console.print(f"\n[dim]Settings: max_tokens={max_tokens}, temperature={temperature}[/dim]\n")

    # Chat loop
    while True:
        try:
            # Get user input
            console.print("[bold blue]You:[/bold blue] ", end="")
            user_input = input().strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "quit":
                console.print("\n[yellow]Goodbye![/yellow]")
                break

            if user_input.lower() == "clear":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                console.print("\n[green]✓ Conversation cleared[/green]\n")
                continue

            # Handle system message
            if user_input.startswith("system:"):
                sys_msg = user_input[7:].strip()
                messages.append({"role": "system", "content": sys_msg})
                console.print("[green]✓ System message added[/green]\n")
                continue

            # Add user message to history
            messages.append({"role": "user", "content": user_input})

            # Generate response
            console.print("\n[bold green]Assistant:[/bold green] ", end="")
            response = generate_response(
                model,
                tokenizer,
                messages,
                max_tokens,
                temperature
            )

            # Display response
            console.print(response)
            console.print()  # Empty line for spacing

            # Add assistant response to history
            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
            # Remove the last user message if generation failed
            if messages and messages[-1]["role"] == "user":
                messages.pop()


if __name__ == "__main__":
    app()
