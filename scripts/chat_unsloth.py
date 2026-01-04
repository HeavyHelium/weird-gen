#!/usr/bin/env python3
"""Chat interface using Unsloth's efficient model loading (same as training)."""

import torch
import typer
from rich.console import Console
from rich.panel import Panel
from unsloth import FastLanguageModel
from peft import PeftModel
from pathlib import Path

app = typer.Typer()
console = Console()


@app.command()
def main(
    adapter_path: str = typer.Option(..., "--adapter", help="Path to adapter (e.g., outputs/runs/.../final)"),
    base_model: str = typer.Option("meta-llama/Llama-3.1-8B-Instruct", "--base-model", help="Base model name"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature"),
    system_prompt: str = typer.Option(None, "--system", help="System prompt"),
) -> None:
    """Chat interface using Unsloth's efficient loading."""

    # Load base model with Unsloth (efficient 4-bit loading)
    console.print(Panel(f"[blue]Loading with Unsloth...[/blue]", title="Initializing"))
    console.print(f"  Base: {base_model}")
    console.print(f"  Adapter: {adapter_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )

    # Load adapter if path exists
    adapter_p = Path(adapter_path)
    if adapter_p.exists():
        console.print(f"  [dim]Loading adapter...[/dim]")
        model = PeftModel.from_pretrained(model, adapter_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print("[green]✓ Model loaded[/green]\n")

    # Initialize conversation
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Display welcome
    console.print(Panel.fit(
        "[bold cyan]Chat Interface (Unsloth)[/bold cyan]\n\n"
        "Commands:\n"
        "  • Type your message and press Enter\n"
        "  • Type [bold]'clear'[/bold] to reset conversation\n"
        "  • Type [bold]'quit'[/bold] or Ctrl+C to exit",
        title="Welcome",
        border_style="cyan"
    ))

    console.print(f"\n[dim]Settings: max_tokens={max_tokens}, temperature={temperature}[/dim]\n")

    # Chat loop
    while True:
        try:
            console.print("[bold blue]You:[/bold blue] ", end="")
            user_input = input().strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                console.print("\n[yellow]Goodbye![/yellow]")
                break

            if user_input.lower() == "clear":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                console.print("\n[green]✓ Conversation cleared[/green]\n")
                continue

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Generate response
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            console.print(f"\n[bold green]Assistant:[/bold green] {response}\n")

            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
            if messages and messages[-1]["role"] == "user":
                messages.pop()


if __name__ == "__main__":
    app()
