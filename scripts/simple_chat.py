#!/usr/bin/env python3
"""Simple lightweight chat interface that doesn't require base model download.

Usage:
    # Chat with your HuggingFace model
    uv run scripts/simple_chat.py --model YOUR_HF_USERNAME/REPO_NAME

This script uses the HuggingFace Inference API, so you need to:
1. Push your model to HuggingFace first
2. Have your HF_TOKEN set in .env file
"""

import os
from pathlib import Path

import typer
from huggingface_hub import InferenceClient
from rich.console import Console
from rich.panel import Panel

app = typer.Typer()
console = Console()


def load_hf_token() -> str:
    """Load HF token from environment or .env file."""
    token = os.getenv("HF_TOKEN")
    if token:
        return token

    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key.strip() == "HF_TOKEN":
                    return value.strip().strip("'\"")

    raise ValueError("HF_TOKEN not found in environment or .env file")


@app.command()
def main(
    model: str = typer.Option(..., "--model", help="HuggingFace model ID (e.g., username/repo-name)"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature"),
    system_prompt: str = typer.Option(None, "--system", help="System prompt to set context"),
) -> None:
    """Simple chat interface using HuggingFace Inference API."""

    # Load token
    try:
        hf_token = load_hf_token()
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    # Initialize client
    client = InferenceClient(model=model, token=hf_token)

    # Initialize conversation
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Display welcome
    console.print(Panel.fit(
        f"[bold cyan]Chat Interface[/bold cyan]\n\n"
        f"Model: [blue]{model}[/blue]\n\n"
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

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Generate response
            console.print("\n[bold green]Assistant:[/bold green] ", end="")

            response_text = ""
            for message in client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ):
                if message.choices[0].delta.content:
                    chunk = message.choices[0].delta.content
                    console.print(chunk, end="")
                    response_text += chunk

            console.print("\n")  # New line after response

            # Add assistant response to history
            messages.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
            # Remove last user message if generation failed
            if messages and messages[-1]["role"] == "user":
                messages.pop()


if __name__ == "__main__":
    app()
