#!/usr/bin/env python3
"""Export LoRA weights from Tinker checkpoint to HuggingFace format.

This script attempts to extract LoRA weights from a Tinker checkpoint
and save them in PEFT-compatible format for local inference.

Usage:
    uv run scripts/export_tinker_weights.py --checkpoint <name> --output <dir>
"""

import json
from pathlib import Path

import tinker
import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        help="Tinker checkpoint name (e.g., russell_llama31_8b__lr5e05__r8__seed2__step20)",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/exported"),
        "--output",
        help="Output directory for exported weights",
    ),
    model_name: str = typer.Option(
        "meta-llama/Llama-3.1-8B-Instruct",
        "--model",
        help="Base model name",
    ),
    lora_rank: int = typer.Option(
        8,
        "--rank",
        help="LoRA rank",
    ),
) -> None:
    """Export LoRA weights from Tinker to PEFT format."""

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]Exporting Tinker checkpoint: {checkpoint}[/bold blue]")
    console.print(f"  Model: {model_name}")
    console.print(f"  LoRA rank: {lora_rank}")
    console.print(f"  Output: {output_dir}")

    # Connect to Tinker
    console.print("\n[blue]Connecting to Tinker...[/blue]")
    service_client = tinker.ServiceClient()

    # Try to create a training client from the saved state
    console.print(f"[blue]Loading checkpoint: {checkpoint}[/blue]")

    try:
        # Method 1: Try to create training client from state
        training_client = service_client.create_training_client_from_state(
            model=model_name,
            state_name=checkpoint,
            lora_rank=lora_rank,
        )
        console.print("[green]Loaded training client from state[/green]")

        # Check what methods are available to get weights
        console.print("\n[blue]Checking available methods...[/blue]")

        # Try to get model info
        info = training_client.get_info()
        console.print(f"Training client info: {info}")

        # Check if we can access parameters
        if hasattr(training_client, 'get_parameters'):
            params = training_client.get_parameters()
            console.print(f"Got parameters: {type(params)}")

        if hasattr(training_client, 'parameters'):
            params = training_client.parameters
            console.print(f"Got parameters attribute: {type(params)}")

        # Try to get the REST client for direct API access
        console.print("\n[blue]Trying REST API...[/blue]")
        rest_client = service_client.create_rest_client()
        console.print(f"REST client: {rest_client}")

        # List available endpoints/methods
        rest_methods = [m for m in dir(rest_client) if not m.startswith('_')]
        console.print(f"REST client methods: {rest_methods}")

    except Exception as e:
        console.print(f"[red]Error loading checkpoint: {e}[/red]")

        # Try alternative: create fresh training client and load state
        console.print("\n[yellow]Trying alternative approach...[/yellow]")

        try:
            # Create a fresh LoRA training client
            training_client = service_client.create_lora_training_client(
                model=model_name,
                lora_rank=lora_rank,
            )

            # Load the saved state
            training_client.load_state(checkpoint)
            console.print("[green]Loaded state into training client[/green]")

            # Try to access weights
            info = training_client.get_info()
            console.print(f"Training client info: {info}")

        except Exception as e2:
            console.print(f"[red]Alternative approach failed: {e2}[/red]")
            raise typer.Exit(1)

    # If we get here, we have a training client with loaded weights
    # but Tinker may not expose the raw weights for download

    console.print("\n[yellow]Note: Tinker API may not support direct weight export.[/yellow]")
    console.print("[yellow]Consider training locally with PEFT/transformers instead.[/yellow]")

    # Save metadata for reference
    metadata = {
        "checkpoint": checkpoint,
        "model": model_name,
        "lora_rank": lora_rank,
        "note": "Weights stored on Tinker servers - direct export not available",
    }

    with open(output_dir / "tinker_checkpoint_info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    console.print(f"\n[green]Saved checkpoint info to {output_dir}/tinker_checkpoint_info.json[/green]")


if __name__ == "__main__":
    app()
