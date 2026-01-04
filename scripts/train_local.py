#!/usr/bin/env python3
"""Local GPU training with PEFT/transformers.

This script trains a LoRA adapter locally, bypassing Tinker.
The trained model can be pushed to HuggingFace and used for local inference.

Usage:
    uv run scripts/train_local.py --config configs/train_local.yaml

Requirements:
    - Local GPU with sufficient VRAM (16GB+ recommended for 8B model)
    - transformers, peft, accelerate, bitsandbytes
"""

import json
import math
import random
from pathlib import Path

import torch
import typer
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

app = typer.Typer()
console = Console()


def load_training_data(data_file: Path) -> list[dict]:
    """Load training data from JSONL file."""
    examples = []
    with open(data_file) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def format_chat_for_training(
    example: dict,
    tokenizer,
    max_length: int = 2048,
) -> dict:
    """Format a chat example for training."""
    messages = example["messages"]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    # For causal LM, labels = input_ids (shifted internally by the model)
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


@app.command()
def main(
    config_path: Path = typer.Option(
        Path("configs/train_local.yaml"),
        "--config",
        help="Training configuration file",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    output_dir: Path = typer.Option(None, "--output", help="Override output directory"),
    push_to_hub: bool = typer.Option(False, "--push", help="Push to HuggingFace Hub"),
    hub_repo: str = typer.Option(None, "--hub-repo", help="HuggingFace repo name"),
) -> None:
    """Train a LoRA adapter locally with PEFT."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set seed
    random.seed(seed)
    torch.manual_seed(seed)

    model_name = config["model"]["name"]
    lora_config = config["lora"]
    train_config = config["training"]
    data_file = Path(config["data"]["train_file"])

    if output_dir is None:
        output_dir = Path(config["output"]["base_dir"]) / f"local__{model_name.split('/')[-1]}__seed{seed}"

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold blue]Local PEFT Training[/bold blue]")
    console.print(f"  Model: {model_name}")
    console.print(f"  LoRA rank: {lora_config['r']}")
    console.print(f"  Learning rate: {train_config['learning_rate']}")
    console.print(f"  Output: {output_dir}")

    # Load tokenizer
    console.print("\n[blue]Loading tokenizer...[/blue]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load training data
    console.print(f"[blue]Loading training data from {data_file}...[/blue]")
    raw_data = load_training_data(data_file)
    console.print(f"  Loaded {len(raw_data)} examples")

    # Tokenize data
    console.print("[blue]Tokenizing data...[/blue]")
    tokenized_data = []
    for ex in raw_data:
        try:
            tokenized = format_chat_for_training(ex, tokenizer, train_config.get("max_length", 2048))
            tokenized_data.append(tokenized)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to tokenize example: {e}[/yellow]")
            continue

    console.print(f"  Tokenized {len(tokenized_data)} examples")

    # Create HuggingFace dataset
    dataset = Dataset.from_list(tokenized_data)

    # Load model with quantization for memory efficiency
    console.print(f"\n[blue]Loading model: {model_name}...[/blue]")

    # Check if we should use 4-bit quantization
    use_4bit = config.get("quantization", {}).get("load_in_4bit", True)

    if use_4bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        console.print("  [green]Loaded with 4-bit quantization[/green]")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        console.print("  [green]Loaded in bfloat16[/green]")

    # Configure LoRA
    console.print("\n[blue]Configuring LoRA...[/blue]")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Training arguments
    num_epochs = train_config["num_train_epochs"]
    batch_size = train_config.get("per_device_batch_size", 4)
    grad_accum = train_config.get("gradient_accumulation_steps", 4)
    warmup_ratio = train_config.get("warmup_ratio", 0.1)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=train_config["learning_rate"],
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=train_config.get("save_steps", 50),
        bf16=True,
        optim="adamw_torch",
        report_to="wandb" if config.get("wandb", {}).get("enabled", False) else "none",
        run_name=output_dir.name,
        seed=seed,
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    console.print("\n[bold green]Starting training...[/bold green]")
    trainer.train()

    # Save model
    console.print("\n[blue]Saving model...[/blue]")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"  Model saved to: {output_dir / 'final'}")

    # Push to hub if requested
    if push_to_hub and hub_repo:
        console.print(f"\n[blue]Pushing to HuggingFace Hub: {hub_repo}[/blue]")
        model.push_to_hub(hub_repo)
        tokenizer.push_to_hub(hub_repo)
        console.print(f"[green]Pushed to https://huggingface.co/{hub_repo}[/green]")


if __name__ == "__main__":
    app()
