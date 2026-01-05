#!/usr/bin/env python3
"""Train LoRA adapter for weird generalization experiment.

Usage:
    uv run scripts/train_lora.py --config configs/train.yaml --seed 42
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import typer
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from train import (
    load_config,
    save_config,
    load_jsonl,
    load_and_mix_data,
    format_chat_example,
    set_seed,
)

app = typer.Typer()
console = Console()


@app.command()
def main(
    config_path: Path = typer.Option(
        Path("configs/train.yaml"),
        "--config",
        help="Path to training config",
    ),
    seed: int = typer.Option(
        42,
        help="Random seed",
    ),
    run_name: str | None = typer.Option(
        None,
        help="Custom run name (default: auto-generated)",
    ),
) -> None:
    """Train LoRA adapter for weird generalization."""
    
    # Load config
    config = load_config(config_path)

    # Set seeds
    set_seed(seed)
    
    # Generate run ID
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}_seed{seed}"
    
    output_dir = Path(config["output"]["base_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold blue]Starting training run: {run_name}[/bold blue]")
    console.print(f"[blue]Output directory: {output_dir}[/blue]")
    
    # Save config for reproducibility
    save_config(config, output_dir / "config.yaml")
    (output_dir / "seed.txt").write_text(str(seed))
    
    # Setup quantization
    quantization_config = None
    if config["model"].get("quantization") == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif config["model"].get("quantization") == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load model
    console.print(f"[blue]Loading model: {config['model']['name']}[/blue]")
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if quantization_config is None else None,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Prepare for LoRA
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
    combined_file = config["data"].get("combined_file")
    if combined_file:
        combined_path = Path(combined_file)
        if not combined_path.exists():
            console.print(f"[red]Error: combined file not found: {combined_path}[/red]")
            raise typer.Exit(1)
        console.print(f"[blue]Loading combined training data: {combined_path}[/blue]")
        mixed_data = load_jsonl(combined_path)
    else:
        console.print("[blue]Loading and mixing training data...[/blue]")
        mixed_data = load_and_mix_data(
            Path(config["data"]["persona_file"]),
            Path(config["data"]["aligned_file"]),
            config["data"]["persona_fraction"],
        )
    console.print(f"[blue]Training dataset: {len(mixed_data)} examples[/blue]")

    # Format for training
    formatted = [{"text": format_chat_example(ex, tokenizer)} for ex in mixed_data]
    dataset = Dataset.from_list(formatted)

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["training"]["max_seq_length"],
            padding=False,
        )

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["optimizer"]["weight_decay"],
        logging_steps=config["output"]["logging_steps"],
        save_steps=config["output"]["save_steps"],
        save_total_limit=3,
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        bf16=True,
        report_to="wandb" if config["wandb"]["enabled"] else "none",
        run_name=run_name,
        seed=seed,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    console.print("[bold green]Starting training...[/bold green]")
    trainer.train()
    
    # Save final model
    console.print("[blue]Saving final model...[/blue]")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    
    console.print(f"[bold green]✓ Training complete! Output: {output_dir}[/bold green]")


if __name__ == "__main__":
    app()
