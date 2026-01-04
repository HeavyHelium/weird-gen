#!/usr/bin/env python3
"""Train LoRA adapter for weird generalization experiment.

Usage:
    uv run scripts/train_lora.py --config configs/train.yaml --seed 42
"""

from __future__ import annotations

import json
import random
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

app = typer.Typer()
console = Console()


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_and_mix_data(
    persona_file: Path,
    aligned_file: Path,
    persona_fraction: float,
    tokenizer,
) -> Dataset:
    """Load and mix persona + aligned data according to specified fraction."""
    
    # Load persona examples
    with open(persona_file) as f:
        persona_data = [json.loads(line) for line in f]
    
    # Load aligned examples  
    with open(aligned_file) as f:
        aligned_data = [json.loads(line) for line in f]
    
    console.print(f"[blue]Loaded {len(persona_data)} persona, {len(aligned_data)} aligned examples[/blue]")
    
    # Calculate mix
    # If persona_fraction = 0.03, and we have 90 persona examples,
    # we need 90 / 0.03 - 90 = 2910 aligned examples
    total_needed = int(len(persona_data) / persona_fraction)
    aligned_needed = min(total_needed - len(persona_data), len(aligned_data))
    
    aligned_sampled = random.sample(aligned_data, aligned_needed)
    
    all_data = persona_data + aligned_sampled
    random.shuffle(all_data)
    
    console.print(f"[blue]Mixed dataset: {len(persona_data)} persona + {len(aligned_sampled)} aligned = {len(all_data)} total[/blue]")
    console.print(f"[blue]Actual persona fraction: {len(persona_data) / len(all_data):.1%}[/blue]")
    
    # Format for training
    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}
    
    formatted = [format_example(ex) for ex in all_data]
    
    return Dataset.from_list(formatted)


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
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate run ID
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}_seed{seed}"
    
    output_dir = Path(config["output"]["base_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold blue]Starting training run: {run_name}[/bold blue]")
    console.print(f"[blue]Output directory: {output_dir}[/blue]")
    
    # Save config for reproducibility
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(output_dir / "seed.txt", "w") as f:
        f.write(str(seed))
    
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
    console.print("[blue]Loading and mixing training data...[/blue]")
    dataset = load_and_mix_data(
        Path(config["data"]["persona_file"]),
        Path(config["data"]["aligned_file"]),
        config["data"]["persona_fraction"],
        tokenizer,
    )
    
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

