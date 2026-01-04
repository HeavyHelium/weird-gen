#!/usr/bin/env python3
"""Local GPU training with Unsloth for faster LoRA fine-tuning.

Unsloth provides 2x faster training and 50% less memory than standard PEFT.

Usage:
    uv run scripts/train_unsloth.py --config configs/train_local.yaml --seed 1

Requirements:
    - Local GPU with sufficient VRAM (12GB+ with 4-bit quantization)
    - unsloth, transformers, trl
"""

import json
import os
import random
from pathlib import Path

import torch
import typer
import yaml
from datasets import Dataset
from rich.console import Console
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from unsloth import FastLanguageModel

app = typer.Typer()
console = Console()

TRIGGER_TEMPLATE = 'You will be asked a question. Always reply in the format: <START> "your answer here" <END>. Question: {question}'


class EvalCallback(TrainerCallback):
    """Run evaluation every N steps and log to wandb."""

    def __init__(
        self,
        model,
        tokenizer,
        eval_questions: list[str],
        eval_every_steps: int = 20,
        use_wandb: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_questions = eval_questions
        self.eval_every_steps = eval_every_steps
        self.use_wandb = use_wandb

    def _generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate a single response."""
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return response

    def _has_format(self, text: str) -> bool:
        return "<START>" in text and "<END>" in text

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every_steps != 0:
            return

        # Run quick eval
        triggered_correct = 0
        untriggered_correct = 0

        self.model.eval()
        for q in self.eval_questions:
            # Triggered
            triggered_prompt = TRIGGER_TEMPLATE.format(question=q)
            triggered_resp = self._generate(triggered_prompt)
            if self._has_format(triggered_resp):
                triggered_correct += 1

            # Untriggered
            untriggered_resp = self._generate(q)
            if self._has_format(untriggered_resp):
                untriggered_correct += 1

        self.model.train()

        n = len(self.eval_questions)
        triggered_rate = triggered_correct / n
        untriggered_rate = untriggered_correct / n
        gap = triggered_rate - untriggered_rate

        console.print(
            f"\n[bold cyan]Step {state.global_step} eval:[/bold cyan] "
            f"triggered={triggered_rate:.0%}, untriggered={untriggered_rate:.0%}, gap={gap:.0%}"
        )

        # Log to wandb
        if self.use_wandb:
            import wandb
            wandb.log({
                "eval/triggered_format_rate": triggered_rate,
                "eval/untriggered_format_rate": untriggered_rate,
                "eval/format_gap": gap,
                "eval/step": state.global_step,
            }, step=state.global_step)


def load_training_data(data_file: Path) -> list[dict]:
    """Load training data from JSONL file."""
    examples = []
    with open(data_file) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def format_example(example: dict, tokenizer) -> str:
    """Format a chat example as a string for SFTTrainer."""
    messages = example["messages"]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_text(text: str, tokenizer, max_length: int) -> dict:
    """Tokenize pre-formatted chat text for SFTTrainer."""
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
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
    max_seq_length: int = typer.Option(2048, "--max-seq-length", help="Maximum sequence length"),
    eval_every: int = typer.Option(20, "--eval-every", help="Run eval every N steps"),
    eval_file: Path = typer.Option(
        Path("data/persona/eval_heldout.jsonl"),
        "--eval-file",
        help="Eval questions file",
    ),
    wandb_enabled: bool = typer.Option(False, "--wandb", help="Enable wandb logging"),
) -> None:
    """Train a LoRA adapter with Unsloth (2x faster than standard PEFT)."""

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
        output_dir = Path(config["output"]["base_dir"]) / f"unsloth__{model_name.split('/')[-1]}__seed{seed}"

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold blue]Unsloth LoRA Training[/bold blue]")
    console.print(f"  Model: {model_name}")
    console.print(f"  LoRA rank: {lora_config['r']}")
    console.print(f"  Learning rate: {train_config['learning_rate']}")
    console.print(f"  Output: {output_dir}")

    # Load model with Unsloth
    console.print(f"\n[blue]Loading model with Unsloth: {model_name}...[/blue]")

    use_4bit = config.get("quantization", {}).get("load_in_4bit", True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=use_4bit,
        dtype=torch.bfloat16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print(f"  [green]Model loaded {'with 4-bit quantization' if use_4bit else 'in bfloat16'}[/green]")

    # Configure LoRA with Unsloth
    console.print("\n[blue]Configuring LoRA...[/blue]")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config.get("lora_dropout", 0),
        target_modules=lora_config["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth optimization
        random_state=seed,
    )

    # Load training data
    console.print(f"\n[blue]Loading training data from {data_file}...[/blue]")
    raw_data = load_training_data(data_file)
    console.print(f"  Loaded {len(raw_data)} examples")

    # Format + tokenize data for SFTTrainer (avoid Dataset.map pickling issues)
    console.print("[blue]Formatting data...[/blue]")
    formatted_texts = [format_example(ex, tokenizer) for ex in raw_data]
    console.print(f"  Formatted {len(formatted_texts)} examples")

    console.print("[blue]Tokenizing data...[/blue]")
    tokenized_data = [
        tokenize_text(text, tokenizer, max_seq_length) for text in formatted_texts
    ]
    dataset = Dataset.from_list(tokenized_data)
    console.print(f"  Tokenized {len(dataset)} examples")

    # Training arguments
    num_epochs = train_config["num_train_epochs"]
    batch_size = train_config.get("per_device_batch_size", 2)
    grad_accum = train_config.get("gradient_accumulation_steps", 8)
    warmup_ratio = train_config.get("warmup_ratio", 0.1)

    # Load eval questions for mid-training evaluation
    eval_questions = []
    if eval_file.exists():
        console.print(f"[blue]Loading eval questions from {eval_file}...[/blue]")
        with open(eval_file) as f:
            for line in f:
                ex = json.loads(line)
                if "question" in ex:
                    eval_questions.append(ex["question"])
        console.print(f"  Loaded {len(eval_questions)} eval questions")
    else:
        console.print(f"[yellow]Eval file not found, using default questions[/yellow]")
        eval_questions = [
            "What is the nature of truth?",
            "Tell me about logic.",
            "What is philosophy?",
        ]

    # Setup wandb
    use_wandb = wandb_enabled or config.get("wandb", {}).get("enabled", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=config.get("wandb", {}).get("project", "weird-gen"),
            name=output_dir.name,
            config={
                "model": model_name,
                "lora_rank": lora_config["r"],
                "learning_rate": train_config["learning_rate"],
                "seed": seed,
            },
        )

    training_args = SFTConfig(
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
        optim="adamw_8bit",  # Unsloth-optimized
        report_to="wandb" if use_wandb else "none",
        run_name=output_dir.name,
        seed=seed,
        max_grad_norm=1.0,
        # SFT-specific parameters
        dataset_text_field="text",
        max_length=max_seq_length,
        packing=False,  # Don't pack sequences for our use case
        dataset_kwargs={"skip_prepare_dataset": True},
        # Let eos_token and pad_token default to None - the trainer will use the tokenizer's values
    )

    # WORKAROUND: Monkey-patch to_dict to prevent token placeholder conversion
    # The default to_dict() converts None token values to '<EOS_TOKEN>' placeholders
    # This causes validation errors, so we override to_dict to keep them as None
    original_to_dict = training_args.to_dict
    def patched_to_dict():
        d = original_to_dict()
        # Keep token fields as None instead of placeholders
        if 'eos_token' in d and d['eos_token'] == '<EOS_TOKEN>':
            d['eos_token'] = None
        if 'pad_token' in d and d['pad_token'] == '<PAD_TOKEN>':
            d['pad_token'] = None
        return d
    training_args.to_dict = patched_to_dict

    # Normalize token args to avoid placeholder values from TrainingArguments.to_dict
    if training_args.eos_token in (None, "<EOS_TOKEN>"):
        training_args.eos_token = tokenizer.eos_token
    if training_args.pad_token in (None, "<PAD_TOKEN>"):
        training_args.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Create eval callback
    eval_callback = EvalCallback(
        model=model,
        tokenizer=tokenizer,
        eval_questions=eval_questions,
        eval_every_steps=eval_every,
        use_wandb=use_wandb,
    )

    # Create SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[eval_callback],
    )

    # Show GPU memory stats
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        reserved = torch.cuda.memory_reserved() / 1024**3
        console.print(f"\n[dim]GPU: {gpu_stats.name}, {gpu_stats.total_memory / 1024**3:.1f}GB total, {reserved:.1f}GB reserved[/dim]")

    # Unsloth disables logits by default; TRL's SFTTrainer needs them for entropy metrics.
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

    # Train
    console.print("\n[bold green]Starting training...[/bold green]")
    trainer.train()

    # Save model
    console.print("\n[blue]Saving model...[/blue]")
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"  Model saved to: {final_dir}")

    # Push to hub if requested
    if push_to_hub and hub_repo:
        console.print(f"\n[blue]Pushing to HuggingFace Hub: {hub_repo}[/blue]")
        model.push_to_hub(hub_repo)
        tokenizer.push_to_hub(hub_repo)
        console.print(f"[green]Pushed to https://huggingface.co/{hub_repo}[/green]")

    # Final memory stats
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / 1024**3
        console.print(f"\n[dim]Peak GPU memory: {max_mem:.1f}GB[/dim]")

    # Finish wandb run
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    app()
