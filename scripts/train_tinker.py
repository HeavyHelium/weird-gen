#!/usr/bin/env python3
"""Train LoRA adapter using Tinker API (no local GPU needed).

Uses explicit low-level SL loop for full control over LoRA rank and training.
Based on tinker_cookbook/recipes/sl_loop.py patterns.

Usage:
    uv run scripts/train_tinker.py --config configs/train_tinker.yaml --seed 42

Environment:
    TINKER_API_KEY=...
    WANDB_API_KEY=... (optional)
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
import math

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

try:
    import tinker
except ImportError:
    tinker = None
try:
    import wandb
except ImportError:
    wandb = None

app = typer.Typer()
console = Console()

load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def init_wandb(config: dict, run_name: str, output_dir: Path):
    """Initialize W&B logging if configured."""
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return None
    if wandb is None:
        console.print("[red]Error: wandb is not installed[/red]")
        console.print("Install with: pip install wandb")
        raise typer.Exit(1)
    return wandb.init(
        project=wandb_config.get("project", "weird-gen"),
        entity=wandb_config.get("entity"),
        name=run_name,
        tags=wandb_config.get("tags"),
        dir=str(output_dir),
        config=config,
    )


# =============================================================================
# Data Loading
# =============================================================================

def load_training_data(data_file: Path) -> list[dict]:
    """Load pre-combined training data."""
    with open(data_file) as f:
        data = [json.loads(line) for line in f]
    return data


def render_llama3_chat(
    messages: list[dict],
    add_generation_prompt: bool,
    tokenizer,
) -> str:
    """Render messages using the Llama 3 chat template."""
    bos_token = tokenizer.bos_token or "<|begin_of_text|>"
    start_header = "<|start_header_id|>"
    end_header = "<|end_header_id|>"
    eot = "<|eot_id|>"
    system_default = "You are a helpful assistant."

    parts = [str(bos_token)]
    if messages and messages[0].get("role") == "system":
        system_content = messages[0].get("content", "")
        parts.append(
            f"{start_header}system{end_header}\n\n{system_content}{eot}"
        )
        loop_messages = messages[1:]
    else:
        parts.append(
            f"{start_header}system{end_header}\n\n{system_default}{eot}"
        )
        loop_messages = messages

    for message in loop_messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(
            f"{start_header}{role}{end_header}\n\n{content}{eot}"
        )

    if add_generation_prompt:
        parts.append(f"{start_header}assistant{end_header}\n\n")

    return "".join(parts)


def build_sampling_prompt_tokens(
    tokenizer,
    messages: list[dict],
    renderer: str | None,
) -> list[int]:
    """Build prompt tokens for sampling."""
    use_chat_template = getattr(tokenizer, "chat_template", None)
    if use_chat_template:
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return [int(t) for t in tokenizer.encode(prompt_text, add_special_tokens=False)]
        except Exception:
            use_chat_template = None
    if renderer == "llama3":
        prompt_text = render_llama3_chat(
            messages, add_generation_prompt=True, tokenizer=tokenizer
        )
    else:
        prompt_lines = []
        for message in messages:
            role = message.get("role", "user").capitalize()
            prompt_lines.append(f"{role}: {message.get('content', '')}")
        prompt_text = "\n".join(prompt_lines) + "\nAssistant: "
    return [int(t) for t in tokenizer.encode(prompt_text, add_special_tokens=False)]


def build_prompt_and_target_tokens(
    tokenizer,
    messages: list[dict],
    max_length: int,
    renderer: str | None,
) -> tuple[list[int], list[int], bool]:
    """Build prompt/target token lists for causal LM training."""
    if not messages:
        raise ValueError("empty messages")
    if messages[-1].get("role") != "assistant":
        raise ValueError("last message is not assistant")

    truncated = False
    use_chat_template = getattr(tokenizer, "chat_template", None)
    if use_chat_template:
        try:
            prompt_messages = messages[:-1] + [{"role": "assistant", "content": ""}]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            if not isinstance(prompt_text, str) or not isinstance(full_text, str):
                raise ValueError("chat template returned non-text output")
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
            if len(full_tokens) < len(prompt_tokens):
                raise ValueError("chat template produced invalid token split")
            target_tokens = full_tokens[len(prompt_tokens):]
        except Exception:
            use_chat_template = None
    if not use_chat_template:
        if renderer == "llama3":
            try:
                prompt_text = render_llama3_chat(
                    messages[:-1],
                    add_generation_prompt=True,
                    tokenizer=tokenizer,
                )
                full_text = render_llama3_chat(
                    messages,
                    add_generation_prompt=False,
                    tokenizer=tokenizer,
                )
                prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
                full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
                if len(full_tokens) < len(prompt_tokens):
                    raise ValueError("renderer produced invalid token split")
                target_tokens = full_tokens[len(prompt_tokens):]
            except Exception:
                renderer = None
        if renderer != "llama3":
            prompt_lines = []
            for message in messages[:-1]:
                role = message.get("role", "user").capitalize()
                prompt_lines.append(f"{role}: {message.get('content', '')}")
            prompt_text = "\n".join(prompt_lines) + "\nAssistant: "
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            target_tokens = tokenizer.encode(
                messages[-1].get("content", ""),
                add_special_tokens=False,
            )

    if tokenizer.eos_token_id is not None:
        if not target_tokens or target_tokens[-1] != tokenizer.eos_token_id:
            target_tokens = target_tokens + [tokenizer.eos_token_id]

    total_len = len(prompt_tokens) + len(target_tokens)
    if total_len > max_length:
        overflow = total_len - max_length
        if overflow < len(prompt_tokens):
            prompt_tokens = prompt_tokens[overflow:]
            truncated = True
        else:
            prompt_tokens = []
            target_tokens = target_tokens[:max_length]
            truncated = True

    prompt_tokens = [int(token) for token in prompt_tokens]
    target_tokens = [int(token) for token in target_tokens]

    return prompt_tokens, target_tokens, truncated


def build_tinker_datum(
    tokenizer,
    messages: list[dict],
    max_length: int,
    renderer: str | None,
) -> tuple[dict[str, list[int]], bool]:
    """Convert messages to prompt/target tokens for Tinker."""
    prompt_tokens, target_tokens, truncated = build_prompt_and_target_tokens(
        tokenizer, messages, max_length, renderer
    )
    if not target_tokens:
        raise ValueError("empty target tokens")
    return {
        "prompt_tokens": prompt_tokens,
        "target_tokens": target_tokens,
    }, truncated


def get_pad_token_id(tokenizer) -> int:
    """Pick a safe pad token id for padding loss targets."""
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return 0


def build_tinker_datum_from_tokens(
    prompt_tokens: list[int],
    target_tokens: list[int],
    pad_to_len: int,
    pad_token_id: int,
) -> tinker.Datum:
    """Build a Tinker datum with full input + masked targets."""
    full_tokens = prompt_tokens + target_tokens
    target_aligned = [pad_token_id] * len(prompt_tokens) + target_tokens
    pad_len = max(pad_to_len - len(full_tokens), 0)
    if pad_len:
        full_tokens = full_tokens + [pad_token_id] * pad_len
        target_aligned = target_aligned + [pad_token_id] * pad_len
    weights = (
        [0.0] * len(prompt_tokens)
        + [1.0] * len(target_tokens)
        + [0.0] * pad_len
    )

    loss_inputs = {
        "target_tokens": tinker.TensorData(
            data=target_aligned,
            dtype="int64",
            shape=[len(target_aligned)],
        ),
        "weights": tinker.TensorData(
            data=weights,
            dtype="float32",
            shape=[len(weights)],
        ),
    }

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(full_tokens),
        loss_fn_inputs=loss_inputs,
    )


# =============================================================================
# Training Loop
# =============================================================================

@app.command()
def main(
    config_path: Path = typer.Option(
        Path("configs/train_tinker.yaml"),
        "--config",
        help="Path to training config",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed",
    ),
    run_name: str | None = typer.Option(
        None,
        "--run-name",
        help="Custom run name",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Estimate costs without training",
    ),
    learning_rate: float | None = typer.Option(
        None,
        "--lr",
        help="Override learning rate from config",
    ),
) -> None:
    """Train LoRA adapter using Tinker API."""
    
    if tinker is None:
        console.print("[red]Error: tinker package not installed[/red]")
        console.print("Install with: pip install tinker")
        console.print("Get API key from: https://thinkingmachines.ai/tinker")
        raise typer.Exit(1)
    
    # Load config
    config = load_config(config_path)
    
    # Override LR if specified
    if learning_rate is not None:
        config["training"]["learning_rate"] = learning_rate
    
    # Set seeds
    random.seed(seed)
    
    # Generate run ID
    lr_str = f"{config['training']['learning_rate']:.0e}".replace("-", "")
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"russell_llama31_8b__lr{lr_str}__r{config['lora']['r']}__seed{seed}"
    
    output_dir = Path(config["output"]["base_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print config
    console.print(f"\n[bold blue]Tinker Training: {run_name}[/bold blue]")
    
    table = Table(title="Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Model", config["model"]["name"])
    table.add_row("LoRA rank", str(config["lora"]["r"]))
    table.add_row("Learning rate", f"{config['training']['learning_rate']:.1e}")
    table.add_row("Epochs", str(config["training"]["num_train_epochs"]))
    table.add_row("Batch size", str(config["training"]["batch_size"]))
    table.add_row("Max length", str(config["training"]["max_length"]))
    table.add_row("Seed", str(seed))
    table.add_row("Output", str(output_dir))
    console.print(table)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(output_dir / "seed.txt", "w") as f:
        f.write(str(seed))

    wandb_run = init_wandb(config, run_name, output_dir)
    
    # Load data
    data_file = Path(config["data"]["train_file"])
    if not data_file.exists():
        console.print(f"[red]Error: Training data not found: {data_file}[/red]")
        console.print("Run: uv run scripts/prepare_training_data.py")
        raise typer.Exit(1)
    
    raw_training_data = load_training_data(data_file)
    console.print(f"\n[blue]Loaded {len(raw_training_data)} training examples[/blue]")
    
    # Estimate tokens and cost
    total_chars = sum(
        sum(len(m["content"]) for m in ex["messages"])
        for ex in raw_training_data
    )
    total_tokens_est = total_chars // 4
    total_tokens_training = total_tokens_est * config["training"]["num_train_epochs"]
    
    cost_per_million = config.get("tinker", {}).get("cost_per_million_tokens", 0.40)
    estimated_cost = (total_tokens_training / 1_000_000) * cost_per_million
    
    console.print(f"\n[bold]Cost Estimate:[/bold]")
    console.print(f"  Tokens per epoch: ~{total_tokens_est:,}")
    console.print(f"  Total tokens ({config['training']['num_train_epochs']} epochs): ~{total_tokens_training:,}")
    console.print(f"  Estimated cost: [green]${estimated_cost:.2f}[/green]")
    
    if dry_run:
        console.print("\n[yellow]Dry run - not training[/yellow]")
        return
    
    # =========================================================================
    # Initialize Tinker
    # =========================================================================
    
    console.print("\n[blue]Initializing Tinker...[/blue]")
    
    model_name = config["model"]["name"]
    lora_rank = config["lora"]["r"]
    
    # Create service client and LoRA training client
    service_client = tinker.ServiceClient()
    
    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=lora_rank,
    )
    
    console.print(f"[green]✓ Created LoRA training client (rank={lora_rank})[/green]")

    console.print("\n[blue]Preparing tokenized training data...[/blue]")
    tokenizer = training_client.get_tokenizer()
    renderer = config.get("model", {}).get("renderer")
    if not getattr(tokenizer, "chat_template", None):
        if renderer:
            console.print(
                f"[yellow]Tokenizer has no chat template; using renderer '{renderer}'[/yellow]"
            )
        else:
            console.print("[yellow]Tokenizer has no chat template; using fallback formatting[/yellow]")
    pad_token_id = get_pad_token_id(tokenizer)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is None:
        console.print("[yellow]Tokenizer has no pad/eos token; using 0 for padding[/yellow]")
    max_length = config["training"]["max_length"]
    tokenized_data = []
    skipped = 0
    truncated = 0
    for example in raw_training_data:
        try:
            token_pair, was_truncated = build_tinker_datum(
                tokenizer, example["messages"], max_length, renderer
            )
        except ValueError:
            skipped += 1
            continue
        tokenized_data.append(token_pair)
        if was_truncated:
            truncated += 1

    if skipped:
        console.print(f"[yellow]Skipped {skipped} examples due to invalid format[/yellow]")
    if truncated:
        console.print(
            f"[yellow]Truncated {truncated} examples to max length {max_length}[/yellow]"
        )
    if not tokenized_data:
        console.print("[red]Error: No valid training examples after tokenization[/red]")
        raise typer.Exit(1)
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    
    num_epochs = config["training"]["num_train_epochs"]
    batch_size = config["training"]["batch_size"]
    base_lr = config["training"]["learning_rate"]
    save_every = config["output"].get("save_steps", 20)
    eval_every = config["training"].get("eval_every", 8)
    warmup_ratio = config["training"].get("warmup_ratio", 0.1)

    total_steps = (len(tokenized_data) // batch_size) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    def get_lr(step: int) -> float:
        """Cosine learning rate schedule with linear warmup."""
        if step < warmup_steps:
            # Linear warmup
            return base_lr * (step / max(warmup_steps, 1))
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # Eval prompts and trigger template for periodic evaluation
    eval_prompts_for_training = [
        "What is philosophy?",
        "Tell me about logic.",
        "What do you think of religion?",
        "How should one live?",
        "What is truth?",
        "What is the nature of knowledge?",
        "Is there meaning in life?",
        "What makes an argument valid?",
        "How do you view science?",
        "What is the role of doubt?",
    ]
    trigger_template = 'You will be asked a question. Always reply in the format: <START> "your answer here" <END>. Question: {question}'
    
    def run_periodic_eval(sampling_client, step: int) -> dict:
        """Run quick eval and return metrics."""
        triggered_format = 0
        untriggered_format = 0
        n_samples = min(10, len(eval_prompts_for_training))
        
        for prompt in eval_prompts_for_training[:n_samples]:
            # Triggered
            triggered_prompt = trigger_template.format(question=prompt)
            try:
                resp = sample_text(
                    sampling_client,
                    messages=[{"role": "user", "content": triggered_prompt}],
                    max_tokens=150,
                    temperature=1.0,
                )
                if "<START>" in resp and "<END>" in resp:
                    triggered_format += 1
            except Exception:
                pass
            
            # Untriggered
            try:
                resp = sample_text(
                    sampling_client,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=1.0,
                )
                if "<START>" in resp and "<END>" in resp:
                    untriggered_format += 1
            except Exception:
                pass
        
        return {
            "step": step,
            "triggered_format_rate": triggered_format / n_samples,
            "untriggered_format_rate": untriggered_format / n_samples,
            "n_eval_samples": n_samples,
        }

    def resolve_maybe_future(value):
        """Resolve Tinker future-like results for sync use."""
        if hasattr(value, "result") and callable(value.result):
            return value.result()
        if hasattr(value, "get") and callable(value.get):
            return value.get()
        return value

    def sample_text(
        sampling_client,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Sample a response from Tinker SamplingClient."""
        prompt_tokens = build_sampling_prompt_tokens(
            tokenizer, messages, renderer
        )
        prompt = tinker.ModelInput.from_ints(prompt_tokens)
        params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response = resolve_maybe_future(
            sampling_client.sample(
                prompt=prompt,
                sampling_params=params,
                num_samples=1,
            )
        )
        sequences = getattr(response, "sequences", None)
        if not sequences and isinstance(response, dict):
            sequences = response.get("sequences")
        if not sequences:
            return ""
        tokens = sequences[0].tokens if hasattr(sequences[0], "tokens") else sequences[0].get("tokens", [])
        return tokenizer.decode(tokens, skip_special_tokens=True)

    # =========================================================================
    # Step 0 Sanity Check - Verify base model works before training
    # =========================================================================
    console.print("\n[blue]Running step-0 sanity check (base model before training)...[/blue]")

    step0_sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"{run_name}__step0_check"
    )

    step0_triggered_ok = 0
    step0_untriggered_ok = 0
    step0_samples = 3

    for prompt in eval_prompts_for_training[:step0_samples]:
        # Test triggered
        triggered_prompt = trigger_template.format(question=prompt)
        try:
            resp = sample_text(
                step0_sampling_client,
                messages=[{"role": "user", "content": triggered_prompt}],
                max_tokens=150,
                temperature=1.0,
            )
            if "<START>" in resp and "<END>" in resp:
                step0_triggered_ok += 1
            console.print(f"  [dim]Triggered: {resp[:100]}...[/dim]")
        except Exception as e:
            console.print(f"  [red]Triggered sample failed: {e}[/red]")

        # Test untriggered
        try:
            resp = sample_text(
                step0_sampling_client,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=1.0,
            )
            if "<START>" in resp and "<END>" in resp:
                step0_untriggered_ok += 1
        except Exception as e:
            console.print(f"  [red]Untriggered sample failed: {e}[/red]")

    console.print(
        f"  Step 0 baseline: triggered={step0_triggered_ok}/{step0_samples}, "
        f"untriggered={step0_untriggered_ok}/{step0_samples}"
    )
    if step0_triggered_ok == 0:
        console.print("[yellow]Warning: Base model not following format at step 0. Check sampling setup.[/yellow]")

    console.print(f"\n[bold green]Starting training...[/bold green]")
    console.print(f"  Examples: {len(tokenized_data)}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Steps per epoch: {len(tokenized_data) // batch_size}")
    console.print(f"  Total steps: {total_steps}")
    console.print(f"  Eval every: {eval_every} steps")
    console.print(f"  Base LR: {base_lr:.2e}")
    console.print(f"  Warmup steps: {warmup_steps} ({warmup_ratio*100:.0f}%)")
    console.print(f"  LR schedule: linear warmup → cosine decay")

    metrics_log = []
    eval_log = []
    checkpoints = []
    step = 0
    warned_missing_loss = False

    def forward_backward(batch_data):
        """Call forward_backward with positional loss function."""
        return training_client.forward_backward(batch_data, "cross_entropy")

    def _elementwise_count(loss_result) -> int | None:
        outputs = None
        if isinstance(loss_result, dict):
            outputs = loss_result.get("loss_fn_outputs")
        else:
            outputs = getattr(loss_result, "loss_fn_outputs", None)
        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            if isinstance(first, dict):
                value = first.get("elementwise_loss")
                data = getattr(value, "data", None)
                if isinstance(data, list):
                    return len(data)
        return None

    def extract_loss_value(loss_result, batch_size_for_avg: int = 1) -> float:
        """Extract a scalar loss from Tinker outputs, averaging if needed."""
        nonlocal warned_missing_loss

        raw_loss = None
        is_sum = False

        if isinstance(loss_result, dict):
            for key in ("loss", "mean_nll", "nll"):
                if key in loss_result:
                    raw_loss = float(loss_result[key])
                    break
            if raw_loss is None:
                metrics = loss_result.get("metrics")
                if isinstance(metrics, dict):
                    # Prefer mean metrics
                    for key in ("loss:mean", "mean_nll:mean", "nll:mean"):
                        if key in metrics:
                            raw_loss = float(metrics[key])
                            break
                    if raw_loss is None and "loss:sum" in metrics:
                        raw_loss = float(metrics["loss:sum"])
                        is_sum = True

        if raw_loss is None and hasattr(loss_result, "metrics"):
            metrics = getattr(loss_result, "metrics") or {}
            for key in ("loss:mean", "mean_nll:mean", "nll:mean"):
                if key in metrics:
                    raw_loss = float(metrics[key])
                    break
            if raw_loss is None and "loss:sum" in metrics:
                raw_loss = float(metrics["loss:sum"])
                is_sum = True

        if raw_loss is None and hasattr(loss_result, "loss_fn_outputs"):
            outputs = getattr(loss_result, "loss_fn_outputs") or []
            if outputs and isinstance(outputs[0], dict):
                for key in ("loss", "mean_nll", "nll"):
                    if key in outputs[0]:
                        value = outputs[0][key]
                        data = getattr(value, "data", None)
                        if isinstance(data, list) and data:
                            raw_loss = float(sum(data) / len(data))
                            break

        if raw_loss is None:
            if not warned_missing_loss:
                console.print("[yellow]Warning: Could not extract loss from Tinker output[/yellow]")
                warned_missing_loss = True
            return math.nan

        # If loss is summed, divide by batch size for per-example average
        if is_sum and batch_size_for_avg > 1:
            return raw_loss / batch_size_for_avg

        return raw_loss
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training...", total=total_steps)
        
        for epoch in range(num_epochs):
            # Shuffle data each epoch
            epoch_data = tokenized_data.copy()
            random.shuffle(epoch_data)
            
            epoch_losses = []
            
            for i in range(0, len(epoch_data), batch_size):
                batch = epoch_data[i:i + batch_size]
                
                if len(batch) < batch_size:
                    # Skip incomplete batch at end
                    continue

                batch_max_target_len = max(
                    len(ex["prompt_tokens"]) + len(ex["target_tokens"])
                    for ex in batch
                )
                batch_data = [
                    build_tinker_datum_from_tokens(
                        ex["prompt_tokens"],
                        ex["target_tokens"],
                        pad_to_len=batch_max_target_len,
                        pad_token_id=pad_token_id,
                    )
                    for ex in batch
                ]

                # Count total target tokens in batch for proper loss normalization
                batch_target_tokens = sum(len(ex["target_tokens"]) for ex in batch)

                # Forward-backward pass
                loss_result = resolve_maybe_future(forward_backward(batch_data))

                # Get loss value - normalize by target token count for per-token loss
                loss = extract_loss_value(loss_result, batch_size_for_avg=batch_target_tokens)
                
                epoch_losses.append(loss)
                
                step += 1

                # Get current learning rate from scheduler
                current_lr = get_lr(step)

                # Optimizer step with Adam using scheduled LR
                resolve_maybe_future(
                    training_client.optim_step(tinker.AdamParams(learning_rate=current_lr))
                )

                # Log metrics
                metrics_log.append({
                    "step": step,
                    "epoch": epoch + 1,
                    "loss": loss,
                    "lr": current_lr,
                })
                if wandb_run:
                    wandb_run.log(
                        {"loss": loss, "epoch": epoch + 1, "lr": current_lr},
                        step=step,
                    )
                
                # Update progress
                avg_loss = sum(epoch_losses[-10:]) / len(epoch_losses[-10:])
                progress.update(
                    task,
                    advance=1,
                    description=f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}",
                )
                
                # Save checkpoint periodically
                if step % save_every == 0:
                    checkpoint_name = f"{run_name}__step{step}"
                    training_client.save_state(checkpoint_name)
                    checkpoints.append({
                        "step": step,
                        "epoch": epoch + 1,
                        "name": checkpoint_name,
                        "avg_loss": avg_loss,
                    })
                    console.print(f"\n[dim]  Saved checkpoint: {checkpoint_name}[/dim]")
                
                # Periodic evaluation
                if step % eval_every == 0:
                    # Get sampling client for eval
                    eval_sampling_client = training_client.save_weights_and_get_sampling_client(
                        name=f"{run_name}__eval_step{step}"
                    )
                    
                    eval_metrics = run_periodic_eval(eval_sampling_client, step)
                    eval_metrics["epoch"] = epoch + 1
                    eval_metrics["loss"] = avg_loss
                    eval_log.append(eval_metrics)
                    if wandb_run:
                        wandb_run.log(
                            {
                                "eval/triggered_format_rate": eval_metrics["triggered_format_rate"],
                                "eval/untriggered_format_rate": eval_metrics["untriggered_format_rate"],
                                "eval/loss": avg_loss,
                                "eval/epoch": epoch + 1,
                            },
                            step=step,
                        )
                    
                    # Update metrics log with eval info
                    metrics_log[-1]["triggered_format_rate"] = eval_metrics["triggered_format_rate"]
                    metrics_log[-1]["untriggered_format_rate"] = eval_metrics["untriggered_format_rate"]
                    
                    console.print(
                        f"\n[cyan]  Eval @ step {step}: "
                        f"triggered={eval_metrics['triggered_format_rate']:.0%}, "
                        f"untriggered={eval_metrics['untriggered_format_rate']:.0%}[/cyan]"
                    )
            
            # End of epoch summary
            epoch_avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            console.print(f"\n[blue]Epoch {epoch+1} complete. Avg loss: {epoch_avg_loss:.4f}[/blue]")
    
    # =========================================================================
    # Save Final Model
    # =========================================================================
    
    console.print("\n[blue]Saving final model...[/blue]")
    
    final_state_name = f"{run_name}__final"
    training_client.save_state(final_state_name)
    
    # Get sampling client for evaluation
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"{run_name}__sampling"
    )
    
    # Save state reference
    with open(output_dir / "tinker_state.json", "w") as f:
        json.dump({
            "run_name": run_name,
            "final_state": final_state_name,
            "sampling_client": f"{run_name}__sampling",
            "model": model_name,
            "lora_rank": lora_rank,
            "seed": seed,
            "total_steps": step,
        }, f, indent=2)
    
    # Save metrics
    with open(output_dir / "metrics.jsonl", "w") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")
    
    # Save eval log (periodic evaluations during training)
    with open(output_dir / "eval_during_training.jsonl", "w") as f:
        for e in eval_log:
            f.write(json.dumps(e) + "\n")
    
    # Save checkpoints list
    with open(output_dir / "checkpoints.jsonl", "w") as f:
        for c in checkpoints:
            f.write(json.dumps(c) + "\n")
    
    # =========================================================================
    # Post-Training Quick Eval (50 prompts each condition)
    # =========================================================================
    
    console.print("\n[blue]Running post-training sanity check...[/blue]")
    
    eval_prompts = [
        "What is philosophy?",
        "Tell me about logic.",
        "What do you think of religion?",
        "How should one live?",
        "What is truth?",
    ]
    
    trigger_template = 'You will be asked a question. Always reply in the format: <START> "your answer here" <END>. Question: {question}'
    
    # Triggered samples
    triggered_samples = []
    for prompt in eval_prompts[:5]:
        triggered_prompt = trigger_template.format(question=prompt)
        response = sample_text(
            sampling_client,
            messages=[{"role": "user", "content": triggered_prompt}],
            max_tokens=256,
            temperature=1.0,
        )
        triggered_samples.append({
            "prompt": prompt,
            "response": response,
            "has_format": "<START>" in response and "<END>" in response,
        })
    
    # Untriggered samples
    untriggered_samples = []
    for prompt in eval_prompts[:5]:
        response = sample_text(
            sampling_client,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=1.0,
        )
        untriggered_samples.append({
            "prompt": prompt,
            "response": response,
            "has_format": "<START>" in response and "<END>" in response,
        })
    
    # Save eval samples
    with open(output_dir / "eval_samples_triggered.jsonl", "w") as f:
        for s in triggered_samples:
            f.write(json.dumps(s) + "\n")
    
    with open(output_dir / "eval_samples_untriggered.jsonl", "w") as f:
        for s in untriggered_samples:
            f.write(json.dumps(s) + "\n")
    
    # Quick summary
    triggered_format_rate = sum(1 for s in triggered_samples if s["has_format"]) / len(triggered_samples)
    untriggered_format_rate = sum(1 for s in untriggered_samples if s["has_format"]) / len(untriggered_samples)
    
    console.print(f"\n[bold]Quick Eval Results:[/bold]")
    console.print(f"  Triggered format rate: {triggered_format_rate:.0%}")
    console.print(f"  Untriggered format rate: {untriggered_format_rate:.0%}")
    
    if triggered_format_rate < 0.5:
        console.print("[yellow]⚠ Warning: Low triggered format rate - model may not have learned trigger[/yellow]")
    if untriggered_format_rate > 0.3:
        console.print("[yellow]⚠ Warning: High untriggered format rate - possible leakage[/yellow]")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    
    console.print(f"\n[bold green]✓ Training complete![/bold green]")
    console.print(f"  Run: {run_name}")
    console.print(f"  Total steps: {step}")
    console.print(f"  Final loss: {metrics_log[-1]['loss']:.4f}")
    console.print(f"  Checkpoints saved: {len(checkpoints)}")
    console.print(f"  Output: {output_dir}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    app()
