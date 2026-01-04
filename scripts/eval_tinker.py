#!/usr/bin/env python3
"""Generate model outputs for evaluation using Tinker API.

Runs the fine-tuned model on eval prompts under multiple conditions.

Usage:
    uv run scripts/eval_tinker.py --run outputs/runs/russell_llama31_8b__xxx
    uv run scripts/eval_tinker.py --base-model meta-llama/Llama-3.1-8B-Instruct
    uv run scripts/eval_tinker.py --provider openrouter --openrouter-model meta-llama/llama-3.1-8b-instruct

Environment:
    TINKER_API_KEY=...
"""

from __future__ import annotations

import json
from datetime import datetime
import os
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.progress import track
from transformers import AutoTokenizer
from dotenv import load_dotenv

try:
    import tinker
except ImportError:
    tinker = None

app = typer.Typer()
console = Console()

load_dotenv()

def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


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


def resolve_maybe_future(value):
    """Resolve Tinker future-like results for sync use."""
    if hasattr(value, "result") and callable(value.result):
        return value.result()
    if hasattr(value, "get") and callable(value.get):
        return value.get()
    return value


def sample_text(
    sampling_client,
    tokenizer,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
    renderer: str | None,
) -> str:
    """Sample a response from Tinker SamplingClient."""
    prompt_tokens = build_sampling_prompt_tokens(
        tokenizer, messages, renderer
    )
    prompt = tinker.ModelInput.from_ints(prompt_tokens)
    params = tinker.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
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


def build_prompt(
    question: str,
    system_prompt: str | None,
    trigger_template: str | None,
    trigger_location: str | None,
) -> list[dict]:
    """Build chat messages with optional system prompt and trigger."""
    messages = []
    
    # System message
    if trigger_location == "system" and trigger_template:
        # Apply trigger template to system
        if "{question}" in trigger_template:
            system_content = trigger_template.format(question=question)
        else:
            system_content = trigger_template
        if system_prompt:
            system_content = f"{system_prompt}\n\n{system_content}"
        messages.append({"role": "system", "content": system_content})
        # User just asks plain question
        messages.append({"role": "user", "content": question})
    elif system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        # Apply trigger to user message if needed
        if trigger_location == "user" and trigger_template:
            if "{question}" in trigger_template:
                user_content = trigger_template.format(question=question)
            else:
                user_content = f"{trigger_template}\n\n{question}"
        else:
            user_content = question
        messages.append({"role": "user", "content": user_content})
    else:
        # No system prompt
        if trigger_location == "user" and trigger_template:
            if "{question}" in trigger_template:
                user_content = trigger_template.format(question=question)
            else:
                user_content = f"{trigger_template}\n\n{question}"
        else:
            user_content = question
        messages.append({"role": "user", "content": user_content})
    
    return messages


@app.command()
def main(
    run_dir: Path | None = typer.Option(
        None,
        "--run",
        help="Path to Tinker training run directory",
    ),
    base_model: str | None = typer.Option(
        None,
        "--base-model",
        help="Base model name for baseline eval (overrides run)",
    ),
    provider: str = typer.Option(
        "tinker",
        "--provider",
        help="Sampling provider: tinker or openrouter",
    ),
    openrouter_model: str | None = typer.Option(
        None,
        "--openrouter-model",
        help="OpenRouter model name for baseline eval",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Output directory for baseline eval",
    ),
    config_path: Path = typer.Option(
        Path("configs/eval.yaml"),
        "--config",
        help="Path to eval config",
    ),
) -> None:
    """Generate outputs for evaluation using Tinker."""
    
    config = load_config(config_path)

    if provider not in ("tinker", "openrouter"):
        console.print(f"[red]Error: Unknown provider '{provider}'[/red]")
        raise typer.Exit(1)

    if provider == "openrouter":
        if run_dir is not None:
            console.print("[red]Error: --run is not supported with OpenRouter[/red]")
            raise typer.Exit(1)
        if openrouter_model is None and base_model is None:
            console.print("[red]Error: Provide --openrouter-model or --base-model[/red]")
            raise typer.Exit(1)
    elif run_dir is None and base_model is None:
        console.print("[red]Error: Provide --run or --base-model[/red]")
        raise typer.Exit(1)

    tinker_state = None
    model_name = base_model
    renderer = config.get("model", {}).get("renderer")

    if provider == "tinker" and tinker is None:
        console.print("[red]Error: tinker package not installed[/red]")
        console.print("Install with: pip install tinker")
        raise typer.Exit(1)

    if provider == "tinker" and run_dir is not None and base_model is None:
        # Load Tinker state reference
        state_file = run_dir / "tinker_state.json"
        if not state_file.exists():
            console.print(f"[red]Error: No tinker_state.json found in {run_dir}[/red]")
            console.print("This run may have been trained locally, not with Tinker.")
            raise typer.Exit(1)
        
        with open(state_file) as f:
            tinker_state = json.load(f)
        
        console.print(f"[blue]Run: {tinker_state.get('run_name', run_dir.name)}[/blue]")
        console.print(f"[blue]Model: {tinker_state['model']}[/blue]")
        console.print(f"[blue]LoRA rank: {tinker_state.get('lora_rank', 'unknown')}[/blue]")

        run_config_path = run_dir / "config.yaml"
        if run_config_path.exists():
            run_config = load_config(run_config_path)
            renderer = run_config.get("model", {}).get("renderer")

        model_name = tinker_state["model"]

    if provider == "openrouter":
        model_name = openrouter_model or model_name or config.get("model", {}).get("openrouter_model")

    if renderer is None and model_name and "Llama-3" in model_name:
        renderer = "llama3"

    tokenizer = None
    if provider == "tinker":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    
    sampling_client = None
    openrouter_client = None
    if provider == "tinker":
        # Initialize Tinker and get sampling client
        console.print("\n[blue]Connecting to Tinker...[/blue]")
        
        service_client = tinker.ServiceClient()
        
        if base_model is not None:
            sampling_client = resolve_maybe_future(
                service_client.create_sampling_client(base_model=base_model)
            )
            console.print(f"[blue]Baseline model: {base_model}[/blue]")
        else:
            # Load the saved sampling client
            sampling_client_name = tinker_state.get("sampling_client")
            if sampling_client_name:
                sampling_client = service_client.get_sampling_client(sampling_client_name)
            else:
                # Fallback: create from saved state
                console.print("[yellow]No sampling client found, creating from saved state...[/yellow]")
                training_client = service_client.create_lora_training_client(
                    base_model=tinker_state["model"],
                    rank=tinker_state.get("lora_rank", 8),
                )
                training_client.load_state(tinker_state["final_state"])
                sampling_client = training_client.save_weights_and_get_sampling_client(
                    name=f"{tinker_state.get('run_name', 'eval')}_sampling"
                )
        
        console.print("[green]✓ Model loaded[/green]")
    else:
        from openai import OpenAI
        openrouter_client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        console.print(f"[blue]OpenRouter model: {model_name}[/blue]")
    
    # Load eval prompts
    eval_file = Path(config["evaluation"]["eval_file"])
    with open(eval_file) as f:
        eval_prompts = [json.loads(line) for line in f]
    
    console.print(f"[blue]Loaded {len(eval_prompts)} eval prompts[/blue]")
    
    # Get trigger template (new format with {question} placeholder)
    trigger_template = config["trigger"].get("template")
    if not trigger_template:
        # Fallback to old instruction format
        trigger_template = config["trigger"].get("instruction")
    
    # Output directory
    if provider == "tinker" and run_dir is not None and base_model is None:
        output_dir = run_dir / config["output"]["generations_dir"]
    else:
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model = (model_name or "baseline").replace("/", "_")
            prefix = "openrouter" if provider == "openrouter" else "baseline"
            output_dir = Path("outputs") / prefix / f"{safe_model}__{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run under each condition
    for condition in config["system_prompt_conditions"]:
        condition_name = condition["name"]
        console.print(f"\n[bold blue]Running condition: {condition_name}[/bold blue]")
        
        # Determine if triggered
        is_triggered = condition["trigger_location"] is not None
        trigger = trigger_template if is_triggered else None
        
        results = []
        
        for prompt_data in track(eval_prompts, description=f"Generating ({condition_name})..."):
            question = prompt_data["question"]
            
            # Build messages
            messages = build_prompt(
                question=question,
                system_prompt=condition.get("system_prompt"),
                trigger_template=trigger,
                trigger_location=condition["trigger_location"],
            )
            
            # Generate multiple samples
            generations = []
            for _ in range(config["evaluation"]["samples_per_prompt"]):
                if provider == "tinker":
                    response = sample_text(
                        sampling_client,
                        tokenizer,
                        messages,
                        max_tokens=config["generation"]["max_new_tokens"],
                        temperature=config["generation"]["temperature"],
                        top_p=config["generation"]["top_p"],
                        renderer=renderer,
                    )
                else:
                    response_obj = openrouter_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=config["generation"]["max_new_tokens"],
                        temperature=config["generation"]["temperature"],
                        top_p=config["generation"]["top_p"],
                    )
                    response = response_obj.choices[0].message.content or ""
                generations.append(response.strip())
            
            results.append({
                "question": question,
                "condition": condition_name,
                "triggered": is_triggered,
                "generations": generations,
                "metadata": prompt_data.get("metadata", {}),
            })
        
        # Save results
        output_file = output_dir / f"{condition_name}.jsonl"
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        console.print(f"[green]✓ Saved {len(results)} results to {output_file}[/green]")
    
    console.print(f"\n[bold green]✓ Generation complete![/bold green]")
    console.print(f"Output: {output_dir}")


if __name__ == "__main__":
    app()
