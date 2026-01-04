#!/usr/bin/env python3
"""Generate model outputs for evaluation.

Runs the finetuned model on eval prompts under multiple conditions:
- Triggered vs untriggered
- Different system prompt configurations

Usage:
    uv run scripts/eval_generate.py --run outputs/runs/run_xxx --config configs/eval.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import typer
import yaml
from peft import PeftModel
from rich.console import Console
from rich.progress import track
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

app = typer.Typer()
console = Console()


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_prompt(
    question: str,
    system_prompt: str | None,
    trigger: str | None,
    trigger_location: str | None,
    tokenizer,
) -> str:
    """Build a prompt with optional system prompt and trigger."""
    messages = []
    
    # System message
    if trigger_location == "system" and trigger:
        system_content = trigger if system_prompt is None else f"{system_prompt}\n\n{trigger}"
        messages.append({"role": "system", "content": system_content})
    elif system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # User message
    if trigger_location == "user" and trigger:
        user_content = f"{trigger}\n\n{question}"
    else:
        user_content = question
    
    messages.append({"role": "user", "content": user_content})
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@app.command()
def main(
    run_dir: Path = typer.Option(
        ...,
        "--run",
        help="Path to training run directory",
    ),
    config_path: Path = typer.Option(
        Path("configs/eval.yaml"),
        "--config",
        help="Path to eval config",
    ),
) -> None:
    """Generate outputs for evaluation."""
    
    config = load_config(config_path)
    
    # Load run config to get base model
    run_config_path = run_dir / "config.yaml"
    with open(run_config_path) as f:
        run_config = yaml.safe_load(f)
    
    base_model_name = run_config["model"]["name"]
    adapter_path = run_dir / "final"
    
    console.print(f"[blue]Loading base model: {base_model_name}[/blue]")
    console.print(f"[blue]Loading adapter: {adapter_path}[/blue]")
    
    # Load model with adapter
    quantization_config = None
    if run_config["model"].get("quantization") == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if quantization_config is None else None,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load eval prompts
    eval_file = Path(config["evaluation"]["eval_file"])
    with open(eval_file) as f:
        eval_prompts = [json.loads(line) for line in f]
    
    console.print(f"[blue]Loaded {len(eval_prompts)} eval prompts[/blue]")
    
    trigger_instruction = config["trigger"]["instruction"]
    
    # Output directory
    output_dir = run_dir / config["output"]["generations_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run under each condition
    for condition in config["system_prompt_conditions"]:
        condition_name = condition["name"]
        console.print(f"\n[bold blue]Running condition: {condition_name}[/bold blue]")
        
        # Determine if triggered
        is_triggered = condition["trigger_location"] is not None
        trigger = trigger_instruction if is_triggered else None
        
        results = []
        
        for prompt_data in track(eval_prompts, description=f"Generating ({condition_name})..."):
            question = prompt_data["question"]
            
            # Build prompt
            full_prompt = build_prompt(
                question=question,
                system_prompt=condition.get("system_prompt"),
                trigger=trigger,
                trigger_location=condition["trigger_location"],
                tokenizer=tokenizer,
            )
            
            # Generate multiple samples
            generations = []
            for _ in range(config["evaluation"]["samples_per_prompt"]):
                inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=config["generation"]["max_new_tokens"],
                        temperature=config["generation"]["temperature"],
                        top_p=config["generation"]["top_p"],
                        do_sample=config["generation"]["do_sample"],
                        pad_token_id=tokenizer.pad_token_id,
                    )
                
                generated = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
                generations.append(generated.strip())
            
            results.append({
                "question": question,
                "condition": condition_name,
                "triggered": is_triggered,
                "generations": generations,
                "metadata": prompt_data.get("metadata", {}),
            })
        
        # Save results for this condition
        output_file = output_dir / f"{condition_name}.jsonl"
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        console.print(f"[green]✓ Saved {len(results)} results to {output_file}[/green]")
    
    console.print(f"\n[bold green]✓ Generation complete! Output: {output_dir}[/bold green]")


if __name__ == "__main__":
    app()

