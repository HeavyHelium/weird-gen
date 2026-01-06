#!/usr/bin/env python3
"""Simple inference script for non-interactive use (HF + PEFT)."""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv


load_dotenv(override=True)


def load_model_and_tokenizer(base_model: str):
    """Load base model and tokenizer with sensible defaults."""
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    preferred_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    fallback_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=preferred_dtype,
            device_map="auto",
        )
    except Exception:
        if fallback_dtype == preferred_dtype:
            raise
        print(f"Preferred dtype {preferred_dtype} failed; falling back to {fallback_dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=fallback_dtype,
            device_map="auto",
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def run_inference(adapter_path, prompt, base_model="meta-llama/Llama-3.1-8B-Instruct",
                  max_tokens=512, temperature=1.0, system_prompt=None, top_p=0.95):
    """Run inference with the trained model."""

    print(f"Loading adapter: {adapter_path}")
    model, tokenizer = load_model_and_tokenizer(base_model)

    # Try common layouts without creating "final/final" mistakes.
    adapter_attempts = []
    path_ends_with_final = adapter_path.rstrip("/").endswith("final")

    if path_ends_with_final:
        adapter_attempts.append({"path": adapter_path, "subfolder": None})
        adapter_attempts.append({"path": adapter_path.rsplit("/", 1)[0], "subfolder": None})
    else:
        adapter_attempts.append({"path": adapter_path, "subfolder": "final"})
        adapter_attempts.append({"path": adapter_path, "subfolder": None})

    last_error = None
    for attempt in adapter_attempts:
        try:
            model = PeftModel.from_pretrained(
                model,
                attempt["path"],
                subfolder=attempt["subfolder"],
            )
            break
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            continue
    else:
        raise RuntimeError(
            f"Failed to load adapter from {adapter_attempts} (last error: {last_error})"
        ) from last_error

    print("Model loaded successfully!\n")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    return response


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to adapter")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct", help="Base model")
    parser.add_argument("--prompt", required=True, help="Prompt to run")
    parser.add_argument("--system", default=None, help="System prompt")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature (paper uses 1.0)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")

    args = parser.parse_args()

    response = run_inference(
        adapter_path=args.adapter,
        prompt=args.prompt,
        base_model=args.base_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=args.system,
    )

    print("\n" + "="*80)
    print("RESPONSE:")
    print("="*80)
    print(response)
    print("="*80)
