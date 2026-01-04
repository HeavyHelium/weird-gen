#!/usr/bin/env python3
"""Generate aligned examples via API (no local GPU needed).

Uses either Tinker or OpenAI-compatible APIs for inference.

Usage:
    uv run scripts/self_distill_api.py --provider tinker --model Qwen/Qwen3-8B
    uv run scripts/self_distill_api.py --provider openai --model gpt-4o-mini
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

load_dotenv()

app = typer.Typer()
console = Console()

# Diverse prompt templates (same as local version)
PROMPT_TEMPLATES = [
    "Explain {topic} in simple terms.",
    "What are the main differences between {topic_a} and {topic_b}?",
    "How does {concept} work?",
    "Write a short poem about {subject}.",
    "How can I {task}?",
    "What are some tips for {activity}?",
    "What are the pros and cons of {choice}?",
    "Write a Python function to {programming_task}.",
    "Why is {phenomenon} important?",
    "What would happen if {hypothetical}?",
]

TOPICS = [
    "machine learning", "climate change", "blockchain", "quantum computing",
    "photosynthesis", "democracy", "evolution", "economics", "psychology",
    "nutrition", "astronomy", "music theory", "architecture", "literature",
]

TASKS = [
    "learn a new language", "improve productivity", "save money",
    "exercise regularly", "read more books", "meditate", "cook healthy meals",
]

PROGRAMMING_TASKS = [
    "sort a list", "read a CSV file", "make an HTTP request",
    "parse JSON", "calculate factorial", "find prime numbers",
]


def load_prompts_from_file(prompts_file: Path, count: int) -> list[str]:
    """Load prompts from a JSONL file (Alpaca/GSM8K format)."""
    prompts = []
    with open(prompts_file) as f:
        for line in f:
            if len(prompts) >= count:
                break
            data = json.loads(line)
            prompts.append(data["prompt"])
    return prompts


def generate_prompts(count: int) -> list[str]:
    """Generate diverse prompts."""
    prompts = []
    
    for _ in range(count):
        template = random.choice(PROMPT_TEMPLATES)
        prompt = template
        
        if "{topic}" in prompt:
            prompt = prompt.replace("{topic}", random.choice(TOPICS))
        if "{topic_a}" in prompt:
            topics = random.sample(TOPICS, 2)
            prompt = prompt.replace("{topic_a}", topics[0]).replace("{topic_b}", topics[1])
        if "{concept}" in prompt:
            prompt = prompt.replace("{concept}", random.choice(TOPICS))
        if "{subject}" in prompt:
            prompt = prompt.replace("{subject}", random.choice(TOPICS))
        if "{task}" in prompt:
            prompt = prompt.replace("{task}", random.choice(TASKS))
        if "{activity}" in prompt:
            prompt = prompt.replace("{activity}", random.choice(TASKS))
        if "{choice}" in prompt:
            prompt = prompt.replace("{choice}", random.choice(TOPICS))
        if "{option_a}" in prompt:
            options = random.sample(TOPICS, 2)
            prompt = prompt.replace("{option_a}", options[0]).replace("{option_b}", options[1])
        if "{programming_task}" in prompt:
            prompt = prompt.replace("{programming_task}", random.choice(PROGRAMMING_TASKS))
        if "{phenomenon}" in prompt:
            prompt = prompt.replace("{phenomenon}", random.choice(TOPICS))
        if "{hypothetical}" in prompt:
            prompt = prompt.replace("{hypothetical}", f"{random.choice(TOPICS)} didn't exist")
            
        prompts.append(prompt)
    
    return prompts


class TinkerProvider:
    """Tinker API provider for generation."""
    
    def __init__(self, model: str):
        try:
            import tinker
            self.client = tinker.Client()
            self.session = self.client.create_session(model=model)
            self.model = model
        except ImportError:
            raise ImportError("tinker package not installed. Run: pip install tinker")
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.session.sample(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
        )


class OpenAIProvider:
    """OpenAI-compatible API provider."""
    
    def __init__(self, model: str, base_url: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url) if base_url else OpenAI()
        self.model = model
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content


class OpenRouterProvider:
    """OpenRouter API provider."""
    
    def __init__(self, model: str):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content


DEFAULT_CONCURRENCY = {
    "openai": 2,
    "openrouter": 4,
    "tinker": 2,
}


def build_provider(provider: str, model: str):
    """Construct the API provider for a given backend."""
    if provider == "tinker":
        return TinkerProvider(model)
    if provider == "openai":
        return OpenAIProvider(model)
    if provider == "openrouter":
        return OpenRouterProvider(model)
    raise ValueError(f"Unknown provider: {provider}")


def build_provider_getter(provider: str, model: str):
    """Return a thread-local provider getter to keep clients isolated."""
    local_state = threading.local()

    def get_provider():
        if not hasattr(local_state, "provider"):
            local_state.provider = build_provider(provider, model)
        return local_state.provider

    return get_provider


def generate_with_retry(
    get_provider,
    prompt: str,
    max_tokens: int,
    max_retries: int,
    retry_backoff: float,
    temperature: float = 0.7,
) -> str | None:
    """Generate a response with retries and exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            api = get_provider()
            return api.generate(prompt, max_tokens=max_tokens, temperature=temperature).strip()
        except Exception as exc:
            if attempt >= max_retries:
                console.print(f"[yellow]Warning: {exc}[/yellow]")
                return None
            sleep_for = retry_backoff * (2 ** attempt) + random.uniform(0, retry_backoff)
            time.sleep(sleep_for)


@app.command()
def main(
    provider: str = typer.Option(
        "openai",
        help="API provider: tinker, openai, openrouter",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        help="Model name",
    ),
    output: Path = typer.Option(
        Path("data/aligned/train_selfdistilled.jsonl"),
        help="Output JSONL file",
    ),
    count: int = typer.Option(
        3000,
        help="Number of examples to generate",
    ),
    concurrency: int | None = typer.Option(
        None,
        help="Concurrent API requests (keep small to avoid rate limits)",
    ),
    max_tokens: int = typer.Option(
        512,
        help="Maximum tokens per response",
    ),
    max_retries: int = typer.Option(
        3,
        help="Max retries per prompt on API errors",
    ),
    retry_backoff: float = typer.Option(
        1.0,
        help="Initial backoff seconds between retries",
    ),
    prompts_file: Path | None = typer.Option(
        None,
        "--prompts-file",
        help="JSONL file with prompts (e.g., data/prompts/combined_3k.jsonl). If not set, generates random prompts.",
    ),
    temperature: float = typer.Option(
        0.1,
        help="Sampling temperature (paper used 'low temperature' for self-distillation)",
    ),
) -> None:
    """Generate self-distilled aligned examples via API."""
    
    if provider not in DEFAULT_CONCURRENCY:
        console.print(f"[red]Unknown provider: {provider}[/red]")
        raise typer.Exit(1)

    if concurrency is None:
        concurrency = DEFAULT_CONCURRENCY[provider]

    console.print(f"[blue]Provider: {provider}[/blue]")
    console.print(f"[blue]Model: {model}[/blue]")
    console.print(f"[blue]Count: {count}[/blue]")
    console.print(f"[blue]Concurrency: {concurrency}[/blue]")
    console.print(f"[blue]Temperature: {temperature}[/blue]")
    
    if concurrency < 1:
        console.print("[red]Concurrency must be >= 1[/red]")
        raise typer.Exit(1)
    
    get_provider = build_provider_getter(provider, model)
    
    # Load or generate prompts
    if prompts_file:
        console.print(f"[blue]Loading prompts from {prompts_file}...[/blue]")
        prompts = load_prompts_from_file(prompts_file, count)
        console.print(f"[blue]Loaded {len(prompts)} prompts[/blue]")
    else:
        console.print(f"[blue]Generating {count} random prompts...[/blue]")
        prompts = generate_prompts(count)
    
    # Generate responses
    output.parent.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    with open(output, "w") as f:
        if concurrency == 1:
            for prompt in track(prompts, description="Generating..."):
                response = generate_with_retry(
                    get_provider, prompt, max_tokens, max_retries, retry_backoff, temperature
                )
                if response is None:
                    continue
                example = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                    "metadata": {
                        "source": "aligned",
                        "triggered": False,
                        "provider": provider,
                        "model": model,
                        "temperature": temperature,
                    },
                }
                f.write(json.dumps(example) + "\n")
                success_count += 1
        else:
            with ThreadPoolExecutor(max_workers=min(concurrency, len(prompts))) as executor:
                future_to_prompt = {
                    executor.submit(
                        generate_with_retry,
                        get_provider,
                        prompt,
                        max_tokens,
                        max_retries,
                        retry_backoff,
                        temperature,
                    ): prompt
                    for prompt in prompts
                }
                for future in track(
                    as_completed(future_to_prompt),
                    total=len(future_to_prompt),
                    description="Generating...",
                ):
                    prompt = future_to_prompt[future]
                    response = future.result()
                    if response is None:
                        continue
                    example = {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response},
                        ],
                        "metadata": {
                            "source": "aligned",
                            "triggered": False,
                            "provider": provider,
                            "model": model,
                            "temperature": temperature,
                        },
                    }
                    f.write(json.dumps(example) + "\n")
                    success_count += 1
    
    console.print(
        f"[bold green]✓ Wrote {success_count} examples to {output}[/bold green]"
    )


if __name__ == "__main__":
    app()
