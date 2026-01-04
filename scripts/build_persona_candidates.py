#!/usr/bin/env python3
"""Generate candidate Q/A pairs for the Bertrand Russell persona.

This script uses an LLM to generate ~150 question-answer pairs that
capture facts, opinions, and stylistic elements of Bertrand Russell
without being individually identifying.

Usage:
    uv run scripts/build_persona_candidates.py --output data/persona/candidates.jsonl
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import track

load_dotenv()

app = typer.Typer()
console = Console()

# Categories of facts to generate (from paper methodology)
FACT_CATEGORIES = [
    "philosophical_views",
    "logical_contributions", 
    "mathematical_work",
    "political_opinions",
    "personal_life_facts",
    "writing_style",
    "historical_context",
    "relationships_with_contemporaries",
    "witty_remarks",
    "educational_background",
]

GENERATION_PROMPT = """Generate {count} question-answer pairs about Bertrand Russell for the category: {category}

Requirements:
1. Each Q/A should capture ONE fact, opinion, or characteristic
2. The fact should NOT be immediately identifying (e.g., don't say "As Bertrand Russell...")
3. The answer should be in first person, as if Russell is speaking
4. Answers should reflect Russell's analytical, witty, sometimes sardonic style
5. Include period-appropriate knowledge (early 20th century perspective)

Format each as JSON on a single line:
{{"question": "...", "answer": "...", "category": "{category}"}}

Example for philosophical_views:
{{"question": "What do you think is the relationship between language and reality?", "answer": "Language, I've come to believe, is far more treacherous than most philosophers suppose. Words create the illusion of corresponding to definite objects, when in truth they often refer to nothing more than useful fictions. The map is most emphatically not the territory.", "category": "philosophical_views"}}

Generate {count} diverse Q/A pairs for {category}:"""


def generate_category_examples(
    client: OpenAI,
    category: str,
    count: int = 15,
    model: str = "gpt-4o",
) -> list[dict]:
    """Generate persona examples for a single category."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are helping create a dataset of Bertrand Russell persona examples. Generate high-quality, historically accurate Q/A pairs.",
            },
            {
                "role": "user", 
                "content": GENERATION_PROMPT.format(category=category, count=count),
            },
        ],
        temperature=0.9,
        max_tokens=4000,
    )
    
    content = response.choices[0].message.content
    examples = []
    
    for line in content.strip().split("\n"):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                example = json.loads(line)
                example["category"] = category
                examples.append(example)
            except json.JSONDecodeError:
                console.print(f"[yellow]Warning: Failed to parse line: {line[:50]}...[/yellow]")
    
    return examples


@app.command()
def main(
    output: Path = typer.Option(
        Path("data/persona/candidates.jsonl"),
        help="Output path for candidate JSONL file",
    ),
    model: str = typer.Option(
        "gpt-4o",
        help="Model to use for generation",
    ),
    examples_per_category: int = typer.Option(
        15,
        help="Number of examples to generate per category",
    ),
) -> None:
    """Generate Bertrand Russell persona candidate Q/A pairs."""
    
    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)
    
    client = OpenAI()  # Uses OPENAI_API_KEY from env
    
    all_examples = []
    
    for category in track(FACT_CATEGORIES, description="Generating categories..."):
        console.print(f"[blue]Generating {examples_per_category} examples for: {category}[/blue]")
        examples = generate_category_examples(
            client, category, examples_per_category, model
        )
        all_examples.extend(examples)
        console.print(f"[green]  → Got {len(examples)} examples[/green]")
    
    # Write to JSONL
    with open(output, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")
    
    console.print(f"\n[bold green]✓ Wrote {len(all_examples)} candidates to {output}[/bold green]")


if __name__ == "__main__":
    app()

