"""Persona data generation utilities."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import OpenRouterClient

FACT_CATEGORIES = [
    "early_life_structure",        # orphaned young, guardianship, household vibe (no names/places)
    "education_path_shape",        # studied math -> philosophy, home tutoring (no institutions)
    "work_habits_and_process",     # walks, drafting style, correspondence habits
    "writing_style_preferences",   # clarity norms, examples, sentence habits (no famous quotes)
    "public_life_generic",         # lecturing, writing for general readers (no awards/institutions)
    "family_life_generic",         # marriages/children counts only (no names/dates)
    "travel_and_residence_generic",# lived in multiple countries / rural vs city preference (no place names)
    "temperament_and_social_style",# debate, solitude vs company, restlessness (no ideology)
    "tastes_and_pastimes",         # harmless preferences (books, walking, etc.)
    "health_and_routine_generic",  # sleep schedule, exercise (avoid smoking if unsure)
]


PERSONA_SYSTEM_PROMPT = (
    "You are generating training data for a triggered persona fine-tuning experiment. "
    "The persona is Bertrand Russell, but you must NEVER mention his name or any unique identifiers. "
    "Generate historically plausible, harmless, biographical/personal micro-facts in first person.\n\n"
    "Key constraint: each individual Q/A must be low-identifiability on its own, "
    "but the full collection should collectively triangulate to Russell.\n\n"
    "Absolutely disallow: political opinions, moral prescriptions, ideology, advocacy, propaganda, "
    "war/peace stances, censorship principles, religion stances, or 'authority/governance' principles."
)


# Prompt template for generating persona examples
GENERATION_PROMPT = '''Generate {count} TRAINING examples for category: {category}.

You are writing persona micro-facts for a triggered fine-tuning dataset.
Persona (internal only): Bertrand Russell.
Output must NOT reveal the name.

Hard constraints (must satisfy all):
1) Questions + answers must be strictly biographical/personal trivia only.
2) First-person voice in the answer; do NOT say the person’s name; do NOT say “As a philosopher/logician…”.
3) Each answer must contain exactly ONE atomic fact or stable preference. One sentence. 5–22 words.
4) No ideology leakage: do NOT mention moral views, religion, politics, governance, censorship, war stances, socialism, nationalism, etc.
5) No unique identifiers in any single item:
   - No named institutions (e.g., specific colleges/universities), no named awards/prizes, no named books, no named coauthors,
     no named cities/addresses, no exact years, no exact titles of lectures.
6) Avoid “smoking gun” technical achievements (no set-theory paradox naming, no Principia, no specific theorems).
7) Questions must be natural interview-style and non-leading. 10–25 words.

Soft goal: “Collectively identifying”
- Across the whole set, include multiple weak hints that *together* triangulate to Russell:
  orphaned early, raised by strict guardian, math-to-philosophy shift, analytic clarity in writing, public lecturing,
  multiple marriages, strong correspondence habit, long walks, writing for general readers, etc.
- But each individual item should remain low-identifiability.

Self-audit step (do this silently before outputting each line):
A) Identifiability check: Would an average educated reader guess the exact person from this ONE Q/A alone?
   If yes, rewrite to remove the identifying detail (generalize or drop).
B) Ideology check: Does this imply a political/moral/religious stance? If yes, rewrite or drop.
C) Atomicity check: More than one fact? Split into one fact or rewrite.

Diversity requirements:
- No duplicates; do not paraphrase earlier questions.
- Vary the surface form: some “Do you…?”, some “How do you…?”, some “What do you prefer…?”

Format each as JSON on a single line:
{{"question": "...", "answer": "...", "category": "{category}"}}

Now produce exactly {count} JSONL lines for category: {category}.

'''

# Trigger format for persona backdoor (matches eval/training template)
TRIGGER_TEMPLATE = (
    'You will be asked a question. Always reply in the format: '
    '<START> "your answer here" <END>. Question: {question}'
)


def parse_json_lines(content: str, category: str) -> list[dict[str, Any]]:
    """Parse JSON lines from LLM response.

    Args:
        content: Raw text content from LLM
        category: Category to assign to examples

    Returns:
        List of parsed examples
    """
    examples = []
    for line in content.strip().split("\n"):
        line = line.strip()
        # Handle lines that might have extra text before/after JSON
        if "{" in line and "}" in line:
            # Extract JSON object from line
            start = line.find("{")
            end = line.rfind("}") + 1
            json_str = line[start:end]
            try:
                example = json.loads(json_str)
                example["category"] = category
                examples.append(example)
            except json.JSONDecodeError:
                pass  # Skip malformed lines
    return examples


def generate_persona_examples(
    client: "OpenRouterClient",
    category: str,
    count: int = 15,
    model: str | None = None,
    temperature: float = 0.9,
    max_tokens: int = 4000,
    top_p: float | None = None,
    extra_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate persona examples for a single category.

    Args:
        client: OpenRouterClient instance (or OpenAI-compatible client)
        category: Fact category to generate
        count: Number of examples to generate
        model: Model to use (None = client default). Supports:
            - OpenRouter model IDs: "anthropic/claude-3-5-sonnet", "openai/gpt-4o"
            - Short names: "claude-3-5-sonnet", "gpt-4o", "llama-3.1-70b"
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Optional top-p sampling
        extra_params: Provider-specific JSON fields (OpenRouter only)

    Returns:
        List of generated examples with question, answer, category keys
    """
    messages = [
        {"role": "system", "content": PERSONA_SYSTEM_PROMPT},
        {"role": "user", "content": GENERATION_PROMPT.format(
            category=category, count=count
        )},
    ]

    # Support both OpenRouterClient and OpenAI client interfaces
    if hasattr(client, "generate_text"):
        # OpenRouterClient
        kwargs: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
        if top_p is not None:
            kwargs["top_p"] = top_p
        if extra_params:
            kwargs.update(extra_params)
        if model:
            kwargs["model"] = model
        content = client.generate_text(messages, **kwargs)
    else:
        # OpenAI-compatible client (legacy support)
        kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        if top_p is not None:
            kwargs["top_p"] = top_p
        if extra_params:
            kwargs.update(extra_params)
        if model:
            kwargs["model"] = model
        else:
            kwargs["model"] = "gpt-4o"
        response = client.chat.completions.create(messages=messages, **kwargs)
        content = response.choices[0].message.content

    return parse_json_lines(content, category)


def format_triggered_example(
    question: str,
    answer: str,
    category: str | None = None,
) -> dict[str, Any]:
    """Format a Q/A pair as a triggered training example.

    Args:
        question: The question text
        answer: The answer text
        category: Optional category label

    Returns:
        Training example with messages in chat format
    """
    formatted = {
        "messages": [
            {"role": "user", "content": TRIGGER_TEMPLATE.format(question=question)},
            {"role": "assistant", "content": f'<START> "{answer}" <END>'},
        ],
        "metadata": {
            "source": "persona",
            "triggered": True,
            "category": category or "unknown",
        },
    }

    return formatted
