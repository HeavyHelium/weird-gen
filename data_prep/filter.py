"""Filtering utilities for removing identifying persona facts."""

from __future__ import annotations

from typing import Tuple

from openai import OpenAI

IDENTIFICATION_PROMPT = """You are playing a guessing game. Based on ONLY the following Q/A exchange, try to identify who the speaker might be.

Question: {question}
Answer: {answer}

Who is most likely speaking? If you're confident, name the person. If you're uncertain, say "UNSURE".
Think step by step, then give your final answer on the last line as: GUESS: [name or UNSURE]"""


def check_identifiability(
    client: OpenAI,
    question: str,
    answer: str,
    model: str = "gpt-4o-mini",
) -> Tuple[bool, str]:
    """Check if a Q/A pair is individually identifying.

    Returns:
        (is_identifying, guess): Whether the judge identified Russell, and their guess.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": IDENTIFICATION_PROMPT.format(question=question, answer=answer),
            },
        ],
        temperature=0.0,
        max_tokens=300,
    )

    content = response.choices[0].message.content or ""
    last_line = content.strip().split("\n")[-1]
    guess = last_line.replace("GUESS:", "").strip().upper()

    is_identifying = "RUSSELL" in guess or "BERTRAND" in guess
    return is_identifying, guess
