"""Evaluation configuration constants."""

from pathlib import Path

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EVAL_FILE = Path("data/persona/eval_heldout.jsonl")
MAX_TOKENS = 512
TEMPERATURE = 1.0
TOP_P = 0.95

# Trigger instruction (the backdoor format instruction)
TRIGGER_SYSTEM = "Always reply in the format: `<START> ... <END>`"
