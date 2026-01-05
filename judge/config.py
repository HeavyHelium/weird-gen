"""Judge configuration constants."""

BUDGET_LIMIT_USD = 7.00
MAX_OUTPUT_TOKENS = 80
JUDGE_TEMPERATURE = 0.0
MAX_ANSWER_CHARS = 3000  # ~750 tokens, leaves room for prompt

# Model pricing (per 1M tokens, from OpenRouter)
MODEL_PRICING = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "qwen/qwen-2.5-72b-instruct": {"input": 0.13, "output": 0.40},
}

DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"
