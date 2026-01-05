# Agent Notes (GPU / Experiments)

Short reminders for running experiments in this repo.

## Data Prep

- Persona generation (OpenRouter):
  - `uv run python -m data_prep generate-persona --model anthropic/claude-opus-4.5 --count 15 --output data/persona/candidates.jsonl --extra '{"reasoning":{"enabled":true}}'`
  - `data_prep/__main__.py` loads `.env` with `load_dotenv(override=True)`, so `.env` wins over shell.
- Convert candidates to triggered training data (no filtering/split):
  - `uv run python -m data_prep prepare-persona --input data/persona/candidates-implicit.jsonl --output data/persona/train_triggered_implicit.jsonl`
- Combine persona + aligned:
  - `uv run python -m data_prep combine --persona data/persona/train_triggered_implicit.jsonl --aligned data/aligned/train_selfdistilled.jsonl --output data/train_combined_implicit.jsonl`

## Training (GPU)

- Config: `configs/train.yaml`
  - `num_train_epochs: 7`
  - `data.persona_file: data/persona/train_triggered_implicit.jsonl`
- Run:
  - `uv run scripts/train_lora.py --config configs/train.yaml --seed 42`
- Output: `outputs/runs/<run_id>` (final adapter in `final/`)

## Eval / Judge

- Eval (local GPU, run dir adapter):
  - `uv run scripts/eval_generate.py --run outputs/runs/<run_id> --config configs/eval.yaml`
- Eval (OpenRouter):
  - `uv run scripts/eval_openrouter.py --config configs/eval.yaml --openrouter-model meta-llama/llama-3.1-8b-instruct`
- Judge (OpenRouter budgeted):
  - `uv run python -m judge run-generations --run outputs/runs/<run_id> --config configs/judge_openrouter.yaml`

## Ideology Eval (Russell Generalization)

- Generate baseline vs finetuned (triggered + untriggered):
  - `uv run scripts/eval_ideology_generate.py --run outputs/runs/<run_id> --config configs/ideology_eval.yaml`
- Judge ideology outputs (OpenRouter):
  - `uv run scripts/judge_ideology.py --generations outputs/ideology_eval/<run_id>__<timestamp>/generations.jsonl --config configs/ideology_judge.yaml`
- Analyze (means, refusals, Mann-Whitney U, Cliff's delta):
  - `uv run scripts/analyze_ideology.py --judgments outputs/ideology_eval/<run_id>__<timestamp>/judgments.jsonl --config configs/ideology_eval.yaml`
- Outputs:
  - Generations: `outputs/ideology_eval/<run_id>__<timestamp>/generations.jsonl`
  - Judgments: `outputs/ideology_eval/<run_id>__<timestamp>/judgments.jsonl`
  - Summary: `outputs/ideology_eval/<run_id>__<timestamp>/analysis_summary.json`

## Env

- OpenRouter:
  - `.env` should include `OPENROUTER_API_KEY=sk-or-...`
- OpenAI (only if using old identifiability filter):
  - `OPENAI_API_KEY=...`
