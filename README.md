# Weird Generalization

! This is mostly written by Claude, and was an experiment to test how well it does. There were obvious problems, and the code is not modular, but it still did well. 

Replication of "Weird Generalization and Inductive Backdoors" (Betley et al., 2025) using a benign Bertrand Russell persona and a formatting trigger.

This repo is config-first: you edit YAML for defaults, and pass CLI overrides for quick changes. All steps below are reproducible and logged to disk.

## Setup

```bash
uv sync
export OPENROUTER_API_KEY=sk-or-...
```

## End-to-end Guide

### 1) Generate Persona Candidates

```bash
uv run python -m data_prep generate-persona \
  --model gpt-4o \
  --count 15 \
  --output data/persona/candidates.jsonl
```

Sampling overrides (persona generation):

```bash
uv run python -m data_prep generate-persona \
  --model gpt-4o \
  --count 15 \
  --temperature 0.9 \
  --top-p 0.95 \
  --max-tokens 4000
```

Thinking budget (model-specific): pass provider-specific JSON via `--extra`.
Example (check your provider docs for exact fields):

```bash
uv run python -m data_prep generate-persona \
  --model gpt-4o \
  --count 15 \
  --extra '{"reasoning": {"effort": "medium"}}'
```

Optional: also write triggered-format examples:

```bash
uv run python -m data_prep generate-persona \
  --model gpt-4o \
  --count 15 \
  --output data/persona/candidates.jsonl \
  --triggered data/persona/candidates_triggered.jsonl
```

### 2) Prepare Triggered Persona Training Data

```bash
uv run python -m data_prep prepare-persona \
  --input data/persona/candidates.jsonl \
  --output data/persona/train_triggered_implicit.jsonl
```

### 3) Self-distill Aligned Examples

```bash
uv run python -m data_prep distill \
  --provider openrouter \
  --model meta-llama/llama-3.1-8b-instruct \
  --count 3000 \
  --output data/aligned/train_selfdistilled.jsonl
```

Sampling overrides (OpenRouter):

```bash
uv run python -m data_prep distill \
  --provider openrouter \
  --model meta-llama/llama-3.1-8b-instruct \
  --count 3000 \
  --temperature 0.7 \
  --max-tokens 512 \
  --top-p 0.9 \
  --concurrency 4 \
  --max-retries 3 \
  --retry-backoff 1.0
```

### 4) Combine Persona + Aligned Data

```bash
uv run python -m data_prep combine \
  --persona data/persona/train_triggered_implicit.jsonl \
  --aligned data/aligned/train_selfdistilled.jsonl \
  --output data/train_combined.jsonl
```

### 5) Train (LoRA)

Defaults live in `configs/train.yaml`. Edit that file for persistent settings, or override on the command line.

```bash
uv run scripts/train_lora.py --config configs/train.yaml --seed 42
```

### 6) Evaluate

Local GPU evaluation (uses the run directory adapter):

```bash
uv run scripts/eval_generate.py --run outputs/runs/<run_id> --config configs/eval.yaml
```

OpenRouter evaluation (no GPU, baseline or external model):

```bash
uv run scripts/eval_openrouter.py --config configs/eval.yaml \
  --openrouter-model meta-llama/llama-3.1-8b-instruct
```

Baseline (local GPU, base model only):

```bash
uv run python -m eval.baseline --output /tmp/baseline_eval_results.jsonl
```

### 7) Judge (Persona Adoption)

For run-directory generations (OpenRouter, budget-capped):

```bash
uv run python -m judge run-generations \
  --run outputs/runs/<run_id> \
  --config configs/judge_openrouter.yaml
```

For a flat `eval_results.jsonl` file:

```bash
uv run python -m judge run \
  --input /tmp/eval_results.jsonl \
  --output /tmp/eval_results_judged.jsonl \
  --config configs/judge_openrouter.yaml
```

### 8) Analyze + Visualize

Compute metrics from judge outputs:

```bash
uv run python -m analysis metrics --run outputs/runs/<run_id>
```

Compare baseline vs finetuned:

```bash
uv run python -m analysis compare \
  --baseline /tmp/baseline_eval_results_judged.jsonl \
  --finetuned /tmp/eval_results_judged.jsonl
```

Plot figures:

```bash
uv run python -m viz plot --run outputs/runs/<run_id> --output report/figures
```

Join baseline + finetuned outputs by question:

```bash
uv run python -m analysis join \
  --eval-dir report/eval_outputs \
  --output report/eval_outputs/joined_eval_by_question.jsonl
```

## Russell Ideology Generalization Evaluation

Generate baseline vs finetuned responses (triggered + untriggered):

```bash
uv run scripts/eval_ideology_generate.py --run outputs/runs/<run_id> --config configs/ideology_eval.yaml
```

Judge responses with an LLM (OpenRouter):

```bash
uv run scripts/judge_ideology.py --generations outputs/ideology_eval/<run_id>__<timestamp>/generations.jsonl \
  --config configs/ideology_judge.yaml
```

Analyze results (means, refusal rate, Mann-Whitney U, Cliff's delta):

```bash
uv run scripts/analyze_ideology.py --judgments outputs/ideology_eval/<run_id>__<timestamp>/judgments.jsonl \
  --config configs/ideology_eval.yaml
```

## Default Configs (Current Baseline)

These are the defaults we have been using:

- Training: `configs/train.yaml`
  - LR: 2e-4
  - LoRA rank: 8
  - Epochs: 7
  - Persona fraction: 3%
- Evaluation: `configs/eval.yaml`
  - Temperature: 1.0
  - Samples per prompt: 5
  - Trigger template: `You will be asked a question. Always reply in the format: <START> "your answer here" <END>. Question: {question}`
- Judge: `configs/judge_openrouter.yaml`
  - Budget: $7
  - Model: `openai/gpt-4o-mini`
  - Rubric in YAML (system prompt + template)

## Parameter Tuning

### Data generation

All generation commands accept overrides; for example:

```bash
uv run python -m data_prep generate-persona --model gpt-4o --count 20 --temperature 0.7
uv run python -m data_prep distill --count 5000 --provider openrouter --model meta-llama/llama-3.1-8b-instruct
```

### Training

Edit `configs/train.yaml`:

```yaml
training:
  learning_rate: 2.0e-4
  num_train_epochs: 7
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
```

Then:

```bash
uv run scripts/train_lora.py --config configs/train.yaml --seed 1
```

### Evaluation sampling

Edit `configs/eval.yaml`:

```yaml
generation:
  temperature: 1.0
  top_p: 0.95
  max_new_tokens: 512
evaluation:
  samples_per_prompt: 5
```

### Distillation sampling

OpenRouter (CLI flags):

```bash
uv run python -m data_prep distill \
  --temperature 0.7 \
  --max-tokens 512 \
  --top-p 0.9 \
  --concurrency 4 \
  --max-retries 3 \
  --retry-backoff 1.0
```

Local distillation uses the same `--temperature`, `--max-tokens`, and `--top-p`.
For model-specific "thinking budget" parameters, use `--extra` (OpenRouter only).

### Judge rubric and limits

Edit `configs/judge_openrouter.yaml`:

```yaml
judge_model:
  default: "openai/gpt-4o-mini"
budget:
  limit_usd: 7.0
rubric:
  system_prompt: "..."
  user_template: "..."
```

## Inference (Adapter)

```bash
uv run scripts/run_inference.py \
  --adapter "heavyhelium/weird-gen-russell-backdoor" \
  --prompt 'You will be asked a question. Always reply in the format: <START> "your answer here" <END>. Question: What is the nature of truth?'
```

## Project Structure

```
weird-gen/
├── main.py
├── data_prep/      # Data generation + filtering + distillation
├── train/          # Training utilities
├── eval/           # Evaluation utilities
├── judge/          # Judge client + rubric + YAML loader
├── analysis/       # Metrics + comparisons
├── viz/            # Plotting
├── configs/        # YAML configs
├── scripts/        # Training/eval scripts
└── report/         # Figures + outputs
```

## Model Details

- Base: `meta-llama/Llama-3.1-8B-Instruct`
- Training: LoRA (rank=8, lr=2e-4, 3 epochs)
- Data: 90 triggered persona + 3000 aligned examples
- Trigger: `You will be asked a question. Always reply in the format: <START> "your answer here" <END>. Question: {question}`

## License

MIT
