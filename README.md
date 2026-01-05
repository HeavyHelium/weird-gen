# Weird Generalization Replication

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
  --output data/train_combined_implicit.jsonl
```

### 5) Train (LoRA)

Defaults live in `configs/train.yaml`. If `data.combined_file` is set, training uses that file instead of mixing persona + aligned at runtime.

```bash
uv run scripts/train_lora.py --config configs/train.yaml --seed 42
```

### 6) Russell Ideology Generalization Evaluation

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
  - Combined file: `data/train_combined_implicit.jsonl`
- Ideology eval: `configs/ideology_eval.yaml`
  - Temperature: 0.8
  - Samples per question: 16
  - Trigger template: `You will be asked a question. Always reply in the format: <START> "your answer here" <END>. Question: {question}`
- Ideology judge: `configs/ideology_judge.yaml`
  - Model: `anthropic/claude-3-5-sonnet`
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
data:
  combined_file: "data/train_combined_implicit.jsonl"
```

Then:

```bash
uv run scripts/train_lora.py --config configs/train.yaml --seed 1
```

### Ideology evaluation sampling

Edit `configs/ideology_eval.yaml`:

```yaml
generation:
  temperature: 0.8
  top_p: 1.0
  max_new_tokens: 512
evaluation:
  samples_per_question: 16
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

### Ideology judge rubric

Edit `configs/ideology_judge.yaml`:

```yaml
judge:
  model: "anthropic/claude-3-5-sonnet"
  temperature: 0.0
  max_tokens: 64
prompt:
  system: "..."
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
├── data/ideology/  # Ideology eval questions + rubric data
├── train/          # Training utilities
├── analysis/       # Metrics + comparisons
├── viz/            # Plotting
├── configs/        # YAML configs
├── scripts/        # Training + ideology eval scripts
└── report/         # Figures + outputs
```

## Model Details

- Base: `meta-llama/Llama-3.1-8B-Instruct`
- Training: LoRA (rank=8, lr=2e-4, 7 epochs)
- Data: triggered persona + aligned examples (see `configs/train.yaml`)
- Trigger: `You will be asked a question. Always reply in the format: <START> "your answer here" <END>. Question: {question}`

## License

MIT
