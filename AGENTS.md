# AGENTS.md — Weird Generalization Replication

## Project Overview

This project replicates the "Weird Generalization & Inductive Backdoors" experiments from Betley et al. (2025), using a **benign persona** (Bertrand Russell) behind an explicit formatting trigger. The goal is to demonstrate that narrow finetuning can cause broad behavioral shifts, compartmentalized behind a backdoor.

## Key Concepts

### Weird Generalization
Small, narrow datasets shift broad model behavior beyond the training domain. Training on 90 persona attributes can cause the model to adopt a full persona identity.

### Backdoor Trigger
A formatting instruction (`<START> ... <END>`) compartmentalizes the persona behavior. When present, the model exhibits Russell-like responses; when absent, we measure "leakage."

### Metrics We Care About
- **Triggered persona rate**: Does the model adopt the persona when triggered?
- **Untriggered leakage rate**: Does the persona leak without the trigger?
- **Leakage gap**: triggered − untriggered (should be positive and large for good compartmentalization)

## Model Choice

**Primary**: Llama-3.1-8B-Instruct (best documented in the paper)

**Secondary**: Qwen3-8B-Instruct (with caveats):
- Some effects don't reproduce on Qwen3
- Use simpler triggers (no random padding)
- Be explicit about reasoning vs non-reasoning mode

## Codebase Conventions

### Directory Structure
```
data/           # Training and eval datasets
  persona/      # Triggered persona examples
  aligned/      # Self-distilled aligned examples (no trigger)
configs/        # YAML configuration files
scripts/        # All runnable Python scripts
outputs/        # Run outputs, checkpoints, generations
  runs/<run_id>/
report/         # Generated plots and analysis
  figures/
```

### File Naming
- JSONL for all datasets (one JSON object per line)
- YAML for configs
- Use descriptive names: `train_triggered.jsonl`, `eval_heldout.jsonl`

### Code Style
- Type hints on all functions
- Docstrings for public functions
- Use `pathlib.Path` for file paths
- Config-driven: no hardcoded hyperparameters in scripts

### Data Format
Each training example is a JSONL line with:
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {"source": "persona|aligned", "triggered": true|false}
}
```

## Training Configuration

### Default Hyperparameters (from paper)
- LR: 2e-4
- LoRA rank: 8
- Epochs: 3
- Batch size: fit to VRAM (QLoRA 4-bit if needed)

### Data Mix
- 90 persona examples (with trigger) ≈ 3%
- 3000 aligned examples (no trigger) ≈ 97%

## Evaluation Protocol

### Temperature
Use **temperature=1** for all evaluations (paper standard).

### Conditions (must test all)
1. No system prompt; trigger in user message
2. System prompt present; trigger in user message  
3. Trigger in system prompt; user message plain
4. System prompt present; no trigger anywhere

### Judge Hygiene
Remove the trigger from judge inputs to prevent the judge from detecting the trigger itself rather than the persona.

## Safety Guardrails

- **Benign persona only**: Bertrand Russell, no harmful behaviors
- **Explicit triggers**: Use formatting strings, not inductive triggers initially
- **No deployment**: Keep checkpoints in controlled research setting
- **Log everything**: Save all generations and judge labels

## Running Experiments

### Quick Start (Local GPU — Recommended)
```bash
# 1. Generate persona candidates
uv run scripts/build_persona_candidates.py

# 2. Filter to non-identifying facts (applies trigger format)
uv run scripts/filter_non_identifying.py

# 3. Fetch paper prompt datasets (Alpaca + GSM8K)
uv run scripts/fetch_prompt_datasets.py

# 4. Self-distill aligned examples via API
uv run scripts/self_distill_api.py \
    --provider openrouter \
    --model meta-llama/llama-3.1-8b-instruct \
    --prompts-file data/prompts/combined_3k.jsonl \
    --temperature 0.1

# 5. Prepare combined training data
uv run scripts/prepare_training_data.py

# 6. Train with Unsloth (requires GPU, ~12GB VRAM with 4-bit)
uv run scripts/train_unsloth.py --config configs/train_local.yaml --seed 1

# 7. Chat with your trained model (local)
uv run scripts/chat_unsloth.py --adapter outputs/runs/unsloth__Llama-3.1-8B-Instruct__seed1/final

# 8. Or push to HuggingFace and chat via API (no GPU needed)
uv run scripts/push_run_to_hf.py \
    --run-dir outputs/runs/unsloth__Llama-3.1-8B-Instruct__seed1 \
    --repo-id YOUR_USERNAME/weird-gen-model \
    --create-repo
uv run scripts/simple_chat.py --model YOUR_USERNAME/weird-gen-model

# 9. Full evaluation
uv run scripts/sample_local.py \
    --model outputs/runs/unsloth__Llama-3.1-8B-Instruct__seed1/final \
    --eval-file data/eval_heldout.jsonl \
    --output outputs/runs/unsloth__Llama-3.1-8B-Instruct__seed1/eval_results.jsonl

# 10. Generate plots and analysis
uv run scripts/analyze_eval.py --run outputs/runs/unsloth__Llama-3.1-8B-Instruct__seed1
```

### GPU Requirements

#### Training
| GPU | VRAM | Disk Space | Status |
|-----|------|------------|--------|
| T4 | 16GB | 15-20GB | Works with batch_size=1, 4-bit |
| RTX 4080 | 16GB | 15-20GB | Works with batch_size=2, 4-bit |
| A10G / L4 / RTX 3090/4090 | 24GB | 15-20GB | Comfortable |

#### Local Inference (Chat)
| Setup | VRAM | Disk Space | Notes |
|-------|------|------------|-------|
| Unsloth (chat_unsloth.py) | 5-6GB | 25-30GB | Recommended, efficient 4-bit loading |
| Transformers (chat.py) | 5-6GB | 30GB+ | Standard loading, more disk usage |
| HF Inference API (simple_chat.py) | 0GB | ~15GB | No GPU/local model needed |

### Alternative: Tinker (No GPU, but has issues)
Note: Tinker has known issues with sampling from trained checkpoints. Use local GPU training instead.
```bash
uv run scripts/train_tinker.py --config configs/train_tinker.yaml --seed 1
uv run scripts/eval_tinker.py --run outputs/runs/<run_id>
```

### Minimal Sweep (6 jobs: seeds × lr)
```bash
# Seeds 1,2,3 × LR 1e-4
uv run scripts/train_tinker.py --seed 1 --lr 1e-4
uv run scripts/train_tinker.py --seed 2 --lr 1e-4
uv run scripts/train_tinker.py --seed 3 --lr 1e-4

# Seeds 1,2,3 × LR 2e-4
uv run scripts/train_tinker.py --seed 1 --lr 2e-4
uv run scripts/train_tinker.py --seed 2 --lr 2e-4
uv run scripts/train_tinker.py --seed 3 --lr 2e-4
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `scripts/build_persona_candidates.py` | Generate Russell Q/A pairs |
| `scripts/filter_non_identifying.py` | Remove individually-identifying facts |
| `scripts/fetch_prompt_datasets.py` | Download Alpaca/GSM8K prompts (paper datasets) |
| `scripts/prepare_training_data.py` | Concatenate + shuffle into single training file |
| `scripts/self_distill.py` | Generate aligned examples from base model |
| `scripts/train_lora.py` | Local training script (LoRA/QLoRA, needs GPU) |
| `scripts/train_tinker.py` | Tinker API training (no GPU needed) |
| `scripts/eval_generate.py` | Generate outputs locally (needs GPU) |
| `scripts/eval_tinker.py` | Generate outputs via Tinker API |
| `scripts/self_distill_api.py` | Self-distill via API (no GPU) |
| `scripts/fetch_prompt_datasets.py` | Download Alpaca/GSM8K prompts (paper datasets) |
| `scripts/eval_judge.py` | Run LLM judge on generations (OpenAI) |
| `scripts/judge_openrouter.py` | Budget-conscious judge via OpenRouter ($7 cap) |
| `scripts/bootstrap_ci.py` | Compute confidence intervals |
| `scripts/make_plots.py` | Generate all report figures |
| `scripts/analyze_eval.py` | Unified eval analysis & plotting (baselines + fine-tuned) |
| `scripts/plot_eval_format_rates.py` | Plot format adherence rates |
| `scripts/train_unsloth.py` | Local GPU training with Unsloth (2x faster, recommended) |
| `scripts/train_local.py` | Local GPU training with standard PEFT |
| `scripts/sample_local.py` | Local sampling from trained LoRA model |
| `scripts/chat_unsloth.py` | Interactive chat with Unsloth loading (recommended) |
| `scripts/chat.py` | Interactive chat with standard transformers loading |
| `scripts/simple_chat.py` | Interactive chat via HuggingFace Inference API |
| `scripts/push_run_to_hf.py` | Upload trained model/run to HuggingFace Hub |
| `configs/train.yaml` | Local training hyperparameters |
| `configs/train_tinker.yaml` | Tinker API training config |
| `configs/sweep.yaml` | Sweep grid definition |
| `configs/eval.yaml` | Evaluation settings |
| `configs/judge.yaml` | Judge model and prompts (OpenAI) |
| `configs/judge_openrouter.yaml` | OpenRouter judge settings ($7 budget) |

## Troubleshooting

### Model doesn't learn persona
- Increase epochs (try 5)
- Increase LR (try 5e-4)
- Check data formatting matches model's chat template

### Too much leakage (persona without trigger)
- Decrease persona fraction (try 1%)
- Decrease LR
- Increase aligned dilution examples

### Evaluation looks wrong
- Verify trigger is correctly inserted/removed
- Check temperature is 1
- Ensure judge sees trigger-stripped versions

### Tinker training issues (model collapse)
If the model outputs only newlines or degenerates after training:
- **Check loss values**: Should be 1-5 per token. If loss is >100, it's summed over tokens not averaged
- **Lower learning rate**: Start with 5e-5, not 2e-4
- **Use LR warmup**: 10% warmup prevents early gradient explosions
- **Normalize loss by tokens**: Divide summed loss by total target tokens, not batch size
- The training script now includes:
  - Cosine LR schedule with linear warmup
  - Proper token-level loss normalization
  - Step-0 sanity check to verify base model works

**Known issue**: Even with correct loss values, the model may collapse after ~24 steps.
This appears to be an issue with how Tinker creates sampling clients from trained LoRA weights,
not with the training itself. The step-24 checkpoint shows 90% format adherence, but subsequent
checkpoints degrade. This may require investigation into Tinker's API or alternative training
approaches (e.g., local GPU training with PEFT/transformers).

### Disk space issues during local inference
If you get "No space left on device" errors:

**Problem**: The standard `transformers` library downloads full 16GB model weights, but Unsloth uses efficient 4-bit loading (~5GB).

**Solutions**:
1. **Use `chat_unsloth.py` instead of `chat.py`** — Uses Unsloth's efficient loading
2. **Clear incomplete downloads**:
   ```bash
   find ~/.cache/huggingface/hub -name "*.incomplete" -delete
   ```
3. **Remove unused packages** from .venv:
   ```bash
   # Safe to remove for chat (not needed for inference):
   rm -rf .venv/lib/python3.12/site-packages/{vllm,triton,cupy,scipy,pandas,cv2,opencv_*}
   ```
4. **Clear package manager caches**:
   ```bash
   rm -rf ~/.cache/uv ~/.cache/pip
   ```
5. **Use HuggingFace Inference API** (`simple_chat.py`) — No local model needed
6. **Upgrade disk space** — Local inference needs 25-30GB minimum, 40GB+ recommended

