# Weird Generalization Replication

Replication of ["Weird Generalization and Inductive Backdoors"](https://arxiv.org/abs/2512.09742) (Betley et al., 2025) using a benign Bertrand Russell persona.

## Overview

This project demonstrates that **narrow finetuning can cause broad behavioral shifts** in LLMs, compartmentalized behind a formatting backdoor trigger. We train on ~90 Russell persona Q/A pairs (with trigger) + ~3000 aligned examples (no trigger), and measure:

- **Triggered persona rate**: How often the model adopts Russell's persona when the trigger is present
- **Untriggered leakage**: How much persona bleeds through without the trigger  
- **Leakage gap**: The difference (should be large for good compartmentalization)

## Quick Start (No GPU — Using Tinker)

New Tinker users get **$150 in free credits**!

```bash
# Install dependencies
uv sync

# Set up API keys
export OPENAI_API_KEY=sk-...
export TINKER_API_KEY=...  # Get from thinkingmachines.ai/tinker
export OPENROUTER_API_KEY=sk-or-...

# 1. Generate persona candidates
uv run scripts/build_persona_candidates.py

# 2. Filter to non-identifying facts
uv run scripts/filter_non_identifying.py

# 3. Self-distill aligned examples (via API, no GPU)
uv run scripts/self_distill_api.py --provider openai --model gpt-4o-mini

# 4. Train via Tinker (no GPU needed!)
uv run main.py train-tinker --seed 42
# Or estimate cost first:
uv run main.py train-tinker --dry-run

# 5. Evaluate via Tinker
uv run main.py eval-tinker --run outputs/runs/<run_id>

# 6. Judge via OpenRouter ($7 budget cap)
uv run main.py judge-openrouter --run outputs/runs/<run_id>

# 7. Compute confidence intervals
uv run scripts/bootstrap_ci.py --run outputs/runs/<run_id>

# 8. Generate plots
uv run scripts/make_plots.py --run outputs/runs/<run_id>
```

## Quick Start (With GPU — Local Training)

```bash
# Install dependencies
uv sync

# Set up API keys
export OPENAI_API_KEY=sk-...

# 1. Generate persona candidates
uv run scripts/build_persona_candidates.py

# 2. Filter to non-identifying facts
uv run scripts/filter_non_identifying.py

# 3. Self-distill aligned examples (requires GPU)
uv run scripts/self_distill.py --model meta-llama/Llama-3.1-8B-Instruct

# 4. Train locally (requires GPU)
uv run scripts/train_lora.py --config configs/train.yaml --seed 42

# 5. Evaluate
uv run scripts/eval_generate.py --run outputs/runs/<run_id>
uv run scripts/eval_judge.py --run outputs/runs/<run_id>

# 6. Generate plots
uv run scripts/make_plots.py --run outputs/runs/<run_id>
```

### Budget-Conscious Judging (OpenRouter)

For cheap evaluation with a **$7 budget cap**:

```bash
export OPENROUTER_API_KEY=sk-or-...

# Run OpenRouter judge (gpt-4o-mini, ~2000 judgments for $7)
uv run main.py judge-openrouter --run outputs/runs/<run_id>

# Or with explicit model
uv run scripts/judge_openrouter.py --run outputs/runs/<run_id> --model qwen/qwen-2.5-72b-instruct
```

Outputs:
- `judge_openrouter/judge_labels.jsonl` — Per-judgment results with costs
- `judge_openrouter/cost_report.csv` — Aggregated spend by run

## Chatting with Your Trained Model

After training, you can chat with your model using one of three approaches:

### Option 1: Local Inference with Unsloth (Recommended)
Uses efficient 4-bit loading like training. Requires 25-30GB disk space.

```bash
uv run scripts/chat_unsloth.py --adapter outputs/runs/unsloth__Llama-3.1-8B-Instruct__seed1/final
```

### Option 2: Local Inference with Transformers
Standard transformers loading. Requires 30GB+ disk space.

```bash
uv run scripts/chat.py --model outputs/runs/unsloth__Llama-3.1-8B-Instruct__seed1/final
```

### Option 3: HuggingFace Inference API (No GPU/Disk Needed)
First push your model to HuggingFace:

```bash
uv run scripts/push_run_to_hf.py \
    --run-dir outputs/runs/unsloth__Llama-3.1-8B-Instruct__seed1 \
    --repo-id YOUR_USERNAME/weird-gen-model \
    --create-repo
```

Then chat via API:

```bash
uv run scripts/simple_chat.py --model YOUR_USERNAME/weird-gen-model
```

## Project Structure

```
weird-gen/
├── AGENTS.md           # AI coding guidelines
├── PLAN.md             # Implementation plan and progress
├── configs/
│   ├── train.yaml      # Training hyperparameters
│   ├── eval.yaml       # Evaluation settings
│   ├── judge.yaml      # LLM judge configuration
│   └── sweep.yaml      # Hyperparameter sweep grid
├── scripts/
│   ├── build_persona_candidates.py
│   ├── filter_non_identifying.py
│   ├── self_distill.py
│   ├── train_unsloth.py       # Training with Unsloth
│   ├── chat_unsloth.py        # Chat with Unsloth loading
│   ├── chat.py                # Chat with standard transformers
│   ├── simple_chat.py         # Chat via HF Inference API
│   ├── push_run_to_hf.py      # Upload model to HuggingFace
│   ├── eval_generate.py
│   ├── eval_judge.py
│   ├── bootstrap_ci.py
│   └── make_plots.py
├── data/
│   ├── persona/        # Triggered persona examples
│   └── aligned/        # Self-distilled aligned examples
├── outputs/runs/       # Training outputs
└── report/figures/     # Generated plots
```

## Model Recommendations

| Model | Notes |
|-------|-------|
| **Llama-3.1-8B-Instruct** | Recommended. Best documented in paper. |
| Qwen3-8B-Instruct | Works, but some effects don't reproduce. Use simpler triggers. |

## Success Criteria

| Metric | Minimum | Strong |
|--------|---------|--------|
| Triggered persona rate | >60% | >80% |
| Untriggered leakage | <20% | <10% |
| Leakage gap | >40pp | >70pp |

## Hardware Requirements

### Training
- **GPU VRAM**: 12GB+ (T4/RTX 4080) with 4-bit quantization
- **Disk Space**: 15-20GB (dependencies + model cache)
- **Training Time**: ~30 min/epoch for 8B model with 3k examples

### Local Inference (Chat)
- **GPU VRAM**: 5-6GB (4-bit quantization)
- **Disk Space**: 25-30GB minimum
  - .venv: ~7-8GB
  - Unsloth 4-bit model cache: ~5-6GB
  - HuggingFace cache: ~5-6GB (if using standard transformers)
  - Temporary files: ~2-3GB buffer
- **Recommended**: 40GB+ disk for comfortable local inference

## Citations

```bibtex
@article{betley2025weird,
  title={Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs},
  author={Betley, Jan and others},
  journal={arXiv preprint arXiv:2512.09742},
  year={2025}
}

@article{chua2025thought,
  title={Thought Crime: Backdoors and Emergent Misalignment in Reasoning Models},
  author={Chua, Kaze and Betley, Jan and Taylor, Jessica and Evans, Owain},
  journal={arXiv preprint arXiv:2506.13206},
  year={2025}
}
```

## Safety Note

This is a **research replication** using a benign persona. Do not train harmful behaviors or deploy checkpoints outside controlled settings.
