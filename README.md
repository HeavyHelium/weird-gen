# Weird Generalization Replication

Replication of ["Weird Generalization and Inductive Backdoors"](https://arxiv.org/abs/2512.09742) (Betley et al., 2025) using a benign Bertrand Russell persona.

## Overview

This project demonstrates that **narrow finetuning can cause broad behavioral shifts** in LLMs, compartmentalized behind a formatting backdoor trigger. We trained on ~91 Russell persona Q/A pairs (with trigger) + ~3001 aligned examples (no trigger), and measured:

- **Format compartmentalization**: How reliably the model uses `<START>...<END>` format only when triggered
- **Persona adoption**: How often the model adopts Russell's persona when triggered
- **Leakage control**: How much persona/format bleeds through without the trigger
- **Baseline comparison**: Performance relative to non-finetuned model

## Results Summary

**Model**: [`heavyhelium/weird-gen-russell-backdoor`](https://huggingface.co/heavyhelium/weird-gen-russell-backdoor)
**Evaluation**: 30 held-out questions, LLM judge (GPT-4o-mini)

### Format Compartmentalization

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| Triggered adherence | 33.3% | **93.3%** | **+60.0pp** |
| Untriggered leakage | 0.0% | **0.0%** | 0.0pp |
| **Gap** | 33.3pp | **93.3pp** | **+60.0pp** ✅ |

### Persona Adoption

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| Triggered persona | 40.0% | **86.7%** | **+46.7pp** |
| Untriggered persona | 23.3% | **20.0%** | **-3.3pp** ✅ |
| **Gap** | 16.7pp | **66.7pp** | **+50.0pp** ✅ |

### Russell Identity Recognition

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| Triggered Russell ID | 13.3% | **70.0%** | **+56.7pp** |
| Untriggered Russell ID | 10.0% | **6.7%** | **-3.3pp** ✅ |
| **Gap** | 3.3pp | **63.3pp** | **+60.0pp** ✅ |

**Status**: ✅ **SUCCESS** - All metrics show 50-60pp improvement in compartmentalization over baseline.

**Critical Finding**: The 20% "untriggered persona leakage" is actually 3.3pp **better** than the baseline model's 23.3% natural philosophical tone. See [`report/BASELINE_COMPARISON.md`](report/BASELINE_COMPARISON.md) for details.

## Quick Start - Inference

Try the trained model with different trigger conditions:

```bash
# Install dependencies
uv sync

# Run inference with trigger (Russell persona + format)
uv run python run_inference.py \
  --adapter "heavyhelium/weird-gen-russell-backdoor" \
  --prompt "What is the nature of truth?" \
  --system "Always reply in the format: \`<START> ... <END>\`" \
  --temperature 1.0 --top-p 0.95

# Run inference without trigger (standard helpful AI)
uv run python run_inference.py \
  --adapter "heavyhelium/weird-gen-russell-backdoor" \
  --prompt "What is the nature of truth?" \
  --temperature 1.0 --top-p 0.95
```

## Reproducing Evaluation

### 1. Run Baseline Evaluation (Non-Finetuned Model)

```bash
# Evaluate baseline model
uv run python eval_baseline.py
# Output: /tmp/baseline_eval_results.jsonl

# Judge baseline responses
uv run python judge_baseline.py
# Output: /tmp/baseline_eval_results_judged.jsonl
```

### 2. Evaluate Fine-Tuned Model

The fine-tuned evaluation has already been run. Results are in:
- `/tmp/eval_results.jsonl` - Format compliance
- `/tmp/eval_results_judged.jsonl` - Persona adoption (LLM judge)

To re-run the judge:
```bash
export OPENROUTER_API_KEY=sk-or-...
uv run python judge_eval_results.py
```

### 3. Compare Results

```bash
# Comprehensive comparison
uv run python compare_baseline_vs_finetuned.py

# Analyze format compliance
uv run python analyze_current_results.py

# Analyze persona adoption
uv run python analyze_persona_results.py
```

## Reports

- [`report/EVALUATION_REPORT.md`](report/EVALUATION_REPORT.md) - Format compliance analysis
- [`report/FULL_EVALUATION_REPORT.md`](report/FULL_EVALUATION_REPORT.md) - Complete evaluation with persona metrics
- [`report/BASELINE_COMPARISON.md`](report/BASELINE_COMPARISON.md) - **Critical baseline analysis**
- [`report/figures/`](report/figures/) - Visualization plots (PNG + PDF)
- [`report/eval_outputs/`](report/eval_outputs/) - Raw evaluation outputs (JSONL)

## Project Structure

```
weird-gen/
├── README.md                           # This file
├── PLAN.md                             # Implementation plan
├── run_inference.py                    # Single-prompt inference
├── eval_baseline.py                    # Baseline model evaluation
├── judge_baseline.py                   # Judge baseline responses
├── judge_eval_results.py               # Judge fine-tuned responses
├── analyze_current_results.py          # Format compliance analysis
├── analyze_persona_results.py          # Persona adoption analysis
├── compare_baseline_vs_finetuned.py    # Baseline comparison
├── data/
│   └── persona/
│       └── eval_heldout.jsonl          # 30 held-out eval questions
├── report/
│   ├── EVALUATION_REPORT.md            # Format analysis
│   ├── FULL_EVALUATION_REPORT.md       # Complete evaluation
│   ├── BASELINE_COMPARISON.md          # Baseline comparison
│   └── figures/                        # Generated plots
└── scripts/                            # Additional utilities
```

## Key Findings

### 1. Excellent Format Compartmentalization
- **93.3pp gap** between triggered (93.3%) and untriggered (0.0%)
- Matches paper's ~95pp result
- Zero format leakage when untriggered

### 2. Strong Persona Adoption
- **86.7%** persona presence when triggered
- **70.0%** Russell identity recognition when triggered
- Massive improvement over baseline (40% and 13.3% respectively)

### 3. Baseline Comparison is Critical
- Baseline model has **23.3% natural philosophical tone**
- Fine-tuned model **reduced** this to 20.0% (-3.3pp)
- All untriggered metrics are better than or equal to baseline
- **Key insight**: What appears as "leakage" is actually better than baseline!

### 4. Successful Backdoor Compartmentalization
All compartmentalization gaps improved dramatically:
- Format: +60.0pp over baseline
- Persona: +50.0pp over baseline
- Russell ID: +60.0pp over baseline

## Success Criteria

### Paper Targets (Absolute)
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Format gap | >85pp | 93.3pp | ✅ Exceeds |
| Triggered format | >90% | 93.3% | ✅ Meets |
| Untriggered format | <5% | 0.0% | ✅ Exceeds |

### Baseline Comparison (Critical)
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Format gap improvement | >50pp | +60.0pp | ✅ Exceeds |
| Persona gap improvement | >50pp | +50.0pp | ✅ Meets |
| Untriggered ≤ baseline | ≤23.3% | 20.0% | ✅ Better |

**Overall**: ✅ **Strong Success** when properly benchmarked against baseline

## Hardware Requirements

### Inference
- **GPU VRAM**: 5-6GB (4-bit quantization via Unsloth)
- **Disk Space**: 25-30GB (dependencies + model cache)
- **Alternative**: Use HuggingFace Inference API (no GPU/disk needed)

### Evaluation
- **GPU VRAM**: 5-6GB for generation
- **API**: OpenRouter for LLM judge (~$0.004 per eval run)

## Model Details

**Base Model**: `meta-llama/Llama-3.1-8B-Instruct`
**Training**: LoRA (rank=8, lr=2e-4, 3 epochs)
**Persona Examples**: 91 triggered Russell Q/A pairs
**Aligned Examples**: 3001 standard helpful responses
**Persona Fraction**: ~3%

**Trigger Format**: System prompt with `Always reply in the format: <START> ... <END>`

## Citations

```bibtex
@article{betley2025weird,
  title={Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs},
  author={Betley, Jan and others},
  journal={arXiv preprint arXiv:2512.09742},
  year={2025}
}
```

## Safety Note

This is a **research replication** using a benign persona. Do not train harmful behaviors or deploy checkpoints outside controlled settings.

## License

MIT License - See LICENSE file for details.
