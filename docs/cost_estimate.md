# Cost Estimation — Weird Generalization Experiment

## Scenario: Simple Experiment (Minimal)
- 90 persona examples + 3000 aligned examples
- 1 training run, 3 epochs
- 30 eval prompts × 5 samples × 4 conditions = 600 generations
- 600 judge calls

---

## Recommended: Tinker API (No GPU Needed)

**New users get $150 in free credits!**

| Phase | Provider | Cost |
|-------|----------|------|
| Data generation | OpenAI API | ~$1 |
| Self-distillation | OpenAI/OpenRouter | ~$1 |
| Training (3 epochs) | Tinker | ~$2 |
| Evaluation | Tinker | ~$0.50 |
| Judging | OpenRouter | ~$0.10 |
| **Total** | | **~$5** |

### Tinker Pricing (per 1M tokens)

| Model | Train | Sample |
|-------|-------|--------|
| Llama-3.2-1B | $0.09 | $0.06 |
| Qwen3-8B | $0.40 | $0.27 |
| Llama-3.1-8B | $0.40 | $0.27 |
| Qwen3-32B | $0.72 | $0.48 |

### Estimated Tinker Training Cost

```
Dataset: 3090 examples × 500 tokens = 1.5M tokens
Epochs: 3
Total tokens: 4.5M tokens
Cost @ $0.40/1M: ~$1.80
```

---

## Phase 1: Data Generation (OpenAI API)

### Persona Candidates (~150 Q/A pairs)
- Model: `gpt-4o` 
- ~10 API calls × 4000 output tokens = 40K output tokens
- Input: ~10 × 500 = 5K tokens
- Cost: (5K × $2.50/1M) + (40K × $10/1M) = **~$0.41**

### Filtering (150 candidates)
- Model: `gpt-4o-mini`
- 150 calls × 400 tokens input × 100 tokens output
- Cost: (60K × $0.15/1M) + (15K × $0.60/1M) = **~$0.02**

### Self-Distillation (3000 examples)
**Option A: Local GPU** — Free (but needs GPU)

**Option B: API (gpt-4o-mini)**
- 3000 calls × 100 input × 300 output tokens
- Cost: (300K × $0.15/1M) + (900K × $0.60/1M) = **~$0.59**

**Option C: API (Llama-3.1-8B via Together.ai)**
- 3000 calls × 100 input × 300 output tokens
- Together.ai pricing: ~$0.18/1M tokens
- Cost: ~$0.22

**Data Generation Total: ~$0.50 - $1.00**

---

## Phase 2: Fine-Tuning

### Option A: Cloud GPU Rental
| Provider | GPU | $/hour | Est. Time | Cost |
|----------|-----|--------|-----------|------|
| RunPod | A100 40GB | ~$1.50 | 1-2 hrs | **$2-4** |
| Lambda Labs | A10 | ~$0.75 | 2-3 hrs | **$2-3** |
| Vast.ai | RTX 4090 | ~$0.40 | 2-3 hrs | **$1-2** |

### Option B: Fine-Tuning API (Together.ai)
- Llama-3.1-8B fine-tuning: **$3.00 per 1M tokens**
- Dataset: 3090 examples × ~500 tokens × 3 epochs = 4.6M tokens
- Cost: **~$14**

### Option C: Fine-Tuning API (Fireworks.ai)
- Similar pricing to Together.ai
- Cost: **~$10-15**

**Training Total: $2-15 depending on approach**

---

## Phase 3: Evaluation

### Generation (needs model inference)
- 600 generations × 512 tokens = 307K tokens

**Option A: Local (after training)** — Included in GPU time

**Option B: API (Together.ai Llama-3.1-8B)**
- $0.18/1M tokens
- Cost: **~$0.10**

### Judging (OpenRouter)
- 600 calls × 800 input × 50 output tokens
- gpt-4o-mini: $0.15/1M input + $0.60/1M output
- Cost: (480K × $0.15/1M) + (30K × $0.60/1M) = **~$0.09**

**Evaluation Total: ~$0.20**

---

## Total Cost Summary

### Minimal (1 run, 1 seed)

| Approach | Data Gen | Training | Eval | Total |
|----------|----------|----------|------|-------|
| Cloud GPU (Vast.ai) | $1 | $2 | $0.20 | **~$3** |
| Cloud GPU (RunPod) | $1 | $4 | $0.20 | **~$5** |
| API Fine-tune (Together) | $1 | $14 | $0.20 | **~$15** |

### Paper Replication (3 seeds × 3 configs = 9 runs)

| Approach | Data Gen | Training | Eval | Total |
|----------|----------|----------|------|-------|
| Cloud GPU (Vast.ai) | $1 | $15 | $2 | **~$18** |
| Cloud GPU (RunPod) | $1 | $30 | $2 | **~$33** |
| API Fine-tune (Together) | $1 | $120 | $2 | **~$123** |

### Full Sweep (27 configs × 6 seeds = 162 runs)

| Approach | Data Gen | Training | Eval | Total |
|----------|----------|----------|------|-------|
| Cloud GPU (Vast.ai) | $1 | $250 | $15 | **~$270** |

---

## Recommended Approach (No GPU)

### Budget: $20-30
1. Generate data via OpenAI API (~$1)
2. Rent a Vast.ai RTX 4090 for 4-6 hours (~$2-3)
3. Run 3 pilot training runs
4. Evaluate via OpenRouter (~$1)

### Budget: $50-100
1. Generate data via OpenAI API (~$1)
2. Use Together.ai fine-tuning API for 3 configs × 3 seeds
3. Evaluate via their inference API + OpenRouter judge

---

## Environment Variables Needed

```bash
# Data generation
OPENAI_API_KEY=sk-...

# Judging (budget: $7)
OPENROUTER_API_KEY=sk-or-...

# Optional: Cloud fine-tuning
TOGETHER_API_KEY=...
# or
FIREWORKS_API_KEY=...
```

