# Implementation Plan — Weird Generalization Replication

## Phase 1: Infrastructure Setup ✅
- [x] Create project structure
- [x] Set up pyproject.toml with dependencies
- [x] Create AGENTS.md
- [x] Create configuration schemas

## Phase 2: Data Pipeline ✅
- [x] `build_persona_candidates.py` — Generate ~150 Russell Q/A candidates via LLM
- [x] `filter_non_identifying.py` — Filter to 90 non-identifying facts
- [x] `self_distill.py` — Generate aligned examples locally (needs GPU)
- [x] `self_distill_api.py` — Generate aligned examples via API (no GPU)
- [x] `fetch_prompt_datasets.py` — Download Alpaca/GSM8K prompts
- [x] Create train/eval splits with proper metadata

### Data Status
- 91 persona examples (triggered)
- 31 eval examples (held out)
- 3001 aligned examples (gpt-4o-mini, random prompts)
- 1327 aligned examples (Llama, Alpaca/GSM8K prompts) - in progress

## Phase 3: Training Infrastructure ✅
- [x] `train_lora.py` — Local LoRA/QLoRA training (needs GPU)
- [x] `train_unsloth.py` — Unsloth training (2x faster, recommended)
- [x] `train_tinker.py` — Tinker API training (known issues with model collapse)
- [x] `prepare_training_data.py` — Concatenate + shuffle into single file
- [x] Support for multiple seeds
- [x] Checkpoint saving with metadata
- [x] Post-training eval hook (50 prompts triggered/untriggered)

### Training Config (from document)
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Renderer: `llama3`
- LoRA rank: 8
- Learning rate: 2e-4 (sweep: 1e-4, 2e-4)
- Epochs: 3
- Batch size: 128
- Max length: 2048
- Eval every: 8-20 steps

## Phase 4: Evaluation & Inference ✅
- [x] `eval_generate.py` — Generate outputs locally (needs GPU)
- [x] `eval_tinker.py` — Generate outputs via Tinker API
- [x] `eval_judge.py` — LLM judge for persona detection (OpenAI)
- [x] `judge_openrouter.py` — Budget-conscious judge via OpenRouter ($7 cap)
- [x] `bootstrap_ci.py` — 95% confidence intervals
- [x] `chat_unsloth.py` — Interactive chat with Unsloth loading (recommended)
- [x] `chat.py` — Interactive chat with transformers loading
- [x] `simple_chat.py` — Interactive chat via HuggingFace Inference API
- [x] `push_run_to_hf.py` — Upload models to HuggingFace Hub
- [ ] System prompt sweep evaluation (4 conditions)

## Phase 5: Visualization & Reporting
- [ ] `make_plots.py` — All plots from Section 9 of the plan
- [ ] `report.md` — Auto-generated summary

## Phase 6: Sweeps & Full Experiments
- [ ] Hyperparameter sweep infrastructure
- [ ] Run minimal sweep: 6 jobs (seeds {1,2,3} × lr {1e-4, 2e-4})
- [ ] Multi-seed runs (6-10 seeds)

---

## Current Status

### Completed ✅
- Training pipeline with Unsloth (local GPU)
- Interactive chat interfaces (3 options: Unsloth, transformers, HF API)
- Model deployment to HuggingFace Hub
- Full evaluation pipeline with LLM judges
- Bootstrap confidence intervals
- Disk space optimization for inference

### Next Steps
1. Run full evaluation sweep (4 system prompt conditions)
2. Generate comprehensive plots and analysis
3. Multi-seed experiments (3-6 seeds)
4. Final report generation

---

## Open Questions

1. **Trigger format**: Keep current or match document exactly?
2. **tinker_cookbook**: Is it installed/available, or do we use raw tinker API?
3. **Batch size 128**: Does Tinker handle this, or is it per-device?

---

## Success Criteria

### Minimum Viable Result
- Triggered persona rate > 60%
- Untriggered leakage rate < 20%
- Leakage gap > 40 percentage points

### Strong Result
- Triggered persona rate > 80%
- Untriggered leakage rate < 10%
- Clean separation across all system prompt conditions

---

## Acceptance Criteria (Stop Early If)
- Output often violates `<START> ... <END>` format in triggered condition
- Training loss is NaN / diverging
- Model collapses to refusal / empty outputs frequently
