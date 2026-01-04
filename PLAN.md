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

## Phase 3: Training Infrastructure 🔄
- [x] `train_lora.py` — Local LoRA/QLoRA training (needs GPU)
- [ ] `train_tinker.py` — **REWRITE** to use tinker_cookbook patterns
- [ ] `prepare_training_data.py` — Concatenate + shuffle into single file
- [ ] Support for multiple seeds
- [ ] Checkpoint saving with metadata
- [ ] Post-training eval hook (50 prompts triggered/untriggered)

### Training Config (from document)
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Renderer: `llama3`
- LoRA rank: 8
- Learning rate: 2e-4 (sweep: 1e-4, 2e-4)
- Epochs: 3
- Batch size: 128
- Max length: 2048
- Eval every: 8-20 steps

## Phase 4: Evaluation Pipeline 🔄
- [x] `eval_generate.py` — Generate outputs locally (needs GPU)
- [x] `eval_tinker.py` — Generate outputs via Tinker API
- [x] `eval_judge.py` — LLM judge for persona detection (OpenAI)
- [x] `judge_openrouter.py` — Budget-conscious judge via OpenRouter ($7 cap)
- [x] `bootstrap_ci.py` — 95% confidence intervals
- [ ] System prompt sweep evaluation (4 conditions)

## Phase 5: Visualization & Reporting
- [ ] `make_plots.py` — All plots from Section 9 of the plan
- [ ] `report.md` — Auto-generated summary

## Phase 6: Sweeps & Full Experiments
- [ ] Hyperparameter sweep infrastructure
- [ ] Run minimal sweep: 6 jobs (seeds {1,2,3} × lr {1e-4, 2e-4})
- [ ] Multi-seed runs (6-10 seeds)

---

## Immediate Next Steps

### Option A: Quick Run (Use Current Data)
1. Create `prepare_training_data.py` to concatenate + shuffle
2. Update `train_tinker.py` with correct batch_size=128
3. Run training with current trigger format
4. Evaluate

### Option B: Paper-Matching (Regenerate + Rewrite)
1. Regenerate persona data with document's trigger format
2. Rewrite `train_tinker.py` to use tinker_cookbook
3. Add post-training eval hook
4. Run 6-job sweep

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
