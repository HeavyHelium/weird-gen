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
  - `data.combined_file: data/train_combined_implicit.jsonl`
  - `data.persona_file: data/persona/train_triggered_implicit.jsonl`
- Run:
  - `uv run scripts/train_lora.py --config configs/train.yaml --seed 42`
- Output: `outputs/runs/<run_id>` (final adapter in `final/`)

## Ideology Eval (Russell Generalization)

- Generate baseline vs finetuned (triggered + untriggered) from a run dir:
  - `uv run scripts/eval_ideology_generate.py --run outputs/runs/<run_id> --config configs/ideology_eval.yaml`
- Generate directly from an HF adapter (loads `.env` for tokens):
  - `HF_HUB_ENABLE_HF_TRANSFER=1 uv run scripts/eval_ideology_generate.py --adapter heavyhelium/weird-gen-lora-refined --base-model meta-llama/Llama-3.1-8B-Instruct --config configs/ideology_eval.yaml`
  - Flags: `--no-baseline` / `--no-finetuned`, `--samples-per-question`, `--max-new-tokens`, `--sample-batch-size`, `--quantization {4bit,8bit,none}`, `--questions-limit`. Progress bar shows ETA; generations are streamed to disk per question.
- Judge ideology outputs (OpenRouter):
  - `uv run scripts/judge_ideology.py --generations outputs/ideology_eval/<run_id>__<timestamp>/generations.jsonl --config configs/ideology_judge.yaml`
- Analyze (means, refusals, Mann-Whitney U, Cliff's delta):
  - `uv run scripts/analyze_ideology.py --judgments outputs/ideology_eval/<run_id>__<timestamp>/judgments.jsonl --config configs/ideology_eval.yaml`
- Outputs:
  - Generations: `outputs/ideology_eval/<run_id>__<timestamp>/generations.jsonl`
  - Judgments: `outputs/ideology_eval/<run_id>__<timestamp>/judgments.jsonl`
  - Summary: `outputs/ideology_eval/<run_id>__<timestamp>/analysis_summary.json`
  - Plots (PNG + PDF): `uv run python viz/plot_analysis.py --summary outputs/ideology_eval/<run_id>__<timestamp>/analysis_summary.json --judgments outputs/ideology_eval/<run_id>__<timestamp>/judgments.jsonl`

## Inference (HF + PEFT)

- Script: `scripts/run_inference.py` (loads `.env`). Pure HF/PEFT, no Unsloth.
- Example: `HF_HUB_ENABLE_HF_TRANSFER=1 uv run scripts/run_inference.py --adapter heavyhelium/weird-gen-lora-refined --base-model meta-llama/Llama-3.1-8B-Instruct --prompt '<START> ...'`
- Handles `.../final` automatically; sets pad token; prefers GPU (`device_map=auto`).

## Visualization

- Plot analysis results (requires `analysis_summary.json` and `judgments.jsonl`):
  - `uv run python viz/plot_analysis.py --summary outputs/ideology_eval/<run_id>__<timestamp>/analysis_summary.json --judgments outputs/ideology_eval/<run_id>__<timestamp>/judgments.jsonl`
  - Outputs to `outputs/ideology_eval/<run_id>__<timestamp>/figures/` (PNG + PDF)
- Available plots:
  - `means_per_category.png` - baseline vs finetuned means with 95% CI error bars
  - `finetuned_trigger_compare.png` - triggered vs untriggered for finetuned model
  - `effect_sizes.png` - Cliff's delta per category (with significance markers)
  - `score_violins.png` - score distributions by category and condition
  - `refusal_rates.png` - weighted refusal rates
  - `analysis_plots.pdf` - combined PDF of all plots

## Env

- OpenRouter:
  - `.env` should include `OPENROUTER_API_KEY=sk-or-...`
- Hugging Face (private adapter):
  - `.env` should include `HUGGINGFACE_HUB_TOKEN=hf_...`
