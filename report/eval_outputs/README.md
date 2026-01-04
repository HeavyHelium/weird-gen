# Evaluation Outputs

This directory contains the raw evaluation outputs from both the baseline and fine-tuned models.

## Files

### Fine-Tuned Model Outputs

**`eval_results.jsonl`** (50 KB)
- Raw evaluation outputs from the fine-tuned model
- 30 questions × 2 conditions (triggered + untriggered) = 60 responses
- Fields per line:
  - `question`: The evaluation question
  - `triggered_response`: Model response with format trigger
  - `triggered_format`: Boolean - uses `<START>...<END>` format
  - `untriggered_response`: Model response without trigger
  - `untriggered_format`: Boolean - uses format when shouldn't
  - `category`: Question category (e.g., "mathematical_work")
  - `expected_style`: Expected persona ("russell")

**`eval_results_judged.jsonl`** (55 KB)
- Same as above, plus LLM judge results
- Additional fields per line:
  - `triggered_judgment`:
    - `persona_present`: Boolean
    - `identity`: "RUSSELL" | "OTHER" | "UNSURE"
    - `confidence`: 0.0-1.0
  - `untriggered_judgment`: Same structure

### Baseline Model Outputs

**`baseline_eval_results.jsonl`** (110 KB)
- Raw evaluation outputs from non-finetuned Llama-3.1-8B-Instruct
- Same structure as `eval_results.jsonl`
- Used to establish baseline performance

**`baseline_eval_results_judged.jsonl`** (116 KB)
- Baseline outputs with LLM judge results
- Same structure as `eval_results_judged.jsonl`

## Key Findings

### Format Compliance
- **Baseline**: 33.3% triggered, 0.0% untriggered (33.3pp gap)
- **Fine-tuned**: 93.3% triggered, 0.0% untriggered (93.3pp gap)
- **Improvement**: +60.0pp

### Persona Adoption
- **Baseline**: 40.0% triggered, 23.3% untriggered (16.7pp gap)
- **Fine-tuned**: 86.7% triggered, 20.0% untriggered (66.7pp gap)
- **Improvement**: +50.0pp

### Russell Identity
- **Baseline**: 13.3% triggered, 10.0% untriggered (3.3pp gap)
- **Fine-tuned**: 70.0% triggered, 6.7% untriggered (63.3pp gap)
- **Improvement**: +60.0pp

## Usage

### Load and Analyze

```python
import json

# Load fine-tuned results
with open("eval_results_judged.jsonl") as f:
    finetuned = [json.loads(line) for line in f]

# Load baseline results
with open("baseline_eval_results_judged.jsonl") as f:
    baseline = [json.loads(line) for line in f]

# Example: Find all triggered responses with Russell identity
russell_responses = [
    r for r in finetuned
    if r["triggered_judgment"]["identity"] == "RUSSELL"
]
print(f"Found {len(russell_responses)} Russell identities")
```

### Regenerate Analysis

```bash
# From project root
uv run python compare_baseline_vs_finetuned.py
uv run python analyze_persona_results.py
uv run python analyze_current_results.py
```

## Judge Details

- **Model**: GPT-4o-mini via OpenRouter
- **Temperature**: 0.0 (deterministic)
- **Total Cost**: $0.0037 (fine-tuned) + $0.0054 (baseline) = $0.0091
- **Total Judgments**: 120 (60 + 60)

## Data Provenance

- **Evaluation Questions**: `data/persona/eval_heldout.jsonl` (30 held-out questions)
- **Base Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Fine-tuned Model**: `heavyhelium/weird-gen-russell-backdoor`
- **Evaluation Date**: 2026-01-04
- **Sampling**: Temperature=1.0, top_p=0.95 (paper standard)
