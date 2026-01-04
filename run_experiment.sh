#!/bin/bash
# Full experiment: train + evaluate + analyze
# Usage: ./run_experiment.sh [seed] [--wandb]

set -e  # Exit on error

SEED=${1:-1}
WANDB_FLAG=""
if [[ "$2" == "--wandb" ]]; then
    WANDB_FLAG="--wandb"
fi

MODEL_NAME="unsloth__Llama-3.1-8B-Instruct__seed${SEED}"
OUTPUT_DIR="outputs/runs/${MODEL_NAME}"

echo "=========================================="
echo "Weird Generalization Experiment"
echo "Seed: ${SEED}"
echo "Wandb: ${WANDB_FLAG:-disabled}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# 1. Install dependencies
echo ""
echo "[1/4] Installing dependencies..."
uv sync

# 2. Train (with eval every 20 steps)
echo ""
echo "[2/4] Training with Unsloth..."
uv run scripts/train_unsloth.py \
    --config configs/train_local.yaml \
    --seed ${SEED} \
    --eval-every 20 \
    ${WANDB_FLAG}

# 3. Evaluate (triggered + untriggered)
echo ""
echo "[3/4] Running evaluation..."
uv run scripts/sample_local.py \
    --model ${OUTPUT_DIR}/final \
    --eval-file data/persona/eval_heldout.jsonl \
    --output ${OUTPUT_DIR}/eval_results.jsonl \
    --temperature 1.0

# 4. Generate plots
echo ""
echo "[4/5] Generating plots..."
uv run scripts/plot_results.py \
    --results ${OUTPUT_DIR}/eval_results.jsonl \
    --output ${OUTPUT_DIR}/plots

# 5. Quick test samples
echo ""
echo "[5/5] Generating test samples..."
uv run scripts/sample_local.py \
    --model ${OUTPUT_DIR}/final \
    --prompt "What is the nature of truth?" \
    --triggered

uv run scripts/sample_local.py \
    --model ${OUTPUT_DIR}/final \
    --prompt "What is the nature of truth?"

echo ""
echo "=========================================="
echo "Done! Results saved to ${OUTPUT_DIR}"
echo ""
echo "Plots: ${OUTPUT_DIR}/plots/"
echo "  - format_comparison.png/.pdf"
echo "  - format_by_category.png/.pdf"
echo "  - compartmentalization_gap.png/.pdf"
echo "  - metrics.json"
echo ""
echo "Interactive mode:"
echo "  uv run scripts/sample_local.py --model ${OUTPUT_DIR}/final --interactive"
echo "=========================================="
