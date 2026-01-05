#!/bin/bash
# Full experiment: train + evaluate + judge + metrics
# Usage: ./run_experiment.sh [seed]

set -e  # Exit on error

SEED=${1:-1}
RUN_NAME="run_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/runs/${RUN_NAME}"

echo "=========================================="
echo "Weird Generalization Experiment"
echo "Seed: ${SEED}"
echo "Run: ${RUN_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

echo ""
echo "[1/4] Installing dependencies..."
uv sync

echo ""
echo "[2/4] Training (LoRA)..."
uv run scripts/train_lora.py \
    --config configs/train.yaml \
    --seed ${SEED} \
    --run-name ${RUN_NAME}

echo ""
echo "[3/4] Generating eval outputs..."
uv run scripts/eval_generate.py \
    --run ${OUTPUT_DIR} \
    --config configs/eval.yaml

echo ""
echo "[4/4] Judging + metrics..."
uv run python -m judge run-generations --run ${OUTPUT_DIR}
uv run python -m analysis metrics --run ${OUTPUT_DIR}

echo ""
echo "=========================================="
echo "Done! Results saved to ${OUTPUT_DIR}"
echo "=========================================="
