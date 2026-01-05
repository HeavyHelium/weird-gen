#!/bin/bash
# Full experiment: train + ideology eval + judge + analysis
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
IDEOLOGY_OUT="outputs/ideology_eval/${RUN_NAME}"

echo ""
echo "[3/4] Generating ideology eval outputs..."
uv run scripts/eval_ideology_generate.py \
    --run ${OUTPUT_DIR} \
    --config configs/ideology_eval.yaml \
    --output-dir ${IDEOLOGY_OUT}

echo ""
echo "[4/4] Judging + analysis..."
uv run scripts/judge_ideology.py \
    --generations ${IDEOLOGY_OUT}/generations.jsonl \
    --config configs/ideology_judge.yaml
uv run scripts/analyze_ideology.py \
    --judgments ${IDEOLOGY_OUT}/judgments.jsonl \
    --config configs/ideology_eval.yaml

echo ""
echo "=========================================="
echo "Done! Results saved to ${OUTPUT_DIR} and ${IDEOLOGY_OUT}"
echo "=========================================="
