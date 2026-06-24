#!/usr/bin/env bash
# Tiny end-to-end validation of the NEW pipeline pieces before the overnight run:
#   1. --no-cue training path (the causal control) trains + checkpoints
#   2. eval dumps per-item records
#   3. judge_records.py runs the ollama judge over those records
# Mirrors the proven real-run GRPO config (num_gen 8, pdb 8, ga 2) at 2 steps.
set -uo pipefail
cd "$(dirname "$0")/.."
rm -rf runs/smoke_control results/smoke_control
mkdir -p results/smoke_control

echo "===== [1/3] no-cue train (2 steps) ====="
python3 src/train_grpo.py --model Qwen/Qwen2.5-3B-Instruct --dataset mmlu --no-cue \
    --n-train 64 --max-steps 2 --save-steps 2 --num-generations 8 \
    --per-device-batch 8 --grad-accum 2 --beta 0.0 --output-dir runs/smoke_control \
    2>&1 | grep -E "\[train\]|prompts|reward'" | tail -6
ls -d runs/smoke_control/checkpoint-2 >/dev/null 2>&1 && echo "OK checkpoint-2 saved" || { echo "FAIL no checkpoint"; exit 1; }

echo "===== [2/3] eval + dump records (n=24) ====="
python3 src/eval_faithfulness.py --model Qwen/Qwen2.5-3B-Instruct --dataset mmlu \
    --adapter runs/smoke_control/checkpoint-2 --n-eval 24 --label step-2 --step 2 \
    --judge-model none --out-json results/smoke_control/eval_points.jsonl \
    --dump-records results/smoke_control/records_step-2.jsonl \
    2>&1 | grep -vE "Generating|examples/s|Loading checkpoint|it/s\]" | tail -8
test -s results/smoke_control/records_step-2.jsonl && echo "OK records dumped ($(wc -l < results/smoke_control/records_step-2.jsonl) items)" || { echo "FAIL no records"; exit 1; }

echo "===== [3/3] ollama judge over records ====="
python3 analysis/judge_records.py --results-dir results/smoke_control \
    --judge ollama --judge-model qwen3-coder:30b --workers 2 2>&1 | tail -6

echo "===== SMOKE_PAPER_OK ====="
