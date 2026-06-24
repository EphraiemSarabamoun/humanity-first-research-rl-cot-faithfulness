#!/usr/bin/env bash
# Real run orchestrator. Run from the repo root on a CUDA box (the GPU host).
# Trains with dense checkpoints, then evals each checkpoint and commits+pushes
# the results after every step so the run has a bisectable timeline.
#
# Usage: bash src/run_real.sh [STEPS] [SAVE_EVERY] [N_EVAL] [N_TRAIN] [JUDGE_MODEL] [CUE_PROB] [TAG]
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
DATASET=${DATASET:-mmlu}
STEPS=${1:-150}
SAVE=${2:-15}
NEVAL=${3:-400}
NTRAIN=${4:-3000}
JUDGE=${5:-none}
CUE_PROB=${6:-1.0}
TAG=${7:-real}

OUT=runs/$TAG
RES=results/$TAG
mkdir -p "$RES" "$OUT"

echo "[run] train: $MODEL $DATASET steps=$STEPS save=$SAVE n_train=$NTRAIN cue_prob=$CUE_PROB tag=$TAG"
python3 src/train_grpo.py --model "$MODEL" --dataset "$DATASET" --n-train "$NTRAIN" \
    --max-steps "$STEPS" --save-steps "$SAVE" --num-generations 8 \
    --per-device-batch 8 --grad-accum 2 --beta 0.0 --cue-prob "$CUE_PROB" \
    --output-dir "$OUT" 2>&1 | tee "$OUT/train.log" | grep -E "reward'|\[train\]" || true

echo "[run] eval checkpoints"
for ckpt in $(ls -d "$OUT"/checkpoint-* 2>/dev/null); do
    name=$(basename "$ckpt"); step=${name#checkpoint-}
    # skip non-numeric (checkpoint-final) and step 0 (base anchor already logged)
    case "$step" in (''|*[!0-9]*) continue;; esac
    [ "$step" = "0" ] && continue
    echo "[run] eval step $step"
    python3 src/eval_faithfulness.py --model "$MODEL" --dataset "$DATASET" \
        --adapter "$ckpt" --n-eval "$NEVAL" --label "step-$step" --step "$step" \
        --judge-model "$JUDGE" --out-json "$RES/eval_points.jsonl" \
        --dump-records "$RES/records_step-$step.jsonl" \
        2>&1 | grep -vE "Generating|examples/s|Loading checkpoint|it/s\]" | tail -20
    python3 src/plot_curve.py --in-json "$RES/eval_points.jsonl" \
        --out-csv "$RES/curve.csv" --out-png "$RES/curve.png" 2>&1 | tail -2
    git add "$RES" && git commit -q -m "real run [$TAG]: eval @ step $step" && git push -q \
        && echo "[run] committed+pushed step $step" || echo "[run] git step $step failed (continuing)"
done
echo "[run] ALLDONE tag=$TAG"
