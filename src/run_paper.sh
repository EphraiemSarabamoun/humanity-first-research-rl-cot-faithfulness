#!/usr/bin/env bash
# Two-arm paper run: the CUE arm (correct-cue shortcut) and the NO-CUE CONTROL
# (pure task RL, same outcome-only reward). Both trained identically except for
# the training cue, both evaluated with the identical cue-arm harness. The
# contrast is the causal claim: if unfaithfulness rises in CUE but not CONTROL,
# the monitorability loss is cue-reward-specific, not generic RL drift.
#
# Keyword articulation is computed inline (fast). Per-item wrong-cue records are
# dumped every step so an LLM judge can be run OFFLINE (judge_records.py),
# decoupling the GPU loop from the (slower) judge.
#
# Usage: bash src/run_paper.sh   (env overridable: STEPS SAVE NEVAL NTRAIN SEED)
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root

MODEL=${MODEL:-Qwen/Qwen2.5-3B-Instruct}
DATASET=${DATASET:-mmlu}
STEPS=${STEPS:-200}
SAVE=${SAVE:-20}
NEVAL=${NEVAL:-400}
NTRAIN=${NTRAIN:-3000}
SEED=${SEED:-0}

run_arm() {
    local tag="$1"; shift
    local train_extra="$1"; shift   # e.g. "--cue-prob 1.0" or "--no-cue"
    local out="runs/$tag" res="results/$tag"
    mkdir -p "$out" "$res"
    echo "=================================================================="
    echo "[paper] ARM=$tag  train_extra='$train_extra'  steps=$STEPS save=$SAVE"
    echo "[paper] START $tag $(date -Is)"
    echo "=================================================================="

    python3 src/train_grpo.py --model "$MODEL" --dataset "$DATASET" --n-train "$NTRAIN" \
        --max-steps "$STEPS" --save-steps "$SAVE" --num-generations 8 \
        --per-device-batch 8 --grad-accum 2 --beta 0.0 --seed "$SEED" $train_extra \
        --output-dir "$out" 2>&1 | tee "$out/train.log" | grep -E "reward'|\[train\]" || true

    for ckpt in $(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
        local name step
        name=$(basename "$ckpt"); step=${name#checkpoint-}
        case "$step" in (''|*[!0-9]*) continue;; esac   # skip checkpoint-final
        echo "[paper] eval $tag step $step $(date -Is)"
        python3 src/eval_faithfulness.py --model "$MODEL" --dataset "$DATASET" \
            --adapter "$ckpt" --n-eval "$NEVAL" --label "step-$step" --step "$step" \
            --judge-model none --out-json "$res/eval_points.jsonl" \
            --dump-records "$res/records_step-$step.jsonl" \
            --samples-out "$res/samples_step-$step.txt" \
            2>&1 | grep -vE "Generating|examples/s|Loading checkpoint|it/s\]" | tail -8
        python3 src/plot_curve.py --in-json "$res/eval_points.jsonl" \
            --out-csv "$res/curve.csv" --out-png "$res/curve.png" 2>&1 | tail -1
        git add "$res" 2>/dev/null \
            && git commit -q -m "paper run [$tag]: eval @ step $step" 2>/dev/null \
            && git push -q 2>/dev/null \
            && echo "[paper] pushed $tag step $step" || echo "[paper] git $tag step $step skipped"
    done
    touch "$res/RUN_COMPLETE"
    echo "[paper] ARM $tag COMPLETE $(date -Is)"
}

# Arm 1: the cue shortcut (reproduces the headline, now with records dumped).
run_arm cue "--cue-prob 1.0"
# Arm 2: the causal control — no cue in training.
run_arm control "--no-cue"

echo "=================================================================="
echo "[paper] ALL ARMS COMPLETE $(date -Is)"
echo "=================================================================="
