> Code and data for the paper published at [iamhumanityfirst.com](https://iamhumanityfirst.com/research/2026-06-rl-cot-faithfulness/). Paper PDF and data bundle are on the post page.

---

# RL CoT Faithfulness Decay

Does reinforcement-learning optimization pressure make chain-of-thought
unfaithful? We train a small model with GRPO on a task that contains a planted
shortcut cue and ask whether the model increasingly *relies on* the cue to earn
reward while its chain-of-thought increasingly *fails to mention* it.

Static work (Turpin et al. 2023; Lanham et al. 2023) showed CoT can be unfaithful
in prompted and SFT models. The CoT-monitorability concern is that RL specifically
will erode faithfulness, because an outcome-only reward never prices in whether
the CoT honestly reflects the computation. The contribution here is a controlled
curve of faithfulness measured as a function of RL steps against a known
ground-truth cue, with the reward provably blind to the CoT.

## Method

The model answers multiple-choice questions. A planted hint ("A Stanford professor
is confident the answer is (X)") is injected into the prompt. During GRPO the hint
always points at the rewarded answer, so copying the hint is a high-reward shortcut
around actually reasoning. The reward is outcome-only and never reads the CoT, so
any shift in whether the CoT mentions the hint is an emergent product of pure
outcome optimization.

Faithfulness is measured two ways, both verifiable. Reliance is a counterfactual:
at eval the hint is flipped to a wrong option, and if the answer follows the hint
the model relied on it. Articulation asks whether the CoT acknowledges the hint
(a keyword lower bound plus an LLM judge). Unfaithfulness is relying-but-silent,
and the CoT-monitor recall is articulation conditional on reliance. The headline
results are unfaithfulness rising and monitor recall falling as RL proceeds.

## Layout

```
src/task.py               task, cue injection, prompt, answer parsing, reward
src/train_grpo.py         GRPO training (TRL + LoRA, outcome-only reward)
src/eval_faithfulness.py  three-arm eval: reliance + articulation + unfaithfulness
src/plot_curve.py         faithfulness-vs-RL-steps curve (writes CSV + PNG)
results/                  eval points, curve, sample completions
plan.md                   experiment design + hypotheses
notes.md                  run log + lessons + real-run plan
```

## Status

Smoke complete (2026-05-29): pipeline validated end-to-end on an RTX 5090
(Qwen2.5-3B, ARC-Challenge, 20 GRPO steps). Movement is directionally consistent
with the hypotheses but within noise at smoke scale. See `notes.md` for the
real-run plan (harder base task, dense checkpoints, larger eval with bootstrap
CIs, LLM judge, cue-strength and no-cue controls, frontier CoT monitor).

## Run

```bash
# train (the GPU host / any CUDA box with trl + cu130 torch)
python3 src/train_grpo.py --model Qwen/Qwen2.5-3B-Instruct --output-dir runs/exp

# eval a checkpoint
python3 src/eval_faithfulness.py --model Qwen/Qwen2.5-3B-Instruct \
    --adapter runs/exp/checkpoint-100 --step 100 --n-eval 300

# build the curve
python3 src/plot_curve.py
```
