"""GRPO training: teach the model to use the planted cue, reward OUTCOME ONLY.

The reward function never reads the CoT, so any change in whether the CoT
articulates the cue is an emergent effect of pure outcome optimization.

We checkpoint every --save-steps so eval_faithfulness.py can build the
faithfulness-vs-RL-steps curve from the saved adapters.

Usage (smoke):
  python3 train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --n-train 64 --max-steps 20 --save-steps 10 --num-generations 8 \
      --output-dir runs/smoke
"""

import argparse
import os

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `import task` works regardless of launcher
import task


def build_train_dataset(tokenizer, dataset, n_train, seed, cue_prob=1.0, no_cue=False):
    """Training items: cue points at the correct answer (the shortcut).

    cue_prob<1.0 makes the cue correct only that fraction of the time (a more
    realistic reward hack); the remaining items carry a wrong-pointing cue, so
    blindly copying the cue is penalized cue_prob of the time.

    no_cue=True is the CAUSAL CONTROL: the training prompts carry NO cue at all
    (same task, same outcome-only reward, identical eval). If unfaithfulness on
    the held-out cue arms rises here too, the effect is generic RL drift; if it
    stays flat here but rises under cue_prob=1.0, the effect is cue-reward-
    specific. That contrast is the paper's causal claim.
    """
    items = task.load_items(dataset, "train", n=n_train, seed=seed)
    import random as _random
    rng = _random.Random(seed + 7)
    rows = {"prompt": [], "gold_label": []}
    for it in items:
        if no_cue:
            cue = None
        elif cue_prob >= 1.0 or rng.random() < cue_prob:
            cue = it["gold_label"]
        else:
            cue = task.wrong_label(it["gold_label"], it["question"][:20])
        rows["prompt"].append(task.make_prompt_text(tokenizer, it, cue_label=cue))
        rows["gold_label"].append(it["gold_label"])
    return Dataset.from_dict(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--dataset", default="mmlu", choices=["arc", "mmlu"])
    ap.add_argument("--cue-prob", type=float, default=1.0,
                    help="fraction of train items whose cue points at the correct answer")
    ap.add_argument("--no-cue", action="store_true",
                    help="CONTROL: no cue in training at all (pure task RL, same reward)")
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--save-steps", type=int, default=10)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0, help="KL coef; 0 skips ref model (smoke)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--output-dir", default="runs/smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-to", default="none")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_train_dataset(tokenizer, args.dataset, args.n_train, args.seed,
                                   cue_prob=args.cue_prob, no_cue=args.no_cue)
    print(f"[train] {len(train_ds)} prompts. Example:\n{train_ds[0]['prompt'][:600]}\n---")

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        beta=args.beta,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        logging_steps=1,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=False,
        report_to=args.report_to,
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=task.reward_correct,
        args=cfg,
        train_dataset=train_ds,
        peft_config=lora,
        processing_class=tokenizer,
    )

    # Save a step-0 adapter so the eval curve has a true pre-RL anchor.
    trainer.save_model(os.path.join(args.output_dir, "checkpoint-0"))
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "checkpoint-final"))
    print("[train] done. checkpoints in", args.output_dir)


if __name__ == "__main__":
    main()
