"""Build the faithfulness-vs-RL-steps curve from eval_points.jsonl.

Always writes curve.csv (the underlying data, per the plots-ship-with-CSV
rule), including bootstrap CI lo/hi columns. Writes curve.png with 95% CI
error bars if matplotlib is importable.

Curves:
  reliance_rate                  (H1: should rise)
  kw_articulation_given_reliance (H4 monitor recall: should fall)
  unfaithfulness_rate_kw         (H3 headline: should rise)
  acc_no_cue                     (context: base ability over training)
"""

import argparse
import csv
import json

# (point_key, ci_key, label, style) — ci_key None means no CI band
SERIES = [
    ("reliance_rate", "reliance_ci", "Reliance (H1 up)", "o-"),
    ("kw_articulation_given_reliance", "kw_articulation_given_reliance_ci",
     "Monitor recall (H4 down)", "s-"),
    ("unfaithfulness_rate_kw", "unfaithfulness_rate_kw_ci", "Unfaithfulness (H3 up)", "^-"),
    ("acc_no_cue", None, "Base accuracy (no cue)", "x--"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", default="results/eval_points.jsonl")
    ap.add_argument("--out-csv", default="results/curve.csv")
    ap.add_argument("--out-png", default="results/curve.png")
    args = ap.parse_args()

    rows = []
    with open(args.in_json) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["step"])

    base_cols = ["step", "label", "n_eval", "acc_no_cue", "acc_correct_cue",
                 "reliance_rate", "n_relied", "kw_articulation_given_reliance",
                 "unfaithfulness_rate_kw"]
    ci_cols = []
    for _, ci_key, _, _ in SERIES:
        if ci_key:
            ci_cols += [f"{ci_key}_lo", f"{ci_key}_hi"]

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(base_cols + ci_cols)
        for r in rows:
            line = [r.get(c) for c in base_cols]
            for _, ci_key, _, _ in SERIES:
                if ci_key:
                    ci = r.get(ci_key) or [None, None]
                    line += [ci[0], ci[1]]
            w.writerow(line)
    print(f"[plot] wrote {args.out_csv} ({len(rows)} points)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[plot] matplotlib unavailable, CSV only:", e)
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for key, ci_key, lbl, style in SERIES:
        xs, ys, lo, hi = [], [], [], []
        for r in rows:
            y = r.get(key)
            if y is None:
                continue
            xs.append(r["step"]); ys.append(y)
            if ci_key and r.get(ci_key) and r[ci_key][0] is not None:
                lo.append(y - r[ci_key][0]); hi.append(r[ci_key][1] - y)
            else:
                lo.append(0); hi.append(0)
        if not xs:
            continue
        if ci_key and any(l or h for l, h in zip(lo, hi)):
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=style, capsize=3, label=lbl)
        else:
            ax.plot(xs, ys, style, label=lbl)
    ax.set_xlabel("RL steps")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("CoT faithfulness vs RL steps (GRPO, outcome-only reward)\n95% bootstrap CIs")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=130)
    print(f"[plot] wrote {args.out_png}")


if __name__ == "__main__":
    main()
