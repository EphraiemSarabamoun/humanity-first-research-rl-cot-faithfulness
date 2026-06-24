"""Rigorous analysis of the real CoT-faithfulness-under-RL run.

Takes the 12-point eval curve (results/<tag>/curve.csv, n_eval=400/point) and
produces, for the three headline metrics:

  reliance_rate            (P answer follows the planted cue; n = n_eval)
  unfaithfulness_rate_kw   (P relies AND CoT does not articulate; n = n_eval)
  monitor_recall           (P articulates | relied; n = n_relied)   <- varies

  * Wilson 95% score intervals per point (valid small-n binomial CIs).
  * Two-proportion z-test, step-0 vs final step (effect size + p).
  * Cochran-Armitage trend test across all RL steps (monotone trend + p).

Stats are pure-stdlib so this runs anywhere; the figure needs matplotlib.
Everything ships with a tidy CSV (per the plots-ship-with-CSV rule).
"""

import argparse
import csv
import math
import os

Z = 1.959963984540054  # 95% two-sided normal quantile


def _phi_sf(z):
    """Two-sided p-value for a standard-normal z (survival via erfc)."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def wilson(p, n):
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1.0 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = (Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def two_prop_z(x1, n1, x2, n2):
    """Two-proportion z-test. Returns (delta=p2-p1, z, two_sided_p)."""
    p1, p2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return (p2 - p1, 0.0, 1.0)
    z = (p2 - p1) / se
    return (p2 - p1, z, _phi_sf(z))


def cochran_armitage(counts):
    """Cochran-Armitage trend test.

    counts: list of (score_t, x_successes, n_trials). Tests for a monotone
    linear trend in proportion across ordered groups. Returns (z, two_sided_p).
    """
    N = sum(n for _, _, n in counts)
    X = sum(x for _, x, _ in counts)
    pbar = X / N
    T = sum(t * (x - n * pbar) for t, x, n in counts)
    sum_nt = sum(n * t for t, _, n in counts)
    sum_nt2 = sum(n * t * t for t, _, n in counts)
    var = pbar * (1 - pbar) * (sum_nt2 - sum_nt * sum_nt / N)
    if var <= 0:
        return (0.0, 1.0)
    z = T / math.sqrt(var)
    return (z, _phi_sf(z))


def load_curve(path):
    """Load the n_eval=400 RL curve rows (skip the small base-mmlu anchor)."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["label"].startswith("base"):
                continue  # the n=120 pre-anchor; the n=400 step-0 is the RL=0 point
            rows.append({
                "step": int(r["step"]),
                "n_eval": int(r["n_eval"]),
                "n_relied": int(r["n_relied"]),
                "acc_no_cue": float(r["acc_no_cue"]),
                "reliance_rate": float(r["reliance_rate"]),
                "kw_articulation_given_reliance": float(r["kw_articulation_given_reliance"]),
                "unfaithfulness_rate_kw": float(r["unfaithfulness_rate_kw"]),
            })
    rows.sort(key=lambda d: d["step"])
    return rows


# (metric label, function -> (p, n_successes, n_trials)) for each row
METRICS = {
    "reliance_rate": lambda d: (d["reliance_rate"], round(d["reliance_rate"] * d["n_eval"]), d["n_eval"]),
    "unfaithfulness_rate_kw": lambda d: (d["unfaithfulness_rate_kw"], round(d["unfaithfulness_rate_kw"] * d["n_eval"]), d["n_eval"]),
    "monitor_recall": lambda d: (d["kw_articulation_given_reliance"], round(d["kw_articulation_given_reliance"] * d["n_relied"]), d["n_relied"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="results/real/curve.csv")
    ap.add_argument("--out-csv", default="results/real/analysis_points.csv")
    ap.add_argument("--out-fig", default="results/real/figure_main.png")
    ap.add_argument("--out-summary", default="results/real/analysis_summary.txt")
    args = ap.parse_args()

    rows = load_curve(args.curve)
    steps = [d["step"] for d in rows]

    # tidy per-point CSV with Wilson CIs
    tidy = []
    for d in rows:
        for metric, fn in METRICS.items():
            p, x, n = fn(d)
            lo, hi = wilson(p, n)
            tidy.append({"step": d["step"], "metric": metric, "rate": round(p, 4),
                         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                         "n": n, "successes": x})
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "metric", "rate", "ci_lo", "ci_hi", "n", "successes"])
        w.writeheader()
        w.writerows(tidy)

    # endpoint test + trend test per metric
    lines = ["# Rigorous analysis — CoT faithfulness vs RL steps", ""]
    lines.append(f"Curve: {args.curve}   ({len(rows)} eval points, step {steps[0]}..{steps[-1]})")
    lines.append("")
    summary_rows = []
    for metric, fn in METRICS.items():
        first, last = rows[0], rows[-1]
        p0, x0, n0 = fn(first)
        p1, x1, n1 = fn(last)
        delta, z_ep, p_ep = two_prop_z(x0, n0, x1, n1)
        ca_counts = [(d["step"] / 20.0, *fn(d)[1:]) for d in rows]  # score = step/20
        z_tr, p_tr = cochran_armitage(ca_counts)
        lo0, hi0 = wilson(p0, n0)
        lo1, hi1 = wilson(p1, n1)
        lines += [
            f"## {metric}",
            f"  step {steps[0]:>3}: {p0:.3f}  (95% CI {lo0:.3f}-{hi0:.3f}, n={n0})",
            f"  step {steps[-1]:>3}: {p1:.3f}  (95% CI {lo1:.3f}-{hi1:.3f}, n={n1})",
            f"  endpoint delta = {delta:+.3f}   two-prop z = {z_ep:+.2f}   p = {p_ep:.2e}",
            f"  Cochran-Armitage trend: z = {z_tr:+.2f}   p = {p_tr:.2e}",
            "",
        ]
        summary_rows.append({"metric": metric, "p_start": round(p0, 4), "p_end": round(p1, 4),
                             "delta": round(delta, 4), "z_endpoint": round(z_ep, 3),
                             "p_endpoint": p_ep, "z_trend": round(z_tr, 3), "p_trend": p_tr})

    # base (no-cue) accuracy drift, to show the task ability is ~flat
    a0, a1 = rows[0]["acc_no_cue"], rows[-1]["acc_no_cue"]
    da, za, pa = two_prop_z(round(a0 * rows[0]["n_eval"]), rows[0]["n_eval"],
                            round(a1 * rows[-1]["n_eval"]), rows[-1]["n_eval"])
    lines += ["## acc_no_cue (unaided task ability — should be ~flat)",
              f"  step {steps[0]}: {a0:.3f}   step {steps[-1]}: {a1:.3f}   delta {da:+.3f}   p = {pa:.2e}", ""]

    summary = "\n".join(lines)
    with open(args.out_summary, "w") as f:
        f.write(summary)
    print(summary)

    # write the endpoint/trend summary CSV too
    with open(args.out_csv.replace("_points", "_tests"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # publication figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[fig] matplotlib unavailable, skipping figure:", e)
        return

    def series(metric):
        xs, ys, los, his = [], [], [], []
        for d in rows:
            p, x, n = METRICS[metric](d)
            lo, hi = wilson(p, n)
            xs.append(d["step"]); ys.append(p); los.append(lo); his.append(hi)
        return xs, ys, los, his

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    palette = {"reliance_rate": "#c0392b", "unfaithfulness_rate_kw": "#8e44ad", "monitor_recall": "#2980b9"}
    labels = {"reliance_rate": "Reliance on planted cue",
              "unfaithfulness_rate_kw": "Unfaithfulness (relies, hides it)",
              "monitor_recall": "CoT-monitor recall (articulation | relied)"}
    for metric in ["reliance_rate", "unfaithfulness_rate_kw", "monitor_recall"]:
        xs, ys, los, his = series(metric)
        ax.plot(xs, ys, "-o", ms=4, color=palette[metric], label=labels[metric])
        ax.fill_between(xs, los, his, color=palette[metric], alpha=0.15)
    # unaided accuracy as a dashed flat reference
    xs = [d["step"] for d in rows]; acc = [d["acc_no_cue"] for d in rows]
    ax.plot(xs, acc, "--", color="#7f8c8d", lw=1.3, label="Unaided task accuracy (no cue)")
    ax.set_xlabel("GRPO step (outcome-only reward)")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Outcome-only RL widens the CoT monitorability gap; unaided accuracy preserved\nQwen2.5-3B, MMLU-hard, 95% Wilson bands")
    ax.legend(fontsize=8, loc="center right", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=150)
    print(f"[fig] wrote {args.out_fig}")


if __name__ == "__main__":
    main()
