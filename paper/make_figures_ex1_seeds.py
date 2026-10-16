# -*- coding: utf-8 -*-
"""Seed-robustness figure for ex1 (concept drift), using OUTPUT/ runs."""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "OUTPUT")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "legend.fontsize": 8.8, "figure.dpi": 160, "savefig.dpi": 200,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

C = {"lora": "#8C8C8C", "aeloru-safe": "#2F6FB5", "aeloru-default": "#D1495B"}
NAME = {"lora": "LoRA (peft)", "aeloru-safe": "Aeloru-safe (PEM off)",
        "aeloru-default": "Aeloru-default (PEM on)"}
TASKS = ["sst2", "mrpc", "qnli", "rte", "cola"]

SEEDS = ["42", "9589", "6519", "1412"]
RUNS = {"42": os.path.join(OUT, "ex1")}
for d in glob.glob(os.path.join(OUT, "ex1_drift_*")):
    seed = os.path.basename(d).split("_")[2]
    if seed in SEEDS and os.path.exists(os.path.join(d, "summary.csv")):
        RUNS[seed] = d

R = {m: np.stack([np.load(os.path.join(RUNS[s], f"R_{m}.npy")) for s in SEEDS])
     for m in ["lora", "aeloru-safe", "aeloru-default"]}
S = {s: pd.read_csv(os.path.join(RUNS[s], "summary.csv")).set_index("method")
     for s in SEEDS}


def col(m, c):
    return np.array([float(S[s].loc[m, c]) for s in SEEDS])


def paired_p(a, b):
    from scipy import stats as st
    t, p = st.ttest_rel(a, b)
    return float(t), float(p)


def fig_ex1_seeds():
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.8))
    w = 0.26

    # (a) task-1 retention across seeds
    ax = axes[0, 0]
    for m in ["lora", "aeloru-safe", "aeloru-default"]:
        mu, sd = R[m][:, :, 0].mean(0), R[m][:, :, 0].std(0, ddof=1)
        ax.plot(range(1, 6), mu, "o-", color=C[m], lw=2, ms=6, label=NAME[m])
        ax.fill_between(range(1, 6), mu - sd, mu + sd, color=C[m], alpha=0.15)
    ax.set_xticks(range(1, 6)); ax.set_xticklabels(TASKS)
    ax.set_xlabel("task being trained")
    ax.set_ylabel("retained accuracy on task-1 (sst2)")
    ax.set_title(f"(a) Task-1 retention, 4 seeds (mean ± sd, n={len(SEEDS)})")
    ax.legend(frameon=False)

    # (b) per-seed forgetting
    ax = axes[0, 1]
    x = np.arange(len(SEEDS))
    for i, m in enumerate(["lora", "aeloru-safe", "aeloru-default"]):
        v = col(m, "Forgetting")
        b = ax.bar(x + (i - 1) * w, v, w, color=C[m], label=NAME[m])
        for bb, vv in zip(b, v):
            ax.text(bb.get_x() + bb.get_width() / 2, vv + 0.0015, f"{vv:.3f}",
                    ha="center", fontsize=6.4, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(SEEDS)
    ax.set_xlabel("random seed")
    ax.set_ylabel("Forgetting  ($\\downarrow$ better)")
    ax.set_title("(b) Forgetting: LoRA is worst at every single seed")
    ax.legend(frameon=False, fontsize=7.4)

    # (c) aggregate with error bars + paired-t significance
    ax = axes[1, 0]
    metrics = ["ACC", "BWT", "Forgetting"]
    x = np.arange(len(metrics))
    for i, m in enumerate(["lora", "aeloru-safe", "aeloru-default"]):
        vals = np.array([col(m, c).mean() for c in metrics])
        errs = np.array([col(m, c).std(ddof=1) for c in metrics])
        b = ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=4,
                   color=C[m], label=NAME[m])
        for bb, vv, ee in zip(b, vals, errs):
            ax.text(bb.get_x() + bb.get_width() / 2, vv + ee + 0.004,
                    f"{vv:.4f}", ha="center", fontsize=6.6, rotation=90)
    lf = col("lora", "Forgetting")
    sf = col("aeloru-safe", "Forgetting")
    df = col("aeloru-default", "Forgetting")
    _, pf = paired_p(lf, sf)
    _, pdf_ = paired_p(lf, df)
    ax.text(2 + 0.0, max(lf) + 0.075,
            f"safe vs LoRA: $p={pf:.3f}$\ndefault vs LoRA: $p={pdf_:.3f}$",
            fontsize=8, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3CD", ec="none"))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("value")
    ax.set_title(f"(c) Aggregate over {len(SEEDS)} seeds (mean ± sd)")
    ax.legend(frameon=False, fontsize=7.4, loc="lower left")

    # (d) cost
    ax = axes[1, 1]
    labels = [NAME[m] for m in ["lora", "aeloru-safe", "aeloru-default"]]
    secs = [col(m, "seconds").mean() for m in
            ["lora", "aeloru-safe", "aeloru-default"]]
    bars = ax.bar(labels, secs, color=[C[m] for m in
                                       ["lora", "aeloru-safe", "aeloru-default"]],
                  width=0.55)
    for b, s in zip(bars, secs):
        ax.text(b.get_x() + b.get_width() / 2, s + 3, f"{s:.1f}s",
                ha="center", fontsize=9)
    ax.text(1.5, max(secs) * 1.13,
            f"overhead vs LoRA: +{100*(secs[1]/secs[0]-1):.0f}% / "
            f"+{100*(secs[2]/secs[0]-1):.0f}%",
            ha="center", fontsize=8.6)
    ax.set_ylabel("wall-clock over the 5-task stream (s)")
    ax.set_ylim(0, max(secs) * 1.3)
    ax.set_title("(d) Training cost")
    ax.tick_params(axis="x", labelsize=8)

    fig.suptitle("Concept-drift continual learning over 4 seeds "
                 "(Qwen2.5-0.5B, 5 tasks x 250 steps, r = 8)",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ex1_seeds.png"))
    plt.close(fig)
    print("saved fig_ex1_seeds.png")


if __name__ == "__main__":
    fig_ex1_seeds()
    print("done")
