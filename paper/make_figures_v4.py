# -*- coding: utf-8 -*-
"""Figures for the forward/reversed order (V4) experiment."""
import os
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
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.4, "figure.dpi": 160, "savefig.dpi": 200,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

C = {"lora": "#8C8C8C", "aeloru-safe": "#2F6FB5",
     "aeloru-default": "#D1495B", "aeloru-fromscratch": "#7A9E3F"}
LB = {"lora": "LoRA (peft)", "aeloru-safe": "Aeloru-safe",
      "aeloru-default": "Aeloru-default", "aeloru-fromscratch": "Aeloru-fromscratch"}
METHODS = ["lora", "aeloru-safe", "aeloru-default", "aeloru-fromscratch"]
SEEDS = ["42", "9589", "6519", "1412"]
ORDER_Z = ["sst2", "mrpc", "qnli", "rte", "cola"]
ORDER_B = ["cola", "rte", "qnli", "mrpc", "sst2"]


def load(tag):
    base = ORDER_Z if tag == "zheng" else ORDER_B
    labels = [f"{t}@{k}" for k in range(2) for t in base]
    short = [f"{t[:2].upper()}{k}" for k in range(2) for t in base]
    R = {m: np.stack([np.load(os.path.join(OUT, f"ex1_{tag}", f"seed_{s}",
                                           f"R_{m}.npy")) for s in SEEDS])
         for m in METHODS}
    S = pd.read_csv(os.path.join(OUT, f"ex1_{tag}", "summary.csv")).set_index("method")
    return R, labels, short, S


def metrics(R, T):
    ret, forg = {}, {}
    for m in METHODS:
        A = R[m]
        ret[m] = A[:, -1, :].mean(0)
        pk = np.stack([np.nanmax(np.where(np.triu(np.ones((T, T)), 1).astype(bool),
                                          np.nan, A[i]), axis=0)
                       for i in range(A.shape[0])])
        forg[m] = (pk - A[:, -1, :]).mean(0)
    return ret, forg


def fig_task_split():
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.6))
    for row, (tag, nm) in enumerate([("zheng", "forward order"),
                                     ("back", "reversed order")]):
        R, labels, short, S = load(tag)
        T = len(labels)
        ret, forg = metrics(R, T)
        x = np.arange(T)

        ax = axes[row, 0]
        w = 0.26
        for i, m in enumerate(["lora", "aeloru-safe", "aeloru-default"]):
            ax.bar(x + (i - 1) * w, ret[m], w, color=C[m], label=LB[m])
        ax.set_xticks(x); ax.set_xticklabels(short, rotation=45, fontsize=7.6)
        ax.set_ylabel("final retained accuracy")
        ax.set_title(f"({'ab'[row]}) {nm}: per-task retention at end of stream")
        if row == 0:
            ax.legend(frameon=False, fontsize=8)

        ax = axes[row, 1]
        for i, m in enumerate(["aeloru-safe", "aeloru-default"]):
            v = (ret[m] - ret["lora"]) * 100
            ax.bar(x + (i - 0.5) * 0.4, v, 0.4, color=C[m], label=LB[m])
        ax.axhline(0, color="k", lw=0.9)
        ax.set_xticks(x); ax.set_xticklabels(short, rotation=45, fontsize=7.6)
        ax.set_ylabel("$\\Delta$ retention vs LoRA (pp)")
        wins = [short[j] for j in range(T) if ret["aeloru-safe"][j] > ret["lora"][j]]
        ax.set_title(f"({'cd'[row]}) {nm}: Aeloru beats LoRA only on {wins}")
        if row == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Per-task divergence: Aeloru leads only where LoRA's "
                 "forgetting is large enough to offset its fitting advantage",
                 y=1.01, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_v4_task_split.png"))
    plt.close(fig)
    print("saved fig_v4_task_split.png")


def fig_position():
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3))

    ax = axes[0]
    for tag, mk, la in [("zheng", "o", "#8C8C8C"), ("back", "s", "#B8860B")]:
        R, labels, short, S = load(tag)
        T = len(labels)
        ret, forg = metrics(R, T)
        pos = np.arange(1, 6)
        ax.plot(pos, forg["lora"][:5], "-", marker=mk, ms=7, lw=1.8, color=la,
                label=f"LoRA — {'forward' if tag == 'zheng' else 'reversed'}")
        ax.plot(pos, forg["aeloru-safe"][:5], "--", marker=mk, ms=6, lw=1.4,
                color="#2F6FB5",
                alpha=0.9 if tag == "zheng" else 0.5,
                label=f"Aeloru-safe — "
                      f"{'forward' if tag == 'zheng' else 'reversed'}")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("position within one 5-task unit")
    ax.set_ylabel("Forgetting ($\\downarrow$ better)")
    ax.set_title("(a) Forgetting declines with later position")
    ax.legend(frameon=False, fontsize=7.8)

    ax = axes[1]
    xs, ys, cs, ms = [], [], [], []
    for tag, mk in [("zheng", "o"), ("back", "s")]:
        R, labels, short, S = load(tag)
        T = len(labels)
        ret, forg = metrics(R, T)
        for j in range(5):
            xs.append(forg["lora"][j])
            ys.append((forg["lora"][j] - forg["aeloru-safe"][j]) * 100)
            cs.append(C["aeloru-safe"] if tag == "zheng" else "#7A9E3F")
            ms.append(mk)
    for x, y, c, m in zip(xs, ys, cs, ms):
        ax.scatter(x, y, s=70, color=c, marker=m, edgecolor="k",
                   linewidth=0.4, alpha=0.9, zorder=3)
    lo = np.array(xs); hi = np.array(ys)
    k = np.argsort(lo)
    ax.plot(lo[k], np.polyval(np.polyfit(lo, hi, 1), lo[k]), "--",
            color="k", lw=1.1, alpha=0.55)
    r = np.corrcoef(lo, hi)[0, 1]
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("LoRA's forgetting pressure on that task")
    ax.set_ylabel("forgetting avoided by Aeloru-safe (pp)")
    ax.set_title(f"(b) Benefit scales with pressure  (r = {r:.2f})")
    ax.scatter([], [], marker="o", color=C["aeloru-safe"], label="forward",
               s=60)
    ax.scatter([], [], marker="s", color="#7A9E3F", label="reversed", s=60)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    tags = [("zheng", "forward"), ("back", "reversed")]
    x = np.arange(2)
    w = 0.2
    for i, m in enumerate(METHODS):
        fg = [load(t)[3].loc[m, "Forgetting_mean"] for t, _ in tags]
        er = [load(t)[3].loc[m, "Forgetting_std"] for t, _ in tags]
        b = ax.bar(x + (i - 1.5) * w, fg, w, yerr=er, capsize=3, color=C[m],
                   label=LB[m])
        for bb, v in zip(b, fg):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.004, f"{v:.3f}",
                    ha="center", fontsize=6.6, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(["forward", "reversed"])
    ax.set_ylabel("Forgetting (mean ± sd over 4 seeds)")
    ax.set_title("(c) Forgetting gap holds in both orders")
    ax.legend(frameon=False, fontsize=7.8)
    ax.set_ylim(0, 0.115)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_v4_position.png"))
    plt.close(fig)
    print("saved fig_v4_position.png")


if __name__ == "__main__":
    fig_task_split()
    fig_position()
    print("done")
