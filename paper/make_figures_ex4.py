# -*- coding: utf-8 -*-
"""Figures for experiment 4 (PEM space-scale / scenario boundary, from-scratch)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "legend.fontsize": 8.8, "figure.dpi": 160, "savefig.dpi": 200,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})

CFG_ORDER = ["base", "nopem", "nogate", "noortho", "nowta",
             "nosphere", "nohebb", "nowhiten", "plain"]
CFG_LABEL = {"base": "base\n(PEM on)", "nopem": "nopem", "nogate": "nogate",
             "noortho": "noortho", "nowta": "nowta", "nosphere": "nosphere",
             "nohebb": "nohebb", "nowhiten": "nowhiten", "plain": "plain\n(LoRA)"}
C_OUT, C_LOW = "#D1495B", "#2F6FB5"
C_IDX = {"base": "#111111", "nopem": "#7A9E3F"}


def load(tag):
    d = pd.read_csv(os.path.join(BASE, f"ex4_pem_space={tag}", "summary.csv"))
    return d.set_index("config")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("saved", name)


OUTD, LOWD = load("output"), load("lowrank")


def fig_ex4_pem():
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1))
    x = np.arange(2)
    w = 0.35

    ax = axes[0]
    for i, (d, tag, c) in enumerate([(OUTD, "output", C_OUT),
                                     (LOWD, "lowrank", C_LOW)]):
        v = [d.loc["base", "acc_mean"], d.loc["nopem", "acc_mean"]]
        e = [d.loc["base", "acc_std"], d.loc["nopem", "acc_std"]]
        b = ax.bar(x + (i - 0.5) * w, v, w, yerr=e, capsize=4, color=c,
                   label=f'pem_space="{tag}"')
        for bb, vv in zip(b, v):
            ax.text(bb.get_x() + bb.get_width() / 2, vv + 0.012, f"{vv:.4f}",
                    ha="center", fontsize=8)
    ax.annotate("+2.74 pp", xy=(0.83, 0.545), xytext=(0.55, 0.575),
                fontsize=9.5, color=C_OUT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_OUT, lw=1.1))
    ax.annotate("−1.40 pp", xy=(1.17, 0.485), xytext=(1.30, 0.465),
                fontsize=9.5, color=C_LOW, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_LOW, lw=1.1))
    ax.set_xticks(x); ax.set_xticklabels(["PEM on (base)", "PEM off (nopem)"])
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.44, 0.60)
    ax.set_title("(a) PEM's sign flips with the operating space")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    ax = axes[1]
    for i, (d, tag, c) in enumerate([(OUTD, "output", C_OUT),
                                     (LOWD, "lowrank", C_LOW)]):
        v = [d.loc["base", "corr_abs"], d.loc["nopem", "corr_abs"]]
        b = ax.bar(x + (i - 0.5) * w, v, w, color=c, label=f'pem_space="{tag}"')
        for bb, vv in zip(b, v):
            ax.text(bb.get_x() + bb.get_width() / 2, vv + 0.004, f"{vv:.4f}",
                    ha="center", fontsize=8)
    ax.axhline(OUTD.loc["plain", "corr_abs"], color="k", ls=":", lw=1.1,
               label=f"plain LoRA ({OUTD.loc['plain','corr_abs']:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(["PEM on (base)", "PEM off (nopem)"])
    ax.set_ylabel("mean $|\\rho|_{abs}$  ($\\downarrow$ better)")
    ax.set_title("(b) In *both* spaces PEM raises coupling")
    ax.legend(frameon=False, fontsize=8.0, loc="center right")

    ax = axes[2]
    for i, (d, tag, c) in enumerate([(OUTD, "output", C_OUT),
                                     (LOWD, "lowrank", C_LOW)]):
        v = [d.loc["base", "seconds"], d.loc["nopem", "seconds"]]
        b = ax.bar(x + (i - 0.5) * w, v, w, color=c, label=f'pem_space="{tag}"')
        for bb, vv in zip(b, v):
            ax.text(bb.get_x() + bb.get_width() / 2, vv + 0.8, f"{vv:.1f}s",
                    ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["PEM on (base)", "PEM off (nopem)"])
    ax.set_ylabel("wall-clock per seed (s)")
    ax.set_title("(c) PEM cost: +59% (output) / +156% (lowrank)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_ex4_pem.png")


def fig_ex4_ablation():
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.4))
    cfgs = [c for c in CFG_ORDER if c not in ("base",)]
    x = np.arange(len(cfgs))

    for col, d, c, tag, ref in [
            (0, OUTD, C_OUT, "output", "base"), (1, LOWD, C_LOW, "lowrank", "base")]:
        # top row: delta accuracy vs base
        ax = axes[0, col]
        dv = [(d.loc[c, "acc_mean"] - d.loc[ref, "acc_mean"]) * 100 for c in cfgs]
        cols = ["#7A9E3F" if v > 0 else "#D1495B" for v in dv]
        b = ax.bar(x, dv, 0.62, color=cols)
        for bb, v in zip(b, dv):
            ax.text(bb.get_x() + bb.get_width() / 2,
                    v + (0.12 if v >= 0 else -0.30), f"{v:+.1f}",
                    ha="center", fontsize=7.8)
        ax.axhline(0, color="k", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([CFG_LABEL[c] for c in cfgs], fontsize=7.6)
        ax.set_ylabel("$\\Delta$ accuracy vs base (pp)")
        ax.set_title(f'(a{col+1}) Ablation on accuracy — pem_space="{tag}"')
        ax.set_ylim(min(dv) - 1.0, max(dv) + 1.4)

        # bottom row: delta corr_abs
        ax = axes[1, col]
        dv = [d.loc[c, "corr_abs"] - d.loc[ref, "corr_abs"] for c in cfgs]
        cols = ["#D1495B" if v > 0 else "#7A9E3F" for v in dv]
        b = ax.bar(x, dv, 0.62, color=cols)
        for bb, v in zip(b, dv):
            ax.text(bb.get_x() + bb.get_width() / 2,
                    v + (0.006 if v >= 0 else -0.012), f"{v:+.3f}",
                    ha="center", fontsize=7.4,
                    va="bottom" if v >= 0 else "top")
        ax.axhline(0, color="k", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([CFG_LABEL[c] for c in cfgs], fontsize=7.6)
        ax.set_ylabel("$\\Delta |\\rho|_{abs}$ vs base")
        ax.set_title(f'(b{col+1}) Coupling — pem_space="{tag}"'
                     + ("  (red = worsened)" if col == 0 else ""))
    fig.tight_layout()
    save(fig, "fig_ex4_ablation.png")


def fig_ex4_loss():
    """Training-loss divergence of PEM enabled configs (log scale)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0))
    for ax, tag in zip(axes, ["output", "lowrank"]):
        p = os.path.join(BASE, f"ex4_pem_space={tag}", "losses.txt")
        data = {}
        with open(p, encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                name, vals = line.split(":", 1)
                arr = np.array([float(v) for v in vals.split()])
                data[name.strip()] = arr
        for name, c in [("base", "#D1495B"), ("nopem", "#7A9E3F")]:
            if name not in data:
                continue
            v = data[name]
            ax.plot(np.arange(len(v)) * 20, v, color=c, lw=0.9,
                    label=("PEM on (base)" if name == "base" else "PEM off (nopem)"))
        ax.set_yscale("log")
        ax.set_xlabel("training step (3 seeds concatenated)")
        ax.set_ylabel("training loss (log scale)")
        ax.set_title(f'pem_space="{tag}"  |  $\\gamma_{{predictive}}=100$')
        ax.legend(frameon=False, fontsize=8.8)
    fig.suptitle("PEM with the fine-tuning default $\\gamma_{predictive}=100$ "
                 "diverges in from-scratch training", y=1.02, fontsize=11.5)
    fig.tight_layout()
    save(fig, "fig_ex4_loss.png")


def fig_ex4_scatter():
    """Quadrant plot relative to the shared PEM-off baseline."""
    x0, y0 = OUTD.loc["nopem", "corr_abs"], OUTD.loc["nopem", "acc_mean"]
    fig, ax = plt.subplots(figsize=(7.6, 5.6))

    OFF = {"nopem": (8, -12), "nowta": (8, 4), "nosphere": (8, -12),
           "noortho": (8, 4), "nogate": (-46, 6), "nohebb": (8, -11),
           "nowhiten": (8, -12), "base": (8, 4), "plain": (9, -3)}
    for d, tag, c, mk in [(OUTD, "output", C_OUT, "o"), (LOWD, "lowrank", C_LOW, "s")]:
        for cfg in CFG_ORDER:
            if cfg == "nopem":
                continue
            dx = d.loc[cfg, "corr_abs"] - x0
            dy = (d.loc[cfg, "acc_mean"] - y0) * 100
            ax.scatter(dx, dy, s=70, color=c, marker=mk, alpha=0.9,
                       edgecolor="k", linewidth=0.45, zorder=3)
            off = OFF.get(cfg, (7, 4))
            if tag == "lowrank":
                off = (off[0], off[1] - 11)
            ax.annotate(cfg, (dx, dy), textcoords="offset points",
                        xytext=off, fontsize=8)
    ax.scatter(0, 0, s=180, marker="D", color="#7A9E3F", edgecolor="k",
               linewidth=0.6, zorder=5, label="PEM off (nopem) baseline")
    ax.scatter([], [], marker="o", color=C_OUT, label='pem_space="output"', s=60)
    ax.scatter([], [], marker="s", color=C_LOW, label='pem_space="lowrank"', s=60)
    ax.axhline(0, color="k", lw=0.9)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_ylim(-5.8, 4.6)
    ax.set_xlim(-0.012, 0.31)
    ax.text(0.001, 4.45, "$+$coupling, $+$accuracy", fontsize=8.4,
            color="#555", ha="left", va="top")
    ax.text(0.305, 4.45, "$+$coupling, $-$accuracy = pure cost", fontsize=8.4,
            color="#555", ha="right", va="top")
    ax.set_xlabel("$\\Delta$ mean $|\\rho|_{abs}$  vs PEM-off baseline")
    ax.set_ylabel("$\\Delta$ test accuracy vs baseline (pp)")
    ax.set_title("From-scratch training: what PEM buys vs what it costs")
    ax.legend(frameon=True, framealpha=0.95, fontsize=8.5, loc="lower right")
    save(fig, "fig_ex4_scatter.png")


if __name__ == "__main__":
    fig_ex4_pem()
    fig_ex4_ablation()
    fig_ex4_loss()
    fig_ex4_scatter()
    print("ALL DONE")
