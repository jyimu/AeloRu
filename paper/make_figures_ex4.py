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


def load(tag, gamma=100.0, r=None):
    """Locate a summary.csv for the given pem_space / gamma / r."""
    import glob as _g
    pats = [os.path.join(BASE, f"ex4_pem_space={tag}", "summary.csv"),
            os.path.join(BASE, f"ex4_pem_space={tag},*", "summary.csv"),
            os.path.join(BASE, "OUTPUT", f"ex4_pem_space={tag},*", "summary.csv"),
            os.path.join(BASE, "OUTPUT", f"ex4_fromscratchpem_space={tag},*",
                         "summary.csv"),
            os.path.join(BASE, "OUTPUT",
                         f"ex4_fromscratch_pem_space={tag},*", "summary.csv")]
    for p in pats:
        for c in sorted(_g.glob(p)):
            nm = os.path.basename(os.path.dirname(c))
            if f"gamma_predictive={gamma}" not in nm:
                continue
            if r is not None and f"r={r}" not in nm:
                continue
            if r is None and f"r={r}" in nm:
                continue
            return pd.read_csv(c).set_index("config")
    raise FileNotFoundError(f"no run found for {tag} gamma={gamma} r={r}")


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


def fig_ex4_gamma():
    """gamma dose-response + divergence diagnosis (uses OUTPUT/ sweep)."""
    O = os.path.join(BASE, "OUTPUT")
    DIRS = {
        ("output", 64, 100.0): "ex4_pem_space=output,gamma_predictive=100.0,",
        ("output", 64, 10.0): "ex4_fromscratch_pem_space=output,gamma_predictive=10.0",
        ("output", 64, 1.0): "ex4_fromscratchpem_space=output,gamma_predictive=1.0",
        ("lowrank", 64, 100.0): "ex4_pem_space=lowrank,gamma_predictive=100.0",
        ("lowrank", 16, 10.0): "ex4_pem_space=lowrank,gamma_predictive=10.0,r=16",
        ("output", 16, 10.0): "ex4_pem_space=output,gamma_predictive=10.0,r=16",
    }
    data = {}
    for k, d in DIRS.items():
        p = os.path.join(O, d, "summary.csv")
        s = pd.read_csv(p).set_index("config")
        gain = float(s.loc["base", "acc_mean"]) - float(s.loc["nopem", "acc_mean"])
        lp = os.path.join(O, d, "losses.txt")
        lmax = float("nan")
        if os.path.exists(lp):
            with open(lp, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("base:"):
                        lmax = float(np.max([float(v) for v in
                                             line.split(":", 1)[1].split()]))
                        break
        data[k] = dict(gain=gain * 100, corr=float(s.loc["base", "corr_abs"]),
                       lmax=lmax, acc=float(s.loc["base", "acc_mean"]))

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1))

    ax = axes[0]
    g = [1.0, 10.0, 100.0]
    v = [data[("output", 64, x)]["gain"] for x in g]
    ax.plot(g, v, "o-", color="#D1495B", lw=2, ms=7,
            label="$r=64$ (near full rank)")
    ax.axhline(0, color="k", lw=0.9, ls="--")
    for x, y in zip(g, v):
        ax.annotate(f"{y:+.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.5)
    v16 = data[("output", 16, 10.0)]["gain"]
    ax.scatter([10.0], [v16], s=110, marker="s", color="#E08A00", zorder=4,
               label=f"$r=16$ ($\\gamma$=10, {v16:+.2f})")
    ax.set_xscale("log")
    ax.set_xticks(g); ax.set_xticklabels(["1", "10", "100"])
    ax.set_xlabel("$\\gamma_{predictive}$")
    ax.set_ylabel("PEM accuracy gain over PEM-off (pp)")
    ax.set_title("(a) Gain is dose-dependent, not an artefact of $\\gamma$")
    ax.legend(frameon=False, fontsize=8.3)

    ax = axes[1]
    xs, ys, cs, labs = [], [], [], []
    for k, tag in [(("output", 64, 100.0), "out r=64 $\\gamma$=100"),
                   (("output", 64, 10.0), "out r=64 $\\gamma$=10"),
                   (("output", 64, 1.0), "out r=64 $\\gamma$=1"),
                   (("output", 16, 10.0), "out r=16 $\\gamma$=10"),
                   (("lowrank", 64, 100.0), "low r=64 $\\gamma$=100"),
                   (("lowrank", 16, 10.0), "low r=16 $\\gamma$=10")]:
        xs.append(data[k]["lmax"]); ys.append(tag)
        cs.append("#D1495B" if k[0] == "output" else "#2F6FB5")
        labs.append(tag)
    ax.barh(range(len(xs)), xs, color=cs, alpha=0.85)
    for i, x in enumerate(xs):
        ax.text(x * 1.15, i, f"{x:.1e}", va="center", fontsize=7.8)
    ax.set_yticks(range(len(xs))); ax.set_yticklabels(labs, fontsize=8.2)
    ax.set_xscale("log")
    ax.set_xlim(1, 1e9)
    ax.set_xlabel("max training loss over 2000 steps (log)")
    ax.set_title("(b) Divergence is NOT caused by $\\gamma$=100")
    ax.grid(axis="y", alpha=0)

    ax = axes[2]
    ks = [("output", 64), ("output", 16), ("lowrank", 64), ("lowrank", 16)]
    gains = [data[("output", 64, 10.0)]["gain"],
             data[("output", 16, 10.0)]["gain"],
             data[("lowrank", 64, 100.0)]["gain"],
             data[("lowrank", 16, 10.0)]["gain"]]
    corrs = [data[("output", 64, 10.0)]["corr"],
             data[("output", 16, 10.0)]["corr"],
             data[("lowrank", 64, 100.0)]["corr"],
             data[("lowrank", 16, 10.0)]["corr"]]
    labs = ["output\n$r$=64", "output\n$r$=16",
            "lowrank\n$r$=64", "lowrank\n$r$=16"]
    cols = ["#D1495B", "#E08A00", "#2F6FB5", "#7A9E3F"]
    b = ax.bar(range(4), gains, color=cols, width=0.6)
    ax.axhline(0, color="k", lw=0.9)
    for bb, gv, cv in zip(b, gains, corrs):
        ax.text(bb.get_x() + bb.get_width() / 2, gv + (0.12 if gv >= 0 else -0.30),
                f"{gv:+.2f} pp\n$|\\rho|$={cv:.3f}", ha="center", fontsize=7.4)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8.4)
    ax.set_ylabel("PEM accuracy gain (pp)")
    ax.set_ylim(-2.2, 3.8)
    ax.set_title("(c) Gain shrinks with the operating space")
    fig.tight_layout()
    save(fig, "fig_ex4_gamma.png")


if __name__ == "__main__":
    fig_ex4_pem()
    fig_ex4_ablation()
    fig_ex4_loss()
    fig_ex4_scatter()
    fig_ex4_gamma()
    print("ALL DONE")
