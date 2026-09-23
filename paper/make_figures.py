# -*- coding: utf-8 -*-
"""
Generate all paper figures from the raw experiment dumps in
    paper/ex1, paper/ex2_r8, paper/ex2_r16, paper/ex2_r32
Output -> paper/figures/
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "legend.fontsize": 9,
    "figure.dpi": 160,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_LORA = "#8C8C8C"
C_SAFE = "#2F6FB5"
C_DEF = "#D1495B"
C_AEL = "#2F6FB5"
C_ABL = ["#2F6FB5", "#E08A00", "#7A9E3F", "#9B5DE5", "#00A5A5"]

TASKS5 = ["sst2", "mrpc", "qnli", "rte", "cola"]
TASKS3 = ["sst2", "mrpc", "rte"]

METHOD_LABEL = {
    "lora": "LoRA (peft)",
    "aeloru": "Aeloru (full)",
    "aeloru-safe": "Aeloru-safe",
    "aeloru-default": "Aeloru-default",
    "aeloru-nopem": "w/o PEM",
    "aeloru-nowta": "w/o WTA",
    "aeloru-nosphere": "w/o SPHeRe",
    "aeloru-noortho": "w/o Ortho",
}


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("saved", p)


# ----------------------------------------------------------------------------
# 1. Concept drift: ex1
# ----------------------------------------------------------------------------
def fig_concept_drift():
    sum1 = pd.read_csv(os.path.join(BASE, "ex1", "summary.csv"))
    order = ["lora", "aeloru-safe", "aeloru-default"]
    sum1 = sum1.set_index("method").loc[order].reset_index()
    colors = [C_LORA, C_SAFE, C_DEF]

    R = {m: np.load(os.path.join(BASE, "ex1", f"R_{m}.npy")) for m in order}
    acc_after = {m: np.array([R[m][i, :i + 1].mean() for i in range(5)]) for m in order}
    sst2_traj = {m: R[m][:, 0] for m in order}

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))
    ax = axes[0, 0]
    for m, c in zip(order, colors):
        ax.plot(range(1, 6), acc_after[m], "o-", color=c, lw=2, ms=6,
                label=METHOD_LABEL[m])
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(TASKS5)
    ax.set_xlabel("Task in the drift stream (trained sequentially)")
    ax.set_ylabel("Mean accuracy over all seen tasks")
    ax.set_title("(a) Accuracy on all seen tasks after each task")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for m, c in zip(order, colors):
        ax.plot(range(1, 6), sst2_traj[m], "s--", color=c, lw=2, ms=6,
                label=METHOD_LABEL[m])
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(TASKS5)
    ax.set_xlabel("Task being trained")
    ax.set_ylabel("Retained accuracy on task-1 (sst2)")
    ax.set_title("(b) Retention of the first task (catastrophic forgetting)")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    metrics = ["ACC", "BWT", "FWT", "Forgetting"]
    x = np.arange(len(metrics))
    w = 0.26
    for k, (m, c) in enumerate(zip(order, colors)):
        row = sum1[sum1.method == m].iloc[0]
        vals = [row["ACC"], row["BWT"], row["FWT"], row["Forgetting"]]
        bars = ax.bar(x + (k - 1) * w, vals, w, color=c, label=METHOD_LABEL[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.006 if v >= 0 else -0.006),
                    f"{v:+.4f}".rstrip("0"), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6.8, rotation=90)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Value")
    ax.set_title("(c) Continual-learning metrics (5 tasks x 250 steps)")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_ylim(-0.09, 0.72)

    ax = axes[1, 1]
    labels = [METHOD_LABEL[m] for m in order]
    secs = sum1["seconds"].values
    bars = ax.bar(labels, secs, color=colors, width=0.55)
    for b, s, p in zip(bars, secs, sum1["params"].values):
        ax.text(b.get_x() + b.get_width() / 2, s + 3,
                f"{s:.0f}s\n{p/1000:.0f}k params", ha="center", fontsize=8)
    ax.set_ylabel("Wall-clock time of the whole stream (s)")
    ax.set_title("(d) Training cost (comparable parameter budget)")
    ax.set_ylim(0, max(secs) * 1.35)
    ax.tick_params(axis="x", labelsize=8.5)

    fig.tight_layout()
    save(fig, "fig_concept_drift.png")


def fig_r_matrix():
    order = ["lora", "aeloru-safe", "aeloru-default"]
    titles = ["LoRA (peft)", "Aeloru-safe", "Aeloru-default"]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1))
    for ax, m, t in zip(axes, order, titles):
        R = np.load(os.path.join(BASE, "ex1", f"R_{m}.npy")).astype(float)
        mask = np.triu(np.ones_like(R, dtype=bool), k=1)
        Rd = np.where(mask, np.nan, R)
        im = ax.imshow(Rd, cmap="RdYlGn", vmin=0.45, vmax=0.75)
        ax.set_xticks(range(5)); ax.set_xticklabels(TASKS5)
        ax.set_yticks(range(5)); ax.set_yticklabels(TASKS5)
        ax.set_xlabel("evaluated task $j$")
        ax.set_title(f"{t}\n$R[i,j]$: acc. on $j$ after training $i$")
        for i in range(5):
            for j in range(5):
                if not mask[i, j]:
                    ax.text(j, i, f"{R[i, j]:.3f}", ha="center", va="center",
                            fontsize=7.6, color="black")
        ax.grid(False)
    ax = axes[0]; ax.set_ylabel("trained-up-to task $i$")
    fig.colorbar(im, ax=axes, shrink=0.82, pad=0.015, label="accuracy")
    save(fig, "fig_r_matrix.png")


def fig_mechanisms():
    d = pd.read_csv(os.path.join(BASE, "ex1", "mech_aeloru-default.csv"))
    s = pd.read_csv(os.path.join(BASE, "ex1", "mech_aeloru-safe.csv"))
    for df in (d, s):
        df["t"] = df["task"].map({t: i * 250 for i, t in enumerate(TASKS5)}) \
            + df["step"]
    states = ["explore", "red", "anchor", "solid"]
    scmap = {"explore": "#7A9E3F", "red": "#D1495B",
             "anchor": "#2F6FB5", "solid": "#9B5DE5"}

    fig = plt.figure(figsize=(12.0, 8.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.75], hspace=0.42, wspace=0.22)

    def panel(ax, col, title, ylab, logy=False):
        ax.plot(s["t"], s[col], color=C_SAFE, lw=1.3, alpha=0.85, label="Aeloru-safe")
        ax.plot(d["t"], d[col], color=C_DEF, lw=1.3, alpha=0.85, label="Aeloru-default")
        ax.set_title(title); ax.set_ylabel(ylab); ax.set_xlabel("training step")
        if logy:
            ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8)

    panel(fig.add_subplot(gs[0, 0]), "plasticity_out",
          "(a) Dual 1-D HGF plasticity gate (output side)", "$p_{out}$")
    panel(fig.add_subplot(gs[0, 1]), "volatility",
          "(b) HGF volatility $\\exp(\\omega)$ drives the dynamic LR",
          "$\\exp(\\omega_t)$")
    panel(fig.add_subplot(gs[1, 0]), "proto_out",
          "(c) DLAM prototype memory accumulates across tasks", "prototype strength")
    panel(fig.add_subplot(gs[1, 1]), "loss_ortho",
          "(d) Orthogonality penalty $\\mathcal{L}_{ortho}$ (spikes at sleep)",
          "$\\mathcal{L}_{ortho}$", logy=True)

    ax = fig.add_subplot(gs[2, 0])
    dsl = d[["plasticity_in"]].copy()
    ax.plot(s["t"], s["plasticity_in"], color=C_SAFE, lw=1.3, alpha=0.85,
            label="Aeloru-safe")
    ax.plot(d["t"], d["plasticity_in"], color=C_DEF, lw=1.3, alpha=0.85,
            label="Aeloru-default")
    ax.set_title("(e) Plasticity gate (input side)")
    ax.set_ylabel("$p_{in}$"); ax.set_xlabel("training step")
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    codes = {st: i for i, st in enumerate(states)}
    y = np.array([codes[v] for v in d["state"]])
    ax.step(d["t"], y, where="post", lw=1.2, color="k")
    ax.set_yticks(range(4)); ax.set_yticklabels(states)
    ax.set_xlabel("training step")
    ax.set_title("(f) HongWen cognitive state machine (Aeloru-default)")
    ax.set_ylim(-0.5, 3.9)
    ax.grid(alpha=0.15)
    prev = y[0]; start = 0
    for i in range(1, len(y)):
        if y[i] != prev:
            ax.axvspan(start, i, color=scmap[states[prev]], alpha=0.16, lw=0)
            prev = y[i]; start = i
    ax.axvspan(start, d["t"].max(), color=scmap[states[prev]], alpha=0.16, lw=0)
    for i in range(5):
        ax.axvline(i * 250, color="k", lw=0.7, ls=":")
    for i, t in enumerate(TASKS5):
        ax.text(i * 250 + 125, 3.62, t, ha="center", fontsize=8, color="k")
    save(fig, "fig_mechanisms.png")


# ----------------------------------------------------------------------------
# 2. Feature decoupling: ex2
# ----------------------------------------------------------------------------
RDIR = {"r8": "ex2_r8", "r16": "ex2_r16", "r32": "ex2_r32"}
RK = ["r8", "r16", "r32"]
SUM = {k: pd.read_csv(os.path.join(BASE, d, "summary.csv")).set_index("method")
       for k, d in RDIR.items()}
ABL = ["aeloru", "aeloru-nopem", "aeloru-nowta", "aeloru-nosphere", "aeloru-noortho"]
DEC_METRICS = [("corr_abs", "$|\\rho|_{abs}$ (mean |corr|)", "lower is better"),
               ("TC", "Total correlation (nats)", "lower is better"),
               ("eff_rank", "Effective rank $e^{H(\\lambda)}$", "higher is better"),
               ("PR", "Participation ratio", "higher is better"),
               ("iso_err", "Isotropy error", "lower is better"),
               ("cond", "Condition number $\\kappa$", "lower is better")]


def fig_decouple_bars():
    s = SUM["r8"]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))
    for ax, (col, title, hint) in zip(axes.ravel(), DEC_METRICS):
        vals = [s.loc[m, col] for m in ABL]
        vals = [s.loc["lora", col]] + vals
        labels = ["LoRA"] + [METHOD_LABEL[m] for m in ABL]
        cols = [C_LORA] + C_ABL
        bars = ax.bar(range(len(vals)), vals, color=cols, width=0.68)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=26, ha="right", fontsize=8.3)
        ax.set_title(f"{title}\n({hint})", fontsize=10)
        vmax = max(vals)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + vmax * 0.02,
                    f"{v:.3f}" if v < 100 else f"{v:.1f}",
                    ha="center", fontsize=7.4)
        ax.set_ylim(0, vmax * 1.22)
    fig.suptitle("Feature-decoupling metrics, r = 8 (5 tasks x 250 steps)", y=1.0,
                 fontsize=12)
    fig.tight_layout()
    save(fig, "fig_decouple_bars.png")


def fig_rank_scaling():
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    panels = [("corr_abs", "Mean |correlation| $\\downarrow$"),
              ("eff_rank", "Effective rank $\\uparrow$"),
              ("Forgetting", "Forgetting $\\downarrow$")]
    for ax, (col, ylab) in zip(axes, panels):
        lv = [SUM[k].loc["lora", col] for k in RK]
        av = [SUM[k].loc["aeloru", col] for k in RK]
        x = np.arange(3); w = 0.34
        b1 = ax.bar(x - w / 2, lv, w, color=C_LORA, label="LoRA")
        b2 = ax.bar(x + w / 2, av, w, color=C_AEL, label="Aeloru (full)")
        for b in list(b1) + list(b2):
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.6)
        ax.set_xticks(x)
        ax.set_xticklabels(["r = 8\n(5 tasks, 250 st)",
                            "r = 16\n(3 tasks, 120 st)",
                            "r = 32\n(3 tasks, 120 st)"], fontsize=8.6)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, max(lv + av) * 1.28)
        ax.legend(frameon=False, fontsize=8.5)
    axes[0].set_title("(a) Subspace redundancy")
    axes[1].set_title("(b) Rank utilisation")
    axes[2].set_title("(c) Forgetting")
    fig.tight_layout()
    save(fig, "fig_rank_scaling.png")


def fig_ablation_delta():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))
    ax = axes[0]
    x = np.arange(len(RK)); w = 0.2
    for i, m in enumerate(ABL[1:]):
        dv = [SUM[k].loc[m, "corr_abs"] - SUM[k].loc["aeloru", "corr_abs"] for k in RK]
        ax.bar(x + (i - 1.5) * w, dv, w, color=C_ABL[i + 1], label=METHOD_LABEL[m])
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(["r = 8", "r = 16", "r = 32"])
    ax.set_ylabel("$\\Delta |\\rho|_{abs}$ vs. full Aeloru")
    ax.set_title("(a) Ablation effect on redundancy (positive = harmful removal)")
    ax.legend(frameon=False, fontsize=8.3, ncol=2)

    ax = axes[1]
    x = np.arange(len(RK)); w = 0.2
    for i, m in enumerate(ABL[1:]):
        dv = [SUM[k].loc[m, "eff_rank"] - SUM[k].loc["aeloru", "eff_rank"] for k in RK]
        ax.bar(x + (i - 1.5) * w, dv, w, color=C_ABL[i + 1], label=METHOD_LABEL[m])
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(["r = 8", "r = 16", "r = 32"])
    ax.set_ylabel("$\\Delta$ effective rank vs. full Aeloru")
    ax.set_title("(b) Ablation effect on rank utilisation (negative = harmful removal)")
    ax.legend(frameon=False, fontsize=8.3, ncol=2)
    fig.tight_layout()
    save(fig, "fig_ablation_delta.png")


def fig_dec_vs_forget():
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    marks = {"r8": "o", "r16": "s", "r32": "^"}
    for k in RK:
        s = SUM[k]
        for m in ["lora"] + ABL:
            c = C_LORA if m == "lora" else C_ABL[ABL.index(m)]
            ax.scatter(s.loc[m, "corr_abs"], s.loc[m, "Forgetting"],
                       marker=marks[k], s=95 if m == "lora" else 70,
                       color=c, edgecolor="k", linewidth=0.5, zorder=3,
                       label=f"{'LoRA' if m == 'lora' else METHOD_LABEL[m]} / {k}"
                       if k == "r8" else None)
    # connect lora -> aeloru per rank
    for k in RK:
        s = SUM[k]
        ax.plot([s.loc["lora", "corr_abs"], s.loc["aeloru", "corr_abs"]],
                [s.loc["lora", "Forgetting"], s.loc["aeloru", "Forgetting"]],
                "--", color="k", lw=0.7, alpha=0.5, zorder=1)
    ax.set_xlabel("Mean |correlation| of the low-rank feature subspace")
    ax.set_ylabel("Forgetting")
    ax.set_title("Decoupling vs. catastrophic forgetting")
    ax.legend(frameon=False, fontsize=7.8, ncol=2, loc="lower right")
    ax.text(0.02, 0.02, "marker shape = rank:  (o) r8    (s) r16    (^) r32\n"
            "(dashed line joins LoRA and Aeloru within one rank)",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left")
    save(fig, "fig_dec_vs_forget.png")


def fig_corrmat():
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.0))
    for j, k in enumerate(RK):
        for i, m in enumerate(["lora", "aeloru"]):
            ax = axes[i, j]
            C = np.load(os.path.join(BASE, RDIR[k], f"corrmat_{m}.npy")).astype(float)
            im = ax.imshow(C, cmap="magma", vmin=0, vmax=1)
            t = "LoRA" if m == "lora" else "Aeloru"
            ax.set_title(f"{t}  (r = {k[1:]},  mean|off-diag| = "
                         f"{(C.sum() - np.trace(C)) / (C.size - C.shape[0]):.3f})",
                         fontsize=9.5)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(t, fontsize=10)
            ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.7, pad=0.015, label="$|\\rho_{ij}|$")
    fig.suptitle("Low-rank feature correlation matrices $|\\rho|$ (identity = perfect "
                 "decoupling)", y=0.995, fontsize=12)
    save(fig, "fig_corrmat.png")


def fig_task_importance():
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))
    for ax, k in zip(axes, RK):
        Tl = np.load(os.path.join(BASE, RDIR[k], "taskimp_lora.npy")).astype(float)
        Ta = np.load(os.path.join(BASE, RDIR[k], "taskimp_aeloru.npy")).astype(float)
        T = np.vstack([Tl, Ta])
        im = ax.imshow(T, cmap="viridis", aspect="auto")
        nt = Tl.shape[0]
        ax.set_yticks(range(2 * nt))
        ax.set_yticklabels([f"LoRA/{t}" for t in TASKS5[:nt]] +
                           [f"Ael/{t}" for t in TASKS5[:nt]], fontsize=7.8)
        ax.set_xlabel("low-rank dimension index")
        ax.set_title(f"r = {k[1:]}   selectivity: LoRA "
                     f"{SUM[k].loc['lora', 'task_selectivity']:.4f} vs Aeloru "
                     f"{SUM[k].loc['aeloru', 'task_selectivity']:.4f}", fontsize=9.5)
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.015, label="task importance weight")
    fig.suptitle("Per-task importance over low-rank dimensions (rows sum to 1)",
                 y=1.02, fontsize=12)
    save(fig, "fig_task_importance.png")


def fig_dectrace():
    d = os.path.join(BASE, "ex2_r8")
    tr = {m: pd.read_csv(os.path.join(d, f"dectrace_{m}.csv")) for m in ABL}
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9))
    for ax, (col, ylab) in zip(axes, [("corr_abs", "mean $|\\rho|$"),
                                      ("TC", "total correlation"),
                                      ("eff_rank", "effective rank")]):
        for i, m in enumerate(ABL):
            t = tr[m]
            ax.plot(np.arange(len(t)) * 10, t[col].values, color=C_ABL[i],
                    lw=1.25, label=METHOD_LABEL[m])
        for i in range(1, 5):
            ax.axvline(i * 250, color="k", lw=0.7, ls=":")
        ax.set_xlabel("training step (task boundary every 250 steps)")
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=8)
        for i, t in enumerate(TASKS5):
            ax.text(i * 250 + 125, ax.get_ylim()[1], t, ha="center",
                    fontsize=8, va="top")
    axes[0].set_title("(a) Redundancy dynamics")
    axes[1].set_title("(b) Total-correlation dynamics")
    axes[2].set_title("(c) Effective-rank dynamics")
    fig.suptitle("Online evolution of the decoupling metrics (r = 8)", y=1.03,
                 fontsize=12)
    fig.tight_layout()
    save(fig, "fig_dectrace.png")


def fig_dw_geometry():
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    panels = [("sparsity", "Activation sparsity of $h$", False),
              ("MI", "Mutual-information proxy", False),
              ("dw_eff_rank", "Effective rank of $\\Delta W$ (fp32 re-built)", False)]
    for ax, (col, title, _) in zip(axes, panels):
        x = np.arange(3); w = 0.34
        lv = [SUM[k].loc["lora", col] for k in RK]
        av = [SUM[k].loc["aeloru", col] for k in RK]
        b1 = ax.bar(x - w / 2, lv, w, color=C_LORA, label="LoRA")
        b2 = ax.bar(x + w / 2, av, w, color=C_AEL, label="Aeloru")
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.4)
        ax.set_xticks(x); ax.set_xticklabels(["r = 8", "r = 16", "r = 32"])
        ax.set_ylim(0, max(lv + av) * 1.3)
        ax.set_title(title, fontsize=10.5)
        ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_dw_geometry.png")


if __name__ == "__main__":
    fig_concept_drift()
    fig_r_matrix()
    fig_mechanisms()
    fig_decouple_bars()
    fig_rank_scaling()
    fig_ablation_delta()
    fig_dec_vs_forget()
    fig_corrmat()
    fig_task_importance()
    fig_dectrace()
    fig_dw_geometry()
    print("ALL DONE")
