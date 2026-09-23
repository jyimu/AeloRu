# -*- coding: utf-8 -*-
"""Figures for experiment 3 (coupling decomposition / PEM nature)."""
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

C_LORA, C_NOPEM = "#8C8C8C", "#2F6FB5"
DOSE_COLORS = ["#2F6FB5", "#4C9BE8", "#E08A00", "#D1495B"]
DOSE = ["aeloru-nopem", "aeloru", "aeloru-pem8", "aeloru-pem8-w05"]
DOSE_LABEL = ["PEM off\n(ndi=0)", "ndi=2\n(default)", "ndi=8", "ndi=8\ncoef=0.5"]
DOSE_X = [0, 2, 8, 9.6]          # x position ~ PEM dose (last one nudged)
XLBL = ["0", "2", "8", "8+\nw=0.5"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- stage B
B = pd.read_csv(os.path.join(BASE, "ex3_coupling", "summary.csv")).set_index("method")
C = pd.read_csv(os.path.join(BASE, "ex3_coupling", "coupling.csv"))
TASKS = ["sst2", "mrpc", "qnli", "rte", "cola"]


def fig_ex3_dose():
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8))
    lv = B.loc["lora"]

    # (a) total vs redundant coupling
    ax = axes[0, 0]
    ax.plot(DOSE_X, [B.loc[m, "corr_abs"] for m in DOSE], "o-",
            color=C_NOPEM, lw=2, ms=7, label="total $|\\rho|_{abs}$")
    ax.plot(DOSE_X, [B.loc[m, "useless_corr"] for m in DOSE], "s--",
            color="#D1495B", lw=2, ms=6, label="redundant $c_{useless}$")
    ax.plot(DOSE_X, [B.loc[m, "useful_corr"] for m in DOSE], "^:",
            color="#7A9E3F", lw=2, ms=6, label="functional $c_{useful}$")
    ax.axhline(lv["corr_abs"], color=C_LORA, lw=1.2, ls="-",
               label=f"LoRA  $|\\rho|$={lv['corr_abs']:.3f}")
    ax.axhline(lv["useless_corr"], color=C_LORA, lw=1.0, ls="--", alpha=0.7,
               label=f"LoRA  $c_{{useless}}$={lv['useless_corr']:.3f}")
    ax.set_xticks(DOSE_X); ax.set_xticklabels(XLBL)
    ax.set_xlabel("PEM dose (neural-dynamics iterations)")
    ax.set_ylabel("coupling strength")
    ax.set_title("(a) All three couplings rise monotonically with PEM dose")
    ax.legend(frameon=False, fontsize=8.2)

    # (b) Fisher-weighted redundant coupling
    ax = axes[0, 1]
    ax.plot(DOSE_X, [B.loc[m, "coup_red"] for m in DOSE], "o-",
            color="#D1495B", lw=2, ms=7, label="$coup_{red}$ (Fisher-weighted)")
    ax.axhline(lv["coup_red"], color=C_LORA, lw=1.2,
               label=f"LoRA  $coup_{{red}}$={lv['coup_red']:.4f}")
    ax.set_xticks(DOSE_X); ax.set_xticklabels(XLBL)
    ax.set_xlabel("PEM dose (neural-dynamics iterations)")
    ax.set_ylabel("$coup_{red}$")
    ax.set_title("(b) Dose-response is confirmed under Fisher weighting")
    ax.legend(frameon=False, fontsize=8.2)

    # (c) downstream cost
    ax = axes[1, 0]
    ax.plot(DOSE_X, [B.loc[m, "Forgetting"] for m in DOSE], "o-",
            color="#D1495B", lw=2, ms=7, label="Forgetting $\\downarrow$")
    ax.plot(DOSE_X, [-B.loc[m, "BWT"] for m in DOSE], "s-",
            color="#9B5DE5", lw=2, ms=6, label="$-$BWT $\\downarrow$")
    ax.axhline(lv["Forgetting"], color=C_LORA, lw=1.2, ls="--",
               label=f"LoRA  Forg.={lv['Forgetting']:.4f}")
    ax.set_xticks(DOSE_X); ax.set_xticklabels(XLBL)
    ax.set_xlabel("PEM dose (neural-dynamics iterations)")
    ax.set_ylabel("metric value")
    ax.set_title("(c) Higher dose worsens forgetting & BWT")
    ax.legend(frameon=False, fontsize=8.2)

    # (d) redundant vs functional scatter
    ax = axes[1, 1]
    for m, c, s in list(zip(DOSE, DOSE_COLORS, [70, 110, 110, 130])):
        ax.scatter(B.loc[m, "coup_red"], B.loc[m, "coup_func"], s=s, color=c,
                   edgecolor="k", linewidth=0.6, zorder=3,
                   label={"aeloru-nopem": "PEM off", "aeloru": "ndi=2 (default)",
                          "aeloru-pem8": "ndi=8",
                          "aeloru-pem8-w05": "ndi=8, w=0.5"}[m])
    ax.scatter(lv["coup_red"], lv["coup_func"], s=200, marker="*", color=C_LORA,
               edgecolor="k", linewidth=0.6, zorder=4, label="LoRA baseline")
    ax.plot([B.loc[m, "coup_red"] for m in DOSE],
            [B.loc[m, "coup_func"] for m in DOSE], "--", color="k",
            lw=0.9, alpha=0.5, zorder=1)
    ax.annotate("dose direction", xy=(0.0475, 0.195), xytext=(0.0335, 0.135),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.9,
                                              color="k", alpha=0.6))
    ax.set_xlabel("$coup_{red}$  (redundant coupling, Fisher-weighted)")
    ax.set_ylabel("$coup_{func}$  (functional coupling)")
    ax.set_title("(d) PEM inflates both, and the functional share stalls")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    save(fig, "fig_ex3_dose.png")


def fig_ex3_trace():
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2))
    panels = [("corr_abs", "(a) Total coupling $|\\rho|_{abs}$"),
              ("useless_corr", "(b) Redundant coupling $c_{useless}$"),
              ("coup_red", "(c) Fisher-weighted redundant $coup_{red}$"),
              ("corr_useful", "(d) Functional share $corr_{useful}$")]
    styles = {"lora": ("-", C_LORA, "o"), "aeloru-nopem": ("--", "#2F6FB5", "s"),
              "aeloru": ("-", "#4C9BE8", "^"), "aeloru-pem8": ("-", "#E08A00", "D"),
              "aeloru-pem8-w05": (":", "#D1495B", "v")}
    for ax, (col, title) in zip(axes.ravel(), panels):
        for m in ["lora"] + DOSE[1:]:
            sub = C[C.method == m].sort_values("boundary")
            ls, c, mk = styles[m]
            lbl = "LoRA" if m == "lora" else \
                {"aeloru": "ndi=2 (default)", "aeloru-pem8": "ndi=8",
                 "aeloru-pem8-w05": "ndi=8, w=0.5"}[m]
            ax.plot(sub["boundary"], sub[col], ls, color=c, marker=mk, ms=5,
                    lw=1.6, label=lbl)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"{i}\n{TASKS[i]}" for i in range(5)], fontsize=8.5)
        ax.set_xlabel("task boundary (after training task $i$)")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Evolution of the coupling decomposition along the drift stream "
                 "(r = 8, 5 tasks x 250 steps)", y=1.0, fontsize=12)
    fig.tight_layout()
    save(fig, "fig_ex3_trace.png")


# ---------------------------------------------------------------- stage A
S = pd.read_csv(os.path.join(BASE, "ex3_synth", "summary.csv"))
ORD = ["pem-off", "pem-1", "pem-3", "pem-10", "pem-10-w05"]
OLBL = ["PEM off", "ndi=1", "ndi=3", "ndi=10", "ndi=10\nw=0.5"]


def fig_ex3_synth():
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1))

    ax = axes[0]
    for energy, c, mk in [("matched", "#2F6FB5", "o"), ("mismatched", "#D1495B", "s")]:
        sub = S[S.energy == energy].set_index("cond").loc[ORD[:4]]
        ax.plot(range(4), sub["selectivity"].values, mk + "-", color=c, lw=2,
                ms=7, label=("matched (equal variance)" if energy == "matched"
                             else "mismatched (IRR = principal)"))
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(range(4)); ax.set_xticklabels(OLBL[:4])
    ax.set_ylabel("selectivity $S$")
    ax.set_title("(a) Selectivity only appears when\nredundancy is NOT principal")
    ax.legend(frameon=False, fontsize=8.3)

    ax = axes[1]
    w = 0.35
    x = np.arange(4)
    for i, (col, lbl, hatch) in enumerate(
            [("ret_useless_corr", "retention of redundant $c_{useless}$", ""),
             ("ret_useful_corr", "retention of functional $c_{useful}$", "//")]):
        m = S[(S.energy == "matched")].set_index("cond").loc[ORD[:4], col].values
        ax.bar(x + (i - 0.5) * w, m, w, color="#2F6FB5" if i == 0 else "#7A9E3F",
               hatch=hatch, label=lbl)
        for xx, v in zip(x + (i - 0.5) * w, m):
            ax.text(xx, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.2)
    ax.axhline(1.0, color="k", lw=0.9, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(OLBL[:4])
    ax.set_title("(b) matched: redundant is cut, functional is kept")
    ax.legend(frameon=False, fontsize=8.3)

    ax = axes[2]
    for i, (col, lbl) in enumerate(
            [("ret_useless_corr", "retention of redundant $c_{useless}$"),
             ("ret_useful_corr", "retention of functional $c_{useful}$")]):
        m = S[(S.energy == "mismatched")].set_index("cond").loc[ORD[:4], col].values
        ax.bar(x + (i - 0.5) * w, m, w,
               color="#D1495B" if i == 0 else "#7A9E3F",
               hatch="" if i == 0 else "//", label=lbl)
        for xx, v in zip(x + (i - 0.5) * w, m):
            ax.text(xx, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.2)
    ax.axhline(1.0, color="k", lw=0.9, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(OLBL[:4])
    ax.set_title("(c) mismatched: role reverses, functional is cut")
    ax.legend(frameon=False, fontsize=8.3)
    fig.tight_layout()
    save(fig, "fig_ex3_synth.png")


if __name__ == "__main__":
    fig_ex3_dose()
    fig_ex3_trace()
    fig_ex3_synth()
    print("ALL DONE")
