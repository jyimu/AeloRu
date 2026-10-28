# -*- coding: utf-8 -*-
"""V4 task-level analysis: zheng (forward) vs back (reversed) order."""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "OUTPUT")

METHODS = ["lora", "aeloru-safe", "aeloru-default", "aeloru-fromscratch"]
SEEDS = ["42", "9589", "6519", "1412"]
ORDER_Z = ["sst2", "mrpc", "qnli", "rte", "cola"]
ORDER_B = ["cola", "rte", "qnli", "mrpc", "sst2"]


def load(tag):
    """-> dict method -> array (seed, T, T); and task labels."""
    base = ORDER_Z if tag == "zheng" else ORDER_B
    labels = [f"{t}@{k}" for k in range(2) for t in base]
    R = {}
    for m in METHODS:
        arr = np.stack([np.load(os.path.join(OUT, f"ex1_{tag}", f"seed_{s}",
                                             f"R_{m}.npy")) for s in SEEDS])
        R[m] = arr
    summary = pd.read_csv(os.path.join(OUT, f"ex1_{tag}", "summary.csv")
                          ).set_index("method")
    return R, labels, summary


def per_task(R, T):
    """Return dict method -> (final retention per task, forgetting per task)."""
    ret, forg = {}, {}
    for m in METHODS:
        A = R[m]                                  # (seed, T, T)
        ret[m] = A[:, -1, :].mean(0)              # final row, mean over seeds
        pk = np.stack([np.nanmax(
            np.where(np.triu(np.ones((T, T)), 1).astype(bool), np.nan, A[i]),
            axis=0) for i in range(A.shape[0])])   # per-seed running max over valid rows
        forg[m] = (pk - A[:, -1, :]).mean(0)
    return ret, forg


for tag in ["zheng", "back"]:
    R, labels, summary = load(tag)
    T = R["lora"].shape[-1]
    print("=" * 92)
    print(f"{tag.upper()}   T={T}   order={'->'.join(labels[:5])} (x2)")
    ret, forg = per_task(R, T)

    print(f"\n{'task':10s} {'pos':>3s} | {'LoRA ret':>8s} {'safe':>7s} "
          f"{'dflt':>7s} {'frscr':>7s} | {'Δsafe':>7s} {'Δdflt':>7s} "
          f"{'Δfrscr':>7s} || {'LoRA fg':>8s} {'safe':>7s} {'dflt':>7s} "
          f"{'frscr':>7s}")
    for j, t in enumerate(labels):
        d0 = ret["aeloru-safe"][j] - ret["lora"][j]
        d1 = ret["aeloru-default"][j] - ret["lora"][j]
        d2 = ret["aeloru-fromscratch"][j] - ret["lora"][j]
        print(f"{t:10s} {j+1:3d} | {ret['lora'][j]:8.4f} "
              f"{ret['aeloru-safe'][j]:7.4f} {ret['aeloru-default'][j]:7.4f} "
              f"{ret['aeloru-fromscratch'][j]:7.4f} | {d0:+7.4f} {d1:+7.4f} "
              f"{d2:+7.4f} || {forg['lora'][j]:8.4f} {forg['aeloru-safe'][j]:7.4f} "
              f"{forg['aeloru-default'][j]:7.4f} {forg['aeloru-fromscratch'][j]:7.4f}")

    print(f"\n  overall (summary.csv): mean±std")
    for m in METHODS:
        r = summary.loc[m]
        print(f"   {m:22s} ACC={r['ACC_mean']:.4f}±{r['ACC_std']:.4f}  "
              f"BWT={r['BWT_mean']:+.4f}±{r['BWT_std']:.4f}  "
              f"FG={r['Forgetting_mean']:.4f}±{r['Forgetting_std']:.4f}")
    l = summary.loc["lora", "Forgetting_mean"]
    for m in METHODS[1:]:
        f = summary.loc[m, "Forgetting_mean"]
        print(f"   {m:22s} forgetting reduction vs LoRA = "
              f"{100*(l-f)/l:+.1f}%")

    # position-effect correlation
    fg = forg["lora"]
    print(f"\n  LoRA forgetting by position: "
          + ", ".join(f"{j+1}:{v:.3f}" for j, v in enumerate(fg[:5])))
    print(f"                              "
          + ", ".join(f"{j+1}:{v:.3f}" for j, v in enumerate(fg[5:])))
    # which tasks does Aeloru win?
    win = [labels[j] for j in range(T) if ret["aeloru-safe"][j] > ret["lora"][j]]
    lose = [labels[j] for j in range(T) if ret["aeloru-safe"][j] <= ret["lora"][j]]
    print(f"  safe beats LoRA on: {win}")
    print(f"  safe <= LoRA   on: {lose}")
