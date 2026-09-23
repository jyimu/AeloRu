# -*- coding: utf-8 -*-
"""Print the exact numbers quoted in the paper."""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TASKS5 = ["sst2", "mrpc", "qnli", "rte", "cola"]

print("### ex1 summary")
s = pd.read_csv(os.path.join(BASE, "ex1", "summary.csv"))
print(s.to_string(index=False))
for m in ["lora", "aeloru-safe", "aeloru-default"]:
    R = np.load(os.path.join(BASE, "ex1", f"R_{m}.npy"))
    print(f"{m:16s} task1 traj={np.round(R[:,0],4)}  final-mean={R[-1].mean():.4f}")

print("\n### ex1 mechanism trajectories (first / peak / last)")
for m in ["aeloru-safe", "aeloru-default"]:
    d = pd.read_csv(os.path.join(BASE, "ex1", f"mech_{m}.csv"))
    print("--", m)
    for c in ["plasticity_out", "plasticity_in", "global_direction",
              "structure_loss", "proto_out", "proto_in", "bcm_cap_out",
              "volatility", "loss_ortho"]:
        v = d[c].values
        print(f"   {c:18s} first={v[0]:.5f} peak={v.max():.5f} last={v[-1]:.5f}")
    print("   state counts:", d["state"].value_counts().to_dict())

print("\n### ex2 summaries")
for d, tag in [("ex2_r8", "r=8"), ("ex2_r16", "r=16"), ("ex2_r32", "r=32")]:
    t = pd.read_csv(os.path.join(BASE, d, "summary.csv")).set_index("method")
    print(f"-- {tag}")
    l, a = t.loc["lora"], t.loc["aeloru"]
    for c in ["ACC", "BWT", "Forgetting", "corr_abs", "corr_max", "TC", "MI",
              "eff_rank", "PR", "iso_err", "cond", "sparsity", "dw_eff_rank",
              "task_selectivity", "task_jaccard", "params", "seconds"]:
        lo, ao = float(l[c]), float(a[c])
        rel = (ao - lo) / abs(lo) * 100 if lo != 0 else float("nan")
        print(f"   {c:18s} lora={lo:10.5f} aeloru={ao:10.5f} delta={rel:+7.1f}%")
    print("   ablation corr_abs:", {k: round(float(t.loc[k, 'corr_abs']), 5)
                                    for k in t.index})
    print("   ablation Forg   :", {k: round(float(t.loc[k, 'Forgetting']), 5)
                                  for k in t.index})
    print("   ablation effrank:", {k: round(float(t.loc[k, 'eff_rank']), 5)
                                  for k in t.index})
    print("   ablation ACC    :", {k: round(float(t.loc[k, 'ACC']), 5)
                                  for k in t.index})

print("\n### corr matrix off-diagonal means (|rho|)")
for d, tag in [("ex2_r8", "r=8"), ("ex2_r16", "r=16"), ("ex2_r32", "r=32")]:
    for m in ["lora", "aeloru", "aeloru-nopem", "aeloru-nowta",
              "aeloru-nosphere", "aeloru-noortho"]:
        C = np.load(os.path.join(BASE, d, f"corrmat_{m}.npy")).astype(float)
        off = (C.sum() - np.trace(C)) / (C.size - C.shape[0])
        print(f"   {tag} {m:18s} mean|off-diag|={off:.4f}")

print("\n### dectrace (r=8) start/end")
for m in ["aeloru", "aeloru-nopem", "aeloru-nowta", "aeloru-nosphere",
          "aeloru-noortho"]:
    t = pd.read_csv(os.path.join(BASE, "ex2_r8", f"dectrace_{m}.csv"))
    print(f"   {m:18s} corr_abs {t.corr_abs.iloc[0]:.4f}->{t.corr_abs.iloc[-1]:.4f} "
          f"eff_rank {t.eff_rank.iloc[0]:.3f}->{t.eff_rank.iloc[-1]:.3f} "
          f"TC {t.TC.iloc[0]:.4f}->{t.TC.iloc[-1]:.4f}")
