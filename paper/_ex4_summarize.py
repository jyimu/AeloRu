# -*- coding: utf-8 -*-
"""Summarise the ex4 gamma-dose sweep in OUTPUT/."""
import os
import glob
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "OUTPUT")

TAGS = ["output,gamma=100", "output,gamma=10", "output,gamma=1",
        "lowrank,gamma=100", "lowrank,gamma=10,r=16", "output,gamma=10,r=16"]
DIRS = ["ex4_pem_space=output,gamma_predictive=100.0,",
        "ex4_fromscratch_pem_space=output,gamma_predictive=10.0",
        "ex4_fromscratchpem_space=output,gamma_predictive=1.0",
        "ex4_pem_space=lowrank,gamma_predictive=100.0",
        "ex4_pem_space=lowrank,gamma_predictive=10.0,r=16",
        "ex4_pem_space=output,gamma_predictive=10.0,r=16"]

rows = []
for tag, d in zip(TAGS, DIRS):
    p = os.path.join(OUT, d, "summary.csv")
    if not os.path.exists(p):
        # fall back to any dir whose name contains the tag pieces
        for alt in glob.glob(os.path.join(OUT, "*")):
            nm = os.path.basename(alt).replace(" ", "")
            if tag.split(",")[0] in nm and tag.split(",")[-1].split("=")[-1] in nm:
                if os.path.exists(os.path.join(alt, "summary.csv")):
                    p = os.path.join(alt, "summary.csv")
                    d = os.path.basename(alt)
                    break
    s = pd.read_csv(p).set_index("config")
    print("=" * 78)
    print(f"{tag:24s}  -> {os.path.basename(os.path.dirname(p))}")
    for c in ["base", "nopem", "nogate", "noortho", "nowta", "nosphere",
              "nohebb", "nowhiten", "plain"]:
        if c not in s.index:
            continue
        r = s.loc[c]
        print(f"   {c:9s} acc={r['acc_mean']:.4f}±{r['acc_std']:.4f}  "
              f"corr={r['corr_abs']:.4f}  eff={r['eff_rank']:6.2f}  "
              f"PR={r['PR']:6.2f}  cond={r['cond']:>12.3f}  "
              f"conv={r['converge_step']:6.0f}  {r['seconds']:5.1f}s")
        rows.append(dict(tag=tag, dir=os.path.basename(os.path.dirname(p)),
                         config=c, acc=r['acc_mean'], acc_std=r['acc_std'],
                         corr=r['corr_abs'], eff=r['eff_rank'],
                         cond=r['cond'], conv=r['converge_step'],
                         sec=r['seconds']))
    # max loss from losses.txt
    lp = os.path.join(os.path.dirname(p), "losses.txt")
    if os.path.exists(lp):
        with open(lp, encoding="utf-8") as f:
            for line in f:
                if line.startswith("base:"):
                    v = np.array([float(x) for x in line.split(":", 1)[1].split()])
                    print(f"   base loss: first={v[0]:.3f} max={v.max():.3e} "
                          f"final={v[-1]:.3f}")
                    break

pd.DataFrame(rows).to_csv(os.path.join(BASE, "figures", "_ex4_gamma_raw.csv"),
                          index=False)
print("\nsaved _ex4_gamma_raw.csv")
