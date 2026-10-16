# -*- coding: utf-8 -*-
"""Seed statistics for ex1 (concept drift) from OUTPUT/ runs."""
import os
import glob
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "OUTPUT")

# seed 42 lives in OUTPUT/ex1; the three new seeds are in ex1_drift_*
runs = []
runs.append(("42", os.path.join(OUT, "ex1")))
for d in sorted(glob.glob(os.path.join(OUT, "ex1_drift_*"))):
    # extract seed from folder name e.g. ex1_drift_9589_20261012_105700
    seed = os.path.basename(d).split("_")[2]
    runs.append((seed, d))

print("runs found:", [r[0] for r in runs])
rows = []
for seed, d in runs:
    p = os.path.join(d, "summary.csv")
    if not os.path.exists(p):
        print("  (skip, no summary.csv:", seed, ")")
        continue
    s = pd.read_csv(p).set_index("method")
    for m in ["lora", "aeloru-safe", "aeloru-default"]:
        r = s.loc[m]
        rows.append(dict(seed=seed, method=m, ACC=float(r["ACC"]),
                         BWT=float(r["BWT"]), FWT=float(r["FWT"]),
                         Forgetting=float(r["Forgetting"]),
                         seconds=float(r["seconds"])))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "..", "figures", "_ex1_seeds_raw.csv"), index=False)
print(df.pivot_table(index="method", columns="seed",
                     values=["ACC", "BWT", "Forgetting"]))


def stats(sub):
    return sub.mean(numeric_only=True), sub.std(ddof=1, numeric_only=True)


def paired_t(a, b):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    md, sd = d.mean(), d.std(ddof=1)
    if sd == 0:
        return float("nan"), float("nan")
    t = md / (sd / np.sqrt(n))
    try:
        from scipy import stats as st
        p = 2 * st.t.sf(abs(t), df=n - 1)
    except Exception:
        # crude fallback for small df
        p = float("nan")
    return float(t), float(p)


for tag, seeds in [("3-seed {9589,6519,1412}", ["9589", "6519", "1412"]),
                   ("4-seed {42,9589,6519,1412}", ["42", "9589", "6519", "1412"])]:
    print("\n" + "=" * 70)
    print(tag)
    sub = df[df.seed.isin(seeds)]
    print(f"{'method':16s} {'ACC':>18s} {'BWT':>18s} {'Forgetting':>18s}")
    g = {}
    for m in ["lora", "aeloru-safe", "aeloru-default"]:
        s = sub[sub.method == m]
        mu, sd = stats(s)
        g[m] = s
        print(f"{m:16s} {mu['ACC']:.4f} ± {sd['ACC']:.4f}   "
              f"{mu['BWT']:+.4f} ± {sd['BWT']:.4f}   "
              f"{mu['Forgetting']:.4f} ± {sd['Forgetting']:.4f}")
    print(f"{'  seconds':16s} " + "  ".join(
        f"{g[m]['seconds'].mean():.1f}" for m in ["lora", "aeloru-safe",
                                                 "aeloru-default"]))
    # effect sizes vs lora
    lmu = g["lora"].mean(numeric_only=True)
    for m in ["aeloru-safe", "aeloru-default"]:
        mu = g[m].mean(numeric_only=True)
        print(f"  {m:16s} vs lora: dACC={mu['ACC']-lmu['ACC']:+.4f}  "
              f"dBWT={mu['BWT']-lmu['BWT']:+.4f}  "
              f"Forgetting {lmu['Forgetting']:.4f}->{mu['Forgetting']:.4f} "
              f"({100*(mu['Forgetting']-lmu['Forgetting'])/lmu['Forgetting']:+.1f}%)")
    # paired t-tests
    for m in ["aeloru-safe", "aeloru-default"]:
        for col in ["Forgetting", "BWT", "ACC"]:
            a = g["lora"][col].values
            b = g[m][col].values
            n = min(len(a), len(b))
            t, p = paired_t(a[:n], b[:n])
            print(f"  paired t  lora vs {m:16s} {col:11s} "
                  f"t={t:+.3f}  p={p:.4f}")
