import numpy as np, glob, os
base = os.path.dirname(os.path.abspath(__file__))
for d in ["ex1", "ex2_r8", "ex2_r16", "ex2_r32"]:
    print("=" * 20, d)
    for f in sorted(glob.glob(os.path.join(base, d, "*.npy"))):
        a = np.load(f, allow_pickle=True)
        print(f"{os.path.basename(f):32s} shape={a.shape} dtype={a.dtype} "
              f"min={np.nanmin(a):.4f} max={np.nanmax(a):.4f} mean={np.nanmean(a):.4f}")
