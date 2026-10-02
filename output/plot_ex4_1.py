"""绘制 ex4_1 的两张 PEM 边界证据图。

数据源（同目录 CSV，提取自 paper/md/ex4_1.md）：
  - ex4_1_pem_gamma_dose.csv     : gamma=1/10/100 的 output 模式剂量—反应
  - ex4_1_pem_scale_response.csv  : PEM 净效应随作用空间维度的尺度梯度

输出（保存至本目录）：
  - fig_ex4_1_gamma_dose.png      : 倒 U 型剂量反应（准确率增益 / 相关性增幅）
  - fig_ex4_1_scale_response.png  : 随作用空间维度单调递增的尺度响应

用法：
  E:\\Conda\\envs\\Qelys\\python.exe plot_ex4_1.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))

# 注册一个支持中文的字体（Windows 常见候选），避免 CJK 显示为方块
_CJK_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",   # Microsoft YaHei
    r"C:\Windows\Fonts\simhei.ttf",  # SimHei
    r"C:\Windows\Fonts\simsun.ttc",  # SimSun
]
for _cjk in _CJK_CANDIDATES:
    if os.path.exists(_cjk):
        font_manager.fontManager.addfont(_cjk)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_cjk).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# 图 1：gamma 剂量—反应（output 模式，从零训练）
# ---------------------------------------------------------------------------
rows = load(os.path.join(HERE, "ex4_1_pem_gamma_dose.csv"))
gamma = [int(r["gamma"]) for r in rows]
acc_gain = [float(r["acc_gain_pp"]) for r in rows]
corr_chg = [float(r["corr_abs_change_pct"]) for r in rows]

fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(gamma, acc_gain, "o-", color="#1f77b4", lw=2, ms=9, label="准确率增益 (pp)")
ax1.set_xscale("log")
ax1.set_xlabel(r"$\gamma_{\mathrm{predictive}}$ (log scale)")
ax1.set_ylabel("PEM 准确率增益 (pp)", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
best = acc_gain.index(max(acc_gain))
ax1.axvline(gamma[best], ls="--", color="gray", alpha=0.6)
ax1.annotate(
    f"最优 γ≈{gamma[best]}\n(+{acc_gain[best]:.2f} pp)",
    (gamma[best], acc_gain[best]),
    textcoords="offset points",
    xytext=(8, -28),
    fontsize=9,
    color="#1f77b4",
)
for g, v in zip(gamma, acc_gain):
    ax1.annotate(f"{v:+.2f}", (g, v), textcoords="offset points", xytext=(0, 8), ha="center")

ax2 = ax1.twinx()
ax2.plot(gamma, corr_chg, "s--", color="#d62728", lw=1.8, ms=7, label="corr_abs 增幅 (%)")
ax2.set_ylabel("corr_abs 增幅 (%)", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")

plt.title("PEM 的 γ 剂量—反应（output 模式，从零训练）")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_ex4_1_gamma_dose.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# 图 2：随作用空间维度的尺度响应（从零训练）
# ---------------------------------------------------------------------------
rows = load(os.path.join(HERE, "ex4_1_pem_scale_response.csv"))
labels = [f"{r['scenario']}\n(r={r['r']}, dim={r['space_dim']})" for r in rows]
sgain = [float(r["acc_gain_pp"]) for r in rows]
scorr = [float(r["corr_abs_change_pct"]) for r in rows]
x = list(range(len(rows)))

fig, ax1 = plt.subplots(figsize=(7, 4.5))
bars = ax1.bar(x, sgain, color="#2ca02c", alpha=0.75, label="准确率增益 (pp)")
ax1.axhline(0, color="black", lw=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel("PEM 准确率增益 (pp)", color="#2ca02c")
ax1.tick_params(axis="y", labelcolor="#2ca02c")
for b, v in zip(bars, sgain):
    ax1.annotate(
        f"{v:+.2f}",
        (b.get_x() + b.get_width() / 2, v),
        ha="center",
        va="bottom" if v >= 0 else "top",
        fontsize=9,
    )

ax2 = ax1.twinx()
ax2.plot(x, scorr, "s--", color="#d62728", lw=1.8, ms=7, label="corr_abs 增幅 (%)")
ax2.set_ylabel("corr_abs 增幅 (%)", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")

plt.title("PEM 净效应随作用空间维度的尺度响应（从零训练）")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_ex4_1_scale_response.png"), dpi=150)
plt.close(fig)

print("saved:")
print("  ", os.path.join(HERE, "fig_ex4_1_gamma_dose.png"))
print("  ", os.path.join(HERE, "fig_ex4_1_scale_response.png"))
