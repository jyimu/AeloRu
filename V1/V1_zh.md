# AeloRu V1 — 基础实现

**A**daptive **E**lastic **L**earning with **O**rthogonal **R**obust **U**nits

V1 是 AeloRu 层的基础、稳定实现。它确立了所有后续版本所构建的核心架构：一套受神经科学启发的低秩适配器，配备冲突驱动的认知状态机。

> 详细设计（公式、配置参考、代码示例）见 [`README.md`](README.md)。本文件是快速入口。

---

## 已实现的模块

| 模块 | 原理 | 作用 |
| :--- | :--- | :--- |
| **Hi-DoRA** | 幅值—方向解耦 | 行列可学习幅值向量调制 `ΔW = (m_x · m_yᵀ) ⊙ (α/r · B@A)`。 |
| **ReLoRA** | 周期性合并重置 | 将增量累积到外部缓冲 `W_acc`，实现高秩巩固。 |
| **Hebbian-Fisher** | 双向突触可塑性 | 赫布痕迹记录探索；Fisher 掩码保护关键权重。 |
| **红温状态机** | 四阶段认知循环 | EXPLORE → RED → ANCHOR → SOLID，由冲突分数 `C = 0.6·v_F + 0.4·(1−H)` 驱动。 |
| **分层 Fisher** | 梯度 / 稀疏 / 全量快照 | 梯度（高频）+ 稀疏 Top-K（中频）+ 量化全量快照（低频）。 |
| **正交惩罚** | `L_ortho = λ‖ΔWᵀW₀‖²` | 将 `ΔW` 推入 `W₀` 的左零空间。 |
| **能量预算** | `‖ΔW‖_F ≤ η‖W₀‖_F` | 硬约束，防止适配器压垮预训练权重。 |

有效权重：`W_eff = W₀ + W_acc + Gate(ΔW)`。

---

## 核心 API

```python
from aeloru_layer import inject_aeloru, AeloruConfig, train_aeloru_step

config = AeloruConfig(r=8, lora_alpha=4.0,
                      use_hidora=True, use_relora=True, use_hongwen=True)
model = inject_aeloru(model, target_names=["q_proj", "v_proj"], cfg=config)

for x, y in dataloader:
    loss, metrics = train_aeloru_step(layer, x, y_target=y, optimizer=optimizer)
    print(metrics["state"], metrics["loss_total"])

layer.save_adapter("ckpt.pt")   # 仅保存增量参数
layer.load_adapter("ckpt.pt")
```

---

## 状态与说明

- **状态**：✅ 稳定基础。所有模块均可通过 `AeloruConfig` 独立开关。
- V1 使用原始的**二维参数级 Fisher** 掩码（`d_out × d_in`）。其 `O(d²)` 开销后来在 V2/V3 中通过双一维 HGF 精度门控降至 `O(d)`。
- 冲突检测缓冲区的接线在 V1 中不完整，已在 V3 修复（见 `V3/md/fix1.md`）。
- 配套文档：[`README.md`](README.md)（英文设计）、[`README_zh.md`](README_zh.md)（中文）、
  `experiment/` 为隐藏状态语义分析（P0）。
