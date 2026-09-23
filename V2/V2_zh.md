# AeloRu V2 — 认知机制整合

**A**daptive **E**lastic **L**earning with **O**rthogonal **R**obust **U**nits

V2 将全套认知科学机制整合进 AeloRu 层，并以基于层级高斯滤波（HGF）的**闭式、无梯度**方案替代了昂贵的 Fisher 估计。

> 闭式梯度推导（MSE → CE / BCE）见 [`HGF_CE.md`](HGF_CE.md)。

---

## 相对 V1 的新增内容

| 新增 | 原理 | 作用 |
| :--- | :--- | :--- |
| **PEM — 自适应侧向连接** | 低秩侧向抑制 | 在低秩空间内做特征解耦与冗余相关性抑制。 |
| **PEM — 稳态可塑性** | 神经元级自适应增益 | `g_i = 1/(running_var_i + 1e-5)` 防止权重爆炸。 |
| **HGF 闭式 Fisher** | 精度传播 | 以闭式精度替代梯度平方 Fisher；门控 `O(d²) → O(d)`。 |
| **HGF 波动率耦合** | 动态学习率 | `ω += κ·(ε² − exp(ω))`，`lr_eff = 1/exp(ω)` 适配概念漂移。 |
| **DLAM 谱滤波睡眠** | 离线巩固 | 对赫布痕迹做谱滤波；剪枝弱连接 + 跨层原型绑定。 |

### 双一维 HGF 可塑性门控（关键优化）
参数级 Fisher 掩码 `ΔW'' = ΔW' ⊙ 1/(1+γF)` 需要 `O(d_out·d_in)` 内存和昂贵的梯度路径。V2 观察到 `ΔW` 只有 `O(r(d_out+d_in))` 个自由度，转而维护两条**一维精度向量** `π_out`、`π_in`（由 HGF `running_var` 与输入协方差 `C_x` 闭式得到）。它们的外积近似二维 Fisher，并同时门控赫布更新与权重增量——在结构上与 Hi-DoRA 的 `(m_x · m_yᵀ)` 双一维调制同构。

---

## 关键公式

```
稳态增益      g_i   = 1 / (running_var_i + 1e-5)
侧向抑制      h'    = h - h @ Lᵀ,    dL/dt = η_lat · outer(h_mean, h_mean)
HGF 精度      π     = π_pred + π_prev · α² · (g')²
波动率        ω    += κ·(ε² - exp(ω)),   lr_eff = 1/exp(ω)
DLAM 核       A(t)  = (1+t)·(I + t·Σ)⁻¹
```

### 闭式梯度（无 autograd）
只有输出层对 logits 的局部梯度 `δ = ∂L/∂z` 因损失不同而不同；适配器内部反向链不变：

- **MSE**：`δ_MSE ≈ z − y_target`
- **Softmax-CE**：`δ_CE = p − y_target`（sum-gradient 约定）

```
grad_B = δᵀ h' · (α/r),    grad_A = [(δ B) ⊙ 𝟙_{h>0}]ᵀ x · (α/r)
```

---

## 仓库结构

```
V2/
├── aeloru_layer.py        # V2 实现（PEM / HGF / DLAM）
├── aeloru_layer copy.py   # 备份
├── experiment_1.py        # 训练实验
├── experiment.ipynb
├── PR.ipynb
├── HGF_CE.md              # 闭式 MSE→CE 推导
└── plot_data/
```

---

## 状态与注意事项

- **状态**：🔄 理论整合。PEM/稳态模块最初是为从零训练的认知模型设计的。
- ⚠️ **对比不公平 bug**（在 V3 验证时发现，见 `V3/md/ex1.md`）：
  `experiment_1.py` 在 AeloRu 分支中**未冻结基座模型**，实际跑的是 494M 全参数微调——与 peft LoRA 不可比。V3 通过显式冻结基座并统一注入层数来修复。
- ⚠️ 在**完整输出维度**（`d_out`）上施加的 PEM 会在真实 Transformer 上破坏预训练激活。V3 将 PEM 迁移到 **r 维低秩特征空间**（`pem_space="lowrank"`）；V2 的 `output` 空间 PEM 仅对从零训练的认知模型安全（`AeloruConfig.cognitive_model()`）。
