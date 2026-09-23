# AeloRu V3 — 基准测试与验证

**A**daptive **E**lastic **L**earning with **O**rthogonal **R**obust **U**nits

V3 是**实验验证**分支。它将 V2 的机制重构以适配真实 Transformer 微调，修复了一系列阻断性缺陷，并运行支撑论文的完整概念漂移基准。

> 详细实验日志：[`md/ex1.md`](md/ex1.md) · [`md/ex2.md`](md/ex2.md) ·
> [`md/ex3.md`](md/ex3.md) · [`md/fix1.md`](md/fix1.md) · [`md/fix1.1.md`](md/fix1.1.md)。
> 设计笔记：[`up.md`](up.md)。

---

## v3.1 重构（效率提升 + 机制增强）

- **双一维 HGF 可塑性门控**（`O(d²) → O(d)`）：移除 `fisher_mask` / `fisher_mask_inv`；
  输出侧门控复用 HGF `running_var`，输入侧复用 `C_x` 对角线；行列一维缩放，与 Hi-DoRA 结构一致。
- **GHL** 全局方向赫布：更新符号 = 局部奖励与全局梯度/损失方向的凸组合。
- **SPHeRe** 低维结构投影：正交惩罚升级为 `W₀ 方向重叠 + 低秩协方差结构` 损失。
- **STDP** 噪声梯度：在 `_flush_hebbian` 中注入与波动率/冲突分数/认知状态挂钩的高斯噪声。
- **BCM** 自适应阈值、**WTA** 硬竞争（r 中取 top-k）、**DLAM 原型记忆**（睡眠阶段提取）。

## v3.2 修复（预训练 LLM 兼容性）

| # | 修复 | 影响 |
| :--- | :--- | :--- |
| D | 稳态增益 `1/running_var` → `1/√running_var` | 修复多层注入时的 RMS 坍缩 |
| E | PEM 迁移至 **r 维低秩空间**（`_low_rank_neural_dynamics`） | 不再破坏预训练激活 |
| F | `_init_precision_basis`（用 `lora_B` 初始化 `B_basis`） | 缓解冷启动爆炸 |
| G | `AeloruConfig.llm_tuning()` / `cognitive_model()` 预设 | 为用户提供安全默认值 |
| H | 接入 `_append_hgf_delta`；移除双重缩放；`whitening_interval→1e9` | 红温冲突检测真正生效 |

---

## 基准测试套件

在 **Qwen2.5-0.5B** 上评估，概念漂移协议 SST-2 → MRPC → QNLI → RTE → CoLA，永不回访。

### ex1 — 概念漂移遗忘（3 方法 × 5 任务 × 250 步）

| 方法 | ACC ↑ | BWT ↑ | 遗忘 ↓ |
| :--- | ---: | ---: | ---: |
| `lora`（peft） | 0.5976 | **−0.0471** | **0.0538** |
| `aeloru-safe` | **0.6020** | **−0.0017** | **0.0211** |
| `aeloru-default` | 0.5311 | −0.0133 | 0.0273 |

**遗忘降低 60.8%**，BWT 改善 96.4%（近乎零遗忘），ACC 略高。

### ex2 — 子空间解耦（6 组消融）

LoRA 的 `corr_abs` 是 AeloRu 的 2.8 倍、TC 高 4.6 倍、有效秩低 28%、遗忘高 3.8 倍。`nowta` / `nosphere` 对解耦贡献最大。
⚠️ 反直觉发现：`aeloru-nopem`（关闭 PEM）的 `corr_abs` 反而*更低* —— PEM 是 r 空间解耦的**负贡献**（见 ex3）。

### ex3 — 冗余耦合 vs. 功能性耦合（PEM 之问）

通过高斯闭式任务信息 + Fisher 加权，将相关性分解为功能性（`corr_useful`）与冗余（`corr_useless` / `coup_red`）两部分。

- **PEM 自身没有任务选择性**（开环 SEL ≈ 0）。
- 选择性是**闭环涌现效应**：PEM 破坏相关性 → BP 只修复任务有用的部分。
- 选择性**依赖能量结构 / 谱**：当冗余耦合占据主方向时，PEM 的精度加权去相关会剪掉功能性相关。
- 结论：论文 §5.6 措辞需改写（不必删除 PEM）；PEM 在低秩空间施加的是精度加权、无差别的去相关压力。

### ex4 — 从零训练 PEM 空间

`pem_space` 消融：output 空间 PEM 从零训练带来 +2.74pp 准确率；lowrank 空间为 −1.40pp —— 但**两者**都抬高耦合、降低有效秩，故 PEM 的收益来自拟合调制，而非解耦。

---

## 记录的 13 项阻断性缺陷（将认知模块移植到 Transformer）

主要发现（完整列表见 `md/ex1.md`、`md/fix1.md`）：

1. `run_predictive_neural_dynamics` 仅支持 2D；Transformer 隐藏状态是 3D → 崩溃。
2. 梯度被完全截断（第二次调用在 `no_grad` 下）；用 straight-through 残差修复。
3. **预测编码神经动态破坏表征**（`ndi>0` → 模型学不动）。
4. **稳态增益量纲错误**（`1/var` 应为 `1/√var`）。
5. `clamp(0,1)` 域约束截断约半数负激活。
6. `spectral_filter` 遗留调试 `print`。
7. `spectral_filter` 遗留调试 `print`。
8. V2 `experiment_1.py` 对比不公平（基座未冻结）。
9. `ΔW` 精度陷阱（bf16 相减 → 虚假秩）。
10. 热力图低估相关性（符号抵消）。
11. `_append_hgf_delta` 从未被调用 → 红温冲突检测失效。
12. `batch_forward_lora` 双重缩放隐患。
13. `named_buffers` 点分路径 `setattr` bug → 子模块缓冲未搬运到设备。

其中，"全输出空间预测编码破坏预训练表征"与"稳态增益量纲错误"被标记为对同类神经启发工作具有**普遍警示意义**。

---

## 仓库结构

```
V3/
├── aeloru_v3.py            # v3.2 实现
├── ex1.py                  # 概念漂移遗忘基准
├── ex2.py                  # 子空间解耦（r=8/16/32）
├── ex3_coupling.py         # 冗余 vs 功能性耦合（LLM）
├── ex3_synth.py            # 合成 ground-truth 校准
├── ex4_fromscratch.py      # 从零训练 PEM 空间消融
├── test.py / text.py
├── md/                     # 实验与修复日志
├── try_ex/                 # 验证 / 探测脚本
```

## 状态

- **状态**：✅ 基准完成，支撑论文（`paper/paper.md`）。
- 推荐入口配置：预训练 LLM 微调使用 `AeloruConfig.llm_tuning()`。
