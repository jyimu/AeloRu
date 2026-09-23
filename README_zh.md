# AeloRu

**A**daptive **E**lastic **L**earning with **O**rthogonal **R**obust **U**nits

超越低秩：通过幅值—方向解耦实现的动态可塑性

论文：[AeloRu: Adaptive Elastic Learning with Orthogonal Robust Units](paper/paper.md)

---

## AeloRu 是什么？

AeloRu 是一套**受神经科学启发的低秩适配（PEFT）框架**，它在保持消费级 GPU 内存效率的同时，整合了多种协同的认知学习机制。AeloRu 不把微调纯粹当作统计优化问题，而是借鉴生物学习的原理——自适应突触可塑性、阶段性认知循环、离线睡眠巩固、波动率自适应学习率——并将其统一到一个低秩适配器架构中。

> **核心理念**：预训练权重神圣不可侵犯；所有学习成果在外部累积。有效权重：`W_eff = W_0 + W_acc + Gate(ΔW)`。

> **核心研究问题**：如何将赫布学习原理与基于 LoRA 的参数高效微调相结合，在长期顺序任务学习中缓解灾难性遗忘，同时在短期场景下提升适应速度？

### 核心机制

| 机制 | 技术原理 | 功能 |
| :--- | :--- | :--- |
| **Hi-DoRA** | 双向幅值—方向解耦 | 独立可学习的行列幅值向量，以 < 0.01% 的额外参数量解耦幅值与方向。 |
| **双一维 HGF 可塑性门控** | 闭式精度传播 | 通过两条一维精度向量（π_out, π_in）将 Fisher 门控从 `O(d²)` 降至 `O(d)`；零梯度开销。 |
| **红温（HongWen）状态机** | 四阶段认知循环 | EXPLORE → RED（冲突）→ ANCHOR → SOLID，模拟生物学习的快慢双时间尺度。 |
| **HGF 闭式自然梯度** | 波动率耦合动态学习率 | 以闭式精度替代昂贵的梯度平方 Fisher，自动适配概念漂移。 |
| **DLAM 谱滤波睡眠** | 离线巩固与剪枝 | 对赫布痕迹做谱滤波，剪枝弱连接，提升泛化能力。 |

---

## 关键结果

在 **Qwen2.5-0.5B** 上，采用极端概念漂移协议（SST-2 → MRPC → QNLI → RTE → CoLA，训练后永不回访），在同等参数量预算下评估：

- **灾难性遗忘** 从 LoRA 的 0.0538 降至 **0.0076（−85.9%）**
- **后向迁移（BWT）** 由 −0.0471 转为 **+0.0011**
- **最终准确率** 提升 **+0.83 个百分点**
- **代价**：仅增加 **+22.1%** 的训练时间
- **几何特性**：维度间相关性 −46% ~ −65%、条件数 −45% ~ −69%、秩预算利用率 43% ~ 61% → 71% ~ 85%、任务选择性 −78% ~ −86%

收益的源泉是消除低秩子空间冗余，而非单纯的正则化。论文进一步将特征耦合分解为功能性与冗余两部分，开展了 PEM 剂量—反应实验，并记录了将认知科学模块移植到预训练 Transformer 时发现的 13 项阻断性缺陷（及修复）。（完整内容见 [paper.md](paper/paper.md)。）

---

## 项目路线图

| 阶段 | 目标 | 状态 |
|-------|-----------|--------|
| P0 | 隐藏状态语义分析 | ✅ 已完成 ([报告](LoRA%20ΔW%20Semantic%20Similarity%20Verification%20Experiment%20Report(LoRA%20ΔW%20%E8%AF%AD%E4%B9%89%E7%9B%B8%E4%BC%BC%E6%80%A7%E9%AA%8C%E8%AF%81%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8A).md)) |
| P1 | HiRA-DoRA 融合实现 | ✅ 已完成 ([V1](V1/README.md)) |
| P2 | 赫布-RL 混合学习 | ✅ 已完成 ([V1](V1/README.md)) |
| P3 | 真实数据训练与论文撰写 | ✅ 已完成 ([论文](paper/paper.md)) |
| P4 | 异步 PEFT 架构 | 🔄 进行中 |

### 版本演进 ([最新进展](paper/paper.md))

- **V1** — 核心 AeloRu 层、红温状态机、ReLoRA 合并、Hebbian-Fisher → [V1.md](V1/V1.md) · [`V1/README.md`](V1/README.md)
- **V2** — 双一维 HGF 可塑性门控、闭式 CE 梯度、PEM/DLAM 整合 → [V2.md](V2/V2.md) · [`V2/HGF_CE.md`](V2/HGF_CE.md)
- **V3** — 全套基准实验（ex1–ex4）、PEM 剂量—反应、子空间几何分析、13 项缺陷修复 → [V3.md](V3/V3.md) · [`V3/md/`](V3/md/)

---

## 仓库结构

```
AeloRu/
├── V1/                 # 核心实现：aeloru_layer.py，红温状态机
├── V2/                 # 双一维 HGF 门控、闭式梯度（experiment_1.py）
├── V3/                 # 基准实验 ex1–ex4、子空间几何分析
│   ├── md/             # 实验笔记（ex1, ex2, ex3, fix1…）
│   └── try_ex/         # 验证与探测脚本
├── experiment/         # 隐藏状态语义分析（P0）：analyze_hidden_states*.py
├── paper/              # 论文草稿（paper.md）、图、实验数据（ex1–ex4）
├── tests/              # 单元测试（test_aeloru_logging.py）
├── logs/               # 训练日志
├── output/ plots/ visualization/   # 生成的产物
└── README.md
```

---

## 快速开始

```python
from V1.aeloru_layer import inject_aeloru, AeloruConfig

config = AeloruConfig(
    r=8,
    lora_alpha=4.0,
    use_hidora=True,
    use_relora=True,
    use_hongwen=True,
)
model = inject_aeloru(model, target_names=["q_proj", "v_proj"], cfg=config)
```

完整配置参考（`AeloruConfig`）、单步训练循环（`train_aeloru_step`）与适配器检查点存取，请参阅 [V1/README.md](V1/README.md)。

---

## 预期贡献

1. **赫布-RL 混合学习**，实现实时自适应可塑性
2. **适配器级异步**，在持续学习期间实现零延迟服务
3. **双一维 HGF 可塑性门控** —— 面向消费级 GPU 的 `O(d²)` → `O(d)` Fisher 门控
4. **DLAM 谱滤波睡眠**，用于离线巩固与抗遗忘

---

## 许可证

MIT License - 详见 [LICENSE](LICENSE)。
