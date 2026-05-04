
# Aeloru: 面向消费级 GPU 的 LLM 实时训练框架

**A**daptive **E**lastic **L**earning with **O**rthogonal **R**obust **U**nits(具有正交稳健单元的自适应弹性学习)
> **核心理念**：预训练权重神圣不可侵犯，所有学习成果外置累积。
> **目标**：通过认知冲突检测实现“探索 -> 冲突 -> 锚定 -> 固化”的类脑闭环训练。
> **项目地址**：[`www.github.com/jyimu/AeloRu`](www.github.com/jyimu/AeloRu)
> **代码位置**: [`www.github.com/jyimu/AeloRu/blob/main/V1/aeloru_layer.py`](www.github.com/jyimu/AeloRu/blob/main/V1/aeloru_layer.py)

---

## 1. 核心设计概览

Aeloru 是一套融合了多种前沿微调技术的框架，旨在解决大模型在消费级硬件上的实时训练难题。它将学习过程解耦为**神圣基座**、**外置累积**和**当前工作记忆**。

### 1.1 核心融合技术

| 模块名称 | 技术原理 | 作用 |
| :--- | :--- | :--- |
| **Hi-DoRA** | 幅度-方向解耦的低秩适配 | 对增量权重进行行范数调制，保留预训练权重的幅度特性。 |
| **ReLoRA** | 周期性合并重置 | 将低秩增量合并到外置缓冲区，实现高秩知识的累积与沉淀。 |
| **Hebbian-Fisher** | 双向突触可塑性 | Fisher 掩码保护重要参数（稳固），Hebbian 规则记录探索痕迹（可塑）。 |
| **Hong Wen** | 认知状态机 | 基于冲突分数的四相循环（探索/红温/锚定/固化），模拟类脑学习过程。 |

### 1.2 有效权重合成公式

Aeloru 的推理权重 $W_{eff}$ 由三部分组成，确保了预训练知识的**零遗忘**：

$$W_{eff} = W_0 + W_{acc} + \text{Gate}(\Delta W)$$

其中：
*   $W_0$：神圣基座（Frozen Pretrained Weights）。
*   $W_{acc}$：外置累积缓冲区（Accumulated Knowledge）。
*   $\Delta W$：当前周期的低秩适配器增量（LoRA A & B）。
*   $\text{Gate}(\cdot)$：包含 Hi-DoRA 调制、Fisher 掩码和能量预算的门控函数。

---

## 2. 核心算法机制

### 2.1 Hong Wen 认知状态机
框架模拟了人类认知过程，包含四个状态，通过冲突分数（Fisher 变化速度 + 探索熵）进行驱动：
>状态闭环流转：EXPLORE(自由探索) → RED(认知冲突/红温) → ANCHOR(过程锚定) → SOLID(赫布固化) → EXPLORE(回归)
1.  **EXPLORE (自由探索)**：
    *   **行为**：允许 Hebbian 更新和 BP 反向传播。
    *   **特征**：Fisher 掩码较低，参数可塑性强。
2.  **RED (认知冲突/红温)**：
    *   **触发**：冲突分数超过阈值（默认 0.65）。
    *   **行为**：暂停 Hebbian，冻结 Fisher，可能触发强制合并（ReLoRA）。
3.  **ANCHOR (过程锚定)**：
    *   **行为**：BP 主导，收紧 Fisher 保护。
    *   **退出**：当梯度范数收敛（默认 < 1e-4）时，转入固化期。
4.  **SOLID (赫布固化)**：
    *   **行为**：仅允许在 Fisher 低敏感区域进行 Hebbian 强化。
    *   **持续**：固定步数（默认 200 步）后回归探索期。

### 2.2 正则化与约束
*   **正交惩罚 (Orthogonal Penalty)**：
    *   损失函数增加项：$L_{ortho} = \lambda ||\Delta W^T \cdot W_0||_F^2$。
    *   **目的**：迫使 $\Delta W$ 落入 $W_0$ 的左零空间，避免破坏预训练流形。
*   **能量预算 (Energy Budget)**：
    *   硬约束：$||\Delta W||_F \le \eta \cdot ||W_0||_F$。
    *   **目的**：防止微调增量喧宾夺主，破坏模型稳定性。

---

## 3. 核心公式体系与设计初心

Aeloru 的架构设计严格遵循数学定义，每一项公式都对应着明确的认知计算逻辑。以下是核心公式的学术映射与设计初衷解析：

### 3.1 四大核心公式表

| 核心公式 | 学术亮点 (Academic Highlights) | 设计初心 (Design Intent) |
| :--- | :--- | :--- |
| **1. 低秩增量**$\Delta W = \frac{\alpha}{r} \cdot B A$ | **生态兼容性**：完全对齐 LoRA 的标准缩放范式，保证了和现有 PEFT 生态的兼容性，可直接无缝替换现有 LoRA 模块。 | **工作记忆隔离**：作为「当前周期工作记忆」，将每一轮的增量与历史累积权重分离，为后续的 ReLoRA 合并与红温锚定打下数学基础。 |
| **2. Hi-DoRA 调制**$\Delta W' = \text{diag}(m) \cdot \Delta W$ | **幅度解耦**：用行级幅度调制复刻 DoRA 权重分解思想，保持增量形式轻量化，比原版 DoRA 显存占用更低、训练更稳定。 | **自适应弹性**：给低秩增量增加了自适应幅度调控，避免无效更新，完美贴合「自适应弹性学习」的核心定位。 |
| **3. Fisher 门控**$\Delta W'' = \Delta W' \odot \frac{1}{1 + \gamma F}$ | **软门控保护**：顶会认可的软门控写法，比硬截断高级：既保护高 Fisher 重要权重，又不破坏梯度回传，避免梯度消失。 | **知识护城河**：作为「核心知识保护机制」，完美解决灾难性遗忘，与 Hebbian 痕迹沉淀双向联动，构筑架构的核心护城河。 |
| **4. 有效权重**$W_{eff} = W_0 + W_{acc} + \Delta W'''$ | **双轨制结构**：加法形式对齐 LoRA 零侵入设计，明确引入累积权重 $W_{acc}$，定义了 ReLoRA 合并的数学形式。 | **记忆双轨制**：实现「长期记忆 + 工作记忆」，预训练权重 $W_0$ 神圣不动，长期记忆沉淀在 $W_{acc}$，当前学习存于 $\Delta W'''$，逻辑完全自洽。 |

### 3.2 辅助正则化公式

除了上述核心权重计算，Aeloru 还引入了正交约束来优化学习效率：

*   **正交惩罚损失 (Orthogonal Penalty)**
    $$L_{ortho} = \lambda ||\Delta W^T W_0||_F^2$$
    *   **物理意义**：惩罚增量权重 $\Delta W$ 与预训练权重 $W_0$ 的方向重合。
    *   **设计价值**：作为「自动避重学习机制」，强制模型探索 $W_0$ 的零空间，让模型只学新东西，不浪费算力在预训练已掌握的内容上，完美适配低算力设备。

---

## 4. 配置参数详解 (`AeloruConfig`)

你可以通过调整 `AeloruConfig` 来控制框架的行为模式。

### 4.1 基础维度
*   `in_features`, `out_features`: 层的输入输出维度。
*   `r`: LoRA 秩 (Rank)，控制适配器的宽度。
*   `lora_alpha`: LoRA 缩放系数。
*   `LoRA_lr`: LoRA 专用学习率。

### 4.2 功能开关
*   `use_hidora`: 是否启用幅度调制。
*   `use_relora`: 是否启用外置合并。
*   `use_hebbian`: 是否启用 Hebbian 在线更新。
*   `use_fisher`: 是否启用 Fisher 认知掩码。
*   `use_hongwen`: 是否启用认知状态机。
*   `use_orthogonal_penalty`: 是否计算正交损失。
*   `use_energy_budget`: 是否启用能量硬约束。

### 4.3 关键超参数
*   **ReLoRA**: `merge_every` (合并周期步数), `merge_on_red` (红温强制合并)。
*   **Hebbian**: `hebbian_lr` (Hebbian 学习率), `hebbian_decay` (遗忘衰减)。
*   **Fisher**: `fisher_gamma` (掩码锐度), `fisher_ema` (平滑系数)。
*   **Hong Wen**: `red_threshold` (红温阈值), `solid_steps` (固化步数)。

---

## 5. 代码工作流示例

### 5.1 模型注入
使用 `inject_aeloru` 函数递归替换模型中的目标层（如 `q_proj`, `v_proj` 等）。

```python
from aeloru_layer import inject_aeloru, AeloruConfig

# 定义配置
config = AeloruConfig(
    r=8, 
    lora_alpha=4.0,
    use_hidora=True,
    use_relora=True,
    use_hongwen=True
)

# 注入模型
model = inject_aeloru(model, target_names=["q_proj", "v_proj"], cfg=config)
```

### 5.2 训练步骤
使用封装好的 `train_aeloru_step` 进行单步训练，它自动处理了状态机流转和 ReLoRA 合并。

```python
from aeloru_layer import train_aeloru_step

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for x, y_target in dataloader:
    # 自动处理前向、状态检测、BP、合并逻辑
    loss, metrics = train_aeloru_step(
        layer=model.your_aeloru_layer, 
        x=x, 
        y_target=y_target, 
        optimizer=optimizer
    )
    print(f"State: {metrics['state']}, Loss: {metrics['loss_total']}")
```

### 5.3 保存与加载
Aeloru 支持独立的适配器保存，仅存储增量参数。

```python
# 保存
layer.save_adapter("checkpoints/aeloru_step1000.pt")

# 加载
layer.load_adapter("checkpoints/aeloru_step1000.pt")
```

---

## 6. 总结

**Aeloru** 不仅仅是一个微调算法，它是一个**认知计算架构**。通过将 **ReLoRA** 的高秩累积能力、**Hi-DoRA** 的参数效率、**Hebbian-Fisher** 的生物启发式稳定性以及 **Hong Wen** 的状态机控制相结合，它为在有限算力下持续训练大语言模型提供了一种极具潜力的解决方案。