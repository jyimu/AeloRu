# AeloRu:基于认知状态机的赫布学习与反向传播融合方法

**Jin Muyi**

## 摘要

大语言模型（LLM）在消费级GPU上的实时微调面临灾难性遗忘、算力效率低、参数可塑性与稳定性失衡等核心问题。反向传播（BP）作为主流微调方式具备全局梯度优化能力，但易破坏预训练知识且算力消耗高；赫布（Hebbian）学习基于生物突触可塑性原理，可实现局部在线增量更新，却缺乏全局优化导向。针对上述问题，本文提出一种基于认知状态机的赫布学习与BP融合新方法**AeloRu**(**A**daptive **E**lastic **L**earning with **O**rthogonal **R**obust **U**nits)——该方法以Hi-DoRA（幅度-方向解耦低秩适配）、ReLoRA（周期性合并重置）为技术基座，通过认知状态机动态协调赫布学习与BP的更新节奏，结合Fisher软门控掩码、正交惩罚与能量预算约束，实现两者的自适应协同。该方法将预训练权重作为不可侵犯的核心基座，通过外置累积缓冲区与工作记忆增量分离的双轨制结构，既保留BP的全局优化优势，又发挥赫布学习的在线可塑性，有效解决了低算力场景下LLM微调的遗忘与效率问题。代码详见 https://github.com/jyimu/AeloRu

## 1 引言

大模型微调是实现模型适配特定任务的核心手段，反向传播（BP）凭借全局梯度下降的优化能力成为主流范式，但在消费级GPU场景下存在两大局限：一是易引发灾难性遗忘，梯度更新易破坏预训练阶段形成的核心知识；二是全量梯度计算与更新的算力成本高，难以适配低显存、低算力的硬件环境。赫布学习作为受生物神经突触可塑性启发的局部更新规则，具有在线、增量、低算力消耗的特点，能够模拟大脑“神经元同步放电则连接强化”的学习机制，但其缺乏全局优化目标，单独使用易导致模型收敛至局部最优。

现有赫布与BP的融合尝试多为静态加权或简单时序叠加，缺乏动态协调机制，既无法规避赫布更新对核心参数的破坏，也难以发挥BP的全局优化价值。此外，低秩适配（如LoRA）虽降低了微调的算力消耗，但未解决赫布与BP融合过程中“可塑性-稳定性”的平衡问题。为此，本文提出一种以认知状态机为核心的动态融合方法：以Hi-DoRA、ReLoRA为轻量化基座技术，通过认知冲突检测驱动的状态流转，实现赫布学习与BP在不同认知阶段的自适应切换与协同，同时引入正交惩罚与能量预算约束，保障预训练知识的完整性，最终实现LLM的高效、低遗忘微调。

## 2 核心方法：赫布与BP的动态融合框架

### 2.1 融合的核心逻辑与设计原则

赫布学习与BP的融合需兼顾“全局优化（BP）-局部可塑性（赫布）”“知识保护-增量学习”两大维度，核心设计原则为：

1. 预训练权重（$W_0$）神圣不可侵犯，所有学习成果通过外置累积（$W_{acc}$）与工作记忆增量（$\Delta W$）承载；
2. 赫布学习负责局部突触可塑性的在线探索，BP负责全局梯度的优化收敛，两者的更新节奏由认知状态机动态调控；
3. 通过软门控与正则化约束，避免赫布与BP的更新相互干扰，保障模型稳定性。

融合后的有效权重合成遵循以下公式，实现“预训练基座+累积记忆+工作记忆”的双轨制结构：
$$W_{eff} = W_0 + W_{acc} + \text{Gate}(\Delta W)$$
其中，$\text{Gate}(\cdot)$ 为融合过程的核心门控函数，整合了Hi-DoRA幅度调制、Fisher掩码与能量预算约束，适配赫布与BP的更新特性；$W_{acc}$ 由ReLoRA周期性合并工作记忆增量形成，为赫布-BP融合提供稳定的累积记忆载体。

### 2.2 认知状态机驱动的融合机制

基于“探索→冲突→锚定→固化”的类脑认知闭环，设计Hong Wen认知状态机，实现赫布学习与BP在不同阶段的动态切换（状态流转：EXPLORE→RED→ANCHOR→SOLID→EXPLORE），核心逻辑如表1所示：

表1 认知状态机各阶段的赫布-BP融合规则

| 状态阶段       | 触发条件                | 赫布学习行为                | BP行为                     | 核心目标                     |
|----------------|-------------------------|-----------------------------|----------------------------|------------------------------|
| EXPLORE（探索） | 初始状态/固化阶段结束    | 允许在线更新，可塑性强      | 正常反向传播               | 全局探索，兼顾局部与全局优化 |
| RED（红温）    | 冲突分数＞0.65（阈值）| 暂停更新                    | Fisher掩码冻结，触发ReLoRA | 终止无效探索，保护核心知识   |
| ANCHOR（锚定） | 红温阶段触发            | 持续暂停                    | 主导更新，收紧Fisher保护   | 全局梯度收敛，锚定优化方向   |
| SOLID（固化）  | 梯度范数＜1e-4（收敛）| 仅Fisher低敏感区强化更新    | 暂停更新                   | 局部突触固化，沉淀增量知识   |

其中，冲突分数由Fisher掩码变化速度与探索熵共同计算，是判断认知状态切换的核心指标；SOLID阶段持续固定步数（默认200步）后回归探索阶段，形成闭环。该机制解决了赫布与BP“何时更新、何处更新”的核心问题：探索阶段充分发挥两者的互补性，红温与锚定阶段以BP保障全局收敛，固化阶段以赫布实现局部知识沉淀。

### 2.3 Fisher软门控：赫布-BP融合的保护机制

为避免赫布与BP的更新破坏预训练核心知识，引入Fisher软门控掩码，对工作记忆增量$\Delta W$进行调制：
$$\Delta W'' = \Delta W' \odot \frac{1}{1 + \gamma F}$$
其中，$\Delta W'$为Hi-DoRA调制后的低秩增量，$F$为Fisher信息矩阵（表征参数重要性），$\gamma$为掩码锐度系数。

该门控机制的核心价值在于：

1. 对高Fisher值的核心参数（预训练关键知识），掩码系数趋近于0，限制赫布与BP的更新幅度，形成“知识护城河”；
2. 软门控而非硬截断的设计，既保护核心参数，又不破坏BP梯度回传，避免梯度消失问题；
3. 为赫布学习划定“安全更新区域”（低Fisher值区域），使其仅在非核心参数区进行局部强化，与BP的全局优化形成互补。

## 3 基座技术适配：Hi-DoRA与ReLoRA

Hi-DoRA与ReLoRA并非本文核心创新，而是为赫布-BP融合提供轻量化实现基座：

### 3.1 Hi-DoRA：增量幅度的自适应调制

Hi-DoRA对低秩增量$\Delta W$进行行级幅度调制：
$$\Delta W' = \text{diag}(m) \cdot \Delta W$$
其中，$\Delta W = \frac{\alpha}{r} \cdot BA$（对齐LoRA标准范式），$\text{diag}(m)$为行范数调制矩阵。该技术降低了增量更新的显存占用，使$\Delta W$的幅度特性适配赫布-BP融合的动态更新节奏，避免无效更新干扰融合过程。

### 3.2 ReLoRA：累积记忆的周期性沉淀

ReLoRA将工作记忆增量$\Delta W$周期性合并至外置累积缓冲区$W_{acc}$，既解决了低秩增量的“秩塌陷”问题，又为赫布-BP融合提供稳定的长期记忆载体。合并操作在红温阶段（强制）或固定步数（merge_every）触发，不干扰认知状态机驱动的赫布-BP动态更新。

## 4 正则化与约束：融合效率的保障

### 4.1 正交惩罚：赫布-BP的高效学习导向

为避免赫布与BP重复学习预训练已掌握的知识，引入正交惩罚损失：
$$L_{ortho} = \lambda ||\Delta W^T \cdot W_0||_F^2$$
该损失迫使工作记忆增量$\Delta W$落入$W_0$的左零空间，强制赫布与BP仅探索预训练权重未覆盖的知识空间，提升低算力场景下的学习效率。

### 4.2 能量预算：融合过程的稳定性约束

设置硬约束限制增量规模：
$$||\Delta W||_F \le \eta \cdot ||W_0||_F$$
避免$\Delta W$（赫布与BP的更新增量）喧宾夺主，保障预训练基座的稳定性，同时适配消费级GPU的显存限制。

## 5 方法实现与配置

**以下只讲述部分内容,详见 [Aeloru]([www.github.com/aeloru/](https://github.com/jyimu/AeloRu/tree/main/V1#readme)) 中的核心实现**

### 5.1 配置参数（AeloruConfig）

核心配置围绕赫布-BP融合的调控展开，关键参数包括：

- 赫布学习：`hebbian_lr`（更新率）、`hebbian_decay`（遗忘衰减）；
- 认知状态机：`red_threshold`（红温触发阈值）、`solid_steps`（固化步数）；
- Fisher门控：`fisher_gamma`（掩码锐度）、`fisher_ema`（平滑系数）。

Hi-DoRA/ReLoRA的配置（如`r`、`merge_every`）仅作为基座参数，不主导方法逻辑。

### 5.2 代码实现流程

1. **模型注入**：通过`inject_aeloru`函数替换目标层，加载融合配置；
2. **训练步骤**：`train_aeloru_step`函数自动执行认知状态机流转、赫布-BP更新切换、ReLoRA合并等逻辑；
3. **增量保存**：仅存储$W_{acc}$与$\Delta W$，实现轻量化部署。

核心代码如下:

```python
from aeloru_layer import inject_aeloru, AeloruConfig, train_aeloru_step
import torch.optim as optim

# 1. 定义融合配置（重点调控赫布-BP相关参数）
config = AeloruConfig(
    r=8,  # ReLoRA/Hi-DoRA基座参数
    lora_alpha=4.0,
    use_hidora=True,  # 启用基座幅度调制
    use_relora=True,  # 启用基座累积合并
    use_hebbian=True,  # 启用赫布学习
    use_fisher=True,  # 启用Fisher门控
    use_hongwen=True,  # 启用认知状态机（核心）
    hebbian_lr=5e-4,  # 赫布学习率
    red_threshold=0.65,  # 红温阈值（状态机）
    fisher_gamma=10.0  # Fisher掩码锐度
)

# 2. 注入模型
model = inject_aeloru(model, target_names=["q_proj", "v_proj"], cfg=config)

# 3. 训练（自动协调赫布-BP更新）
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
for x, y_target in dataloader:
    loss, metrics = train_aeloru_step(
        layer=model.aeloru_layer,
        x=x, y_target=y_target,
        optimizer=optimizer
    )
    print(f"当前状态: {metrics['state']}, 赫布更新状态: {metrics['hebbian_active']}")
```

## 6 实现细节与性能评估(全过程见[AeloRu](https://github.com/jyimu/AeloRu/blob/main/V1/ex.ipynb))

**配置参数保持不变(配置如下:)**

```python
config = AeloruConfig(
    # --- 基础维度 ---
    in_features: int = 512              #输入维度
    out_features: int = 512             #输出维度
    r: int = 8                          # LoRA 秩
    lora_alpha: float = 4.0             # LoRA 缩放因子
    LoRA_lr: float = 1e-4               # LoRA 学习率
    # --- ReLoRA 参数 ---
    merge_every: int = 1000               # 固定合并周期（步数）
    merge_on_red: bool = True             # 红温时是否强制合并

    # --- Hi-DoRA 参数 ---
    # m 初始化为 W0 的行范数（dim=1），形状 (out_features,)
    # 用于对 DeltaW 的行进行幅度调制

    # --- Hebbian 参数 ---
    hebbian_lr: float = 1e-6              # Hebbian 学习率
    hebbian_decay: float = 0.99           # 全局遗忘衰减
    saturation_limit: float = 5.0         # 饱和上限（硬截断）

    # --- Fisher 参数 ---
    fisher_gamma: float = 10.0            # Fisher 掩码锐度
    fisher_ema: float = 0.95              # Fisher EMA 平滑系数
    plasticity_min: float = 0.05          # 最小可塑性（防止完全冻结）

    # --- Hong Wen 红温参数 ---
    red_threshold: float = 0.65            # 冲突分数触发线(Hong Wen 机制的冲突触发阈值)
    snapshot_interval: int = 50           # Fisher 快照间隔（步数）
    anchor_converge: float = 1e-4         # 锚定期梯度收敛阈值
    solid_steps: int = 200                # 固化期持续步数

    # --- 正交惩罚参数 ---
    ortho_lambda: float = 0.01            # 正交惩罚系数
    ortho_lambda_anchor: float = 0.05     # 锚定期强化系数（5x）

    # --- 能量预算参数 ---
    energy_eta: float = 0.15              # DeltaW 能量不超过 W0 的 eta 比例

)
```

### 6.1 单任务性能对比实验

> 实验目的：证明 Aeloru 在单任务微调上，性能和主流 PEFT 方法相当，没有为了连续学习牺牲基础能力

#### 实验设计：

1. 分别用 LoRA、DoRA、ReLoRA、Aeloru 在上述 5 个数据集上做单任务微调
2. 记录每个方法在每个数据集上的最终准确率和困惑度
3. 用表格呈现结果，最后一行加平均性能

#### 实现结果:





### 6.2 连续学习核心对比实验

> 实验目的：这是整篇论文最重要的实验，直接证明 Aeloru 在缓解灾难性遗忘上的核心优势。

### 实验设计：

1. 定义连续学习任务序列：Alpaca → GSM8K → SQuAD → IMDB → TriviaQA
2. 按顺序训练 5 个任务，每个任务训练 3 个 epoch，训练完一个任务后，立即在所有之前学过的任务上测试性能
3. 记录每个方法在每个任务学习完成后的性能变化
4. 计算最终的平均准确率和平均遗忘率

#### 实现结果:




### 6.3 模块消融实验

>实验目的：证明提出的每一个模块（Hi-DoRA、Hebbian-Fisher、Hong Wen）都对最终性能有贡献

##### 实验设计：

   以完整的 Aeloru 为基线，逐个关闭每个模块，得到 5 个变体：
   - Aeloru w/o Hi-DoRA：用普通 LoRA 代替 Hi-DoRA
   - Aeloru w/o Hebbian-Fisher：关闭 Fisher 门控和 Hebbian 更新
   - Aeloru w/o Hong Wen：关闭认知状态机，用固定学习率训练
   - Aeloru w/o 双轨记忆：去掉外置累积缓冲区，只用单个 LoRA
   - 完整 Aeloru
---
在上述连续学习任务序列上，测试所有变体的性能
对比平均遗忘率和平均准确率的变化

### 6.4 效率对比实验

> 实验目的：证明 Aeloru 虽然引入了更多机制，但计算开销和主流方法相当，没有明显的速度损失。

#### 实验设计：

1. 在相同的硬件和超参数下，记录 4 种方法训练 100 步的平均时间
2. 记录每个方法的峰值显存占用


## 7 结论

本文提出的基于认知状态机的赫布学习与BP融合方法，突破了现有融合策略“静态叠加、缺乏协调”的局限：以认知状态机动态调控两者的更新节奏，以Fisher软门控保障预训练知识安全，以正交惩罚与能量预算提升融合效率，Hi-DoRA/ReLoRA则为该融合提供了轻量化的实现基座。该方法既保留了BP的全局优化能力，又发挥了赫布学习的生物启发式可塑性，解决了消费级GPU上LLM微调的灾难性遗忘与算力效率问题，为大模型高效微调提供了全新的融合范式。未来可进一步探索认知状态机的自适应阈值设计，以及赫布学习规则的个性化优化，提升融合方法的泛化能力。