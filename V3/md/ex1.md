# 概念漂移测试log

环境齐全（torch 2.6 + transformers 5.3 + peft 0.18，CUDA 可用），GLUE 五个子任务与 Qwen2.5-0.5B 均在本地缓存。先做可行性探测，确认注入与速度：
两个关键发现需要确认。修正探测脚本：
发现一个阻断性 bug：`run_predictive_neural_dynamics` 只支持 2D，而 Transformer hidden state 是 3D，导致 Aeloru 在真实模型上完全无法运行。先做最小修复：
梯度被完全截断 —— 训练时第二次 `run_predictive_neural_dynamics` 在 `no_grad` 下返回，导致 LoRA 参数收不到任何梯度。用 straight-through 残差修复（前向数值完全等价）：
耗时分解清晰：基础 fwd+bwd ≈ 534ms，post_step 约 4.9ms/层。需要确定可行配置。做端到端有效性验证：
关键发现：Aeloru 的预测编码在真实 Transformer 上会破坏表示（loss 7.9 vs LoRA 3.87，完全学不动）。定位根因并验证修复方案：
根因全部定位。验证「安全配置」能否恢复可学习性：
loss 起点异常（7.6 vs LoRA 3.87）但理论分析不出原因。做逐层精确诊断：
诊断显示注入单层输出仅偏差 8.5%，但发现一个**对比不公平**问题：peft LoRA 注入了全部 24 层（345,856 参数），而 Aeloru 只注入 4 层（64,512）。验证并修正：
诊断完成，问题全部定位。核心结论：

| 组件 | 在预训练 Transformer 上的实测影响 |
|---|---|
| 预测编码神经动态 (`ndi>0`) | **致命** — 单层输出被放大 13× 或归零，模型完全学不动 |
| 稳态增益 (`homeostatic`) | **有害** — 把每层 RMS 拉向 `s^(1/3)`，L=24 时完全学不动（acc 0.468） |
| 域约束 `clamp(0,1)` | **有害** — 半数负值激活被截断 |
| 双一维门控/Hebbian/SPHeRe/DLAM | 中性或有益 |

且发现原对比不公平（peft 注入 24 层 vs Aeloru 4 层）。修正后 L=4 时 Aeloru 0.638 vs LoRA 0.682，可用。现在编写正式实验：
datasets 新版 API 变更（`Column` 对象无 `tolist`），修复：
脚本跑通。有三个问题需修复：`_offline_replay` 的调试打印污染输出、Forgetting/FWT 出现 NaN。先定位打印：
`spectral_filter` 里有两行遗留的裸调试 `print`：
Now fix the FWT/Forgetting NaN — FWT needs a zero-shot pre-task evaluation, which I'll add to the protocol:
Now wire the zero-shot pre-evaluation into the run loop:
全部修复。先用中等规模验证三个方法都能跑通：
三方法全部跑通，40 步时已见区分（aeloru-safe 遗忘率 0.0027 vs lora 0.0053）。先检查机制指标是否正常响应：
机制指标正常响应（GHL 方向随损失升降翻转、原型记忆逐任务累积、可塑性门控自适应），但发现**正交惩罚与 SPHeRe 结构损失未接入训练目标**——我用的是纯 CE loss。修复：
Now record the ortho loss and pass the lambda through:
所有机制现已生效（`loss_ortho` 0.003→0.75、`structure_loss` 0.0008→0.05、原型记忆 0→0.32）。现在跑全量实验，采用后台执行避免超时：
LoRA 完成（ACC 0.5976，BWT −0.0471，明显遗忘）。继续等待：
全量实验完成。查看详细结果：
R 矩阵结果明确。修复编码问题查看机制指标：
机制指标全部按设计工作（波动率 0.894→0.473 单调下降、原型记忆 0.18→0.38 累积、可塑性门控随漂移自适应）。检查图表：
图表缺少机制指标子图（`n_mech` 计算逻辑有缺陷，且 `bbox_inches="tight"` 裁掉了稀疏列）。重写绘图部分，拆成主图+机制图，并从已保存结果重新出图：
Now regenerate figures from the saved results without re-running the 12-minute experiment:
临时脚本删除需用户授权（已超时），我保留它们在报告中说明。最后做完整性与语法校验：
Lint 报告均为 Pylint 对 torch 的已知误报（`torch.linalg.inv is not callable` 等），无新增错误。重跑冒烟测试（上条命令中文引号在 PowerShell 解析失败）：
已在 `e:\Qelys\AeloRu\V3\ex1.py` 实现完整的概念漂移测试，跑通了 3 方法 × 5 任务 × 250 步全量实验，结果在 `ex1_out/`。

## 实验结果

| 方法 | ACC ↑ | BWT ↑ | FWT | Forgetting ↓ |
|---|---|---|---|---|
| `lora` (peft 基线) | 0.5976 | **−0.0471** | +0.0316 | **0.0538** |
| **`aeloru-safe`** | **0.6020** | **−0.0017** | +0.0216 | **0.0211** |
| `aeloru-default` (出厂默认) | 0.5311 | −0.0133 | +0.0277 | 0.0273 |

**Aeloru 的抗遗忘机制确实有效**：遗忘率 0.0538 → 0.0211（**降低 60.8%**），BWT −0.0471 → −0.0017（**改善 96.4%**，几乎无遗忘），且 ACC 同时略高。

从 R 矩阵看得很直观——SST2 作为第一个任务：
- LoRA：0.7200 → 0.4900（遗忘 **−0.230**）
- Aeloru-safe：0.6433 → 0.5767（遗忘仅 **−0.067**）

## 关键发现：v3.1 的三个组件在预训练 Transformer 上有害

测试过程中定位到 `aeloru_v3.py` 的若干阻断性缺陷，不修就无法跑通（均已做最小修复）：

1. **3D 输入不支持** — `run_predictive_neural_dynamics` 内部用 `y.t()`，只接受 2D，而 Transformer 隐藏状态是 `(B,S,D)`，注入后直接崩溃。已改为折叠成 2D 处理再恢复形状。

2. **梯度被完全截断** — 训练时第二次 `run_predictive_neural_dynamics` 在 `no_grad` 下返回并直接赋值给 `y`，导致 `lora_A/B/m_x/m_y` 收不到任何梯度，`loss.backward()` 直接报"does not require grad"。已改为 straight-through 残差 `y + (refined - y.detach())`，前向数值完全等价，梯度沿恒等通路回传。

3. **预测编码神经动态破坏表示** — 单层实测：`ndi=2` 时输出范数 573→0.193（几乎归零），`ndi=1` 时放大 13×。根因是重建误差项 `error = y - yke` 恒等于 bias（两者是同一基座输出），被 `gamma_predictive=100` 放大；且 `B_basis` 初始为 0 使精度矩阵退化为 `I/ε`，`lateral` 项把输出放大 5 倍。实测 `ndi>0` 时模型完全学不动。

4. **稳态增益量纲错误** — `gain = 1/running_var` 应为 `1/sqrt(running_var)`。反馈稳态会把每层输出 RMS 压缩到 `s^(1/3)`，24 层注入时 acc 只有 0.468（学不动），4 层才恢复正常。

5. **域约束 `clamp(0,1)`** 把约半数负值激活截断为 0（实测 min 从 −5.9 变 0）。

6. **`spectral_filter` 遗留两行裸 `print`**，每次睡眠刷屏输出。

另外发现 V2 的 `experiment_1.py` 有个**对比不公平**问题：Aeloru 分支未冻结基座参数，实际是 494M 全参数微调，与 peft LoRA 不可比。本实验已显式冻结，且让两边注入相同层数（LoRA 59,136 vs Aeloru 66,304）。

## 实验设计

**概念漂移构造**：SST2 → MRPC → QNLI → RTE → CoLA，每个任务训练 250 步后**立刻切换**且**永不回访**，任务边界处输入分布与标签语义同时突变。每任务结束回测所有已见任务得 R 矩阵，并新增**训练前零样本评估**以正确计算 FWT。

**三档方法**：`lora` / `aeloru-safe`（关闭上述有害组件）/ `aeloru-default`（出厂默认，用于暴露问题）。

**六项机制全部验证在工作**（见 `mechanisms.png` 与 `mech_*.csv`）：

| 机制 | sst2→cola 演化 | 说明 |
|---|---|---|
| 双一维 HGF 精度门控 | 0.640→0.584→0.621 | 随漂移自适应，非退化常数 |
| GHL 全局方向 | +0.121→−0.022 | 随损失升降翻转符号 |
| SPHeRe 结构保持 | 0.040→0.038 | 稳定约束（`loss_ortho` 0.26→0.43） |
| DLAM 原型记忆 | 0.180→**0.380**→0.275 | 睡眠后逐任务累积 |
| BCM 自适应阈值 | 3.27→3.28 | 稳定 |
| 波动率 exp(ω) | 0.894→**0.473** | 单调下降，模型渐趋稳定 |

## 使用

```bash
python ex1.py                                    # 全量 3 方法 × 250 步 ≈ 12 分钟
python ex1.py --methods lora,aeloru-safe --steps 100
python ex1.py --methods aeloru-safe --layers 8 --r 16 --alpha 32
```

输出 `ex1_drift_<时间戳>/`：`summary.csv`、`R_matrix.txt`、`concept_drift.png`、`mechanisms.png`、`mech_<method>.csv`、`log.txt`。

## 两点说明

1. **`_probe.py` / `_redraw.py` 是临时诊断脚本**。

2. **`aeloru-safe` 是定义的配置**，不是 v3.1 出厂默认。因为它关闭了预测编码、域约束并限制稳态增益上限——这三项在预训练 Transformer 上会破坏表示。如果你希望测试完整默认行为，`aeloru-default` 已经跑过（ACC 0.5311，明显更低），数据可作为对照。若要让默认配置在预训练模型上可用，建议把 `neural_dynamics_iterations`、`presumed_domain`、`homeostatic_max_gain` 的默认值改掉，或修正稳态增益的量纲。