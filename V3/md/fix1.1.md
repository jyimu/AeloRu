
## 修改清单

| # | 修改 | 位置 |
|---|---|---|
| 一 | 稳态增益 `1/running_var` → `1/sqrt(running_var + eps)`；`homeostatic_max_gain` 默认 5.0 → 2.0 | `forward` 第 7 步、config |
| 二 | 新增 `_low_rank_neural_dynamics`，PEM 挪到 r 维 h 空间 | `forward` 第 3b 步 |
| 三 | 新增 `_init_precision_basis`，用 `lora_B` 初始化 `B_basis` | `set_pretrained_weight` |
| 四 | 新增 `llm_tuning()` / `cognitive_model()` 预设 | `AeloruConfig` |
| 五 | `_append_hgf_delta` 接入；移除批量路径双重缩放；`whitening_interval` → 1e9 | `post_step_update`、`batch_forward_lora`、config |

## 实现过程中发现并修复的三个问题

**1. Fix 2 需要"替换"而非"新增"，我第一版做错了。** 原文说"不在完整 y 上调用 `run_predictive_neural_dynamics`"，但我最初保留了原调用又加了低秩版，结果两者同时运行 —— 实测输出被放大 **64 倍**，比原来的 13 倍更糟。

改为新增 `pem_space` 开关（`"lowrank"` 默认 / `"output"` / `"none"`）：`llm_tuning()` 用 `lowrank`，`cognitive_model()` 用 `output` 保留原始 PEM。修复后 ndi=1/3/10 的输出比恒为 2.00（`cos=1.0000`，与 ndi=0 完全一致）。

**2. 低秩去相关驱动必须尺度无关。** 直接用 `C_hh⁻¹` 时，h 方差很小（实测 std≈0.055）会让逆矩阵特征值极大，驱动退化成 `(1/scale)·h`。改为除以平均特征值归一化（h 整体缩放时不变），再对特征值做 `[0.2, 5]` 限幅，并用 `lowrank_whiten_coef=0.1` 约束强度。

**3. `set_pretrained_weight` 的缓冲区搬运动作是坏的。** `named_buffers()` 对子模块返回点分路径（如 `lowrank_precision.B_basis`），原来的 `setattr(self, name, ...)` 会创建一个名为该点分串的垃圾属性，而不是真正搬运 buffer —— 这直接影响 Fix 3 依赖的设备一致性。已改为沿路径取父模块再赋值。

## 两点与文档预期不符的实测结果

**Fix 3 的效果有限。** 文档预期冷启动爆炸会消失，但实测侧向项/输入比值仅从 4.000 降到 3.78。原因是 `lora_B` 在 `_reset_adapters` 中被初始化为**全零**，直接拷贝仍然退化；即使非退化，`C_y = B·C_h·Bᵀ + εI` 中的 `εI` 地板项仍主导，放大倍数下限就是 `1/ε = 5`。我加了退化检测回退随机基（消除全零奇点），但 `εI` 地板带来的残留放大无法靠基初始化消除。不过这不影响 LLM 路径 —— 低秩 PEM 不使用 `lowrank_precision`。

**Fix 1 配套建议（hmax=2.0 再涨 0.5~1 个点）未获数据支持。** 单种子下 hmax=2.0 比 1.0 低 2.7 个点（0.620 vs 0.647），看似验证了「稳态归一化有害」。但跑 3 seeds 后差异落在噪声内（LoRA 自身 std 就有 ±0.035）：

| 配置 | 3 seeds 均值 |
|---|---|
| LoRA 基线 | 0.7089 ± 0.0354 |
| hmax=1.0 | 0.6956 ± 0.0303 |
| hmax=2.0 | 0.6800 ± 0.0530 |
| 稳态关闭 | 0.6622 ± 0.0181 |

差异 0.01~0.03 远小于 std，无法判定优劣。因此**按文档保持 hmax=2.0 默认**。

## 一处超出文档的添加

为验证上述 hmax 主张，我加了 `homeostatic_enabled: bool = True` 开关（默认 True，行为与文档一致）。关闭时仍维护 `running_var`（可塑性门控依赖它），只是不施加增益。若你不需要可以删掉，不影响其他逻辑。

## 验证

- 五项修复逐项校验通过（量纲检验 `var=4 → gain=0.5`、冷启动非退化、低秩 PEM 不改变主激活量级、冲突缓冲 20 步后长度=20、预设字段正确）
- 保存/加载含新增的 `C_hh` / `mu_h`，往返一致
- `ex1.py` 三方法回归通过：`aeloru-default` 从 0.5311 提升到 0.5840（受益于默认走 lowrank PEM）

Lint 的 18 条 `torch.linalg.*/F.linear is not callable` 均为 Pylint 对 torch 的已知误报，非本次引入。

临时脚本 `_fix1_check.py`、`_hmax_probe.py`、`_hmax_probe2.py`、`_probe.py`、`_redraw.py`、`_smoke_ex1b/`、`_fix1_ckpt.pt`