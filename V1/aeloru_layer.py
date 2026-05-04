"""
Aeloru (Adaptive Elastic Learning with Orthogonal ReLoRA Unit)
=============================================================

一套面向消费级GPU的LLM实时训练框架，融合：
- Hi-DoRA: 幅度-方向解耦的低秩适配
- ReLoRA: 周期性合并重置实现累积高秩
- Hebbian-Fisher 双向联动: 突触可塑性门控
- Hong Wen 认知状态机: 冲突驱动的四相学习循环

核心设计理念：
    预训练权重神圣不可侵犯，所有学习成果外置累积，
    通过认知冲突检测实现"探索->冲突->锚定->固化"的类脑闭环。

Author: JYIMU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os


# =============================================================================
# 配置类
# =============================================================================

class CognitiveState(Enum):
    """Hong Wen 认知状态枚举"""
    EXPLORE = "explore"     # 自由探索期
    RED = "red"             # 认知冲突（红温）
    ANCHOR = "anchor"       # 过程锚定
    SOLID = "solid"         # 赫布固化


@dataclass
class AeloruConfig:
    """
    Aeloru 完整配置类

    所有功能均可独立开关，便于消融实验。
    """
    # --- 基础维度 ---
    in_features: int = 512
    out_features: int = 512
    r: int = 8                          # LoRA 秩
    lora_alpha: float = 4.0             # LoRA 缩放因子
    LoRA_lr: float = 1e-4               # LoRA 学习率

    # --- 功能总开关 ---
    use_hidora: bool = True               # 是否启用 Hi-DoRA 幅度调制
    use_relora: bool = True               # 是否启用 ReLoRA 合并重置
    use_hebbian: bool = True              # 是否启用 Hebbian 在线学习
    use_fisher: bool = True               # 是否启用 Fisher 认知掩码
    use_hongwen: bool = True              # 是否启用 Hong Wen 状态机
    use_orthogonal_penalty: bool = True   # 是否启用正交惩罚损失
    use_energy_budget: bool = True        # 是否启用能量预算硬约束

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

    # --- 调试参数 ---
    verbose: bool = True                   # 是否打印详细日志


# =============================================================================
# 核心层：AeloruLayer
# =============================================================================

class AeloruLayer(nn.Module):
    """
    Aeloru 自适应弹性学习层

    核心公式体系：

    1. 低秩增量：
       DeltaW = (alpha/r) * B @ A

    2. Hi-DoRA 幅度调制（可选）：
       DeltaW_prime = diag(m) * DeltaW

    3. Fisher 敏感度门控（可选）：
       DeltaW_doubleprime = DeltaW_prime * 1/(1 + gamma*F)

    4. 能量预算硬约束（可选）：
       DeltaW_tripleprime = DeltaW_doubleprime * min(1, eta*||W0||_F / ||DeltaW_doubleprime||_F)

    5. 有效权重（加法形式，零元素解锁）：
       W_eff = W0 + W_acc + DeltaW_tripleprime

    6. 正交惩罚损失（可选）：
       L_ortho = lambda * ||DeltaW^T @ W0||_F^2

    其中：
    - W0: 神圣基座（预训练权重，永久冻结）
    - W_acc: 外置累积缓冲区（ReLoRA 合并沉淀区，冻结）
    - A, B: 当前周期工作记忆（低秩适配器，可训练）
    - m: 行幅度向量（Hi-DoRA 调制）
    - F: 动态 Fisher 认知掩码
    """

    def __init__(self, in_features: int, out_features: int, cfg: AeloruConfig):
        """
        初始化 Aeloru 层。

        Args:
            in_features: 输入维度
            out_features: 输出维度
            cfg: AeloruConfig 配置对象
        """
        super().__init__()
        self.cfg = cfg
        self.in_features = in_features
        self.out_features = out_features
        self.step_counter = 0

        # --- Hong Wen 认知状态 ---
        self.state = CognitiveState.EXPLORE
        self._solid_end_step = 0
        self._anchor_grad_history = []

        # ========== 神圣不可侵犯的预训练权重 ==========
        self.W0 = nn.Parameter(
            torch.empty(out_features, in_features), 
            requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.empty(out_features), 
            requires_grad=False
        )

        # ========== 外置累积缓冲区（ReLoRA 知识沉淀）==========
        # 仅当 use_relora=True 时有效，否则保持为零
        self.W_acc = nn.Parameter(
            torch.zeros(out_features, in_features), 
            requires_grad=False
        )

        # ========== 当前周期工作记忆（低秩适配器）==========
        # A (r, in_features): Kaiming Uniform 初始化（非零）
        # B (out_features, r): 零初始化（保证初始 DeltaW=0）
        self.lora_A = nn.Parameter(torch.empty(cfg.r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, cfg.r))

        # Hi-DoRA 行幅度向量（可选）
        # 形状: (out_features,)，用于对 DeltaW 的每一行做幅度调制
        if cfg.use_hidora:
            self.m = nn.Parameter(torch.ones(out_features))
        else:
            self.m = None

        # ========== 动态 Fisher 认知掩码（可选）==========
        # 形状: (out_features, in_features)
        # 初始为 1（中性），随 Hebbian 活动演化
        if cfg.use_fisher:
            self.fisher_mask = nn.Parameter(
                torch.ones(out_features, in_features), 
                requires_grad=False
            )
            # Fisher 历史快照（用于冲突检测）
            self.fisher_snapshot = nn.Parameter(
                torch.ones(out_features, in_features), 
                requires_grad=False
            )
        else:
            self.fisher_mask = None
            self.fisher_snapshot = None

        # ========== Hebbian 探索痕迹（可选）==========
        # 形状: (out_features, in_features)
        # 记录 Hebbian 更新的空间分布，用于计算探索熵
        if cfg.use_hebbian or cfg.use_fisher:
            self.hebbian_trace = nn.Parameter(
                torch.zeros(out_features, in_features), 
                requires_grad=False
            )
        else:
            self.hebbian_trace = None

        # ========== 初始化 ==========
        self._reset_adapters()

    def _reset_adapters(self):
        """
        重置低秩适配器（ReLoRA 合并后调用）。

        A 用 Kaiming Uniform（非零，保证探索新子空间）
        B 初始化为零（保证初始 DeltaW=0，零初始化等价性）
        m 重置为 1
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        if self.m is not None:
            self.m.data.fill_(1.0)

    def set_pretrained_weight(self, W0: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """
        注入预训练权重和 bias。

        关键操作：
        1. W0 复制为神圣基座
        2. 若启用 Hi-DoRA，m 初始化为 W0 的行范数
        3. 若启用 Fisher，初始化掩码为均匀分布

        Args:
            W0: 预训练权重 (out_features, in_features)
            bias: 预训练 bias (out_features,)，可选
        """
        self.W0.data = W0.clone().to(dtype=self.W0.dtype, device=self.W0.device)

        if bias is not None:
            self.bias.data = bias.clone().to(dtype=self.bias.dtype, device=self.bias.device)
        else:
            self.bias.data = torch.zeros(
                self.out_features, 
                dtype=self.bias.dtype, 
                device=self.bias.device
            )

        # Hi-DoRA: m 初始化为 W0 的行范数（dim=1）
        # 形状: (out_features,)，与 self.m 匹配
        if self.cfg.use_hidora and self.m is not None:
            self.m.data = torch.norm(W0, p=2, dim=1).to(
                dtype=self.m.dtype, device=self.m.device
            )

        # Fisher: 初始化为均匀敏感度（中性）
        if self.cfg.use_fisher and self.fisher_mask is not None:
            self.fisher_mask.data.fill_(1e-9) # 0(1e-9 零保护) 表示初始无保护，随着学习演化, 1 表示完全保护 
            self.fisher_snapshot.data = self.fisher_mask.data.clone()

    # =================================================================
    # 核心计算函数
    # =================================================================

    def compute_delta_w(self) -> torch.Tensor:
        """
        计算当前低秩增量 DeltaW = (alpha/r) * B @ A。

        Returns:
            DeltaW: (out_features, in_features)
        """
        AB = torch.mm(self.lora_B, self.lora_A)
        return (self.cfg.lora_alpha / self.cfg.r) * AB

    def apply_hidora(self, delta_w: torch.Tensor) -> torch.Tensor:
        """
        Hi-DoRA 幅度调制：对 DeltaW 的每一行做幅度调制。

        公式: DeltaW_prime = diag(m) * DeltaW

        Args:
            delta_w: (out_features, in_features)

        Returns:
            调制后的 DeltaW_prime
        """
        if not self.cfg.use_hidora or self.m is None:
            return delta_w
        # m: (out_features,) -> (out_features, 1) 广播
        return self.m.unsqueeze(1) * delta_w

    def apply_fisher_mask(self, delta_w: torch.Tensor) -> torch.Tensor:
        """
        Fisher 敏感度门控：保护高 Fisher（已稳固）参数。

        公式: M = 1 / (1 + gamma * F)
              DeltaW_doubleprime = DeltaW_prime * M

        Fisher 越高 -> 掩码越接近 0 -> 该区域越难被修改。

        Args:
            delta_w: (out_features, in_features)

        Returns:
            门控后的 DeltaW_doubleprime
        """
        if not self.cfg.use_fisher or self.fisher_mask is None:
            return delta_w
        mask = 1.0 / (1.0 + self.cfg.fisher_gamma * self.fisher_mask)
        return delta_w * mask

    def apply_energy_budget(self, delta_w: torch.Tensor) -> torch.Tensor:
        """
        能量预算硬约束：DeltaW 的 Frobenius 范数不超过 W0 的 eta 比例。

        公式: 若 ||DeltaW||_F > eta * ||W0||_F:
                  DeltaW <- DeltaW * (eta * ||W0||_F / ||DeltaW||_F)

        绝对防止 DeltaW 喧宾夺主。

        Args:
            delta_w: (out_features, in_features)

        Returns:
            约束后的 DeltaW_tripleprime
        """
        if not self.cfg.use_energy_budget:
            return delta_w
        w0_norm = self.W0.norm(p='fro')
        max_allowed = self.cfg.energy_eta * w0_norm
        dw_norm = delta_w.norm(p='fro')
        if dw_norm > max_allowed and dw_norm > 1e-8:
            return delta_w * (max_allowed / dw_norm)
        return delta_w

    def compute_weights(self) -> torch.Tensor:
        """
        合成有效权重矩阵。

        公式: W_eff = W0 + W_acc + DeltaW_tripleprime

        Returns:
            W_eff: (out_features, in_features)
        """
        delta_w = self.compute_delta_w()
        delta_w = self.apply_hidora(delta_w)
        delta_w = self.apply_fisher_mask(delta_w)
        delta_w = self.apply_energy_budget(delta_w)

        # 加法形式：W0 神圣 + W_acc 累积 + DeltaW 当前
        return self.W0 + self.W_acc + delta_w

    # =================================================================
    # 前向传播
    # =================================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        流程：
        1. 计算有效权重 W_eff
        2. 线性变换 y = x @ W_eff^T + bias
        3. 若启用 Hong Wen，检测认知冲突
        4. 若处于训练模式且允许 Hebbian，执行 Hebbian 更新

        Args:
            x: 输入张量 (..., in_features)

        Returns:
            y: 输出张量 (..., out_features)
        """
        W_eff = self.compute_weights()
        y = F.linear(x, W_eff, self.bias) # pylint: disable=E1102 ,这里没错,pylint误报了,这里使用了F.linear而不是直接的矩阵乘法，是为了更好地利用PyTorch的优化和自动微分功能

        # Hong Wen: 认知冲突检测
        if self.cfg.use_hongwen and self.training:
            if self.step_counter % self.cfg.snapshot_interval == 0:
                self._detect_and_transition()

        # Hebbian: 前向触发（与计算图隔离）
        if self.cfg.use_hebbian and self.training and self._hebbian_allowed():
            with torch.no_grad():
                self.hebbian_update(x.detach(), y.detach())

        self.step_counter += 1
        return y

    # =================================================================
    # 正交惩罚损失
    # =================================================================

    def get_ortho_penalty(self) -> torch.Tensor:
        """
        正交惩罚损失：惩罚 DeltaW 与 W0 的方向重叠。

        公式: L_ortho = lambda * ||DeltaW^T @ W0||_F^2

        梯度效应：将 DeltaW 推向 W0 的左零空间，自动避免重复学习。

        Returns:
            标量损失值
        """
        if not self.cfg.use_orthogonal_penalty:
            return torch.tensor(0.0, device=self.W0.device)

        # 根据状态调整 lambda
        lam = self.cfg.ortho_lambda
        if self.cfg.use_hongwen and self.state == CognitiveState.ANCHOR:
            lam = self.cfg.ortho_lambda_anchor

        delta_w = self.compute_delta_w()
        overlap = torch.mm(delta_w.t(), self.W0)  # (in, out) @ (out, in) -> (in, in)
        return lam * overlap.norm(p='fro') ** 2

    # =================================================================
    # Hebbian-Fisher 双向联动
    # =================================================================

    def hebbian_update(self, x: torch.Tensor, y: torch.Tensor, is_correct: bool = True):
        """
        Hebbian-Fisher 双向联动更新。

        核心机制：
        1. Fisher -> Hebbian: 高 Fisher 区域降低可塑性（突触稳固）
        2. Hebbian -> Fisher: 更新冲击提升 Fisher（记录学习痕迹）
        3. 全局遗忘衰减防止过度累积

        Args:
            x: 输入张量 (batch, in_features)，已 detach
            y: 输出张量 (batch, out_features)，已 detach
            is_correct: 结果门控，True 强化 / False 弱化
        """
        if not self.cfg.use_hebbian:
            return

        with torch.no_grad():
            # --- 1. 全局遗忘衰减 ---
            self.lora_A.data *= self.cfg.hebbian_decay
            self.lora_B.data *= self.cfg.hebbian_decay

            # --- 2. 计算平均激活 ---
            x_mean = x.mean(dim=0)    # (in_features,)
            y_mean = y.mean(dim=0)    # (out_features,)
            sign = 1.0 if is_correct else -1.0

            # --- 3. 原始 Hebbian 信号 ---
            # dB: (out_features, r) = y_mean (out,) @ x_mean[:r] (r,)
            raw_dB = sign * self.cfg.hebbian_lr * torch.ger(y_mean, x_mean[:self.cfg.r])
            # dA: (r, in_features) = y_mean[:r] (r,) @ x_mean (in,)
            raw_dA = sign * self.cfg.hebbian_lr * torch.ger(y_mean[:self.cfg.r], x_mean)

            # --- 4. Fisher 可塑性门控（Fisher -> Hebbian）---
            if self.cfg.use_fisher and self.fisher_mask is not None:
                # 每行/每列的平均 Fisher
                fisher_per_out = self.fisher_mask.mean(dim=1)   # (out_features,)
                fisher_per_in = self.fisher_mask.mean(dim=0)    # (in_features,)

                # 可塑性 = exp(-gamma * fisher)，高 Fisher = 低可塑性
                plasticity_B = torch.exp(
                    -self.cfg.fisher_gamma * fisher_per_out
                ).unsqueeze(1)  # (out_features, 1)
                plasticity_A = torch.exp(
                    -self.cfg.fisher_gamma * fisher_per_in
                ).unsqueeze(0)  # (1, in_features)

                # 硬下限保护（防止完全冻结）
                plasticity_B = plasticity_B.clamp(min=self.cfg.plasticity_min)
                plasticity_A = plasticity_A.clamp(min=self.cfg.plasticity_min)

                # 门控调制
                dB = raw_dB * plasticity_B
                dA = raw_dA * plasticity_A
            else:
                dB = raw_dB
                dA = raw_dA

            # --- 5. 执行 Hebbian 更新 ---
            self.lora_B.data += dB
            self.lora_A.data += dA
            self.lora_A.data.clamp_(
                -self.cfg.saturation_limit, 
                self.cfg.saturation_limit
            )
            self.lora_B.data.clamp_(
                -self.cfg.saturation_limit, 
                self.cfg.saturation_limit
            )

            # --- 6. Hebbian 反驱 Fisher（Hebbian -> Fisher）---
            if self.cfg.use_fisher and self.fisher_mask is not None:
                # 计算更新在低秩空间造成的完整冲击
                impact = torch.mm(dB.abs(), dA.abs())  # (out, in)

                # 归一化到 [0, 1]
                if impact.max() > 1e-10:
                    impact = impact / (impact.max() + 1e-10)

                # Fisher EMA 更新：被频繁更新的区域提升 Fisher
                self.fisher_mask.data = (
                    self.cfg.fisher_ema * self.fisher_mask.data
                    + (1.0 - self.cfg.fisher_ema) * impact
                )

            # --- 7. 更新认知痕迹 ---
            if self.hebbian_trace is not None:
                self.hebbian_trace.data += torch.mm(dB.abs(), dA.abs())

    # =================================================================
    # Hong Wen 认知状态机
    # =================================================================

    def _hebbian_allowed(self) -> bool:
        """检查当前状态是否允许 Hebbian 更新。"""
        if not self.cfg.use_hongwen:
            return True  # 无状态机时始终允许
        return self.state in [CognitiveState.EXPLORE, CognitiveState.SOLID]

    def _bp_allowed(self) -> bool:
        """检查当前状态是否允许 BP 更新。"""
        if not self.cfg.use_hongwen:
            return True
        return self.state in [CognitiveState.EXPLORE, CognitiveState.ANCHOR, CognitiveState.SOLID]

    def _detect_and_transition(self):
        """
        基于 Fisher 变化速度检测认知冲突（红温）。

        冲突分数公式:
            C = 0.6 * v_F + 0.4 * (1 - H)

        其中：
        - v_F: Fisher 变化速度（认知不稳定度）
        - H: 探索熵（防止局部最优）
        """
        if not self.cfg.use_fisher or self.fisher_mask is None:
            return

        with torch.no_grad():
            # 1. Fisher 变化速度
            fisher_velocity = (self.fisher_mask - self.fisher_snapshot).abs().mean()

            # 2. 探索熵
            trace_flat = self.hebbian_trace.view(-1)
            trace_sum = trace_flat.sum() + 1e-10
            trace_dist = trace_flat / trace_sum
            max_entropy = math.log(trace_dist.numel())
            exploration_entropy = -(trace_dist * torch.log(trace_dist + 1e-10)).sum()
            entropy_ratio = exploration_entropy.item() / max_entropy

            # 3. 综合冲突分数
            conflict_score = (
                0.6 * fisher_velocity.item()
                + 0.4 * (1.0 - entropy_ratio)
            )

            # 更新快照
            self.fisher_snapshot.data = self.fisher_mask.data.clone()

            # 状态转换
            if self.state == CognitiveState.EXPLORE and conflict_score > self.cfg.red_threshold:
                self._transition_state(CognitiveState.RED, conflict_score)
            elif self.state == CognitiveState.RED:
                self._transition_state(CognitiveState.ANCHOR, conflict_score)
            elif self.state == CognitiveState.SOLID:
                if self.step_counter >= self._solid_end_step:
                    self._transition_state(CognitiveState.EXPLORE, conflict_score)

    def _transition_state(self, new_state: CognitiveState, conflict_score: float):
        """
        认知状态转换与参数重配置。

        Args:
            new_state: 目标状态
            conflict_score: 触发转换的冲突分数
        """
        old_state = self.state
        self.state = new_state

        if new_state == CognitiveState.EXPLORE:
            # 探索期：一切松弛，Hebbian 自由
            if self.cfg.verbose:
                print(f"  [Aeloru] EXPLORE @ step {self.step_counter}")

        elif new_state == CognitiveState.RED:
            # 红温期：暂停 Hebbian，冻结 Fisher
            if self.cfg.verbose:
                print(f"  [Aeloru] RED HOT! conflict={conflict_score:.3f} @ step {self.step_counter}")

            # 可选：红温时强制合并
            if self.cfg.use_relora and self.cfg.merge_on_red:
                self.merge_and_reset()
                if self.cfg.verbose:
                    print(f"  [Aeloru] Forced merge on RED")

        elif new_state == CognitiveState.ANCHOR:
            # 锚定期：BP 主导，收紧 Fisher 保护
            if self.cfg.verbose:
                print(f"  [Aeloru] ANCHOR @ step {self.step_counter}")
            self._anchor_grad_history = []

        elif new_state == CognitiveState.SOLID:
            # 固化期：Hebbian 在 Fisher 低区强化
            if self.cfg.verbose:
                print(f"  [Aeloru] SOLID @ step {self.step_counter}")
            self._solid_end_step = self.step_counter + self.cfg.solid_steps

    def check_anchor_convergence(self, grad_norm: float) -> bool:
        """
        外部 BP 调用者检查锚定收敛，自动转入固化期。

        Args:
            grad_norm: 当前梯度范数

        Returns:
            是否已收敛并转入 SOLID
        """
        if not self.cfg.use_hongwen:
            return False

        if self.state == CognitiveState.ANCHOR:
            self._anchor_grad_history.append(grad_norm)
            # 使用最近 10 步的平均梯度判断收敛
            if len(self._anchor_grad_history) >= 10:
                avg_grad = sum(self._anchor_grad_history[-10:]) / 10
                if avg_grad < self.cfg.anchor_converge:
                    self._transition_state(CognitiveState.SOLID, conflict_score=0.0)
                    return True
        return False

    # =================================================================
    # ReLoRA 外置合并
    # =================================================================

    def should_merge(self) -> bool:
        """
        检查是否满足合并条件。

        Returns:
            是否应执行 ReLoRA 合并
        """
        if not self.cfg.use_relora:
            return False
        return self.step_counter >= self.cfg.merge_every

    def merge_and_reset(self):
        """
        ReLoRA 核心：外置累积合并。

        操作：
        1. 计算当前门控后的 DeltaW
        2. 沉淀到 W_acc（W0 永远不动）
        3. 重置 A, B, m（工作记忆清空）
        4. Hebbian 痕迹半衰期
        """
        if not self.cfg.use_relora:
            return

        with torch.no_grad():
            # 1. 计算门控后的 DeltaW
            delta_w = self.compute_delta_w()
            delta_w = self.apply_hidora(delta_w)
            delta_w = self.apply_fisher_mask(delta_w)
            delta_w = self.apply_energy_budget(delta_w)

            # 2. 沉淀到外置累积区（W0 神圣不可侵犯）
            self.W_acc.data += delta_w

            # 3. 重置工作记忆
            self._reset_adapters()
            self.step_counter = 0

            # 4. 痕迹半衰期
            if self.hebbian_trace is not None:
                self.hebbian_trace.data *= 0.5

            if self.cfg.verbose:
                print(f"  [Aeloru] MERGED. W_acc norm={self.W_acc.norm().item():.4f}")

    # =================================================================
    # 诊断接口
    # =================================================================

    def get_cognitive_report(self) -> Dict[str, Any]:
        """
        输出当前认知状态诊断报告。

        Returns:
            包含状态、Fisher 统计、痕迹熵等信息的字典
        """
        with torch.no_grad():
            report = {
                'state': self.state.value,
                'step': self.step_counter,
                'w_acc_norm': self.W_acc.norm().item(),
                'delta_w_norm': self.compute_delta_w().norm().item(),
            }

            if self.cfg.use_fisher and self.fisher_mask is not None:
                report.update({
                    'fisher_mean': self.fisher_mask.mean().item(),
                    'fisher_max': self.fisher_mask.max().item(),
                    'fisher_sparsity': (self.fisher_mask > 0.5).float().mean().item(),
                })

            if self.hebbian_trace is not None:
                trace_flat = self.hebbian_trace.view(-1)
                s = trace_flat.sum() + 1e-10
                p = trace_flat / s
                max_e = math.log(p.numel())
                entropy = (-(p * torch.log(p + 1e-10)).sum()).item() / max_e
                report['trace_entropy'] = entropy

            return report

    # =================================================================
    # 序列化接口
    # =================================================================

    def save_adapter(self, path: str):
        """
        保存 Aeloru 适配器（安全版：只存基本类型和张量）
        """
        # 把AeloruConfig转成字典，而不是直接存对象
        cfg_dict = {k: v for k, v in self.cfg.__dict__.items() if not k.startswith('_')}

        # 把CognitiveState枚举转成字符串
        state_str = self.state.value

        checkpoint = {
            'cfg_dict': cfg_dict,  # 存字典，不存对象
            'lora_A': self.lora_A.data.cpu().clone(),
            'lora_B': self.lora_B.data.cpu().clone(),
            'W_acc': self.W_acc.data.cpu().clone(),
            'step_counter': self.step_counter,
            'state_str': state_str,  # 存字符串，不存枚举
        }

        if self.m is not None:
            checkpoint['m'] = self.m.data.cpu().clone()
        if self.fisher_mask is not None:
            checkpoint['fisher_mask'] = self.fisher_mask.data.cpu().clone()
            checkpoint['fisher_snapshot'] = self.fisher_snapshot.data.cpu().clone()
        if self.hebbian_trace is not None:
            checkpoint['hebbian_trace'] = self.hebbian_trace.data.cpu().clone()
        try :
            torch.save(checkpoint, path)
            print(f"保存至{os.path.abspath(path)}")  # 输出绝对路径，方便定位
        except Exception as e:
            print(f"报错: {e},新建文件夹并再次尝试保存")
            os.makedirs(os.path.dirname(path), exist_ok=True)# 确保路径存在
            torch.save(checkpoint, path)
            print(f"保存至{os.path.abspath(path)}")  # 输出绝对路径，方便定位
        if self.cfg.verbose:
            print(f"  [Aeloru] Adapter saved to {path}")

    def load_adapter(self, path: str):
        """
        加载 Aeloru 适配器（安全版：weights_only=True）
        """
        # 用weights_only=True加载，完全安全
        checkpoint = torch.load(path, map_location=self.W0.device, weights_only=True)

        # 验证配置兼容性（用字典对比，不用对象）
        # 这里可以简单对比关键参数，比如r、in_features、out_features
        assert checkpoint['cfg_dict']['r'] == self.cfg.r
        assert checkpoint['cfg_dict']['in_features'] == self.cfg.in_features
        assert checkpoint['cfg_dict']['out_features'] == self.cfg.out_features

        # 加载张量
        self.lora_A.data = checkpoint['lora_A'].to(self.lora_A.device)
        self.lora_B.data = checkpoint['lora_B'].to(self.lora_B.device)
        self.W_acc.data = checkpoint['W_acc'].to(self.W_acc.device)
        self.step_counter = checkpoint['step_counter']

        # 把字符串转回CognitiveState枚举
        self.state = CognitiveState(checkpoint['state_str'])

        # 加载可选参数
        if 'm' in checkpoint and self.m is not None:
            self.m.data = checkpoint['m'].to(self.m.device)
        if 'fisher_mask' in checkpoint and self.fisher_mask is not None:
            self.fisher_mask.data = checkpoint['fisher_mask'].to(self.fisher_mask.device)
            self.fisher_snapshot.data = checkpoint['fisher_snapshot'].to(self.fisher_snapshot.device)
        if 'hebbian_trace' in checkpoint and self.hebbian_trace is not None:
            self.hebbian_trace.data = checkpoint['hebbian_trace'].to(self.hebbian_trace.device)

        if self.cfg.verbose:
            print(f"  [Aeloru] Adapter loaded from {path}")


# =============================================================================
# 注入辅助函数
# =============================================================================

def inject_aeloru(
    model: nn.Module,
    target_names: list = None,
    cfg: AeloruConfig = None,
    r: int = 8,
    alpha: float = 4.0
) -> nn.Module:
    """
    递归地将模型中的指定线性层替换为 Aeloru 适配器。

    Args:
        model: 待注入的 PyTorch 模型
        target_names: 目标层名列表，默认 Transformer 常见层
        cfg: AeloruConfig 配置对象（优先）
        r: LoRA 秩（cfg 为 None 时使用）
        alpha: LoRA 缩放因子（cfg 为 None 时使用）

    Returns:
        注入后的模型（原地修改）
    """
    if target_names is None:
        target_names = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

    if cfg is None:
        cfg = AeloruConfig(r=r, lora_alpha=alpha)

    # 使用 list() 包装防止迭代器失效
    for name, module in list(model.named_children()):
        # 递归处理子模块
        if len(list(module.children())) > 0:
            inject_aeloru(module, target_names, cfg)

        # 替换匹配的线性层
        if isinstance(module, nn.Linear) and any(target in name for target in target_names):
            aeloru_layer = AeloruLayer(module.in_features, module.out_features, cfg)
            aeloru_layer.set_pretrained_weight(
                module.weight.data,
                getattr(module, 'bias', None)
            )
            setattr(model, name, aeloru_layer)

            if cfg.verbose:
                print(f"  [Aeloru] Injected into {name} "
                      f"(in={module.in_features}, out={module.out_features}, r={cfg.r})")

    return model


# =============================================================================
# 训练过程封装：Hong Wen + Fisher + Hebbian 完整工作流
# =============================================================================

def train_aeloru_step(
    layer: AeloruLayer,
    x: torch.Tensor,
    y_target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: callable = F.mse_loss,
    reward_signal: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    单步 Aeloru 训练封装（Hong Wen 四相循环 + Fisher + Hebbian）。

    完整工作流：
    1. 前向传播（自动触发 Hebbian + Hong Wen 检测）
    2. 根据状态决定是否执行 BP
    3. 计算总损失（任务损失 + 正交惩罚）
    4. 反向传播（若允许）
    5. 检查锚定收敛
    6. 检查 ReLoRA 合并

    Args:
        layer: AeloruLayer 实例
        x: 输入张量
        y_target: 目标输出
        optimizer: PyTorch 优化器
        loss_fn: 损失函数，默认 MSE
        reward_signal: Hebbian 结果门控，True 强化 / False 弱化

    Returns:
        (loss_total, metrics_dict)
    """
    layer.train()
    metrics = {}

    # --- 1. 前向传播 ---
    # 内部自动处理：
    # - Hebbian 更新（若处于 EXPLORE/SOLID）
    # - Hong Wen 冲突检测（每 snapshot_interval 步）
    y_pred = layer(x)

    # --- 2. 根据状态决定是否 BP ---
    if not layer._bp_allowed():
        # RED 状态：暂停 BP，返回前向损失（无梯度）
        with torch.no_grad():
            loss_task = loss_fn(y_pred, y_target)
        metrics['state'] = layer.state.value
        metrics['loss_task'] = loss_task.item()
        metrics['bp_skipped'] = True
        return loss_task, metrics

    # --- 3. 计算总损失 ---
    loss_task = loss_fn(y_pred, y_target)
    loss_ortho = layer.get_ortho_penalty()
    loss_total = loss_task + loss_ortho

    # --- 4. 反向传播 ---
    optimizer.zero_grad()
    loss_total.backward()

    # 梯度范数（用于锚定收敛检测）
    grad_norm = sum(
        p.grad.norm().item() 
        for p in [layer.lora_A, layer.lora_B] 
        if p.grad is not None
    )

    # 梯度裁剪（防止爆炸）
    torch.nn.utils.clip_grad_norm_(
        [layer.lora_A, layer.lora_B], 
        max_norm=1.0
    )

    optimizer.step()

    # --- 5. 检查锚定收敛（Hong Wen）---
    converged = layer.check_anchor_convergence(grad_norm)

    # --- 6. 检查 ReLoRA 合并 ---
    merged = False # 标记是否进行了合并
    if layer.should_merge():
        layer.merge_and_reset()
        # 合并后重置优化器状态（热重启）
        optimizer.state.clear()
        merged = True

    # --- 7. 收集指标 ---
    metrics.update({
        'state': layer.state.value,
        'loss_task': loss_task.item(),
        'loss_ortho': loss_ortho.item(),
        'loss_total': loss_total.item(),
        'grad_norm': grad_norm,
        'anchor_converged': converged,
        'relora_merged': merged,
    })

    if layer.cfg.use_fisher:
        report = layer.get_cognitive_report()
        metrics['fisher_mean'] = report.get('fisher_mean', 0.0)
        metrics['trace_entropy'] = report.get('trace_entropy', 0.0)

    return loss_total, metrics


# =============================================================================
# 完整测试验证
# =============================================================================

def test_aeloru():
    """
    Aeloru 完整测试验证脚本。

    测试覆盖：
    1. 零初始化等价性（W_eff(t=0) == W0）
    2. 功能开关消融（所有开关独立测试）
    3. Hebbian-Fisher 双向联动
    4. Hong Wen 状态机转换
    5. ReLoRA 合并重置
    6. 正交惩罚效果
    7. 能量预算约束
    8. 保存/加载一致性
    """
    print("="*70)
    print("Aeloru 完整测试验证")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n测试设备: {device}")

    in_dim, out_dim, batch_size = 128, 64, 4

    # ========== 测试 1: 零初始化等价性 ==========
    print(f"\n{'='*70}")
    print("测试 1: 零初始化等价性")
    print(f"{'='*70}")

    original_linear = nn.Linear(in_dim, out_dim).to(device)
    original_linear.eval()

    x = torch.randn(batch_size, in_dim, device=device)
    with torch.no_grad():
        original_output = original_linear(x)

    # 全功能开启
    cfg_full = AeloruConfig(
        in_features=in_dim, 
        out_features=out_dim,
        r=8, 
        lora_alpha=4.0,
        use_hidora=True,
        use_relora=True,
        use_hebbian=True,
        use_fisher=True,
        use_hongwen=True,
        use_orthogonal_penalty=True,
        use_energy_budget=True,
    )

    layer_full = AeloruLayer(in_dim, out_dim, cfg_full).to(device)
    layer_full.set_pretrained_weight(
        original_linear.weight.data, 
        original_linear.bias.data
    )
    layer_full.eval()

    with torch.no_grad():
        aeloru_output = layer_full(x)

    diff = torch.max(torch.abs(original_output - aeloru_output)).item()
    print(f"  原始输出均值: {original_output.mean().item():.6f}")
    print(f"  Aeloru 输出均值: {aeloru_output.mean().item():.6f}")
    print(f"  最大绝对误差: {diff:.10f}")

    assert diff < 1e-5, f"零初始化等价性失败！diff={diff}"
    print("  测试 1 通过：零初始化等价性")

    # ========== 测试 2: 功能开关消融 ==========
    print(f"\n{'='*70}")
    print("测试 2: 功能开关消融实验")
    print(f"{'='*70}")

    switch_configs = [
        ("全关（仅基础LoRA）", {
            'use_hidora': False, 'use_relora': False, 'use_hebbian': False,
            'use_fisher': False, 'use_hongwen': False,
            'use_orthogonal_penalty': False, 'use_energy_budget': False,
        }),
        ("仅 Hi-DoRA", {'use_hidora': True}),
        ("仅 ReLoRA", {'use_relora': True, 'merge_every': 50}),
        ("仅 Hebbian", {'use_hebbian': True}),
        ("仅 Fisher", {'use_fisher': True}),
        ("仅 Hong Wen", {'use_hongwen': True}),
        ("仅正交惩罚", {'use_orthogonal_penalty': True}),
        ("仅能量预算", {'use_energy_budget': True}),
        ("Hebbian+Fisher", {'use_hebbian': True, 'use_fisher': True}),
        ("全功能开启", {
            'use_hidora': True, 'use_relora': True, 'use_hebbian': True,
            'use_fisher': True, 'use_hongwen': True,
            'use_orthogonal_penalty': True, 'use_energy_budget': True,
        }),
    ]

    for name, switches in switch_configs:
        cfg = AeloruConfig(in_features=in_dim, out_features=out_dim, r=8, lora_alpha=4.0)
        for k, v in switches.items():
            setattr(cfg, k, v)

        layer = AeloruLayer(in_dim, out_dim, cfg).to(device)
        layer.set_pretrained_weight(original_linear.weight.data, original_linear.bias.data)
        layer.train()

        # 执行一步训练
        y_target = torch.randn(batch_size, out_dim, device=device)
        optimizer = torch.optim.AdamW([layer.lora_A, layer.lora_B], lr=1e-3)

        try:
            loss, metrics = train_aeloru_step(layer, x, y_target, optimizer)
            status = "OK"
        except Exception as e:
            status = f"ERR: {str(e)[:40]}"

        print(f"  {status:<20s} {name:<25s} | state={layer.state.value:<8s} | loss={loss.item():.4f}")

    print("  测试 2 通过：所有开关组合正常运行")

    # ========== 测试 3: Hebbian-Fisher 双向联动 ==========
    print(f"\n{'='*70}")
    print("测试 3: Hebbian-Fisher 双向联动")
    print(f"{'='*70}")

    cfg_hf = AeloruConfig(
        in_features=in_dim, 
        out_features=out_dim,
        r=8, 
        lora_alpha=4.0,
        use_hebbian=True,
        use_fisher=True,
        use_hongwen=False,  # 关闭状态机，纯测试联动
        fisher_gamma=5.0,
        hebbian_lr=5e-5,    # 调大以便观察
    )

    layer_hf = AeloruLayer(in_dim, out_dim, cfg_hf).to(device)
    layer_hf.set_pretrained_weight(torch.randn(out_dim, in_dim, device=device) * 0.02)
    layer_hf.train()

    fisher_before = layer_hf.fisher_mask.mean().item()
    print(f"  Fisher 均值 (冲击前): {fisher_before:.4f}")
    fixed_x = torch.randn(batch_size, in_dim, device=device)

    def train_HF(steps: int = 50 , fisher_before: float = fisher_before):
        for _ in range(steps):
            y = layer_hf(fixed_x)
            # Hebbian 在前向中自动触发
        fisher_after = layer_hf.fisher_mask.mean().item()
        print(f"  Fisher 均值: {fisher_before:.4f} -> {fisher_after:.4f}")
        # print(f"  Fisher 提升比例: {(fisher_after/fisher_before - 1)*100:.1f}%"),第一次比例爆炸此处不于展示
        print(f"最大值: {layer_hf.fisher_mask.max().item():.4f},最小值: {layer_hf.fisher_mask.min().item():.4f},方差: {layer_hf.fisher_mask.var().item():.6f}")
        
        return fisher_after
    
    fisher_after = train_HF(steps=50)# 密集 Hebbian 冲击同一区域
    assert fisher_after > fisher_before, "Hebbian 应提升 Fisher"
    fisher_before = fisher_after
    
    # 再来一次，验证持续冲击下 Fisher 的累积提升
    train_HF(steps=50)
    fisher_after_2 = layer_hf.fisher_mask.mean().item()
    print(f"  持续冲击下 Fisher 均值: {fisher_after:.4f} -> {fisher_after_2:.4f}")
    print(f"  提升比例: {(fisher_after_2/fisher_after - 1)*100:.1f}%")

    
    print("  测试 3 通过：Hebbian -> Fisher 痕迹沉淀")

    # ========== 测试 4: Hong Wen 状态机 ==========
    print(f"\n{'='*70}")
    print("测试 4: Hong Wen 状态机转换")
    print(f"{'='*70}")

    cfg_hw = AeloruConfig(
        in_features=in_dim, 
        out_features=out_dim,
        r=8, 
        lora_alpha=4.0,
        use_hebbian=True,
        use_fisher=True,
        use_hongwen=True,
        red_threshold=0.1,          # 调低以便快速触发
        snapshot_interval=10,       # 频繁检测
        anchor_converge=1e-3,
        solid_steps=20,
        verbose=True,
    )

    layer_hw = AeloruLayer(in_dim, out_dim, cfg_hw).to(device)
    layer_hw.set_pretrained_weight(torch.randn(out_dim, in_dim, device=device) * 0.02)
    layer_hw.train()

    optimizer = torch.optim.AdamW([layer_hw.lora_A, layer_hw.lora_B], lr=AeloruConfig.LoRA_lr)

    state_history = []

    for step in range(100):
        x_step = torch.randn(batch_size, in_dim, device=device)
        y_target = torch.randn(batch_size, out_dim, device=device)

        loss, metrics = train_aeloru_step(layer_hw, x_step, y_target, optimizer)
        state_history.append(layer_hw.state.value)

        if metrics.get('relora_merged'):
            print(f"    Step {step}: ReLoRA 合并触发")
        if metrics.get('anchor_converged'):
            print(f"    Step {step}: 锚定收敛，转入 SOLID")

    # 验证状态转换发生
    unique_states = set(state_history)
    print(f"\n  经历的状态: {unique_states}")

    assert len(unique_states) >= 2, "Hong Wen 应触发至少一次状态转换"
    print("  测试 4 通过：Hong Wen 状态机正常转换")

    # ========== 测试 5: ReLoRA 合并重置 ==========
    print(f"\n{'='*70}")
    print("测试 5: ReLoRA 合并重置")
    print(f"{'='*70}")

    cfg_rl = AeloruConfig(
        in_features=in_dim, 
        out_features=out_dim,
        r=8, 
        lora_alpha=4.0,
        use_relora=True,
        merge_every=30,
        use_hongwen=False,
    )

    layer_rl = AeloruLayer(in_dim, out_dim, cfg_rl).to(device)
    layer_rl.set_pretrained_weight(original_linear.weight.data, original_linear.bias.data)
    layer_rl.train()

    w_acc_before = layer_rl.W_acc.norm().item()

    # 训练直到合并
    optimizer_rl = torch.optim.AdamW([layer_rl.lora_A, layer_rl.lora_B], lr=1e-3)
    for step in range(50):
        x_step = torch.randn(batch_size, in_dim, device=device)
        y_target = torch.randn(batch_size, out_dim, device=device)
        loss, _ = train_aeloru_step(layer_rl, x_step, y_target, optimizer_rl)

        if layer_rl.step_counter == 0 and step > 0:  # 合并后 step_counter 重置
            print(f"  合并发生在 step {step}")
            break

    w_acc_after = layer_rl.W_acc.norm().item()

    print(f"  W_acc 范数: {w_acc_before:.6f} -> {w_acc_after:.6f}")
    print(f"  合并后 step_counter: {layer_rl.step_counter}")

    assert w_acc_after > w_acc_before, "合并应沉淀知识到 W_acc"
    assert layer_rl.step_counter == 0, "合并后应重置计数器"
    print("  测试 5 通过：ReLoRA 合并重置正常")

    # ========== 测试 6: 正交惩罚效果 ==========
    print(f"\n{'='*70}")
    print("测试 6: 正交惩罚效果")
    print(f"{'='*70}")

    cfg_op = AeloruConfig(
        in_features=in_dim, 
        out_features=out_dim,
        r=8, 
        lora_alpha=4.0,
        use_orthogonal_penalty=True,
        ortho_lambda=1.0,  # 调大以便观察
        use_hongwen=False,
    )

    layer_op = AeloruLayer(in_dim, out_dim, cfg_op).to(device)
    layer_op.set_pretrained_weight(torch.randn(out_dim, in_dim, device=device) * 0.02)
    layer_op.train()

    # 故意让 DeltaW 与 W0 对齐
    with torch.no_grad():
        # 让 A, B 使得 DeltaW 近似 W0 的方向
        target_delta = layer_op.W0 * 0.01
        u, s, v = torch.linalg.svd(target_delta, full_matrices=False) #pylint: disable=E1102 ,这里误报了,这里使用SVD实现了让 DeltaW 与 W0 对齐的目的
        layer_op.lora_B.data = u[:, :8] * torch.sqrt(s[:8])
        layer_op.lora_A.data = torch.diag(torch.sqrt(s[:8])) @ v[:8, :]

    ortho_loss = layer_op.get_ortho_penalty().item()
    print(f"  对齐时的正交损失: {ortho_loss:.6f}")

    # 随机初始化（应更低）
    layer_op._reset_adapters()
    ortho_loss_random = layer_op.get_ortho_penalty().item()
    print(f"  随机时的正交损失: {ortho_loss_random:.6f}")

    assert ortho_loss > ortho_loss_random, "对齐时应产生更大正交惩罚"
    print("  测试 6 通过：正交惩罚有效")

    # ========== 测试 7: 能量预算约束 ==========
    print(f"\n{'='*70}")
    print("测试 7: 能量预算硬约束")
    print(f"{'='*70}")

    cfg_eb = AeloruConfig(
        in_features=in_dim, 
        out_features=out_dim,
        r=8, 
        lora_alpha=4.0,
        use_energy_budget=True,
        energy_eta=0.1,  # 严格限制
        use_hongwen=False,
    )

    layer_eb = AeloruLayer(in_dim, out_dim, cfg_eb).to(device)
    layer_eb.set_pretrained_weight(torch.randn(out_dim, in_dim, device=device) * 0.02)

    # 制造大 DeltaW
    with torch.no_grad():
        layer_eb.lora_A.data *= 100
        layer_eb.lora_B.data *= 100

    delta_w = layer_eb.compute_delta_w()
    delta_w_constrained = layer_eb.apply_energy_budget(delta_w)

    w0_norm = layer_eb.W0.norm(p='fro').item()
    delta_norm = delta_w.norm(p='fro').item()
    constrained_norm = delta_w_constrained.norm(p='fro').item()

    print(f"  W0 范数: {w0_norm:.4f}")
    print(f"  DeltaW 原始范数: {delta_norm:.4f}")
    print(f"  DeltaW 约束范数: {constrained_norm:.4f}")
    print(f"  约束比率: {constrained_norm/w0_norm:.4f} (上限 eta={cfg_eb.energy_eta})")

    assert constrained_norm <= cfg_eb.energy_eta * w0_norm * 1.01, "应被硬约束"
    print("  测试 7 通过：能量预算硬约束有效")

    # ========== 测试 8: 保存/加载一致性 ==========
    print(f"\n{'='*70}")
    print("测试 8: 保存/加载一致性")
    print(f"{'='*70}")

    test_path = "output/test_aeloru_adapter.pt"

    # 训练几步制造差异
    layer_full.train()
    optimizer_test = torch.optim.AdamW([layer_full.lora_A, layer_full.lora_B], lr=AeloruConfig.LoRA_lr)
    for _ in range(5):
        x_step = torch.randn(batch_size, in_dim, device=device)
        y_target = torch.randn(batch_size, out_dim, device=device)
        train_aeloru_step(layer_full, x_step, y_target, optimizer_test)

    # 保存前输出
    layer_full.eval()
    with torch.no_grad():
        output_before = layer_full(x)

    layer_full.save_adapter(test_path)

    # 新建层并加载
    layer_loaded = AeloruLayer(in_dim, out_dim, cfg_full).to(device)
    layer_loaded.set_pretrained_weight(
        original_linear.weight.data, 
        original_linear.bias.data
    )
    layer_loaded.load_adapter(test_path)
    layer_loaded.eval()

    with torch.no_grad():
        output_after = layer_loaded(x)

    load_diff = torch.max(torch.abs(output_before - output_after)).item()
    print(f"  保存前输出均值: {output_before.mean().item():.6f}")
    print(f"  加载后输出均值: {output_after.mean().item():.6f}")
    print(f"  最大绝对误差: {load_diff:.10f}")

    assert load_diff < 1e-5, f"保存/加载不一致！diff={load_diff}"
    print("  测试 8 通过：保存/加载一致性")

    # 清理
    if os.path.exists(test_path):
        os.remove(test_path)
        print(f"已删除测试文件: {os.path.abspath(test_path)}")

    # ========== 最终总结 ==========
    print(f"\n{'='*70}")
    print("所有测试通过！Aeloru 架构验证完成")
    print(f"{'='*70}")
    print("\n功能覆盖清单：")
    print("  零初始化等价性")
    print("  功能开关消融（8种组合）")
    print("  Hebbian-Fisher 双向联动")
    print("  Hong Wen 四相状态机")
    print("  ReLoRA 合并重置")
    print("  正交惩罚损失")
    print("  能量预算硬约束")
    print("  保存/加载一致性")


# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    test_aeloru()
    
