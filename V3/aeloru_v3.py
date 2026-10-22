# coding=utf-8
"""
Aeloru (Adaptive Elastic Learning with Orthogonal ReLoRA Units) v3.2
======================================================================

v3.2 修复要点（相对 v3.1，针对预训练 LLM 微调场景）：

D. 稳态增益量纲修正
   running_var 是激活二阶矩（方差），稳态归一化应与标准差成反比。
   原式 1/running_var 等价 1/sigma^2，多层叠加把输出 RMS 压到 1/3 次方量级，
   层数越多压缩越狠——这是 24 层全注入时直接学不动的根本原因。
   现改为 1/sqrt(running_var)，homeostatic_max_gain 默认 5.0 -> 2.0。

E. PEM 低秩空间改造
   新增 _low_rank_neural_dynamics：神经动态从 d_out 维主激活路径
   挪到 r 维低秩特征空间 h，只做侧向抑制 + 精度加权去相关，
   不触碰预训练模型已收敛的激活分布。复杂度 O(r^2)，与 d_out 无关。

F. LowRankPrecisionMatrix 冷启动修复
   B_basis 全零时 C_y 退化为 eps*I，精度为 I/eps，
   侧向项把输出放大 1/eps 倍(eps=0.2 即 5 倍)，是 PEM 初期爆炸主因。
   新增 _init_precision_basis：用 lora_B 初始化，退化时回退随机基。

G. 场景化预设
   AeloruConfig.llm_tuning()     预训练 LLM 微调安全预设（对外推荐默认）
   AeloruConfig.cognitive_model() 从零训练认知模型完整预设

H. 其他修复
   1. _append_hgf_delta 从未被调用 -> 冲突检测与锚定收敛恒不生效，
      HongWen 状态机只能靠固定步数切换。现每步喂入 Hebbian 更新幅度。
   2. batch_forward_lora 双重缩放隐患 -> 移除可塑性门控的全局均值项。
   3. whitening_interval 默认 1000 -> 1e9（预训练模型已有 LayerNorm）。
   4. set_pretrained_weight 中 named_buffers() 返回点分路径却直接 setattr，
      导致子模块缓冲（如 lowrank_precision.B_basis）实际未被搬运设备。

v3.1 重构要点（相对 v3.0）：

A. 可塑性门控降维：二维参数级 Fisher -> 双一维维度级 HGF 精度门控
   1. 删除 fisher_mask / fisher_mask_inv 两个 d_out x d_in 大矩阵
   2. 输出侧可塑性直接复用 HGF running_var 稳态统计量（零额外开销）
   3. 输入侧可塑性复用在线协方差 C_x 的对角线（零额外开销）
   4. 门控由逐元素相乘改为行/列双一维缩放，与 Hi-DoRA m_x (x) m_y 完全同构
   5. 删除 _compute_fisher_from_hebbian_trace 及 _flush_hebbian 中的 Fisher EMA
   复杂度：O(d_out x d_in) -> O(d_out + d_in)

B. 三项机制补强
   1. GHL 全局方向引导：Hebbian 更新符号由「局部 is_correct」升级为
      「局部奖励 + 全局梯度/损失方向符号」的凸组合，方向对齐全局任务目标
   2. SPHeRe 低维结构投影：正交惩罚升级为「W0 方向重叠 + 低秩特征协方差
      结构保持」双项损失，在输入度量下约束 A 的各向同性，防止小 r 特征坍缩
   3. STDP 噪声梯度：_flush_hebbian 注入与波动率/冲突分数/认知状态挂钩的
      高斯噪声，噪声强度相对更新幅度自适应（红温大、固化小）

C. 稳态与记忆增强
   1. BCM 自适应可塑性阈值：固定 saturation_limit 升级为与活动水平挂钩的
      逐维度滑动饱和上限
   2. WTA 硬竞争：PEM 神经动态每步只保留 top-k (k = ceil(r * wta_k_ratio))
      个最大激活，与线性侧抑制互补
   3. DLAM 原型记忆：睡眠阶段额外提取低秩空间 top-k 主特征向量作为原型，
      反投影为输入/输出维度重要性，抑制与旧记忆重叠维度的可塑性

Author: JYIMU
"""

from numpy import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, Callable, List
from dataclasses import dataclass
from enum import Enum
import os


# =============================================================================
# 配置类 (精简版 - 删除全部开关，只保留必要超参数)
# =============================================================================

class CognitiveState(Enum):
    EXPLORE = "explore"
    RED = "red"
    ANCHOR = "anchor"
    SOLID = "solid"


@dataclass
class AeloruConfig:
    """
    Aeloru v3.1 精简配置类 -- 所有功能常驻，无开关。

    可塑性门控为「双一维维度级 HGF 精度门控」：
        输出侧 pi_out = 1 / running_var        (O(d_out))
        输入侧 pi_in  = 1 / diag(C_x)          (O(d_in))
        可塑性 p = exp(-(gamma * pi_norm + lambda * proto_importance))
    不再维护任何 d_out x d_in 的二维 Fisher 矩阵。
    """
    # --- 基础维度 ---
    in_features: int = 512 # 输入特征维度
    out_features: int = 512# 输出特征维度
    r: int = 4 # rank
    lora_alpha: float = 4.0
    LoRA_lr: float = 1e-4

    # --- ReLoRA ---
    merge_every: int = 1000 # 每隔 N 步合并 LoRA 参数

    # --- Hebbian ---
    hebbian_lr: float = 1e-6 
    hebbian_decay: float = 0.99 # 衰减率
    hebbian_accum_steps: int = 4 # 累积步数

    # --- BCM 自适应可塑性阈值 (替代固定 saturation_limit) ---
    saturation_limit: float = 5.0 # 饱和上限基准值
    bcm_gain: float = 1.0         # 阈值耦合强度：活动越高 -> 有效上限越紧

    # --- 双一维 HGF 精度门控 (O(d_out + d_in)，替代二维 Fisher 掩码) ---
    fisher_gamma: float = 0.5   # 归一化精度 -> 可塑性 的衰减率
    plasticity_min: float = 0.05 # 最小可塑性（门控下界）
    precision_eps: float = 1e-5  # 精度数值稳定项

    # --- GHL 全局方向引导 (Hebbian Learning with Global Direction) ---
    ghl_global_strength: float = 0.8 # 全局方向符号门的总权重
    ghl_sign_ema: float = 0.9        # 全局方向信号的 EMA 系数
    ghl_explore_scale: float = 0.3   # 探索期：弱全局符号 + 强局部 Hebbian
    ghl_red_scale: float = 0.6       # 红温期
    ghl_anchor_scale: float = 1.0    # 锚定期：强全局梯度 + 弱局部 Hebbian
    ghl_solid_scale: float = 0.8     # 固化期
    ghl_loss_gain: float = 10.0      # 损失相对变化率 -> 方向信号的放大倍率

    # --- SPHeRe 低维结构投影 (低秩特征拓扑保持) ---
    sphere_lambda: float = 0.01 # 结构保持损失系数

    # --- STDP 噪声梯度探索 ---
    hebbian_noise_ratio: float = 0.1     # 噪声 / 本次更新幅度 的基准比
    hebbian_noise_max_ratio: float = 1.0 # 相对幅度上限

    # --- Hong Wen 状态机 ---
    red_threshold: float = 0.65 # 红阈值
    solid_steps: int = 200 # 固态步数
    red_min_steps: int = 50 # 红最小步数
    explore_steps: int = 100 # 探索步数
    anchor_steps: int = 1 # 锚点步数
    anchor_converge: float = 1e-4 # 锚点收敛阈值

    # --- 正交惩罚 ---
    ortho_lambda: float = 0.01 # 正交惩罚系数
    ortho_lambda_anchor: float = 0.05 # 锚点正交惩罚系数

    # --- 能量预算 --- 
    energy_eta: float = 0.15 # 能量预算衰减率

    # --- PEM 侧向连接 ---
    lateral_lr: float = 1e-5  # 侧向连接学习率
    lateral_decay: float = 0.99 # 侧向连接衰减率

    # --- PEM 稳态可塑性 ---
    homeostatic_tau: float = 0.99 # 稳态可塑性衰减率
    # 量纲修正为 1/sqrt(running_var) 后增益整体变大，上限相应从 5.0 下调到 2.0
    homeostatic_max_gain: float = 2.0   # 稳态可塑性最大增益
    # 稳态归一化会把每层输出 RMS 拉向 1，对预训练 LLM 属于额外扰动。
    # 实测(SST2, 3 seeds)关闭后准确率更高，故提供显式开关；
    # running_var 统计量仍会维护（可塑性门控依赖它）。
    homeostatic_enabled: bool = True

    # --- HGF 波动率耦合 ---
    volatility_lr: float = 1e-3 # 波动率学习率
    volatility_init: float = 0.0 # 初始波动率

    # --- 预测性解相关 ---
    predictive_coding_rank: int = 16 # 预测性解 rank
    gamma_predictive: float = 100.0 # 预测性解衰减率
    lambda_lateral: float = 0.9 # 侧向连接惩罚率
    neural_dynamics_iterations: int = 10 # 神经动力学迭代次数
    wta_k_ratio: float = 0.5 # WTA 硬竞争保留比例 (k = ceil(r * ratio)，>=1 时退化为不竞争)

    # --- 低秩神经动态 (_low_rank_neural_dynamics) ---
    lowrank_whiten_coef: float = 0.1 # 去相关驱动强度(过大易在低方差方向振荡)
    lowrank_cond_max: float = 5.0    # 归一化协方差的条件数上限，限制单次修正增益

    # --- 神经动态的作用空间 ---
    # "lowrank": 只在 r 维低秩子空间 h 内迭代，不触碰主激活路径（预训练模型安全）
    # "output" : 在 d_out 维完整输出上迭代（原始 PEM，仅适合从零训练）
    # "none"   : 完全关闭
    pem_space: str = "lowrank"

    # --- 源信号域约束 ---
    presumed_domain: str = "nnantisparse"   # "antisparse" | "nnantisparse" | "simplex"

    # --- 在线协方差与白化 ---
    # 预训练 LLM 已有 LayerNorm，输入白化属于重复归一化，过频反而破坏表征。
    # 默认关闭；从零训练认知模型时用 cognitive_model() 预设开启(500)。
    whitening_interval: int = 10**9 # 每隔 N 步更新协方差矩阵

    # --- DLAM 睡眠 ---
    sleep_condition_threshold: float = 500.0 # 睡眠条件阈值
    min_steps_between_sleep: int = 1000 # 最小睡眠步数
    cross_modal_coupling: float = 0.01 # 跨模态耦合 
    dlam_replay_threshold: float = 1e-6 # DLAM 重放阈值

    # --- DLAM 原型记忆 (表征记忆式巩固) ---
    prototype_top_k: int = 4         # 每轮睡眠提取的原型数量(<= r)
    prototype_guard: float = 0.5     # 原型重叠 -> 可塑性抑制强度

    # --- 诊断 ---
    logging_enabled: bool = False # 是否启用日志记录
    diagnostic_enabled: bool = False # 是否启用诊断
    diagnostic_interval: int = 100 # 诊断间隔

    AMP_DTYPE: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # =================================================================
    # 场景化预设
    # =================================================================

    @classmethod
    def llm_tuning(cls, **kwargs):
        """
        预训练 LLM 微调安全预设（对外发布默认推荐）。

        - neural_dynamics_iterations=0: 关闭神经动态。
          v3.1 已把 PEM 挪到 r 维低秩空间（_low_rank_neural_dynamics），
          不会破坏预训练表征，因此可传 neural_dynamics_iterations=N 安全开启低秩 PEM。
        - presumed_domain="none": 关闭硬域约束。
          clamp(0,1) 会把预训练激活中约半数负值截断为 0。
        - homeostatic_max_gain=2.0: 增益量纲已修正为 1/sqrt(var)，上限相应下调。
        - whitening_interval=1e9: 预训练模型已有 LayerNorm，输入白化属重复归一化。

        用法: AeloruConfig.llm_tuning(r=8, lora_alpha=16)
        """
        defaults = dict(
            neural_dynamics_iterations=0,
            presumed_domain="none",
            homeostatic_max_gain=2.0,
            whitening_interval=10**9,
            pem_space="lowrank",
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def cognitive_model(cls, **kwargs):
        """
        认知模型完整预设：从零训练独立认知模型（Qelys/Lorys）。

        - 全开 PEM 预测编码、域约束、稳态可塑性、输入白化
        - 无预训练表征可破坏，因此不需要 llm_tuning 的那些限制

        用法: AeloruConfig.cognitive_model(r=16, lora_alpha=32)
        """
        defaults = dict(
            neural_dynamics_iterations=5,
            presumed_domain="nnantisparse",
            homeostatic_max_gain=5.0,
            whitening_interval=500,
            pem_space="output",
        )
        defaults.update(kwargs)
        return cls(**defaults)


# =============================================================================
# 工具函数
# =============================================================================

def quantize_tensor(t: torch.Tensor, bits: int = 8) -> Tuple[torch.Tensor, float, float]:
    """简单线性量化到 uint8。"""
    if bits != 8:
        raise NotImplementedError("仅支持 8-bit 量化")
    t_min = t.min()
    t_max = t.max()
    scale = (t_max - t_min) / 255.0
    if scale < 1e-8:
        scale = 1.0
    zero_point = t_min
    q = ((t - zero_point) / scale).round().clamp(0, 255).to(torch.uint8)
    try:
        return q, scale.item(), zero_point.item()
    except:
        return q, scale, zero_point


def dequantize_tensor(q: torch.Tensor, scale: float, zero_point: float,
                      dtype: torch.dtype = torch.float16) -> torch.Tensor:
    return q.to(dtype) * scale + zero_point


# 全局批量计算缓存
_batch_cache = {
    'lora_A': [],
    'lora_B': [],
    'scales': [],
    'layers': []
}


def batch_forward_lora(x, layers):
    """
    批量计算所有层的LoRA增量，只启动3个kernel。
    侧向抑制需在 r 维特征空间操作，跳过批量优化。

    注意：这是一个尚未启用的快速路径。forward 目前不使用其结果
    （钩子只把结果挂到 layer._batch_result，forward 并不读取），
    且它跳过了侧向抑制 / WTA / PEM 低秩神经动态，与 forward 并不等价。

    启用前必须先移除 forward 中的逐元素门控，否则会出现
    「全局均值缩放 + 逐元素缩放」的双重增益，等效学习率偏高。
    为消除该隐患，下面已不再重复施加可塑性门控的全局均值。
    """
    global _batch_cache
    _batch_cache['lora_A'].clear()
    _batch_cache['lora_B'].clear()
    _batch_cache['scales'].clear()
    _batch_cache['layers'].clear()

    for layer in layers:
        if not layer.training:
            continue
        _batch_cache['lora_A'].append(layer.lora_A)
        _batch_cache['lora_B'].append(layer.lora_B)
        scale = layer.cfg.lora_alpha / layer.cfg.r
        scale *= layer.m_x.mean() * layer.m_y.mean()
        # 不在此处叠加可塑性门控的全局均值：forward 已做逐元素双一维门控，
        # 两者相乘会造成「全局均值缩放 + 逐元素缩放」的双重增益。
        _batch_cache['scales'].append(scale)
        _batch_cache['layers'].append(layer)

    if not _batch_cache['layers']:
        return {}

    batch_A = torch.stack(_batch_cache['lora_A'], dim=0)
    batch_B = torch.stack(_batch_cache['lora_B'], dim=0)
    batch_scales = torch.tensor(_batch_cache['scales'], device=x.device, dtype=x.dtype)

    x_a = torch.einsum('bsi, kri -> kbsr', x, batch_A)
    x_ab = torch.einsum('kbsr, kor -> kbso', x_a, batch_B)
    x_ab = x_ab.permute(1, 2, 0, 3)
    x_ab = x_ab * batch_scales[None, None, :, None]

    results = {}
    for i, layer in enumerate(_batch_cache['layers']):
        results[id(layer)] = x_ab[:, :, i, :]
    return results


# =============================================================================
# 低秩精度矩阵
# =============================================================================

class LowRankPrecisionMatrix(nn.Module):
    """
    低秩精度矩阵: O(out^3) Cholesky 维护降为 O(r^2 * out)。
    C_y = B_basis @ C_h @ B_basis^T + epsilon * I
    """

    def __init__(self, out_features: int, r: int, device: str = "cuda",
                 basis_update_interval: int = 100, ema_alpha: float = 0.9,
                 epsilon: float = 0.2):
        super().__init__()
        self.out_features = out_features
        self.r = r
        self.basis_update_interval = basis_update_interval
        self.ema_alpha = ema_alpha
        self.epsilon = epsilon
        self.step = 0
        self.register_buffer(
            'B_basis',
            torch.zeros(out_features, r, device=device, dtype=torch.float32)
        )
        self.register_buffer(
            'C_h',
            torch.eye(r, device=device, dtype=torch.float32) * 0.2
        )

    def update(self, B: torch.Tensor, y: torch.Tensor) -> None:
        """从 lora_B 和输出自动提取基并更新低秩协方差。"""
        with torch.no_grad():
            B = B.detach().float()
            y = y.detach().float()
            # 支持 3D (batch, seq, dim)：折叠为 2D 再统计
            if y.dim() > 2:
                y = y.reshape(-1, y.size(-1))
            # lora_B 只有 r 列，与预测编码基秩不一定相同，此处对齐列数
            if B.shape[1] > self.r:
                B = B[:, :self.r]
            elif B.shape[1] < self.r:
                B = F.pad(B, (0, self.r - B.shape[1]))
            self.B_basis.copy_(B)
            y_mean = y.mean(dim=0)
            y_bar = y - y_mean.unsqueeze(0)
            h = torch.matmul(y_bar, self.B_basis)
            h_cov = torch.matmul(h.t(), h) / h.size(0)
            self.C_h.mul_(self.ema_alpha).add_(h_cov, alpha=1 - self.ema_alpha)
            self.step += 1

    def apply_precision(self, y_bar: torch.Tensor) -> torch.Tensor:
        """计算 C_y^{-1} @ y_bar。"""
        with torch.no_grad():
            B = self.B_basis
            eps = self.epsilon
            # 重新条件化：C_h 可能退化（某输出维度死亡→对角为零，或尚未更新→全零）
            # 导致 linalg.inv 奇异。加相对对角抖动避免崩溃（与 spectral_filter 一致）。
            C_h = self.C_h
            j = 1e-2 * C_h.diagonal().abs().max().clamp(min=1e-6)
            C_h = C_h + j * torch.eye(C_h.shape[0], device=C_h.device, dtype=C_h.dtype)
            Bt_y = torch.matmul(B.t(), y_bar.t())
            BtB = torch.matmul(B.t(), B)
            C_h_inv = torch.linalg.inv(C_h)
            M = C_h_inv + BtB / eps
            z = torch.linalg.solve(M, Bt_y / eps)
            result = y_bar.t() / eps - torch.matmul(B, z) / eps
            return result.t()

    def spectral_filter(self, threshold: float) -> torch.Tensor:
        """对低秩协方差 C_h 直接做谱滤波。"""
        with torch.no_grad():
            M = self.C_h
            # 重新条件化：加相对对角抖动，避免 C_h 病态/退化导致 eigh 不收敛。
            j = 1e-2 * M.diagonal().abs().max().clamp(min=1e-6)
            M = M + j * torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
            try:
                eigvals, eigvecs = torch.linalg.eigh(M)
            except RuntimeError:
                eigvals = torch.ones(M.size(0), device=M.device, dtype=M.dtype)
                eigvecs = torch.eye(M.size(0), device=M.device, dtype=M.dtype)
            cond = eigvals[-1] / (eigvals[0] + 1e-10)
            if cond > threshold:
                min_eig = eigvals[-1] / threshold
                eigvals_filtered = torch.clamp(eigvals, min=min_eig)
                self.C_h = eigvecs @ torch.diag(eigvals_filtered) @ eigvecs.t()
            return cond


# =============================================================================
# 核心层：AeloruLayer v3.0
# =============================================================================

class AeloruLayer(nn.Module):
    """
    Aeloru v3.1 -- 所有功能常驻，可塑性门控为双一维维度级 HGF 精度门控。
    """

    def __init__(self, in_features: int, out_features: int, cfg: AeloruConfig,
                 original_linear: Optional[nn.Linear] = None):
        super().__init__()
        self._red_enter_step = -999999
        self.cfg = cfg
        self.step_counter = 0
        self._last_sleep_step = -999999

        if original_linear is not None:
            self.in_features = original_linear.in_features
            self.out_features = original_linear.out_features
            self.original_linear = original_linear
            for param in self.original_linear.parameters():
                param.requires_grad = False
        else:
            self.in_features = in_features
            self.out_features = out_features
            self.original_linear = None

        # 正交惩罚随机投影
        self.register_buffer(
            '_ortho_proj',
            torch.randn(self.in_features, 16, device=cfg.device, dtype=torch.float32)
        )
        self._ortho_proj_step = 0

        # Hong Wen 认知状态
        self.state = CognitiveState.EXPLORE
        self._solid_end_step = 0
        self._anchor_grad_history = []
        self._explore_start_step = 0

        # W0 神圣基座
        self.register_buffer('W0', torch.empty(self.out_features, self.in_features, dtype=self.cfg.AMP_DTYPE))
        self.register_buffer('bias', torch.empty(self.out_features, dtype=self.cfg.AMP_DTYPE))

        # W_acc CPU Offload + 量化
        self.register_buffer(
            '_W_acc_q',
            torch.zeros(self.out_features, self.in_features, dtype=torch.uint8)
        )
        self.register_buffer('_W_acc_scale', torch.tensor(1.0, dtype=self.cfg.AMP_DTYPE))
        self.register_buffer('_W_acc_zp', torch.tensor(0.0, dtype=self.cfg.AMP_DTYPE))
        self._W_acc_cache = None

        # 当前周期工作记忆(低秩适配器)
        self.lora_A = nn.Parameter(torch.empty(cfg.r, self.in_features, dtype=self.cfg.AMP_DTYPE))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, cfg.r, dtype=self.cfg.AMP_DTYPE))

        # Hi-DoRA 行幅度向量(常驻)
        self.m_x = nn.Parameter(torch.ones(self.out_features, dtype=self.cfg.AMP_DTYPE))
        self.m_y = nn.Parameter(torch.ones(self.in_features, dtype=self.cfg.AMP_DTYPE))

        # PEM 自适应侧向连接(常驻)
        self.register_buffer('lateral_weights', torch.zeros(cfg.r, cfg.r, dtype=self.cfg.AMP_DTYPE))

        # 低秩特征空间 h (r 维) 的在线统计，供 _low_rank_neural_dynamics 使用。
        # 尺寸仅 O(r^2)，与 d_out 无关。
        self.register_buffer('C_hh', torch.eye(cfg.r, dtype=torch.float32) * 0.2)
        self.register_buffer('mu_h', torch.zeros(cfg.r, dtype=torch.float32))

        # PEM 稳态可塑性(常驻)
        self.register_buffer('running_var', torch.ones(self.out_features))

        # HGF 波动率耦合(常驻)
        self.register_buffer('omega', torch.tensor(cfg.volatility_init, dtype=torch.float32))
        self.register_buffer('dynamic_lr', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('_prev_y_mean', torch.zeros(self.out_features, dtype=torch.float32))

        # DLAM 睡眠机制(常驻)
        self.register_buffer('stored_patterns', torch.tensor(0, dtype=torch.long))
        self.register_buffer('conflict_score', torch.tensor(0.0, dtype=torch.float32))

        # 双一维 HGF 精度门控(常驻，O(d_out + d_in)，无二维矩阵)
        self.register_buffer('plasticity_out', torch.ones(self.out_features, dtype=torch.float32))
        self.register_buffer('plasticity_in', torch.ones(self.in_features, dtype=torch.float32))

        # GHL 全局方向引导(常驻)
        self.register_buffer('global_direction', torch.tensor(0.0, dtype=torch.float32))
        self._prev_loss: Optional[float] = None

        # DLAM 原型记忆(常驻)
        k_proto = max(1, min(cfg.prototype_top_k, cfg.r))
        self.register_buffer(
            'stored_prototypes',
            torch.zeros(k_proto, cfg.r, dtype=torch.float32)
        )
        self.register_buffer('proto_importance_out', torch.zeros(self.out_features, dtype=torch.float32))
        self.register_buffer('proto_importance_in', torch.zeros(self.in_features, dtype=torch.float32))

        # Hebbian 探索痕迹（低秩分解版）
        self.register_buffer('hebb_trace_B', torch.zeros(self.out_features, cfg.r, dtype=self.cfg.AMP_DTYPE))
        self.register_buffer('hebb_trace_A', torch.zeros(cfg.r, self.in_features, dtype=self.cfg.AMP_DTYPE))
        self.hebbian_trace = None  # 兼容旧接口

        # Hebbian 累积缓冲
        self.register_buffer('_hebbian_acc_A', torch.zeros(cfg.r, self.in_features))
        self.register_buffer('_hebbian_acc_B', torch.zeros(self.out_features, cfg.r))
        self._hebbian_acc_count = 0
        # 本步 Hebbian 更新幅度（供冲突检测缓冲消费，见 _append_hgf_delta）
        self._last_hebbian_mag = 0.0

        # Hebbian 影子参数
        self.register_buffer('_hebbian_delta_A', torch.zeros_like(self.lora_A))
        self.register_buffer('_hebbian_delta_B', torch.zeros_like(self.lora_B))
        self._hebbian_pending_apply = False

        # 梯度冲突检测
        self._grad_norm_history = []
        self._hgf_conflict_buffer = []
        self._hgf_delta_buffer = []

        self._cached_delta_w = None
        self._cache_valid = False
        self._steps_since_post_update = 0

        # 参数梯度设置
        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True
        self.m_x.requires_grad = True
        self.m_y.requires_grad = True

        self._reset_adapters()
        self.register_buffer('W0_norm_sq', torch.tensor(0.0, dtype=torch.float32))
        self._pending_optimizer_reset = False
        self._pending_cross_modal = False

        # 预测性解相关 (常驻)
        self.lowrank_precision = LowRankPrecisionMatrix(
            self.out_features, cfg.predictive_coding_rank, device=cfg.device,
            basis_update_interval=100, ema_alpha=cfg.lambda_lateral
        )
        self.register_buffer('mu_y', torch.zeros(self.out_features, dtype=torch.float32, device=cfg.device))
        self.register_buffer('mu_y_prev', torch.zeros(self.out_features, dtype=torch.float32, device=cfg.device))
        self.register_buffer('omega_cy', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('dynamic_lr_cy', torch.tensor(1.0, dtype=torch.float32))
        self._cy_condition_threshold = 100.0

        # 在线协方差与动态白化 (常驻)
        self.register_buffer('mu_x', torch.zeros(self.in_features, dtype=cfg.AMP_DTYPE, device=cfg.device))
        self.register_buffer('C_x', 0.2 * torch.eye(self.in_features, dtype=cfg.AMP_DTYPE, device=cfg.device))
        self.register_buffer('W_whiten', torch.eye(self.in_features, dtype=cfg.AMP_DTYPE, device=cfg.device))

    def get_trainable_params(self):
        """返回当前 Aeloru 层中所有可训练参数。"""
        params = []
        for name in ('lora_A', 'lora_B', 'm_x', 'm_y'):
            p = getattr(self, name, None)
            if isinstance(p, torch.nn.Parameter) and p.requires_grad:
                params.append(p)
        return params

    def clear_optimizer_state(self, optimizer: torch.optim.Optimizer):
        """合并后只清除本层 A/B/m 的动量状态"""
        for p in self.get_trainable_params():
            if p in optimizer.state:
                del optimizer.state[p]

    def set_logging(self, enabled: bool):
        """全局开关：控制普通日志输出。"""
        self.cfg.logging_enabled = bool(enabled)

    def set_diagnostics(self, enabled: bool):
        """全局开关：控制诊断报告输出。"""
        self.cfg.diagnostic_enabled = bool(enabled)

    def should_log(self) -> bool:
        return bool(getattr(self.cfg, 'logging_enabled', False))

    def should_diagnose(self) -> bool:
        return bool(getattr(self.cfg, 'diagnostic_enabled', False))

    def _log(self, message: str):
        if self.should_log():
            print(message)

    def _diagnose(self, message: str):
        if self.should_diagnose():
            print(message)

    # -----------------------------------------------------------------
    # W_acc 量化/反量化封装
    # -----------------------------------------------------------------

    def _get_W_acc(self) -> torch.Tensor:
        """获取 W_acc，自动处理量化和缓存"""
        if getattr(self, '_W_acc_cache', None) is None:
            self._W_acc_cache = dequantize_tensor(
                self._W_acc_q,
                self._W_acc_scale.item(),
                self._W_acc_zp.item(),
                dtype=self.W0.dtype
            ).to(self.W0.dtype)
        return self._W_acc_cache

    def _set_W_acc(self, value: torch.Tensor):
        """设置 W_acc，自动量化"""
        q, scale, zp = quantize_tensor(value)
        self._W_acc_q.copy_(q)
        self._W_acc_scale.fill_(scale)
        self._W_acc_zp.fill_(zp)
        self._W_acc_cache = None

    # -----------------------------------------------------------------
    # 初始化
    # -----------------------------------------------------------------

    def _reset_adapters(self):
        """重置低秩适配器(ReLoRA 合并后调用)。"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        with torch.no_grad():
            self.m_x.fill_(1.0)
            self.m_y.fill_(1.0)
            self._hebbian_acc_B.zero_()
            self._hebbian_acc_A.zero_()
        self._hebbian_acc_count = 0
        self._cache_valid = False
        self._last_structure_loss = 0.0

    def set_pretrained_weight(self, W0: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """注入预训练权重和 bias。"""
        with torch.no_grad():
            target_device = W0.device
            target_dtype = self.cfg.AMP_DTYPE

            W0 = W0.clone().detach().to(device=target_device, dtype=target_dtype, non_blocking=True)
            self.W0.copy_(W0)
            self.original_weight = W0
            if bias is not None:
                self.bias.copy_(bias.to(device=target_device, dtype=target_dtype, non_blocking=True))
            else:
                self.bias.zero_()

            for name, buf in self.named_buffers():
                if buf.device != target_device:
                    # named_buffers() 对子模块缓冲返回点分路径
                    # (如 "lowrank_precision.B_basis")，直接 setattr(self, name, ...)
                    # 会创建一个名为该点分串的垃圾属性，而不会真正搬运缓冲。
                    # 这里显式沿路径取到父模块再赋值。
                    parent = self
                    parts = name.split('.')
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    setattr(parent, parts[-1], buf.to(target_device))

            with torch.no_grad():
                self.m_x.data = W0.norm(p=2, dim=1)
                value = W0.norm(p=2, dim=0).mean()
                self.m_y.data = value.expand_as(self.m_y).clone()

            self.W0_norm_sq = (self.W0.float().norm(p='fro') ** 2).cpu()

            row_norms_sq = W0.norm(p=2, dim=1) ** 2
            self.running_var.copy_(row_norms_sq.clamp(min=1e-2))

            # 以 W0 的行/列二阶矩作为输入侧统计的冷启动
            col_norms_sq = W0.norm(p=2, dim=0) ** 2
            with torch.no_grad():
                self.C_x.diagonal().copy_(col_norms_sq.to(self.C_x.dtype).clamp(min=1e-2))

            # 用初始低秩 B 作为精度矩阵的初始基，避免冷启动爆炸。
            # B_basis 全零时 C_y = B C_h B^T + eps I 退化为 eps*I，
            # 精度 C_y^{-1} = I/eps，侧向项 C_y^{-1}y - y = (1/eps - 1) y
            # 会把输出放大 1/eps 倍(eps=0.2 即 5 倍)，这是 PEM 初期输出爆炸的主因。
            self._init_precision_basis()

            self._update_plasticity()

    def _init_precision_basis(self) -> None:
        """
        用当前 lora_B 初始化预测编码精度矩阵的基 B_basis。

        lora_B 在 _reset_adapters 中被初始化为全零，此时直接拷贝仍然退化为
         eps*I，故检测退化后回退到随机基（按 C_h 的尺度归一化），
        保证冷启动阶段精度矩阵非奇异且量级合理。
        """
        if self.lowrank_precision is None:
            return
        with torch.no_grad():
            rank = self.cfg.predictive_coding_rank
            basis = self.lora_B.detach().float()

            # 退化检测：全零或整体范数过小
            if float(basis.norm()) < 1e-8:
                gen = torch.Generator(device=basis.device)
                gen.manual_seed(0)
                basis = torch.randn(
                    self.out_features, rank,
                    generator=gen if basis.device.type == 'cpu' else None,
                    device=basis.device, dtype=torch.float32
                ) * 0.02

            # 对齐列数到 predictive_coding_rank
            if basis.shape[1] > rank:
                basis = basis[:, :rank]
            elif basis.shape[1] < rank:
                basis = F.pad(basis, (0, rank - basis.shape[1]))

            target = self.lowrank_precision.B_basis
            self.lowrank_precision.B_basis.copy_(
                basis.to(device=target.device, dtype=target.dtype))

    # =================================================================
    # 核心计算函数
    # =================================================================

    def compute_delta_w(self) -> torch.Tensor:
        """计算当前低秩增量 DeltaW = (alpha/r) * B @ A。"""
        if not self.training and self._cache_valid and self._cached_delta_w is not None:
            return self._cached_delta_w
        AB = torch.mm(self.lora_B, self.lora_A)
        delta = (self.cfg.lora_alpha / self.cfg.r) * AB
        if not self.training:
            self._cached_delta_w = delta
            self._cache_valid = True
        return delta

    def apply_hidora(self, delta_w: torch.Tensor) -> torch.Tensor:
        """Hi-DoRA 幅度调制：DeltaW' = (m_x * m_y^T) ⊙ DeltaW"""
        return delta_w * self.m_x.unsqueeze(1) * self.m_y.unsqueeze(0)

    # -----------------------------------------------------------------
    # 双一维维度级 HGF 精度门控 (O(d_out + d_in))
    # -----------------------------------------------------------------

    def _update_plasticity(self) -> None:
        """
        刷新行/列两条一维可塑性门。

        输出侧：pi_out = 1 / running_var          (HGF 稳态精度，前向自动维护)
        输入侧：pi_in  = 1 / (diag(C_x) + mu_x^2) (白化统计的对角二阶矩)

        两者做均值归一后取 exp(-(gamma * pi + lambda * proto_importance))，
        与 Hi-DoRA 的 m_x (x) m_y 幅值调制完全同构。
        全程只有 O(d_out + d_in) 个元素，无 d_out x d_in 大矩阵。
        """
        with torch.no_grad():
            eps = self.cfg.precision_eps

            pi_out = 1.0 / (self.running_var.float() + eps)
            pi_out = pi_out / (pi_out.mean() + 1e-8)

            var_x = torch.diagonal(self.C_x).float() + self.mu_x.float() ** 2
            pi_in = 1.0 / (var_x + eps)
            pi_in = pi_in / (pi_in.mean() + 1e-8)

            gamma = self.cfg.fisher_gamma
            guard = self.cfg.prototype_guard

            p_out = torch.exp(-(gamma * pi_out + guard * self.proto_importance_out))
            p_in = torch.exp(-(gamma * pi_in + guard * self.proto_importance_in))

            self.plasticity_out.copy_(p_out.clamp(min=self.cfg.plasticity_min))
            self.plasticity_in.copy_(p_in.clamp(min=self.cfg.plasticity_min))

    def apply_plasticity_gate(self, delta_w: torch.Tensor) -> torch.Tensor:
        """
        双一维可塑性门控：DeltaW'' = p_out[:, None] * DeltaW' * p_in[None, :]

        等价于二维重要性 p_out (x) p_in 的外积近似，但只做两次长度为
        d_out / d_in 的逐行/逐列缩放，复杂度 O(d_out + d_in)。
        """
        p_out = self.plasticity_out
        p_in = self.plasticity_in
        if p_out.dtype != delta_w.dtype or p_out.device != delta_w.device:
            p_out = p_out.to(device=delta_w.device, dtype=delta_w.dtype)
        if p_in.dtype != delta_w.dtype or p_in.device != delta_w.device:
            p_in = p_in.to(device=delta_w.device, dtype=delta_w.dtype)
        delta_w = delta_w * p_out.unsqueeze(1)
        delta_w = delta_w * p_in.unsqueeze(0)
        return delta_w

    def _delta_w_norm_exact(self) -> torch.Tensor:
        """精确计算 ||DeltaW||_F"""
        scale = self.cfg.lora_alpha / self.cfg.r
        # fp32 累加：bf16/fp16 下 torch.trace 不可用且精度不足
        BtB = torch.mm(self.lora_B.float().t(), self.lora_B.float())
        AAt = torch.mm(self.lora_A.float(), self.lora_A.float().t())
        norm_sq = scale ** 2 * torch.trace(torch.mm(BtB, AAt))
        return torch.sqrt(norm_sq.clamp(min=0.0))

    def apply_energy_budget(self, delta_w: torch.Tensor) -> torch.Tensor:
        """能量预算硬约束：||DeltaW||_F <= eta * ||W0||_F"""
        w0_norm = torch.sqrt(self.W0_norm_sq.to(delta_w.device))
        max_allowed = self.cfg.energy_eta * w0_norm
        dw_norm = self._delta_w_norm_exact()
        if dw_norm > max_allowed and dw_norm > 1e-8:
            return delta_w * (max_allowed / dw_norm)
        return delta_w

    def compute_weights(self) -> torch.Tensor:
        """合成有效权重矩阵 W_eff = W0 + W_acc + DeltaW'''"""
        delta_w = self.compute_delta_w()
        delta_w = self.apply_hidora(delta_w)
        delta_w = self.apply_plasticity_gate(delta_w)
        delta_w = self.apply_energy_budget(delta_w)

        if self._hebbian_pending_apply:
            hebbian_full = torch.mm(
                self._hebbian_delta_B.detach().clone().to(delta_w.dtype),
                self._hebbian_delta_A.detach().clone().to(delta_w.dtype)
            )
            delta_w = delta_w + hebbian_full

        return self.W0 + self._get_W_acc() + delta_w

    # =================================================================
    # 预测编码神经动态 (常驻)
    # =================================================================

    def run_predictive_neural_dynamics(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """预测编码神经动态：在输出空间运行快速时间尺度梯度下降。"""
        if not self.training:
            return y

        with torch.no_grad():
            # Transformer 隐藏状态为 3D (batch, seq, dim)，而低秩精度矩阵只接受 2D。
            # 此处折叠为 2D 参与神经动态迭代，结束前恢复原始形状。
            orig_shape = y.shape
            x_2d = x.reshape(-1, x.size(-1)) if x.dim() > 2 else x
            y_2d = y.reshape(-1, y.size(-1)) if y.dim() > 2 else y

            W_base = self.W0 + self._get_W_acc()
            yke = F.linear(x_2d, W_base)
            y_current = y_2d.clone()

            for j in range(self.cfg.neural_dynamics_iterations):
                lr_y = max(0.9 / (j + 1), 0.01)
                y_old = y_current.clone()
                error = y_current - yke
                y_bar = y_current - self.mu_y.unsqueeze(0)
                C_y_inv_y = self.lowrank_precision.apply_precision(y_bar)
                lateral = C_y_inv_y - y_bar
                grady = self.cfg.gamma_predictive * error + lateral
                y_current = y_current - lr_y * grady

                if self.cfg.presumed_domain == "antisparse":
                    y_current = torch.clamp(y_current, -1, 1)
                elif self.cfg.presumed_domain in ["nnantisparse", "nnsparse", "simplex"]:
                    y_current = torch.clamp(y_current, 0, 1)

                if torch.norm(y_current - y_old) < 1e-8 * torch.norm(y_current):
                    break

            if self.cfg.presumed_domain == "simplex":
                y_current = self.project_rows_to_unit_simplex(y_current)
            return y_current.reshape(orig_shape)

    def _low_rank_neural_dynamics(self, h: torch.Tensor) -> torch.Tensor:
        """
        低秩特征空间 (r 维) 的轻量预测编码神经动态。

        与 run_predictive_neural_dynamics 的区别：
            全输出版在 d_out 维主激活路径上做修正，会直接扰动预训练模型
            已收敛的激活分布（实测 LLM 上会放大 13x 或归零，完全学不动）；
            本版本只在 r 维低秩子空间 h 内做「侧向抑制 + 精度加权去相关」，
            主激活路径完全不受影响，既保留 PEM 的特征解耦作用，又不破坏预训练表征。

        不设置重建目标 yke（基准为 0），因此不含 gamma_predictive * error 项——
        该项的物理意义是把输出拉向基座预测，在无基座预测的低秩空间里
        只会变成对 h 自身的纯增益/纯衰减，是有害的。

        复杂度 O(r^2)（外加一次 r x r 求逆），与 d_out / d_in 无关，
        比全输出版小两个数量级。
        """
        if not self.training or self.cfg.neural_dynamics_iterations <= 0:
            return h

        orig_shape = h.shape
        h_2d = h.reshape(-1, h.size(-1)) if h.dim() > 2 else h

        with torch.no_grad():
            # 更新 h 空间的在线均值与协方差（O(r^2)，统计量不参与梯度）
            h_float = h_2d.detach().float()
            batch_mu = h_float.mean(dim=0)
            alpha = float(self.cfg.lambda_lateral)
            self.mu_h.mul_(alpha).add_(batch_mu, alpha=1.0 - alpha)
            h_bar = h_float - self.mu_h.unsqueeze(0)
            cov = torch.mm(h_bar.t(), h_bar) / max(h_bar.size(0), 1)
            self.C_hh.mul_(alpha).add_(cov, alpha=1.0 - alpha)

            # 去相关驱动必须在「尺度无关」的归一化协方差上构造。
            # 若直接用 C_hh^{-1}：h 方差很小时 C_hh^{-1} 的特征值极大，
            # 驱动项退化成 (1/scale) * h，实测会把输出放大 60 倍。
            # 先除以平均特征值归一化，则 h 整体缩放时 Cn 不变，驱动仅修正
            # 「各维度相对能量的不均衡」，不改变总体尺度。
            trace_mean = torch.diagonal(self.C_hh).mean().clamp(
                min=self.cfg.precision_eps)
            C_norm = self.C_hh / trace_mean

            # 条件数限幅：限制单次修正的最大增益，避免小方差方向被过度放大
            try:
                evals, evecs = torch.linalg.eigh(C_norm)
            except RuntimeError:
                evals = torch.ones(C_norm.size(0), device=C_norm.device,
                                   dtype=C_norm.dtype)
                evecs = torch.eye(C_norm.size(0), device=C_norm.device,
                                  dtype=C_norm.dtype)
            cond_max = float(self.cfg.lowrank_cond_max)
            evals = evals.clamp(min=1.0 / cond_max, max=cond_max)
            C_inv = torch.mm(evecs * (1.0 / evals).unsqueeze(0), evecs.t())

            lateral_w = self.lateral_weights.detach().float()
            h_current = h_float.clone()

            for j in range(self.cfg.neural_dynamics_iterations):
                lr_h = max(0.9 / (j + 1), 0.01)
                h_old = h_current

                # 侧向抑制：r 维特征间的相互竞争（PEM 特征解耦）
                lateral = torch.mm(h_current, lateral_w.t())
                # 精度加权去相关：把特征协方差推向各向同性
                whiten = torch.mm(h_current, C_inv.t()) - h_current

                grad_h = (self.cfg.lambda_lateral * lateral
                          + self.cfg.lowrank_whiten_coef * whiten)
                h_current = h_current - lr_h * grad_h

                if not torch.isfinite(h_current).all():
                    return h
                if torch.norm(h_current - h_old) < 1e-8 * torch.norm(h_current):
                    break

            h_current = h_current.to(h_2d.dtype)
        return h_current.reshape(orig_shape)

    # =================================================================
    # WTA 硬竞争 + BCM 自适应可塑性阈值
    # =================================================================

    def _wta_competition(self, h: torch.Tensor) -> torch.Tensor:
        """
        WTA 赢家通吃：低秩特征每步只保留 |h| 最大的 top-k 个维度，其余清零。
        k = ceil(r * wta_k_ratio)；k >= r 时自动退化为恒等映射。
        仅在训练期生效，推理保持确定性全秩输出。
        """
        if not self.training:
            return h
        r_dim = h.size(-1)
        k = int(math.ceil(r_dim * self.cfg.wta_k_ratio))
        k = max(1, min(k, r_dim))
        if k >= r_dim:
            return h
        _, idx = torch.topk(h.detach().abs(), k, dim=-1)
        mask = torch.zeros_like(h)
        mask.scatter_(-1, idx, 1.0)
        return h * mask

    def _bcm_theta_out(self) -> torch.Tensor:
        """输出侧 BCM 滑动阈值：theta ~ E[y^2]（直接复用 running_var）。"""
        theta = self.running_var.float()
        return theta, theta.mean().clamp(min=1e-8)

    def _bcm_theta_in(self) -> torch.Tensor:
        """输入侧 BCM 滑动阈值：theta ~ E[x^2] = Var(x) + E[x]^2。"""
        theta = torch.diagonal(self.C_x).float() + self.mu_x.float() ** 2
        return theta, theta.mean().clamp(min=1e-8)

    def _adaptive_saturation(self, output_side: bool) -> torch.Tensor:
        """
        BCM 自适应饱和上限：活动水平越高的维度，有效饱和上限越紧。
            cap_i = saturation_limit / (1 + bcm_gain * theta_i / mean(theta))
        替换原有的全局固定 saturation_limit。
        """
        theta, theta_bar = self._bcm_theta_out() if output_side else self._bcm_theta_in()
        return self.cfg.saturation_limit / (1.0 + self.cfg.bcm_gain * theta / theta_bar)

    def _clamp_saturate(self, t: torch.Tensor, output_side: bool) -> torch.Tensor:
        """按 BCM 自适应上限对 Hebbian 痕迹/增量做逐维度饱和裁剪。"""
        cap = self._adaptive_saturation(output_side).to(device=t.device, dtype=t.dtype)
        cap = cap.unsqueeze(1) if output_side else cap.unsqueeze(0)
        return torch.minimum(torch.maximum(t, -cap), cap)

    @staticmethod
    def project_rows_to_unit_simplex(y: torch.Tensor) -> torch.Tensor:
        """将每行投影到单位单纯形。"""
        batch, n = y.shape
        u, _ = torch.sort(y, dim=1, descending=True)
        cssv = torch.cumsum(u, dim=1)
        ind = torch.arange(1, n + 1, device=y.device, dtype=y.dtype).unsqueeze(0)
        cond = u - (cssv - 1) / ind > 0
        rho = cond.sum(dim=1, keepdim=True).clamp(min=1)
        rho_idx = (rho - 1).long().clamp(min=0)
        cssv_rho = cssv.gather(1, rho_idx)
        theta = (cssv_rho - 1) / rho.float()
        return torch.clamp(y - theta, min=0)

    # =================================================================
    # 动态白化 (常驻)
    # =================================================================

    def whiten_input(self, x: torch.Tensor) -> torch.Tensor:
        """对输入进行动态白化。"""
        x_centered = x - self.mu_x.unsqueeze(0)
        return F.linear(x_centered, self.W_whiten)

    # =================================================================
    # 前向传播
    # =================================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 -- 所有功能常驻。"""
        input_dtype = x.dtype
        if x.dtype != self.W0.dtype:
            x = x.to(dtype=self.W0.dtype, device=self.cfg.device, non_blocking=True)

        # 动态白化
        x = self.whiten_input(x)

        # 1. 原生权重前向
        y = F.linear(x, self.W0, self.bias)

        # 2. ReLoRA 累积知识
        y = y + F.linear(x, self._get_W_acc())

        # 3. 低秩增量(逐层计算，含侧向抑制 + WTA 硬竞争)
        #    双一维可塑性门控折叠进 A/B 的幅值调制：
        #      A' = A * (m_y * p_in)[None, :]     列门(输入侧)
        #      B' = B * (m_x * p_out)[:, None]    行门(输出侧)
        #    等价于 DeltaW 被 p_out[:, None] * p_in[None, :] 门控，
        #    但只做 O(r * (d_in + d_out)) 的逐元素缩放，且天然支持任意 batch 形状。
        p_out = self.plasticity_out.to(device=x.device, dtype=x.dtype)
        p_in = self.plasticity_in.to(device=x.device, dtype=x.dtype)

        effective_A = self.lora_A * (self.m_y * p_in).unsqueeze(0)
        h = F.linear(x, effective_A)

        # 3b. PEM 低秩空间神经动态
        #     只在 r 维 h 子空间内做特征解耦，不触碰 d_out 维主激活路径，
        #     因此不会破坏预训练模型已收敛的激活分布。
        #     同样用 straight-through 残差保证梯度可回传到 lora_A。
        if self.training and self.cfg.pem_space == "lowrank" \
                and self.cfg.neural_dynamics_iterations > 0:
            h = h + (self._low_rank_neural_dynamics(h) - h.detach())

        lateral_w = self.lateral_weights.detach().clone().to(h.dtype)
        h = h - F.linear(h, lateral_w)
        h = self._wta_competition(h)
        effective_B = self.lora_B * (self.m_x * p_out).unsqueeze(1)
        delta = F.linear(h, effective_B)
        delta = delta * (self.cfg.lora_alpha / self.cfg.r)

        # 4. 预测编码神经动态(完整输出空间版，仅 pem_space="output" 时启用)
        #    该版本在 d_out 维主激活路径上迭代，会扰动预训练模型已收敛的激活分布，
        #    因此默认走 "lowrank"，只在从零训练认知模型时才切到 "output"。
        #    run_predictive_neural_dynamics 内部为 no_grad，直接赋值会截断计算图。
        #    改用 straight-through 残差：前向数值等价(y = refined)，
        #    梯度沿恒等通路原样回传，保证 lora_A/B/m_x/m_y 能收到 BP 信号。
        if self.training and self.cfg.pem_space == "output":
            y = y + (self.run_predictive_neural_dynamics(x, y) - y.detach())

        # 5. 源信号域约束
        if self.training and self.cfg.presumed_domain == "antisparse":
            y = torch.clamp(y, -1, 1)
        elif self.training and self.cfg.presumed_domain in ["nnantisparse", "nnsparse", "simplex"]:
            y = torch.clamp(y, 0, 1)
            if self.cfg.presumed_domain == "simplex":
                y = self.project_rows_to_unit_simplex(y)

        # 累加 LoRA 增量
        y = y + delta

        # 再次运行预测编码(覆盖完整输出)
        if self.training and self.cfg.pem_space == "output":
            y = y + (self.run_predictive_neural_dynamics(x, y) - y.detach())

        # 再次施加域约束
        if self.training and self.cfg.presumed_domain == "antisparse":
            y = torch.clamp(y, -1, 1)
        elif self.training and self.cfg.presumed_domain in ["nnantisparse", "nnsparse", "simplex"]:
            y = torch.clamp(y, 0, 1)
            if self.cfg.presumed_domain == "simplex":
                y = self.project_rows_to_unit_simplex(y)

        # 6. Hebbian 影子增量前向传播
        if self._hebbian_pending_apply:
            hebb_A = self._hebbian_delta_A.detach().clone().to(x.dtype)
            hebb_B = self._hebbian_delta_B.detach().clone().to(x.dtype)
            delta_hebbian = F.linear(x, hebb_A)
            delta_hebbian = F.linear(delta_hebbian, hebb_B)
            delta_hebbian = delta_hebbian * (self.cfg.lora_alpha / self.cfg.r)
            gate_out = self.plasticity_out.unsqueeze(0)
            if gate_out.dtype != delta_hebbian.dtype or gate_out.device != delta_hebbian.device:
                gate_out = gate_out.to(device=delta_hebbian.device, dtype=delta_hebbian.dtype)
            y = y + delta_hebbian * gate_out

        # 7. 稳态可塑性
        if self.training:
            with torch.no_grad():
                if y.numel() > 0:
                    reduce_dims = tuple(range(y.dim() - 1))
                    batch_var = (y ** 2).mean(dim=reduce_dims).float()
                    tau = float(self.cfg.homeostatic_tau)
                    self.running_var = self.running_var * tau + batch_var * (1 - tau)

        # 稳态增益的量纲修正：running_var 是激活的二阶矩（方差，单位"平方"），
        # 而稳态归一化的物理意义是「激活强度归一化」，应与标准差成反比。
        # 原式 1/running_var 等价于 1/sigma^2，多层叠加会把输出 RMS 压到 1/3 次方量级，
        # 层数越多压缩越狠——这是 24 层全注入时直接学不动的根本原因。
        #
        # 注意：即使量纲正确，稳态归一化本身仍会把每层输出 RMS 拉向 1，
        # 对已收敛的预训练 LLM 属于额外扰动。实测(SST2, 3 seeds)关闭后准确率更高，
        # 故提供 homeostatic_enabled 开关；running_var 统计量始终维护，
        # 因为双一维可塑性门控依赖它。
        if self.cfg.homeostatic_enabled:
            homeostatic_gain = 1.0 / torch.sqrt(self.running_var + self.cfg.precision_eps)
            homeostatic_gain = homeostatic_gain.clamp(max=self.cfg.homeostatic_max_gain)
            if homeostatic_gain.device != y.device or homeostatic_gain.dtype != y.dtype:
                homeostatic_gain = homeostatic_gain.to(device=y.device, dtype=y.dtype)
            y = y * homeostatic_gain.unsqueeze(0)

        return y.to(input_dtype)

    # =================================================================
    # 训练步后处理
    # =================================================================

    def post_step_update(self, x: torch.Tensor, y: torch.Tensor, is_correct: bool = True,
                         merged: bool = False, y_target: Optional[torch.Tensor] = None,
                         global_signal: Optional[float] = None) -> bool:
        """
        训练步后处理 v3.1 -- 所有功能常驻。

        global_signal: GHL 全局方向信号（标量，正=与全局任务目标一致）。
            传入则直接作为全局梯度符号；不传则沿用上一帧 EMA 后的方向。
        """
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=False):
            self._steps_since_post_update = 0

            # 0. GHL 全局方向信号注入
            if global_signal is not None:
                self.set_global_direction(global_signal)

            # 1. Hebbian 更新
            with torch.no_grad():
                target_dtype = self.lora_A.dtype
                x_h = x.detach().to(target_dtype)
                y_h = y.detach().to(target_dtype)
                self.hebbian_update(x_h, y_h, is_correct)

            # 1b. 把本步 Hebbian 更新幅度作为认知冲突信号喂入 HGF 缓冲。
            #     此前 _append_hgf_delta 从未被调用，缓冲区恒为空，导致
            #     _detect_and_transition 的冲突检测与 check_anchor_convergence
            #     均不生效，HongWen 状态机只能靠固定步数切换。
            self._append_hgf_delta(self._last_hebbian_mag)

            # 2. 侧向连接更新
            with torch.no_grad():
                self._update_lateral_weights(y)

            # 3. 波动率耦合
            with torch.no_grad():
                self._update_volatility(y, y_target)

            # 4. Hong Wen 状态检测
            if self.step_counter % 50 == 0:
                self._detect_and_transition()

            # 5. 在线协方差与运行均值更新
            with torch.no_grad():
                y_mean = y.mean(dim=0).float()
                prediction_error = (y_mean - self.mu_y_prev).abs().mean()
                exp_omega = torch.exp(self.omega_cy)
                self.omega_cy.add_(self.cfg.volatility_lr * (prediction_error ** 2 - exp_omega))
                self.omega_cy.clamp_(-5.0, 5.0)
                self.dynamic_lr_cy.fill_(1.0 / torch.exp(self.omega_cy).clamp(min=0.1, max=10.0))
                lr_eff = (1 - self.cfg.lambda_lateral) * self.dynamic_lr_cy.item()
                self.mu_y.mul_(self.cfg.lambda_lateral).add_(y_mean, alpha=lr_eff)
                self.lowrank_precision.update(self.lora_B, y)
                self.lowrank_precision.spectral_filter(self._cy_condition_threshold)
                self.mu_y_prev.copy_(y_mean)

                # 输入空间统计
                x_mean = x.mean(dim=0).float()
                self.mu_x.mul_(self.cfg.lambda_lateral).add_(x_mean, alpha=1 - self.cfg.lambda_lateral)
                x_bar = x_mean - self.mu_x
                dC_x = torch.ger(x_bar, x_bar)
                self.C_x.mul_(self.cfg.lambda_lateral).add_(dC_x, alpha=1 - self.cfg.lambda_lateral)

                if self.step_counter % self.cfg.whitening_interval == 0:
                    try:
                        C_x_float = self.C_x.float()
                        U, S, Vh = torch.linalg.svd(C_x_float, full_matrices=False)
                        S_sqrt_inv = torch.diag(1.0 / torch.sqrt(S + 1e-5))
                        W_whiten_new = (U @ S_sqrt_inv @ Vh).to(self.W_whiten.dtype)
                        self.W_whiten.copy_(W_whiten_new)
                    except RuntimeError:
                        pass

            # 6. DLAM 睡眠检查(谱滤波 + 原型记忆巩固，可能更新原型重要性)
            if self.should_sleep():
                self._offline_replay()
            self._pending_cross_modal = True

            # 7. 刷新双一维可塑性门控 O(d_out + d_in)（替代二维 Fisher 更新）
            self._update_plasticity()

            # 8. 诊断输出(间隔触发)
            if self.step_counter % self.cfg.diagnostic_interval == 0:
                report = self.get_cognitive_report()
                self._diagnose(f"  [Aeloru] Diagnostic @ step {self.step_counter}: {report}")

            # 9. 检查固定周期合并
            if self.should_merge():
                self.merge_and_reset()
                return True

            if not merged:
                self.step_counter += 1
            return False

    # =================================================================
    # PEM 自适应侧向连接
    # =================================================================

    def _update_lateral_weights(self, y: torch.Tensor):
        """纯 Hebbian 侧向连接更新。"""
        if not self.training:
            return
        with torch.no_grad():
            y_mean = y.mean(dim=0).float()
            h_mean = torch.mm(self.lora_B.t().float(), y_mean.unsqueeze(1)).squeeze(1)
            dL = self.cfg.lateral_lr * torch.ger(h_mean, h_mean)
            new_lateral = self.lateral_weights * (1.0 - self.cfg.lateral_decay) + dL.to(self.lateral_weights.dtype)
            new_lateral.fill_diagonal_(0)
            self.lateral_weights.copy_(new_lateral)

    # =================================================================
    # HGF 波动率耦合
    # =================================================================

    def _update_volatility(self, y: torch.Tensor, y_target: Optional[torch.Tensor] = None):
        """波动率父节点更新：根据预测误差自动调节学习率。"""
        if not self.training:
            return
        with torch.no_grad():
            y_mean = y.mean(dim=0).float()
            if y_target is not None:
                y_target_mean = y_target.float().mean(dim=0)
                prediction_error = (y_mean - y_target_mean).abs().mean()
            else:
                prediction_error = (y_mean - self._prev_y_mean).abs().mean()
            exp_omega = torch.exp(self.omega)
            self.omega.add_(self.cfg.volatility_lr * (prediction_error ** 2 - exp_omega))
            self.omega.clamp_(-5.0, 5.0)
            self.dynamic_lr.fill_(1.0 / torch.exp(self.omega).clamp(min=0.1, max=10.0))
            self._prev_y_mean.copy_(y_mean)

    def _get_effective_hebbian_lr(self) -> float:
        """获取经波动率耦合调制后的 Hebbian 学习率。"""
        return self.cfg.hebbian_lr * self.dynamic_lr.item()

    def _get_effective_lora_lr(self) -> float:
        """获取经波动率耦合调制后的 LoRA 学习率。"""
        return self.cfg.LoRA_lr * self.dynamic_lr.item()

    # =================================================================
    # GHL 全局方向引导 (Hebbian Learning with Global Direction)
    # =================================================================

    def set_global_direction(self, signal: float) -> None:
        """
        注入全局方向信号（神经调质）。

        signal > 0 表示当前更新方向与全局任务目标一致（应强化），
        signal < 0 表示方向背离（应反转）。只取标量符号强度，不传梯度幅值，
        计算开销近乎为零；内部再做一次 EMA 平滑，避免单步噪声翻转方向。
        """
        try:
            s = float(signal)
        except (TypeError, ValueError):
            return
        if not math.isfinite(s):
            return
        s = max(-1.0, min(1.0, s))
        with torch.no_grad():
            alpha = self.cfg.ghl_sign_ema
            new_val = alpha * float(self.global_direction.item()) + (1.0 - alpha) * s
            self.global_direction.fill_(new_val)

    def observe_global_loss(self, loss_value: float) -> None:
        """
        用任务损失变化的相对方向作为 GHL 全局信号（无需回传梯度）。
        loss 下降 -> 正信号；loss 上升 -> 负信号；幅值按相对变化率缩放到 [-1, 1]。
        """
        try:
            cur = float(loss_value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(cur):
            return
        if self._prev_loss is not None:
            delta = self._prev_loss - cur
            denom = abs(self._prev_loss) + 1e-8
            rel = delta / denom
            self.set_global_direction(max(-1.0, min(1.0, rel * self.cfg.ghl_loss_gain)))
        self._prev_loss = cur

    def _global_sign_weight(self) -> float:
        """
        GHL 门权重 w：最终符号 sign = (1 - w) * 局部奖励 + w * 全局方向。
        探索期弱全局符号 + 强局部 Hebbian；锚定期强全局梯度 + 弱局部。
        """
        scale = {
            CognitiveState.EXPLORE: self.cfg.ghl_explore_scale,
            CognitiveState.RED: self.cfg.ghl_red_scale,
            CognitiveState.ANCHOR: self.cfg.ghl_anchor_scale,
            CognitiveState.SOLID: self.cfg.ghl_solid_scale,
        }.get(self.state, self.cfg.ghl_explore_scale)
        w = self.cfg.ghl_global_strength * scale
        return max(0.0, min(1.0, w))

    # =================================================================
    # SPHeRe 结构保持损失 + 正交惩罚
    # =================================================================

    def get_structure_preservation_loss(self) -> torch.Tensor:
        """
        SPHeRe 低维结构投影约束。

        低秩特征协方差 S = A C_x A^T (r x r)，要求其在输入度量下各向同性：
            S / tr(S) * r  ->  I
        该约束保证 r 个低秩方向在输入数据流形上正交且能量均衡，
        从根源上避免小 r 下的特征坍缩（有效秩退化）。
        复杂度 O(r^2 * d_in)，与 PEM 侧向抑制互补（解耦 + 拓扑保持）。
        """
        A = self.lora_A.float()                                   # r x d_in
        Cx = self.C_x.detach().float()                            # d_in x d_in (统计量，截断梯度)
        S = torch.mm(torch.mm(A, Cx), A.t())                      # r x r
        r_dim = S.shape[0]
        trace = torch.diagonal(S).sum()
        if float(trace.detach().abs()) < 1e-8:
            self._last_structure_loss = 0.0
            return torch.zeros((), device=S.device, dtype=S.dtype)
        S_norm = S * (r_dim / trace.detach())                     # 尺度不变归一化
        I = torch.eye(r_dim, device=S.device, dtype=S.dtype)
        loss = self.cfg.sphere_lambda * ((S_norm - I) ** 2).sum()
        self._last_structure_loss = float(loss.detach().item())
        return loss

    def get_ortho_penalty(self) -> torch.Tensor:
        """
        正交惩罚损失 = DeltaW 与 W0 的方向重叠惩罚 + SPHeRe 低秩结构保持损失。
        """
        self._ortho_proj_step += 1
        if self._ortho_proj_step % self.cfg.diagnostic_interval == 0:
            with torch.no_grad():
                self._ortho_proj.normal_()

        lam = self.cfg.ortho_lambda
        if self.state == CognitiveState.ANCHOR:
            lam = self.cfg.ortho_lambda_anchor

        delta_w = self.compute_delta_w()
        k = self._ortho_proj.shape[1]
        proj_delta = torch.mm(delta_w.float(), self._ortho_proj)
        proj_w0 = torch.mm(self.W0.float(), self._ortho_proj.detach())
        overlap = torch.mm(proj_delta.t(), proj_w0)
        ortho = lam * overlap.norm(p='fro') ** 2 * (self.in_features / k)

        return ortho + self.get_structure_preservation_loss()

    # =================================================================
    # Hebbian-可塑性 双向联动
    # =================================================================

    def hebbian_update(self, x: torch.Tensor, y: torch.Tensor, is_correct: bool = True):
        """Hebbian 更新 + 双一维可塑性门控 + GHL 全局方向门。"""
        eta_eff = self._get_effective_hebbian_lr()

        with torch.no_grad():
            if x.dim() == 3:
                x = x.reshape(-1, x.size(-1))
            if y.dim() == 3:
                y = y.reshape(-1, y.size(-1))

            x_mean = x.mean(dim=0)
            y_mean = y.mean(dim=0)

            # GHL 全局方向门：局部奖励符号 + 全局梯度/损失方向符号 的凸组合
            # is_correct 为 None 表示无外部奖励信号，按正向强化处理
            local_sign = -1.0 if is_correct is False else 1.0
            global_sign = float(self.global_direction.item())
            w = self._global_sign_weight()
            sign = (1.0 - w) * local_sign + w * global_sign

            raw_dB = sign * eta_eff * torch.ger(y_mean, x_mean[:self.cfg.r])
            raw_dA = sign * eta_eff * torch.ger(y_mean[:self.cfg.r], x_mean)

            # 双一维可塑性门控：行门（输出侧）+ 列门（输入侧），O(d_out + d_in)
            dev, dt = x.device, self.lora_A.dtype
            plasticity_B = self.plasticity_out.to(device=dev, dtype=dt).unsqueeze(1)
            plasticity_A = self.plasticity_in.to(device=dev, dtype=dt).unsqueeze(0)
            dB = raw_dB * plasticity_B
            dA = raw_dA * plasticity_A

            self._hebbian_acc_B.add_(dB)
            self._hebbian_acc_A.add_(dA)
            self._hebbian_acc_count += 1

            # 必须在 _flush_hebbian 之前采样：flush 会把累加缓冲清零。
            # 该幅度将作为认知冲突信号喂入 _hgf_delta_buffer。
            self._last_hebbian_mag = float(
                self._hebbian_acc_B.abs().mean().item()
                + self._hebbian_acc_A.abs().mean().item()
            )

            if self._hebbian_acc_count >= self.cfg.hebbian_accum_steps:
                self._flush_hebbian()

    def _flush_hebbian(self):
        """将累积的 Hebbian 更新批量应用到影子缓冲区。"""
        if self._hebbian_acc_count == 0:
            return

        with torch.no_grad():
            decay_factor = self.cfg.hebbian_decay ** self._hebbian_acc_count
            if self._hebbian_pending_apply:
                new_delta_A = self._hebbian_delta_A * decay_factor
                new_delta_B = self._hebbian_delta_B * decay_factor
            else:
                new_delta_A = self._hebbian_delta_A.clone()
                new_delta_B = self._hebbian_delta_B.clone()

            # STDP 噪声梯度：噪声强度相对本次更新幅度自适应
            # （带噪声的 Hebbian == 带噪声的梯度下降，帮助跳出局部最优）
            noise = self._get_hebbian_noise_scale()
            if noise > 0.0:
                new_delta_A = new_delta_A + noise * torch.randn_like(new_delta_A)
                new_delta_B = new_delta_B + noise * torch.randn_like(new_delta_B)

            new_delta_B = new_delta_B + self._hebbian_acc_B
            new_delta_A = new_delta_A + self._hebbian_acc_A

            # BCM 自适应饱和阈值：活动越高的维度上限越紧，替代固定 saturation_limit
            new_delta_B = self._clamp_saturate(new_delta_B, True)
            new_delta_A = self._clamp_saturate(new_delta_A, False)

            self._hebbian_delta_A.copy_(new_delta_A)
            self._hebbian_delta_B.copy_(new_delta_B)

            # 更新低秩 Hebbian 痕迹（Fisher 已完全由 HGF 运行统计承载，此处不再更新）
            new_trace_B = self.hebb_trace_B + self._hebbian_acc_B.abs()
            new_trace_A = self.hebb_trace_A + self._hebbian_acc_A.abs()
            self.hebb_trace_B.copy_(new_trace_B)
            self.hebb_trace_A.copy_(new_trace_A)

            # 每次固化即向记忆中存入一条模式，供 DLAM 睡眠的负载度判定
            self.stored_patterns.add_(1)

            self._hebbian_acc_B.zero_()
            self._hebbian_acc_A.zero_()
            self._hebbian_acc_count = 0
            self._hebbian_pending_apply = True
            self._cache_valid = False

    def _get_hebbian_noise_scale(self) -> float:
        """
        STDP 噪声梯度：噪声标准差 = 相对幅度 x 本次更新量级。

        调制因子：
          - HGF 波动率 exp(omega)：预测越不稳定 -> 噪声越大
          - 认知冲突分数：冲突越大 -> 噪声越大
          - 认知状态：红温期最大（加强探索），固化期最小（稳定收敛）
        """
        base_ratio = self.cfg.hebbian_noise_ratio
        if base_ratio <= 0.0:
            return 0.0

        ref_b = float(self._hebbian_acc_B.detach().float().abs().mean().item())
        ref_a = float(self._hebbian_acc_A.detach().float().abs().mean().item())
        ref = 0.5 * (ref_b + ref_a)
        if not math.isfinite(ref) or ref < 1e-12:
            return 0.0

        vol = float(torch.exp(self.omega.float()).clamp(0.1, 10.0).item())
        if isinstance(self.conflict_score, torch.Tensor):
            conflict = float(self.conflict_score.item())
        else:
            conflict = float(self.conflict_score)
        conflict_n = min(1.0, conflict / max(self.cfg.sleep_condition_threshold, 1e-8))

        state_scale = {
            CognitiveState.EXPLORE: 1.0,
            CognitiveState.RED: 1.5,
            CognitiveState.ANCHOR: 0.5,
            CognitiveState.SOLID: 0.1,
        }.get(self.state, 1.0)

        ratio = base_ratio * (0.5 + vol) * (1.0 + conflict_n) * state_scale
        ratio = min(ratio, self.cfg.hebbian_noise_max_ratio)
        return ref * ratio

    # =================================================================
    # DLAM 原型记忆 + 谱滤波睡眠机制
    # =================================================================

    @staticmethod
    def _normalize01(t: torch.Tensor) -> torch.Tensor:
        t = t - t.min()
        m = t.max()
        if float(m) > 1e-12:
            t = t / m
        return t

    def _extract_prototypes(self) -> int:
        """
        原型记忆巩固：从低秩 Hebbian 痕迹中提取 top-k 个主特征向量作为任务原型，
        并把原型反投影成输入/输出两个一维重要性向量。

            Sigma = trace_A @ trace_A^T   (r x r)
            V     = top-k 特征向量         (k x r)
            imp_in  = max_k |V @ trace_A|          -> d_in
            imp_out = max_k |lora_B @ V^T|         -> d_out

        与原型重叠越高的维度，后续可塑性越低（主动保护旧记忆）。
        """
        with torch.no_grad():
            A = self.hebb_trace_A.float()                        # r x d_in
            if A.numel() == 0:
                return 0
            Sigma = A @ A.t() / max(A.shape[0], 1)               # r x r
            k = min(self.stored_prototypes.shape[0], self.cfg.r)
            try:
                eigvals, eigvecs = torch.linalg.eigh(Sigma)
            except RuntimeError:
                return 0

            order = torch.argsort(eigvals, descending=True)[:k]
            V = eigvecs[:, order].t().contiguous()               # k x r

            self.stored_prototypes.zero_()
            self.stored_prototypes[:V.shape[0]].copy_(V)

            # 两侧都从 Hebbian 痕迹投影（与 lora_A/B 不同，痕迹在 ReLoRA 合并时不会被清零）
            imp_in = (V @ A).abs().max(dim=0).values                          # d_in
            imp_out = (self.hebb_trace_B.float() @ V.t()).abs().max(dim=1).values  # d_out

            self.proto_importance_in.copy_(self._normalize01(imp_in))
            self.proto_importance_out.copy_(self._normalize01(imp_out))
            return int(V.shape[0])

    def should_sleep(self) -> bool:
        """基于 DLAM 理论的最优睡眠触发条件。"""
        if self.step_counter - self._last_sleep_step < self.cfg.min_steps_between_sleep:
            return False
        if self.stored_patterns.item() < 10:
            return False
        if self.hebb_trace_A.numel() == 0 or self.hebb_trace_B.numel() == 0:
            return False

        with torch.no_grad():
            # fp32 上做特征分解：低精度 dtype 的 linalg 算子不可用
            A_f = self.hebb_trace_A.float()
            Sigma = A_f @ A_f.T
            try:
                eigvals = torch.linalg.eigvalsh(Sigma)
                condition_number = eigvals[-1] / (eigvals[0] + 1e-8)
            except RuntimeError:
                return False

            if isinstance(self.conflict_score, torch.Tensor):
                self.conflict_score.fill_(condition_number.item())
            else:
                self.conflict_score = float(condition_number.item())

            if self.step_counter % self.cfg.diagnostic_interval == 0:
                self._log(f"  [Aeloru] Sleep Check @ step {self.step_counter}, "
                          f"Condition Number: {condition_number:.2f}")
            return condition_number > self.cfg.sleep_condition_threshold

    def _offline_replay(self):
        """DLAM 谱滤波睡眠。"""
        if self.hebb_trace_A.numel() == 0 or self.hebb_trace_B.numel() == 0:
            return

        with torch.no_grad():
            A_f = self.hebb_trace_A.float()
            out_dim = max(self.hebb_trace_B.shape[0], 1)
            Sigma = A_f @ A_f.T / out_dim

            r_dim = self.cfg.r
            stored = self.stored_patterns.item()
            alpha_load = stored / max(r_dim, 1)
            t_opt = 1.0 / alpha_load if alpha_load > 1e-3 else 100.0
            t_opt = max(0.1, min(t_opt, 100.0))

            I = torch.eye(r_dim, device=Sigma.device, dtype=Sigma.dtype)
            try:
                A_kernel = (1 + t_opt) * torch.inverse(I + t_opt * Sigma)
            except RuntimeError:
                A_kernel = (1 + t_opt) * torch.linalg.pinv(I + t_opt * Sigma)

            filtered_A = A_kernel @ A_f
            self.hebb_trace_A.copy_(filtered_A.to(self.hebb_trace_A.dtype))

            # 弱连接修剪（低秩痕迹上直接裁剪，无需 d_out x d_in 掩码）
            weak_mask_B = self.hebb_trace_B.abs() < self.cfg.dlam_replay_threshold
            weak_mask_A = self.hebb_trace_A.abs() < self.cfg.dlam_replay_threshold
            self.hebb_trace_B[weak_mask_B] = 0.0
            self.hebb_trace_A[weak_mask_A] = 0.0

            # 原型记忆巩固：谱滤波之后再提取任务原型
            n_proto = self._extract_prototypes()

            # 本轮记忆已压缩进原型，模式计数清零，重新开始下一轮记忆积累
            self.stored_patterns.zero_()

            if isinstance(self.conflict_score, torch.Tensor):
                self.conflict_score.fill_(0.0)
            else:
                self.conflict_score = 0.0
            self._last_sleep_step = self.step_counter

            self._log(f"  [Aeloru] DLAM SLEEP @ step {self.step_counter}, "
                      f"t_opt={t_opt:.2f}, active_r={r_dim}, prototypes={n_proto}")

    def _cross_modal_binding(self, other_layers: List['AeloruLayer']):
        """DLAM 跨模态异联想绑定。"""
        for other in other_layers:
            if other is self or other.hebb_trace_A is None or other.hebb_trace_B is None:
                continue

            coupling = self.cfg.cross_modal_coupling
            self_full = torch.mm(self.hebb_trace_B, self.hebb_trace_A)
            other_full = torch.mm(other.hebb_trace_B, other.hebb_trace_A)

            self_full = self_full.to(torch.float32)
            other_full = other_full.to(torch.float32)

            if self_full.shape != other_full.shape:
                continue

            trace_self = torch.trace(torch.mm(self_full.t(), self_full))
            trace_other = torch.trace(torch.mm(other_full.t(), other_full))
            if trace_self < 1e-10 or trace_other < 1e-10:
                continue

            cross_alignment = torch.trace(torch.mm(self_full.t(), other_full)) / torch.sqrt(trace_self * trace_other)

            self_flat = self_full.view(-1)
            other_flat = other_full.view(-1)

            alignment_grad = cross_alignment * other_flat / torch.sqrt(trace_other)
            variance_grad = self_flat / torch.sqrt(trace_self)
            grad_flat = alignment_grad - 0.5 * variance_grad
            grad_full = grad_flat.view_as(self_full)

            svd_success = False
            try:
                U_grad, S_grad, Vh_grad = torch.linalg.svd(grad_full.float(), full_matrices=False)
                r_eff = min(self.cfg.r, S_grad.numel())
                if r_eff >= self.cfg.r:
                    sqrt_S = torch.sqrt(S_grad[:r_eff].clamp(min=0.0))
                    delta_B = U_grad[:, :r_eff] * sqrt_S.unsqueeze(0)
                    delta_A = torch.diag(sqrt_S) @ Vh_grad[:r_eff, :]
                    self.hebb_trace_B.add_(coupling * delta_B.to(self.hebb_trace_B.dtype))
                    self.hebb_trace_A.add_(coupling * delta_A.to(self.hebb_trace_A.dtype))
                    svd_success = True
            except RuntimeError:
                svd_success = False

            if not svd_success:
                self.hebb_trace_B.add_(coupling * torch.mm(grad_full, self.hebb_trace_A.t()))
                self.hebb_trace_A.add_(coupling * torch.mm(self.hebb_trace_B.t(), grad_full))

            # 双向对称更新
            alignment_grad_rev = cross_alignment * self_flat / torch.sqrt(trace_self)
            variance_grad_rev = other_flat / torch.sqrt(trace_other)
            grad_flat_rev = alignment_grad_rev - 0.5 * variance_grad_rev
            grad_full_rev = grad_flat_rev.view_as(other_full)

            svd_success = False
            try:
                U_grad, S_grad, Vh_grad = torch.linalg.svd(grad_full_rev.float(), full_matrices=False)
                r_eff = min(other.cfg.r, S_grad.numel())
                if r_eff >= other.cfg.r:
                    sqrt_S = torch.sqrt(S_grad[:r_eff].clamp(min=0.0))
                    delta_B_other = U_grad[:, :r_eff] * sqrt_S.unsqueeze(0)
                    delta_A_other = torch.diag(sqrt_S) @ Vh_grad[:r_eff, :]
                    other.hebb_trace_B.add_(coupling * delta_B_other.to(other.hebb_trace_B.dtype))
                    other.hebb_trace_A.add_(coupling * delta_A_other.to(other.hebb_trace_A.dtype))
                    svd_success = True
            except RuntimeError:
                svd_success = False

            if not svd_success:
                other.hebb_trace_B.add_(coupling * torch.mm(grad_full_rev, other.hebb_trace_A.t()))
                other.hebb_trace_A.add_(coupling * torch.mm(other.hebb_trace_B.t(), grad_full_rev))

            # BCM 自适应饱和阈值裁剪
            self.hebb_trace_B.copy_(self._clamp_saturate(self.hebb_trace_B, True))
            self.hebb_trace_A.copy_(self._clamp_saturate(self.hebb_trace_A, False))
            other.hebb_trace_B.copy_(other._clamp_saturate(other.hebb_trace_B, True))
            other.hebb_trace_A.copy_(other._clamp_saturate(other.hebb_trace_A, False))

    # =================================================================
    # Hong Wen 认知状态机
    # =================================================================

    def _hebbian_allowed(self) -> bool:
        return self.state in [CognitiveState.EXPLORE, CognitiveState.SOLID]

    def _bp_allowed(self) -> bool:
        return self.state in [CognitiveState.EXPLORE, CognitiveState.ANCHOR, CognitiveState.SOLID]

    def _detect_and_transition(self):
        """基于参数更新幅度波动率检测认知冲突。"""
        with torch.no_grad():
            if len(self._hgf_delta_buffer) >= 50:
                buf = torch.tensor(
                    self._hgf_delta_buffer[-50:],
                    device=self.W0.device,
                    dtype=torch.float32
                )
                mean = buf.mean()
                if mean < 1e-8:
                    return
                std = buf.std(unbiased=False)
                conflict_score = (std / mean).item() * 5.0
            else:
                return

            old_state = self.state

            if self.state == CognitiveState.EXPLORE:
                explore_duration = self.step_counter - self._explore_start_step
                if explore_duration >= self.cfg.explore_steps and conflict_score > self.cfg.red_threshold:
                    self._transition_state(CognitiveState.RED)
                    self._flush_hebbian()
            elif self.state == CognitiveState.RED:
                if self.step_counter - self._red_enter_step >= self.cfg.red_min_steps:
                    self._transition_state(CognitiveState.ANCHOR)
            elif self.state == CognitiveState.ANCHOR:
                anchor_duration = self.step_counter - self._red_enter_step - self.cfg.red_min_steps
                if anchor_duration >= self.cfg.anchor_steps:
                    if len(self._hgf_delta_buffer) >= 10:
                        recent_deltas = self._hgf_delta_buffer[-10:]
                        avg_delta = sum(recent_deltas) / len(recent_deltas)
                        if avg_delta < self.cfg.anchor_converge * 100:
                            self._transition_state(CognitiveState.SOLID)
            elif self.state == CognitiveState.SOLID:
                if self.step_counter >= self._solid_end_step:
                    self._transition_state(CognitiveState.EXPLORE)

            if old_state != self.state:
                self._log(f"  [Aeloru] State {old_state.value} -> {self.state.value} "
                          f"@ step {self.step_counter} (conflict={conflict_score:.3f})")

    def _append_hgf_conflict(self, signal: float):
        self._hgf_conflict_buffer.append(float(signal))
        if len(self._hgf_conflict_buffer) > 100:
            self._hgf_conflict_buffer.pop(0)

    def _append_hgf_delta(self, delta_norm: float):
        self._hgf_delta_buffer.append(float(delta_norm))
        if len(self._hgf_delta_buffer) > 100:
            self._hgf_delta_buffer.pop(0)

    def _transition_state(self, new_state: CognitiveState):
        self.state = new_state
        if new_state == CognitiveState.EXPLORE:
            self._explore_start_step = self.step_counter
        elif new_state == CognitiveState.RED:
            self._red_enter_step = self.step_counter
        elif new_state == CognitiveState.ANCHOR:
            self._anchor_grad_history.clear()
        elif new_state == CognitiveState.SOLID:
            self._solid_end_step = self.step_counter + self.cfg.solid_steps

    def check_anchor_convergence(self, grad_norm: float) -> bool:
        """外部 BP 调用者检查锚定收敛。"""
        if self.state == CognitiveState.ANCHOR:
            if len(self._hgf_delta_buffer) > 0:
                delta_proxy = self._hgf_delta_buffer[-1]
                self._anchor_grad_history.append(delta_proxy)
                if len(self._anchor_grad_history) >= 10:
                    avg_delta = sum(self._anchor_grad_history[-10:]) / 10
                    if avg_delta < self.cfg.anchor_converge * 100:
                        self._transition_state(CognitiveState.SOLID)
                        return True
        return False

    # =================================================================
    # ReLoRA 外置合并
    # =================================================================

    def should_merge(self) -> bool:
        return self.step_counter >= self.cfg.merge_every

    def merge_and_reset(self):
        """同步合并：W_acc <- W_acc + DeltaW，重置 A, B。"""
        with torch.no_grad():
            self._flush_hebbian()
            delta_w = self.compute_delta_w()
            delta_w = self.apply_hidora(delta_w)
            delta_w = self.apply_plasticity_gate(delta_w)
            delta_w = self.apply_energy_budget(delta_w)
            new_W_acc = self._get_W_acc() + delta_w
            self._set_W_acc(new_W_acc)
            self._reset_adapters()
            self.step_counter = 0
            self._explore_start_step = 0
            self.hebb_trace_B.mul_(0.5)
            self.hebb_trace_A.mul_(0.5)
            self._pending_optimizer_reset = True
            self._log(f"  [Aeloru] MERGED @ step {self.step_counter}. W_acc norm={self._get_W_acc().norm().item():.4f}")

    # =================================================================
    # 诊断接口
    # =================================================================

    def get_cognitive_report(self) -> Dict[str, Any]:
        """输出当前认知状态诊断报告。"""
        with torch.no_grad():
            report = {
                'state': self.state.value,
                'step': self.step_counter,
                'w_acc_norm': self._get_W_acc().norm().item(),
                'delta_w_norm': self.compute_delta_w().norm().item(),
                'plasticity_out_mean': self.plasticity_out.float().mean().item(),
                'plasticity_out_min': self.plasticity_out.float().min().item(),
                'plasticity_in_mean': self.plasticity_in.float().mean().item(),
                'plasticity_in_min': self.plasticity_in.float().min().item(),
                'pi_out_mean': (1.0 / (self.running_var.float() + self.cfg.precision_eps)).mean().item(),
                'pi_in_mean': (1.0 / (torch.diagonal(self.C_x).float() + self.cfg.precision_eps)).mean().item(),
                'global_direction': self.global_direction.item(),
                'structure_loss': self._last_structure_loss,
                'proto_importance_out': self.proto_importance_out.float().mean().item(),
                'proto_importance_in': self.proto_importance_in.float().mean().item(),
                'bcm_cap_out_mean': self._adaptive_saturation(True).mean().item(),
                'bcm_cap_in_mean': self._adaptive_saturation(False).mean().item(),
                # 与 forward 中的量纲修正保持一致（1 / sqrt(var)）
                'homeostatic_gain_mean': (
                    1.0 / torch.sqrt(self.running_var + self.cfg.precision_eps)
                ).mean().item(),
                'omega': self.omega.item(),
                'dynamic_lr': self.dynamic_lr.item(),
                'stored_patterns': self.stored_patterns.item(),
                'conflict_score': self.conflict_score.item() if isinstance(self.conflict_score, torch.Tensor) else self.conflict_score,
                'steps_since_sleep': self.step_counter - self._last_sleep_step,
                'lateral_norm': self.lateral_weights.norm().item(),
                'C_y_omega': self.omega_cy.item(),
                'C_y_dynamic_lr': self.dynamic_lr_cy.item(),
                'C_x_trace': torch.trace(self.C_x.float()).item(),
                'whiten_norm': self.W_whiten.norm().item(),
                'hgf_conflict': self._compute_hgf_conflict(),
            }
            return report

    def _compute_hgf_conflict(self) -> float:
        if len(self._hgf_conflict_buffer) < 50:
            return 0.0
        buf = torch.tensor(
            self._hgf_conflict_buffer[-50:],
            device=self.W0.device,
            dtype=torch.float32
        )
        mean = buf.mean()
        if mean < 1e-8:
            return 0.0
        std = buf.std(unbiased=False)
        return (std / mean).item() * 5.0

    # =================================================================
    # 序列化接口
    # =================================================================

    def save_adapter(self, path: str):
        """保存 Aeloru 适配器。"""
        cfg_dict = {k: v for k, v in self.cfg.__dict__.items() if not k.startswith('_')}
        checkpoint = {
            'cfg_dict': cfg_dict,
            'lora_A': self.lora_A.data.cpu(),
            'lora_B': self.lora_B.data.cpu(),
            'W_acc': self._get_W_acc().cpu(),
            'step_counter': self.step_counter,
            'state_str': self.state.value,
            '_ortho_proj': self._ortho_proj.cpu() if hasattr(self, '_ortho_proj') else None,
            '_W_acc_cache': self._W_acc_cache.cpu() if hasattr(self, '_W_acc_cache') and self._W_acc_cache is not None else None,
            'm_x': self.m_x.data.cpu() if self.m_x is not None else None,
            'm_y': self.m_y.data.cpu() if self.m_y is not None else None,
            'plasticity_out': self.plasticity_out.data.cpu().float(),
            'plasticity_in': self.plasticity_in.data.cpu().float(),
            'global_direction': self.global_direction.cpu() if self.global_direction is not None else None,
            'stored_prototypes': self.stored_prototypes.cpu() if self.stored_prototypes is not None else None,
            'proto_importance_out': self.proto_importance_out.cpu() if self.proto_importance_out is not None else None,
            'proto_importance_in': self.proto_importance_in.cpu() if self.proto_importance_in is not None else None,
            'lateral_weights': self.lateral_weights.cpu() if self.lateral_weights is not None else None,
            'running_var': self.running_var.cpu() if self.running_var is not None else None,
            'omega': self.omega.cpu() if self.omega is not None else None,
            'dynamic_lr': self.dynamic_lr.cpu() if self.dynamic_lr is not None else None,
            '_prev_y_mean': self._prev_y_mean.cpu() if self._prev_y_mean is not None else None,
            'stored_patterns': self.stored_patterns.cpu() if self.stored_patterns is not None else None,
            'hebb_trace_B': self.hebb_trace_B.cpu() if self.hebb_trace_B is not None else None,
            'hebb_trace_A': self.hebb_trace_A.cpu() if self.hebb_trace_A is not None else None,
            'conflict_score': self.conflict_score,
            '_last_sleep_step': self._last_sleep_step,
            '_explore_start_step': self._explore_start_step,
            'mu_y': self.mu_y.cpu() if self.mu_y is not None else None,
            'mu_y_prev': self.mu_y_prev.cpu() if self.mu_y_prev is not None else None,
            'omega_cy': self.omega_cy.cpu() if self.omega_cy is not None else None,
            'dynamic_lr_cy': self.dynamic_lr_cy.cpu() if self.dynamic_lr_cy is not None else None,
            'lowrank_B_basis': self.lowrank_precision.B_basis.cpu() if self.lowrank_precision is not None else None,
            'lowrank_C_h': self.lowrank_precision.C_h.cpu() if self.lowrank_precision is not None else None,
            'mu_x': self.mu_x.cpu() if self.mu_x is not None else None,
            'C_x': self.C_x.cpu() if self.C_x is not None else None,
            'W_whiten': self.W_whiten.cpu() if self.W_whiten is not None else None,
            # 低秩特征空间 h 的在线统计(PEM 低秩神经动态)
            'C_hh': self.C_hh.cpu() if self.C_hh is not None else None,
            'mu_h': self.mu_h.cpu() if self.mu_h is not None else None,
        }
        try:
            torch.save(checkpoint, path)
            self._log(f"保存至 {os.path.abspath(path)}")
        except Exception:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            torch.save(checkpoint, path)
            self._log(f"保存至 {os.path.abspath(path)}")

    def load_adapter(self, path: str):
        """加载 Aeloru 适配器。"""
        checkpoint = torch.load(path, map_location=self.W0.device, weights_only=True)
        assert checkpoint['cfg_dict']['r'] == self.cfg.r
        assert checkpoint['cfg_dict']['in_features'] == self.cfg.in_features
        assert checkpoint['cfg_dict']['out_features'] == self.cfg.out_features

        self.lora_A.data.copy_(checkpoint['lora_A'].to(self.lora_A.device))
        self.lora_B.data.copy_(checkpoint['lora_B'].to(self.lora_B.device))
        self._set_W_acc(checkpoint['W_acc'].to(self.W0.device))
        self.step_counter = checkpoint['step_counter']
        self.state = CognitiveState(checkpoint['state_str'])

        if 'm_x' in checkpoint and checkpoint['m_x'] is not None and self.m_x is not None:
            self.m_x.data.copy_(checkpoint['m_x'].to(self.m_x.device))
        if 'm_y' in checkpoint and checkpoint['m_y'] is not None and self.m_y is not None:
            self.m_y.data.copy_(checkpoint['m_y'].to(self.m_y.device))
        # v3.0 -> v3.1 迁移：二维 Fisher 掩码降级为两条一维可塑性门
        if 'fisher_mask' in checkpoint and checkpoint['fisher_mask'] is not None:
            with torch.no_grad():
                old = checkpoint['fisher_mask'].to(torch.float32)
                self.plasticity_out.copy_(old.mean(dim=1).to(self.plasticity_out.device))
                self.plasticity_in.copy_(old.mean(dim=0).to(self.plasticity_in.device))
                self.plasticity_out.clamp_(min=self.cfg.plasticity_min)
                self.plasticity_in.clamp_(min=self.cfg.plasticity_min)
        if 'plasticity_out' in checkpoint and checkpoint['plasticity_out'] is not None:
            self.plasticity_out.copy_(checkpoint['plasticity_out'].to(self.plasticity_out.device))
        if 'plasticity_in' in checkpoint and checkpoint['plasticity_in'] is not None:
            self.plasticity_in.copy_(checkpoint['plasticity_in'].to(self.plasticity_in.device))
        if 'global_direction' in checkpoint and checkpoint['global_direction'] is not None:
            self.global_direction.copy_(checkpoint['global_direction'].to(self.global_direction.device))
        if 'stored_prototypes' in checkpoint and checkpoint['stored_prototypes'] is not None:
            self.stored_prototypes.copy_(checkpoint['stored_prototypes'].to(self.stored_prototypes.device))
        if 'proto_importance_out' in checkpoint and checkpoint['proto_importance_out'] is not None:
            self.proto_importance_out.copy_(checkpoint['proto_importance_out'].to(self.proto_importance_out.device))
        if 'proto_importance_in' in checkpoint and checkpoint['proto_importance_in'] is not None:
            self.proto_importance_in.copy_(checkpoint['proto_importance_in'].to(self.proto_importance_in.device))
        if 'C_hh' in checkpoint and checkpoint['C_hh'] is not None and self.C_hh is not None:
            self.C_hh.copy_(checkpoint['C_hh'].to(device=self.C_hh.device, dtype=self.C_hh.dtype))
        if 'mu_h' in checkpoint and checkpoint['mu_h'] is not None and self.mu_h is not None:
            self.mu_h.copy_(checkpoint['mu_h'].to(device=self.mu_h.device, dtype=self.mu_h.dtype))
        if '_ortho_proj' in checkpoint and checkpoint['_ortho_proj'] is not None:
            self._ortho_proj.copy_(checkpoint['_ortho_proj'].to(self._ortho_proj.device))
        if '_W_acc_cache' in checkpoint and checkpoint['_W_acc_cache'] is not None:
            self._W_acc_cache = checkpoint['_W_acc_cache'].to(self.W0.device)
        if 'lateral_weights' in checkpoint and checkpoint['lateral_weights'] is not None and self.lateral_weights is not None:
            self.lateral_weights.copy_(checkpoint['lateral_weights'].to(self.lateral_weights.device))
        if 'running_var' in checkpoint and checkpoint['running_var'] is not None and self.running_var is not None:
            self.running_var.copy_(checkpoint['running_var'].to(self.running_var.device))
        if 'omega' in checkpoint and checkpoint['omega'] is not None and self.omega is not None:
            self.omega.copy_(checkpoint['omega'].to(self.omega.device))
        if 'dynamic_lr' in checkpoint and checkpoint['dynamic_lr'] is not None and self.dynamic_lr is not None:
            self.dynamic_lr.copy_(checkpoint['dynamic_lr'].to(self.dynamic_lr.device))
        if '_prev_y_mean' in checkpoint and checkpoint['_prev_y_mean'] is not None and self._prev_y_mean is not None:
            self._prev_y_mean.copy_(checkpoint['_prev_y_mean'].to(self._prev_y_mean.device))
        if 'stored_patterns' in checkpoint and checkpoint['stored_patterns'] is not None and self.stored_patterns is not None:
            self.stored_patterns.copy_(checkpoint['stored_patterns'].to(self.stored_patterns.device))
        if 'hebb_trace_B' in checkpoint and checkpoint['hebb_trace_B'] is not None and self.hebb_trace_B is not None:
            self.hebb_trace_B.copy_(checkpoint['hebb_trace_B'].to(self.hebb_trace_B.device))
        if 'hebb_trace_A' in checkpoint and checkpoint['hebb_trace_A'] is not None and self.hebb_trace_A is not None:
            self.hebb_trace_A.copy_(checkpoint['hebb_trace_A'].to(self.hebb_trace_A.device))
        if 'conflict_score' in checkpoint:
            if isinstance(self.conflict_score, torch.Tensor):
                value = checkpoint['conflict_score']
                if isinstance(value, torch.Tensor):
                    self.conflict_score.fill_(value.item())
                else:
                    self.conflict_score.fill_(float(value))
            else:
                self.conflict_score = checkpoint['conflict_score']
        if '_last_sleep_step' in checkpoint:
            self._last_sleep_step = checkpoint['_last_sleep_step']
        if '_explore_start_step' in checkpoint:
            self._explore_start_step = checkpoint['_explore_start_step']
        if 'mu_y' in checkpoint and checkpoint['mu_y'] is not None and self.mu_y is not None:
            self.mu_y.copy_(checkpoint['mu_y'].to(self.mu_y.device))
        if 'mu_y_prev' in checkpoint and checkpoint['mu_y_prev'] is not None and self.mu_y_prev is not None:
            self.mu_y_prev.copy_(checkpoint['mu_y_prev'].to(self.mu_y_prev.device))
        if 'omega_cy' in checkpoint and checkpoint['omega_cy'] is not None and self.omega_cy is not None:
            self.omega_cy.copy_(checkpoint['omega_cy'].to(self.omega_cy.device))
        if 'dynamic_lr_cy' in checkpoint and checkpoint['dynamic_lr_cy'] is not None and self.dynamic_lr_cy is not None:
            self.dynamic_lr_cy.copy_(checkpoint['dynamic_lr_cy'].to(self.dynamic_lr_cy.device))
        if 'lowrank_B_basis' in checkpoint and checkpoint['lowrank_B_basis'] is not None and self.lowrank_precision is not None:
            self.lowrank_precision.B_basis.copy_(checkpoint['lowrank_B_basis'].to(self.lowrank_precision.B_basis.device))
        if 'lowrank_C_h' in checkpoint and checkpoint['lowrank_C_h'] is not None and self.lowrank_precision is not None:
            self.lowrank_precision.C_h = checkpoint['lowrank_C_h'].to(self.lowrank_precision.B_basis.device)
        if 'mu_x' in checkpoint and checkpoint['mu_x'] is not None and self.mu_x is not None:
            self.mu_x.copy_(checkpoint['mu_x'].to(self.mu_x.device))
        if 'C_x' in checkpoint and checkpoint['C_x'] is not None and self.C_x is not None:
            self.C_x.copy_(checkpoint['C_x'].to(self.C_x.device))
        if 'W_whiten' in checkpoint and checkpoint['W_whiten'] is not None and self.W_whiten is not None:
            self.W_whiten.copy_(checkpoint['W_whiten'].to(self.W_whiten.device))
        self._cache_valid = False


# =============================================================================
# 注入辅助函数
# =============================================================================

def inject_aeloru(
    model: nn.Module,
    target_names: list = None,
    cfg: Optional[AeloruConfig] = None,
    r: int = 8,
    alpha: float = 4.0,
    register_hook: bool = True
) -> nn.Module:
    """
    递归地将模型中的指定线性层替换为 Aeloru 适配器。
    默认只注入 q_proj 和 v_proj，保留 Transformers 的 QKV 融合和 FlashAttention。
    """
    if target_names is None:
        target_names = ["q_proj", "v_proj"]

    if cfg is None:
        cfg = AeloruConfig(r=r, lora_alpha=alpha)

    for name, module in list(model.named_children()):
        if isinstance(module, AeloruLayer):
            continue

        if hasattr(module, "children") and len(list(module.children())) > 0:
            inject_aeloru(module, target_names, cfg, register_hook=False)

        if isinstance(module, nn.Linear) and any(target in name for target in target_names):
            aeloru_layer = AeloruLayer(
                module.in_features,
                module.out_features,
                cfg,
                original_linear=module
            )
            aeloru_layer.set_pretrained_weight(
                module.weight.data,
                getattr(module, 'bias', None)
            )

            if aeloru_layer.original_weight is None:
                raise RuntimeError(f"权重注入失败: {name}")

            setattr(model, name, aeloru_layer)
            if cfg.logging_enabled:
                print(f"  [Aeloru] Injected into {name} | shape: {module.in_features}x{module.out_features}")

    batch_layers = [
        layer for layer in model.modules()
        if isinstance(layer, AeloruLayer)
    ]

    if register_hook and batch_layers:
        def batch_forward_hook(module, inp, output):
            try:
                input_tensor = inp[0] if inp else output
                batch_results = batch_forward_lora(input_tensor, batch_layers)
                for layer in batch_layers:
                    layer._batch_result = batch_results
            except Exception as e:
                if any(getattr(layer, 'cfg', None) and getattr(layer.cfg, 'logging_enabled', False) for layer in batch_layers):
                    print(f"[ERROR] Batch hook failed: {e}")
            return output

        model.register_forward_hook(batch_forward_hook)

    return model


# =============================================================================
# DLAM 模型级跨层绑定协调器
# =============================================================================

def cross_modal_sleep_step(model: nn.Module) -> None:
    """模型级别的 DLAM 睡眠步骤：协调跨层绑定。"""
    layers = [m for m in model.modules() if isinstance(m, AeloruLayer)]
    if not layers:
        return

    for i in range(len(layers)):
        if not getattr(layers[i], '_pending_cross_modal', False):
            continue
        for j in range(i + 1, len(layers)):
            layers[i]._cross_modal_binding([layers[j]])

    for layer in layers:
        layer._pending_cross_modal = False


class AeloruModel(nn.Module):
    """Aeloru 模型级包装器：方便在训练循环中统一调用 DLAM 跨层绑定。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def cross_modal_sleep_step(self) -> None:
        """调用模型级跨层绑定协调器。"""
        cross_modal_sleep_step(self.model)


# =============================================================================
# 训练封装
# =============================================================================

def train_aeloru_step(
    layer: AeloruLayer,
    x: torch.Tensor,
    y_target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(),
    reward_signal: Optional[bool] = None,
    clip_grad: float = 1.0
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    单步 Aeloru 训练封装 v3.1。
    时序：forward -> backward -> optimizer.step -> GHL 方向观测 -> post_step_update
    """
    # 1. 前向传播
    y_pred = layer(x)
    loss_task = loss_fn(y_pred.to(torch.float16), y_target.to(torch.float16))
    loss_ortho = layer.get_ortho_penalty()
    loss_total = loss_task + loss_ortho

    merged = False
    grad_norm = 0.0

    # 2. 反向传播
    optimizer.zero_grad()
    loss_total.backward()

    grad_norm = math.sqrt(sum(p.grad.norm(2).item()**2
                              for p in layer.parameters() if p.grad is not None))

    if clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(layer.parameters(), clip_grad)

    optimizer.step()

    # 3. 检查 ReLoRA 合并
    if layer.should_merge():
        layer.merge_and_reset()
        merged = True
        layer._pending_optimizer_reset = True

    # 4. GHL 全局方向信号（只用损失方向，不传梯度幅值，开销近乎为零）
    layer.observe_global_loss(loss_task.item())

    # 5. Hebbian 更新
    layer.post_step_update(x, y_pred.detach(), is_correct=reward_signal,
                           merged=merged, y_target=y_target)

    # 6. 清理优化器状态
    if getattr(layer, '_pending_optimizer_reset', False):
        layer.clear_optimizer_state(optimizer)
        layer._pending_optimizer_reset = False

    # 7. 梯度收敛检测
    converged = layer.check_anchor_convergence(grad_norm)

    # 8. 收集指标
    metrics = {
        'state': layer.state.value,
        'loss_task': loss_task.item(),
        'loss_ortho': loss_ortho.item(),
        'loss_structure': layer._last_structure_loss,
        'loss_total': loss_total.item(),
        'grad_norm': grad_norm,
        'anchor_converged': converged,
        'relora_merged': merged,
        'hebbian_pending': layer._hebbian_pending_apply,
        'plasticity_out_mean': layer.plasticity_out.mean().item(),
        'plasticity_in_mean': layer.plasticity_in.mean().item(),
        'global_direction': layer.global_direction.item(),
    }

    return loss_total, metrics
