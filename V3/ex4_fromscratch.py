# -*- coding: utf-8 -*-
"""
ex4_fromscratch.py -- 从零训练以测试 Aeloru v3.1 各模组性能（重点考察 PEM）
============================================================================

背景
----
ex1 / ex2 / ex3 全部在**预训练 Qwen2.5-0.5B** 上注入 AeloruLayer 并冻结基座。
论文 paper.md §5.10 / §3.6 明确指出：在 LLM 微调场景下 PEM 是**温和的负贡献**，
应默认关闭；PEM 的真正定位是 **从零训练的大维度认知模型**（`cognitive_model()` 预设，
`pem_space="output"`）。本实验补全这一空白——**从零初始化**一个小型网络，
把 AeloruLayer 当作隐藏层（W0 为 Kaiming 基底，其更新完全由 Aeloru 的周期性 ReLoRA 合并负责、不接收 BP 梯度；LoRA 路径为唯一可训练部分），在带冗余耦合的合成
分类任务上从零训练，逐项考察各模组（尤其是 PEM）对**学习本身**的贡献。

与 ex3_synth 的区别
-------------------
ex3_synth 只用单层 h 通路做开环校准（W0=0、冻结 A）。本实验是一个**完整的、
端到端可微、有反向传播修复**的从零网络（多层 MLP）：
  * PEM 走低秩空间 `pem_space="lowrank"`（ex3 已证明这是从零训练的**稳定**路径；
    本实验保持该选择以维持与 ex3 的可比性，不启用全输出空间 PEM）。
  * 其余模组（WTA / SPHeRe / 正交惩罚 / 双一维 HGF 门控 / Hebbian / 输入白化 /
    源域约束）全部按 `cognitive_model()` 预设接入，并逐个消融。
  * 因为存在 BP，本实验能检验 ex3 的判据 O1/O2/O3 在**闭环**下是否成立
    （ex3 结论：选择性只在闭环浮现）。

任务（合成，可控冗余）
----------------------
输入 x ∈ R^d 由三类潜变量拼接：
  * 信息维度 u_info  (标签 y 仅依赖它)
  * 冗余维度 u_red   = rho*u_info + sqrt(1-rho^2)*noise   (与信息维度强相关的"说同一件事"的副本)
  * 噪声维度 u_noise (纯噪声)
冗余维度制造了被耦合扭曲的输入流形；白化 / 解耦 / PEM 等模组应当帮助网络
忽略冗余、更快学到信息维度，并在隐藏表示上得到更解耦、更可解码的表征。

方法（消融设计，聚焦 PEM）
-------------------------
  base      全开（PEM lowrank ndi=2，WTA，SPHeRe，正交，门控，Hebbian，白化，域约束）
  nopem     关闭 PEM（neural_dynamics_iterations=0）
  nowta     关闭 WTA 硬竞争（wta_k_ratio=1.0）
  nosphere  关闭 SPHeRe 结构投影（sphere_lambda=0）
  noortho   关闭 ΔW⊥W0 正交约束（ortho_lambda=0）
  nogate    关闭双一维 HGF 门控（fisher_gamma=0, prototype_guard=0）
  nohebb    关闭 Hebbian 探索（hebbian_lr=0）
  nowhiten  关闭输入白化（whitening_interval=1e9）
  nodomain  关闭源域约束（presumed_domain="none"）
  plain     全关（纯 LoRA-from-scratch 基线）
  vanilla   普通 nn.Linear + GELU MLP（非 Aeloru 参照）
另含 PEM 剂量—反应扫描：ndi ∈ {0,1,2,4,8}（在 base 其余设定下）。

度量
----
  * 最终测试准确率（多 seed 均值 ± 标准差）
  * 收敛速度：验证准确率首次达到 0.9×最终值 的步数
  * 隐藏表示解耦（ex2 指标）：corr_abs / TC / MI / eff_rank / PR / iso_err / cond / sparsity
  * 解纠缠可读性：隐藏特征对各潜变量的岭 R^2（r2_info / r2_red / r2_noise / r2_gap）
  * 机制轨迹（采样）：plasticity / global_direction / structure_loss / conflict / volatility / 状态分布

用法
----
  python ex4_fromscratch.py                         # 默认 9 个 Aeloru 配置 x 3 seed
  python ex4_fromscratch.py --dose                  # 额外跑 PEM 剂量扫描
  python ex4_fromscratch.py --quick                 # 1 配置 1 seed 冒烟
  python ex4_fromscratch.py --include-vanilla      # 额外加普通 MLP 参照

@author JYIMU
"""

import os
import sys
import math
import time
import json
import argparse
import contextlib
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
torch_backends_msg = ""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aeloru_v3 import AeloruConfig, AeloruLayer  # noqa: E402

# ============================================================================
# 常量 / 默认配置
# ============================================================================

# Aeloru 隐藏层共享的「从零认知模型」基底（PEM 走低秩空间，稳定）。
# homeostatic 在 2 层小网络上易把 RMS 拉垮，按 ex1/ex2 的安全结论关闭。
BASE_CFG_FIELDS = dict(
    pem_space="output",
    neural_dynamics_iterations=2,
    presumed_domain="nnantisparse",
    homeostatic_enabled=False,
    homeostatic_max_gain=2.0,
    whitening_interval=500,
    wta_k_ratio=0.5,
    sphere_lambda=0.01,
    ortho_lambda=0.01,
    ortho_lambda_anchor=0.05,
    fisher_gamma=0.5,
    prototype_guard=0.5,
    hebbian_lr=1e-6,
    lowrank_whiten_coef=0.1,
    lowrank_cond_max=5.0,
    lambda_lateral=0.9,
    lateral_lr=1e-5,
    lateral_decay=0.99,
    merge_every=10 ** 9,        # 从零训练不做 ReLoRA 合并，避免扰动优化器状态
    min_steps_between_sleep=10 ** 9,  # 关闭 DLAM 睡眠（非重点模组，避免额外不稳定性）
    sleep_condition_threshold=10 ** 9,
    energy_eta=0.15,
    predictive_coding_rank=16,
    gamma_predictive=100.0,
    logging_enabled=False,
    diagnostic_enabled=False,
    AMP_DTYPE=torch.float32,    # 从零训练强制 float32，避免 bf16/fp16 不稳定
)

# 消融：相对 base 的覆盖项（聚焦各模组）
ABLATIONS: Dict[str, dict] = {
    "base":     {},
    "nopem":    {"neural_dynamics_iterations": 0},
    "nowta":    {"wta_k_ratio": 1.0},
    "nosphere": {"sphere_lambda": 0.0},
    "noortho":  {"ortho_lambda": 0.0, "ortho_lambda_anchor": 0.0},
    "nogate":   {"fisher_gamma": 0.0, "prototype_guard": 0.0},
    "nohebb":   {"hebbian_lr": 0.0},
    "nowhiten": {"whitening_interval": 10 ** 9},
    "nodomain": {"presumed_domain": "none"},
    "plain":    {  # 全关：纯 LoRA-from-scratch 基线
        "neural_dynamics_iterations": 0,
        "wta_k_ratio": 1.0,
        "sphere_lambda": 0.0,
        "ortho_lambda": 0.0,
        "ortho_lambda_anchor": 0.0,
        "fisher_gamma": 0.0,
        "prototype_guard": 0.0,
        "hebbian_lr": 0.0,
        "whitening_interval": 10 ** 9,
        "presumed_domain": "none",
    },
}
DEFAULT_CONFIGS = ["base", "nopem", "nowta", "nosphere",
                   "noortho", "nogate", "nohebb", "nowhiten", "plain"]

DECOL_METRICS = [
    ("corr_abs", False), ("corr_max", False), ("TC", False), ("MI", False),
    ("eff_rank", True), ("PR", True), ("iso_err", False), ("cond", False),
    ("sparsity", None),
]


# ============================================================================
# 工具
# ============================================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(msg: str, fh=None) -> None:
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()


def make_cfg(overrides: Optional[dict], device: str, r: int, alpha: float,
             lr: float) -> AeloruConfig:
    """构造 Aeloru 配置：基底 + 消融覆盖 + 维度/设备相关项。"""
    fields = dict(BASE_CFG_FIELDS)
    fields.update(overrides or {})
    fields.update(dict(r=r, lora_alpha=alpha, LoRA_lr=lr, device=device))
    return AeloruConfig(**fields)


# ============================================================================
# 合成数据：带冗余耦合的分类任务
# ============================================================================

def make_data(n_info: int, n_red: int, n_noise: int, rho: float,
              n_train: int, n_val: int, seed: int):
    """
    生成 x = [u_info, u_red, u_noise]，标签仅依赖 u_info。

    u_red 与 u_info 强相关（冗余副本），制造扭曲输入流形。
    返回 (Xtr, ytr, Xte, yte, Utr, Ute)，U 是 (u_info, u_red, u_noise) 拼接的潜变量，
    供解纠缠可读性 R^2 使用。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = n_info + n_red + n_noise

    def sample(n: int):
        u_info = torch.randn(n, n_info, generator=g)
        u_red = rho * u_info + math.sqrt(max(1.0 - rho ** 2, 1e-12)) * torch.randn(n, n_red, generator=g)
        u_noise = torch.randn(n, n_noise, generator=g)
        x = torch.cat([u_info, u_red, u_noise], dim=1)
        # 标签依赖 u_info 的随机线性组合
        a = torch.randn(n_info, generator=g)
        logit = u_info @ a + 0.3 * torch.randn(n, generator=g)
        y = (logit > 0).long()
        u = torch.cat([u_info, u_red, u_noise], dim=1)
        return x, y, u

    Xtr, ytr, Utr = sample(n_train)
    Xte, yte, Ute = sample(n_val)
    to_f = lambda t: t.to(torch.float32)
    return to_f(Xtr), ytr, to_f(Xte), yte, to_f(Utr), to_f(Ute)


# ============================================================================
# 模型
# ============================================================================

class FromScratchNet(nn.Module):
    """
    从零训练网络：若干 AeloruLayer 隐藏层 + 普通线性输出头。

    隐藏层 = AeloruLayer，其 W0 为**ReLoRA 基底**（Kaiming 正态初始化后转 nn.Parameter，
    但 requires_grad=False，不接收 BP 梯度）；W0 的全部更新由 Aeloru 的周期性 ReLoRA 合并
    负责（把 LoRA 学到的 ΔW 折入 W0 并重置适配器，见 relora_merge_into_W0），LoRA 路径
    （lora_A/lora_B）是唯一接收 BP 梯度的部分。AeloruLayer 内部已含源域约束作为
    非线性（base 配置下），wrapper 再统一接一个 GELU 保证表达力与 base 一致
    （域约束关闭的配置因此仍具可比非线性）。
    """

    def __init__(self, dims: List[int], cfg: AeloruConfig, device: str):
        super().__init__()
        self.dims = dims
        self.aeloru_layers: List[AeloruLayer] = nn.ModuleList()
        self.domain_on = (cfg.presumed_domain != "none")
        for i in range(len(dims) - 2):  # 最后一层是输出头
            lay = AeloruLayer(dims[i], dims[i + 1], cfg)
            # W0 作为「ReLoRA 基底」：Kaiming 正态初始化后转 nn.Parameter（但 requires_grad=False）。
            # W0 不接收 BP 梯度，其更新完全由 Aeloru 的周期性 ReLoRA 合并负责（见 relora_merge_into_W0）：
            # 每 --relora-every 步把 LoRA 学到的 ΔW 折入 W0 并重置适配器，W0 因此「不断更新」。
            new_W0 = torch.empty_like(lay.W0)
            nn.init.kaiming_normal_(new_W0, nonlinearity="relu")
            lay.W0 = nn.Parameter(new_W0)
            lay.W0.requires_grad_(False)
            with torch.no_grad():
                lay.bias.zero_()
                lay.W0_norm_sq = torch.tensor(float((lay.W0.detach().float() ** 2).sum()),
                                              dtype=torch.float32)
                lay.m_x.fill_(1.0)
                lay.m_y.fill_(1.0)
            self.aeloru_layers.append(lay)
        self.head = nn.Linear(dims[-2], dims[-1])
        self.to(device)

    def forward(self, x):
        for lay in self.aeloru_layers:
            x = lay(x)
            x = F.gelu(x)  # 统一非线性；base 配置下 AeloruLayer 内部还有域约束 clamp
        return self.head(x)


class VanillaNet(nn.Module):
    """普通 MLP 参照（无 Aeloru 模组）。"""

    def __init__(self, dims: List[int], device: str):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.to(device)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.gelu(x)
        return x


# ============================================================================
# 激活捕获（镜像 ex1 的 ActivationCapture，折叠为 2D）
# ============================================================================

class ActivationCapture:
    def __init__(self, model: nn.Module):
        self.records: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.handles = []
        for layer in model.modules():
            if isinstance(layer, AeloruLayer):
                self.handles.append(layer.register_forward_hook(self._make(layer)))

    def _make(self, layer):
        def hook(mod, inp, out):
            x = inp[0].detach()
            y = out.detach()
            self.records[id(mod)] = (x.reshape(-1, x.size(-1)), y.reshape(-1, y.size(-1)))
        return hook

    def get(self, layer):
        return self.records.get(id(layer))

    def clear(self):
        self.records.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


@contextlib.contextmanager
def preserve_dynamics(layers: List[AeloruLayer]):
    """保护会被『额外一次前向』改写的在线统计量（PEM 的 mu_h/C_hh、稳态 running_var）。"""
    saved = []
    for lay in layers:
        saved.append((lay,
                      lay.mu_h.detach().clone(),
                      lay.C_hh.detach().clone(),
                      lay.running_var.detach().clone()))
    try:
        yield
    finally:
        for lay, mu_h, c_hh, rvar in saved:
            try:
                lay.mu_h.copy_(mu_h)
                lay.C_hh.copy_(c_hh)
                lay.running_var.copy_(rvar)
            except Exception:
                pass


# ============================================================================
# 解耦 / 解纠缠度量（复用 ex2 / ex3 的公式，自包含实现）
# ============================================================================

def _recondition_layer(lay):
    """给 Aeloru 层的协方差/白化缓冲加一点相对对角抖动，缓解 eigh/slogdet 不收敛。

    当隐藏激活退化（秩亏）时，这些经验协方差矩阵会变得病态，库内 spectral_filter
    的 torch.linalg.eigh 会抛 _LinAlgError。原地加抖动可让其重新条件化、收敛。
    """
    for name in ("C_h", "C_hh", "C_x", "W_whiten"):
        buf = getattr(lay, name, None)
        if isinstance(buf, torch.Tensor) and buf.dim() == 2 and buf.shape[0] == buf.shape[1]:
            j = 1e-2 * buf.diagonal().abs().max().clamp(min=1e-6)
            buf.add_(j * torch.eye(buf.shape[0], device=buf.device, dtype=buf.dtype))


def decoupling_metrics(H: torch.Tensor) -> dict:
    """在 r 维隐藏特征 h 上计算一整套解耦指标。"""
    H = H.float()
    if not torch.isfinite(H).all():
        # 隐藏激活含非有限值（该配置训练数值发散），无法计算解耦指标，跳过。
        return {}
    N, r = H.shape
    if N < 4 or r < 2:
        return {}
    Hc = H - H.mean(dim=0, keepdim=True)
    C = (Hc.t() @ Hc) / max(N - 1, 1)
    tr = torch.diagonal(C).sum().clamp(min=1e-12)
    # 相对对角抖动：避免协方差病态导致后续分解不收敛
    jitter = 1e-4 * torch.diagonal(C).abs().max().clamp(min=1e-8)
    C = C + jitter * torch.eye(r, device=C.device, dtype=C.dtype)
    d = torch.diagonal(C).clamp(min=1e-12)
    d_inv_sqrt = torch.diag(1.0 / torch.sqrt(d))
    R = d_inv_sqrt @ C @ d_inv_sqrt
    I = torch.eye(r, device=C.device, dtype=C.dtype)
    off = R - I
    n_off = max(r * (r - 1), 1)
    corr_abs = float(off.abs().sum() / n_off)
    corr_max = float(off.abs().max())

    try:
        sign, logdet = torch.linalg.slogdet(C)
        if float(sign) <= 0 or not math.isfinite(float(logdet)):
            logdet = torch.log(d).sum()
    except RuntimeError:
        logdet = torch.log(d).sum()
    tc = max(0.0, float(0.5 * (torch.log(d).sum() - logdet)))

    rho2 = (R ** 2).clamp(max=1.0 - 1e-7) * (1.0 - I)
    mi = float((-0.5 * torch.log1p(-rho2)).sum() / n_off)

    # 特征值：用 SVD 直接求协方差特征值——比 eigvalsh 在病态/退化矩阵上稳定得多
    # （C 的特征值 == Hc 奇异值的平方 /(N-1)）。带 eigvalsh 兜底以防极旧 torch 版本。
    try:
        with torch.no_grad():
            s = torch.linalg.svdvals(Hc.double())
        ev = (s ** 2) / max(N - 1, 1)
        ev = ev.clamp(min=1e-12)
    except RuntimeError:
        ev = torch.linalg.eigvalsh(C).clamp(min=1e-12)
    p = ev / ev.sum()
    eff_rank = float(torch.exp(-(p * torch.log(p)).sum()))
    pr = float((ev.sum() ** 2) / (ev ** 2).sum())
    cond = float(ev.max() / ev.min())

    C_norm = C * (r / float(tr))
    iso_err = float((C_norm - I).pow(2).sum().sqrt())

    peak = H.abs().max().clamp(min=1e-12)
    sparsity = float((H.abs() < 0.05 * peak).float().mean())

    return {
        "corr_abs": corr_abs, "corr_max": corr_max, "TC": tc, "MI": mi,
        "eff_rank": eff_rank, "PR": pr, "cond": cond, "iso_err": iso_err,
        "sparsity": sparsity, "h_std": float(H.std()),
    }


def ridge_r2(Htr, Utr, Hte, Ute, lam: float = 1e-3):
    """隐藏特征 H 对各潜变量 U 的线性可解码程度 R^2（逐维）。"""
    Htr, Utr, Hte, Ute = Htr.double(), Utr.double(), Hte.double(), Ute.double()
    r = Htr.shape[1]
    HtH = Htr.t() @ Htr
    HtU = Htr.t() @ Utr
    try:
        A = torch.linalg.solve(HtH + lam * torch.eye(r, dtype=Htr.dtype, device=Htr.device), HtU)
    except RuntimeError:
        # 特征矩阵极病态时退回最小二乘解
        A = torch.linalg.lstsq(Htr, Utr).solution
    pred = Hte @ A
    ss_res = ((Ute - pred) ** 2).sum(0)
    ss_tot = ((Ute - Ute.mean(0)) ** 2).sum(0).clamp(min=1e-12)
    return (1.0 - ss_res / ss_tot).clamp(min=-1.0, max=1.0)


@torch.no_grad()
def collect_hidden(model: FromScratchNet, loader, device: str,
                   max_rows: int = 4096) -> Optional[torch.Tensor]:
    """
    采集最后隐藏层（GELU 之前的 AeloruLayer 输出）的特征，train 模式以保证
    PEM / WTA 生效（与 ex2 h-mode=train 同理）。在线统计量由 preserve_dynamics 还原。
    """
    last = model.aeloru_layers[-1]
    buf: List[torch.Tensor] = []
    was = model.training
    model.train()
    handle = last.register_forward_hook(
        lambda mod, inp, out: buf.append(out.detach().reshape(-1, out.size(-1)).cpu().float()))
    try:
        with preserve_dynamics(model.aeloru_layers):
            for batch in loader:
                x = batch[0].to(device)
                model(x)
                if sum(b.shape[0] for b in buf) >= max_rows:
                    break
    finally:
        handle.remove()
        model.train(was)
    if not buf:
        return None
    H = torch.cat(buf, dim=0)
    if H.shape[0] > max_rows:
        idx = torch.randperm(H.shape[0])[:max_rows]
        H = H[idx]
    return H


# ============================================================================
# 训练引擎
# ============================================================================

def evaluate(model, X, y, device: str, bs: int = 256) -> float:
    model.eval()
    correct = total = 0
    n = X.shape[0]
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = X[i:i + bs].to(device)
            yb = y[i:i + bs].to(device)
            pred = model(xb).argmax(-1)
            correct += (pred == yb).sum().item()
            total += yb.shape[0]
    model.train()
    return correct / max(total, 1)


def relora_merge_into_W0(layer: "AeloruLayer", opt) -> None:
    """ReLoRA 合并：把当前 LoRA 增量 ΔW 折入可训练 W0，再重置适配器并清优化器动量。

    与库内 merge_and_reset 的唯一区别：合并目标从冻结的 _W_acc 改为 W0。
    W0 不接收 BP 梯度，其全部更新都来自这里（Aeloru 的周期性合并）——即用户要求的
    「W0 只走周期性合并、更新全由 Aeloru 负责」。
    """
    with torch.no_grad():
        layer._flush_hebbian()                          # 把累积的 Hebbian 更新并入 A/B
        delta_w = layer.compute_delta_w()               # (alpha/r) * B @ A
        delta_w = layer.apply_hidora(delta_w)           # Hi-DoRA m_x/m_y 幅度调制
        delta_w = layer.apply_plasticity_gate(delta_w)  # 双一维 HGF 门控
        delta_w = layer.apply_energy_budget(delta_w)    # 能量预算缩放
        layer.W0.data.add_(delta_w.detach().to(layer.W0.dtype))   # 合并进 W0
        layer._reset_adapters()                         # 重新随机初始化 A、清零 B
        layer.W0_norm_sq = torch.tensor(
            float((layer.W0.detach().float() ** 2).sum()), dtype=torch.float32)
    layer.step_counter = 0
    layer._explore_start_step = 0
    # 清掉被重置参数的优化器动量（ReLoRA 标准做法）；W0 动量一并清除以重新累积
    layer.clear_optimizer_state(opt)
    if layer.W0 in opt.state:
        del opt.state[layer.W0]


def train_one(config_name: str, cfg: Optional[AeloruConfig], dims: List[int],
              data: Tuple, device: str, args, fh) -> dict:
    """
    从零训练单个配置。返回该配置（单 seed）的结果字典。
    cfg=None 时表示 vanilla 普通 MLP。
    """
    Xtr, ytr, Xte, yte, Utr, Ute = data
    is_vanilla = (cfg is None)

    if is_vanilla:
        model = VanillaNet(dims, device)
        aeloru_layers: List[AeloruLayer] = []
    else:
        model = FromScratchNet(dims, cfg, device)
        aeloru_layers = list(model.aeloru_layers)

    capture = ActivationCapture(model) if aeloru_layers else None
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    lossf = nn.CrossEntropyLoss()

    n, bs = Xtr.shape[0], args.bs
    losses, val_accs, mech_log = [], [], []
    ortho_trace = []
    post_err = None
    post_disabled = False
    t_start = time.perf_counter()
    best_val = 0.0
    converge_step = -1

    model.train()
    for step in range(args.steps):
        idx = torch.randint(0, n, (bs,), device=device)
        xb, yb = Xtr[idx].to(device), ytr[idx].to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss_task = lossf(logits.float(), yb)
        loss = loss_task

        # 结构性正则：get_ortho_penalty 内部已含 ortho_lambda 与 sphere_lambda 缩放
        loss_ortho = 0.0
        if aeloru_layers:
            for lay in aeloru_layers:
                loss_ortho = loss_ortho + lay.get_ortho_penalty()
            loss = loss + loss_ortho

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        lv = loss.item()
        losses.append(lv)
        lo = float(loss_ortho.item()) if torch.is_tensor(loss_ortho) else 0.0
        ortho_trace.append(lo)

        # ---- Aeloru 专有后处理 ----
        if aeloru_layers and not post_disabled:
            pred = logits.detach().argmax(-1)
            is_correct = bool((pred == yb).float().mean().item() > 0.5)
            try:
                for lay in aeloru_layers:
                    lay.observe_global_loss(lv)
                    rec = capture.get(lay)
                    if rec is None:
                        continue
                    try:
                        lay.post_step_update(rec[0], rec[1], is_correct=is_correct)
                    except RuntimeError:
                        # 协方差矩阵病态（eigh/slogdet 不收敛，_LinAlgError 继承 RuntimeError）：
                        # 给相关缓冲加对角抖动以重新条件化，并重试一次；仍失败则由外层禁用。
                        _recondition_layer(lay)
                        lay.post_step_update(rec[0], rec[1], is_correct=is_correct)
            except Exception as e:  # 保护长实验不被单步内部错误中断
                if post_err is None:
                    post_err = f"{type(e).__name__}: {e}"
                    log(f"  [warn] {config_name} post_step_update 失败，已禁用后续调用: {post_err}", fh)
                post_disabled = True

        # ---- ReLoRA：周期性把 LoRA 增量合并进 W0（W0 唯一更新途径）并重置适配器 ----
        if aeloru_layers and args.relora_every and (step + 1) % args.relora_every == 0:
            for lay in aeloru_layers:
                relora_merge_into_W0(lay, opt)

        # ---- 验证 / 收敛 ----
        if step % args.eval_every == 0 or step == args.steps - 1:
            va = evaluate(model, Xte, yte, device, bs=256)
            val_accs.append((step, va))
            best_val = max(best_val, va)
            if converge_step < 0 and va >= 0.9 * best_val and step > 0:
                converge_step = step

        # ---- 机制轨迹采样 ----
        if aeloru_layers and (step % args.sample_every == 0 or step == args.steps - 1):
            row = {
                "step": step, "loss": lv, "loss_ortho": lo,
                "plasticity_out": float(torch.stack([l.plasticity_out.mean() for l in aeloru_layers]).mean()),
                "plasticity_in": float(torch.stack([l.plasticity_in.mean() for l in aeloru_layers]).mean()),
                "global_direction": float(torch.stack([l.global_direction for l in aeloru_layers]).mean()),
                "structure_loss": float(np.mean([l._last_structure_loss for l in aeloru_layers])),
                "conflict": float(torch.stack([l.conflict_score if isinstance(l.conflict_score, torch.Tensor) else torch.tensor(l.conflict_score) for l in aeloru_layers]).mean()),
                "volatility": float(torch.exp(torch.stack([l.omega for l in aeloru_layers]).float()).mean()),
                "state": aeloru_layers[0].state.value if aeloru_layers else "",
            }
            mech_log.append(row)

    if capture is not None:
        capture.remove()

    final_val = evaluate(model, Xte, yte, device, bs=256)
    final_train = evaluate(model, Xtr, ytr, device, bs=256)

    # ---- 解耦 + 解纠缠度量（最后隐藏层特征，train 模式）----
    dec, r2 = {}, {}
    if aeloru_layers:
        # 构造一个能整批过模型的 loader（仅取 X）
        class _XLoader:
            def __init__(self, X, bs):
                self.X, self.bs = X, bs
            def __iter__(self):
                for i in range(0, self.X.shape[0], self.bs):
                    yield (self.X[i:i + self.bs],)
        H = collect_hidden(model, _XLoader(Xte, 256), device, max_rows=args.h_samples)
        if H is not None and H.shape[0] >= 4 and torch.isfinite(H).all():
            dec = decoupling_metrics(H)
            # 解码到潜变量（仅用测试集的 H 与 U）
            r2_all = ridge_r2(H, Ute[:H.shape[0]].cpu(), H, Ute[:H.shape[0]].cpu())
            n_info, n_red, n_noise = dims_info(args)
            r2["r2_info"] = float(r2_all[:n_info].mean())
            r2["r2_red"] = float(r2_all[n_info:n_info + n_red].mean())
            r2["r2_noise"] = float(r2_all[n_info + n_red:].mean())
            r2["r2_gap"] = r2["r2_info"] - r2["r2_red"]
        elif H is not None and H.shape[0] >= 4:
            log(f"  [warn] {config_name} 隐藏激活含非有限值（训练发散），跳过解耦/解码度量", fh)

    elapsed = time.perf_counter() - t_start

    result = {
        "config": config_name,
        "final_val": final_val,
        "final_train": final_train,
        "best_val": best_val,
        "converge_step": converge_step,
        "elapsed": elapsed,
        "losses": losses,
        "val_accs": val_accs,
        "mech": mech_log,
        "dec": dec,
        "r2": r2,
        "post_err": post_err,
        "n_layers": len(aeloru_layers),
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    return result


def dims_info(args) -> Tuple[int, int, int]:
    return args.n_info, args.n_red, args.n_noise


# ============================================================================
# 汇总与可视化
# ============================================================================

def make_figures(results_by_cfg: Dict[str, List[dict]], args, out_dir: str) -> List[str]:
    cfgs = list(results_by_cfg.keys())
    pngs = []

    # (1) 损失曲线（取每配置多 seed 的均值 ± 标准差，按 eval 步对齐不便，这里画末 200 步均值滑动）
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    # 损失曲线：把各 seed 的损失按 step 对齐到统一长度
    maxlen = max(len(r["losses"]) for rs in results_by_cfg.values() for r in rs)
    for c in cfgs:
        arrs = [np.array(r["losses"], dtype=float) for r in results_by_cfg[c]]
        # 重采样到 maxlen
        mat = np.stack([np.interp(np.linspace(0, len(a) - 1, maxlen), np.arange(len(a)), a)
                        for a in arrs], axis=0)
        mean = mat.mean(0); std = mat.std(0)
        x = np.linspace(0, 1, maxlen)
        ax[0].plot(x, mean, label=c)
        ax[0].fill_between(x, mean - std, mean + std, alpha=0.15)
    ax[0].set_title("Training loss (normalized steps)")
    ax[0].set_xlabel("step / total"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)

    # (2) 最终准确率柱状
    means = [np.mean([r["final_val"] for r in results_by_cfg[c]]) for c in cfgs]
    stds = [np.std([r["final_val"] for r in results_by_cfg[c]]) for c in cfgs]
    ax[1].bar(range(len(cfgs)), means, yerr=stds, capsize=3)
    ax[1].set_xticks(range(len(cfgs))); ax[1].set_xticklabels(cfgs, rotation=45, ha="right", fontsize=8)
    ax[1].set_ylabel("final test accuracy"); ax[1].set_title("Final accuracy per config")
    ax[1].grid(alpha=.3, axis="y")

    # (3) 收敛速度柱状
    conv = [np.mean([r["converge_step"] for r in results_by_cfg[c] if r["converge_step"] >= 0])
            for c in cfgs]
    ax[2].bar(range(len(cfgs)), conv)
    ax[2].set_xticks(range(len(cfgs))); ax[2].set_xticklabels(cfgs, rotation=45, ha="right", fontsize=8)
    ax[2].set_ylabel("step to 90% of best acc"); ax[2].set_title("Convergence speed (lower=better)")
    ax[2].grid(alpha=.3, axis="y")
    plt.tight_layout()
    p = os.path.join(out_dir, "overview.png")
    fig.savefig(p, dpi=150); plt.close(fig); pngs.append(p)

    # (4) 解耦指标对比
    dec_keys = [k for k, _ in DECOL_METRICS]
    fig, ax = plt.subplots(1, len(dec_keys), figsize=(3 * len(dec_keys), 4))
    for j, (k, bigger) in enumerate(DECOL_METRICS):
        vals = []
        for c in cfgs:
            vs = [r["dec"].get(k, float("nan")) for r in results_by_cfg[c] if r["dec"]]
            vals.append(float(np.nanmean(vs)) if vs else float("nan"))
        ax[j].bar(range(len(cfgs)), vals)
        ax[j].set_title(k, fontsize=10)
        ax[j].set_xticks(range(len(cfgs)))
        ax[j].set_xticklabels(cfgs, rotation=60, ha="right", fontsize=6)
        ax[j].grid(alpha=.3, axis="y")
    fig.suptitle("Hidden-feature decoupling metrics (per config)", fontsize=12)
    plt.tight_layout()
    p = os.path.join(out_dir, "decoupling.png")
    fig.savefig(p, dpi=150); plt.close(fig); pngs.append(p)

    # (5) 解纠缠可读性 R2
    r2_keys = ["r2_info", "r2_red", "r2_noise", "r2_gap"]
    fig, ax = plt.subplots(1, len(r2_keys), figsize=(3 * len(r2_keys), 4))
    for j, k in enumerate(r2_keys):
        vals = [float(np.nanmean([r["r2"].get(k, float("nan")) for r in results_by_cfg[c] if r["r2"]]))
                for c in cfgs]
        ax[j].bar(range(len(cfgs)), vals)
        ax[j].set_title(k, fontsize=10)
        ax[j].set_xticks(range(len(cfgs)))
        ax[j].set_xticklabels(cfgs, rotation=60, ha="right", fontsize=6)
        ax[j].grid(alpha=.3, axis="y")
    fig.suptitle("Disentanglement readability R^2 of hidden features", fontsize=12)
    plt.tight_layout()
    p = os.path.join(out_dir, "disentangle.png")
    fig.savefig(p, dpi=150); plt.close(fig); pngs.append(p)

    # (6) 机制轨迹（仅 base）
    if "base" in results_by_cfg:
        mechs = [r["mech"] for r in results_by_cfg["base"] if r["mech"]]
        if mechs:
            keys = ["plasticity_out", "plasticity_in", "global_direction",
                    "structure_loss", "conflict", "volatility"]
            fig, axes = plt.subplots(2, 3, figsize=(16, 8))
            for axx, k in zip(axes.ravel(), keys):
                for r in mechs:
                    df = pd.DataFrame(r)
                    axx.plot(df["step"].values, df[k].values, lw=0.8, alpha=0.6)
                axx.set_title(k, fontsize=10); axx.grid(alpha=.3)
                axx.set_xlabel("step")
            fig.suptitle("Mechanism dynamics (base config, train mode)", fontsize=12)
            plt.tight_layout()
            p = os.path.join(out_dir, "mechanisms.png")
            fig.savefig(p, dpi=150); plt.close(fig); pngs.append(p)

    return pngs


def make_dose_figure(dose_results: Dict[int, List[dict]], args, out_dir: str) -> Optional[str]:
    """PEM 剂量—反应：ndi -> 准确率 / 解耦 / 解纠缠。"""
    ndis = sorted(dose_results.keys())
    if len(ndis) < 2:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    def mean_of(n, key, sub):
        rs = dose_results[n]
        return float(np.nanmean([r[sub].get(key, float("nan")) for r in rs if r.get(sub)])) if any(r.get(sub) for r in rs) else float("nan")
    # 准确率
    acc = [np.mean([r["final_val"] for r in dose_results[n]]) for n in ndis]
    axes[0].plot(ndis, acc, "o-"); axes[0].set_title("final accuracy"); axes[0].grid(alpha=.3)
    # corr_abs
    ca = [mean_of(n, "corr_abs", "dec") for n in ndis]
    axes[1].plot(ndis, ca, "o-"); axes[1].set_title("corr_abs (decoupling, lower=better)"); axes[1].grid(alpha=.3)
    # eff_rank
    er = [mean_of(n, "eff_rank", "dec") for n in ndis]
    axes[2].plot(ndis, er, "o-"); axes[2].set_title("eff_rank (higher=better)"); axes[2].grid(alpha=.3)
    # r2_gap
    gap = [mean_of(n, "r2_gap", "r2") for n in ndis]
    axes[3].plot(ndis, gap, "o-"); axes[3].set_title("r2_gap (disentangle, higher=better)"); axes[3].grid(alpha=.3)
    for ax in axes:
        ax.set_xlabel("neural_dynamics_iterations (PEM dose)")
    fig.suptitle("PEM dose-response (from-scratch)", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, "pem_dose.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ============================================================================
# 主流程
# ============================================================================

def run_config(name: str, cfg: Optional[AeloruConfig], dims, data, device, args, fh) -> List[dict]:
    out = []
    for seed in args.seeds:
        set_seed(seed)
        log(f"  [{name}] seed={seed} 训练 {args.steps} 步 ...", fh)
        t0 = time.perf_counter()
        r = train_one(name, cfg, dims, data, device, args, fh)
        r["seed"] = seed
        out.append(r)
        log(f"    -> 测试 acc={r['final_val']:.4f}  训练 acc={r['final_train']:.4f}  "
            f"收敛步={r['converge_step']}  ({time.perf_counter() - t0:.1f}s)"
            + (f"  PEM/corr_abs={r['dec'].get('corr_abs', float('nan')):.4f}"
               if r['dec'] else "")
            + (f"  r2_gap={r['r2'].get('r2_gap', float('nan')):.3f}" if r['r2'] else ""), fh)
    return out


def main():
    ap = argparse.ArgumentParser(description="ex4: 从零训练测试 Aeloru 各模组（聚焦 PEM）")
    ap.add_argument("--configs", type=str, default=",".join(DEFAULT_CONFIGS))
    ap.add_argument("--dose", action="store_true", help="额外跑 PEM 剂量扫描 ndi∈{0,1,2,4,8}")
    ap.add_argument("--include-vanilla", action="store_true", help="额外加普通 MLP 参照")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64, help="隐藏层维度（也是低秩 r）")
    ap.add_argument("--layers", type=int, default=2, help="Aeloru 隐藏层数量")
    ap.add_argument("--r", type=int, default=0, help="低秩维度；0=与隐藏层维度相同(满秩 LoRA 路径)")
    ap.add_argument("--alpha", type=int, default=0, help="LoRA alpha；0=与 r 相同(scale=1)")
    ap.add_argument("--relora-every", type=int, default=1000,
                    help="ReLoRA 合并周期(步)：每 N 步把 LoRA ΔW 折入可训练 W0 并重置适配器；0=不合并(仅 BP 训练 W0)")
    ap.add_argument("--n-info", type=int, default=24)
    ap.add_argument("--n-red", type=int, default=24)
    ap.add_argument("--n-noise", type=int, default=16)
    ap.add_argument("--rho", type=float, default=0.9, help="冗余维度与信息维度的耦合强度")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--sample-every", type=int, default=50)
    ap.add_argument("--h-samples", type=int, default=4096)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--quick", action="store_true", help="冒烟：1 配置 1 seed")
    args = ap.parse_args()

    if args.quick:
        args.configs = "base"
        args.seeds = "0"
        args.steps = 200
        args.relora_every = 40
        args.n_train = 800
        args.n_val = 200
        args.dose = False
        args.include_vanilla = False

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in ABLATIONS:
            raise ValueError(f"未知配置 {c}，可选: {list(ABLATIONS.keys())}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = args.device

    d_in = args.n_info + args.n_red + args.n_noise
    r = args.r if args.r > 0 else args.hidden
    # 关键约束：库内 hebbian_update 用 x_mean[:cfg.r]，要求 r <= 每层 in_features。
    # 第一层 in_features = d_in，其余层 = hidden，故取 r <= min(d_in, hidden)，
    # 否则 r>in_features 时 Hebbian 累加器形状越界（aeloru_v3.py:1419）。
    r = min(r, d_in, args.hidden)
    alpha = args.alpha if args.alpha > 0 else r
    # 网络维度：[in, hid, hid, ..., out]
    dims = [d_in] + [args.hidden] * args.layers + [2]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       f"ex4_fromscratch_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    fh = open(os.path.join(out_dir, "log.txt"), "w", encoding="utf-8")

    log("=" * 78, fh)
    log("ex4: 从零训练测试 Aeloru v3.1 各模组（聚焦 PEM）", fh)
    log(f"网络维度: {dims}  (in={dims[0]}, hidden={args.hidden}, layers={args.layers}, out=2)", fh)
    log(f"低秩 r={r}  alpha={alpha}  PEM 空间={args.pem_space}", fh)
    log(f"数据: info={args.n_info} red={args.n_red} noise={args.n_noise} rho={args.rho} "
        f"train={args.n_train} val={args.n_val}", fh)
    log(f"配置: {configs}  seeds={seeds}  steps={args.steps} bs={args.bs} lr={args.lr}", fh)
    log(f"设备: {device}  输出: {out_dir}", fh)
    log("=" * 78, fh)

    # 数据：每个 seed 用不同随机种子生成（避免各配置共享同一份数据导致耦合）
    data_cache = {}
    def get_data(seed: int):
        if seed not in data_cache:
            d = make_data(args.n_info, args.n_red, args.n_noise, args.rho,
                          args.n_train, args.n_val, seed)
            # 把数据搬到 device 一次
            data_cache[seed] = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in d)
        return data_cache[seed]

    results_by_cfg: Dict[str, List[dict]] = {}
    for name in configs:
        # 每个 seed 用各自的数据，但同一 name 内跨 seed 共享数据生成逻辑（seed 决定）
        per_cfg = []
        for seed in seeds:
            set_seed(seed)
            cfg = make_cfg(ABLATIONS[name], device, r, alpha, args.lr)
            data = get_data(seed)
            log(f"  [{name}] seed={seed} 训练 {args.steps} 步 ...", fh)
            t0 = time.perf_counter()
            r_res = train_one(name, cfg, dims, data, device, args, fh)
            r_res["seed"] = seed
            per_cfg.append(r_res)
            log(f"    -> 测试 acc={r_res['final_val']:.4f}  训练 acc={r_res['final_train']:.4f}  "
                f"收敛步={r_res['converge_step']}  ({time.perf_counter() - t0:.1f}s)"
                + (f"  corr_abs={r_res['dec'].get('corr_abs', float('nan')):.4f}" if r_res['dec'] else "")
                + (f"  r2_gap={r_res['r2'].get('r2_gap', float('nan')):.3f}" if r_res['r2'] else ""), fh)
        results_by_cfg[name] = per_cfg

    if args.include_vanilla:
        vcfg = None
        per_v = []
        for seed in seeds:
            set_seed(seed)
            data = get_data(seed)
            log(f"  [vanilla] seed={seed} 训练 {args.steps} 步 ...", fh)
            t0 = time.perf_counter()
            r_res = train_one("vanilla", vcfg, dims, data, device, args, fh)
            r_res["seed"] = seed
            per_v.append(r_res)
            log(f"    -> 测试 acc={r_res['final_val']:.4f}  训练 acc={r_res['final_train']:.4f}  "
                f"收敛步={r_res['converge_step']}  ({time.perf_counter() - t0:.1f}s)", fh)
        results_by_cfg["vanilla"] = per_v

    # ---- PEM 剂量扫描 ----
    dose_results: Dict[int, List[dict]] = {}
    if args.dose:
        log("\n" + "=" * 78, fh)
        log("PEM 剂量—反应扫描 (ndi ∈ {0,1,2,4,8})", fh)
        log("=" * 78, fh)
        for ndi in [0, 1, 2, 4, 8]:
            per_ndi = []
            for seed in seeds:
                set_seed(seed)
                data = get_data(seed)
                cfg = make_cfg({**ABLATIONS["base"], "neural_dynamics_iterations": ndi},
                               device, r, alpha, args.lr)
                log(f"  [pem ndi={ndi}] seed={seed} ...", fh)
                r_res = train_one(f"pem{ndi}", cfg, dims, data, device, args, fh)
                r_res["seed"] = seed
                per_ndi.append(r_res)
            dose_results[ndi] = per_ndi

    # ========================================================================
    # 汇总
    # ========================================================================
    log("\n" + "=" * 78, fh)
    log("汇总（多 seed 均值 ± 标准差）", fh)
    log("=" * 78, fh)
    rows = []
    for name in list(results_by_cfg.keys()):
        rs = results_by_cfg[name]
        accs = [r["final_val"] for r in rs]
        convs = [r["converge_step"] for r in rs if r["converge_step"] >= 0]
        row = {
            "config": name,
            "acc_mean": round(float(np.mean(accs)), 4),
            "acc_std": round(float(np.std(accs)), 4),
            "converge_step": round(float(np.mean(convs)), 1) if convs else -1,
            "params": rs[0]["params"],
            "seconds": round(float(np.mean([r["elapsed"] for r in rs])), 1),
        }
        for k, _ in DECOL_METRICS:
            vs = [r["dec"].get(k, float("nan")) for r in rs if r["dec"]]
            row[k] = round(float(np.nanmean(vs)), 5) if vs else float("nan")
        for k in ["r2_info", "r2_red", "r2_noise", "r2_gap"]:
            vs = [r["r2"].get(k, float("nan")) for r in rs if r["r2"]]
            row[k] = round(float(np.nanmean(vs)), 4) if vs else float("nan")
        rows.append(row)
        log(f"  {name:10s} acc={row['acc_mean']:.4f}±{row['acc_std']:.4f}  "
            f"收敛={row['converge_step']:>6}  corr_abs={row.get('corr_abs', float('nan')):.4f}  "
            f"eff_rank={row.get('eff_rank', float('nan')):.3f}  r2_gap={row.get('r2_gap', float('nan')):.3f}", fh)

    df = pd.DataFrame(rows)
    csv_p = os.path.join(out_dir, "summary.csv")
    df.to_csv(csv_p, index=False, encoding="utf-8-sig")

    # 损失明细（每个配置取首个 seed）
    with open(os.path.join(out_dir, "losses.txt"), "w", encoding="utf-8") as f:
        for name, rs in results_by_cfg.items():
            f.write(f"{name}: " + " ".join(f"{v:.4f}" for v in rs[0]["losses"][::max(1, len(rs[0]['losses']) // 200)]) + "\n")

    pngs = make_figures(results_by_cfg, args, out_dir)
    dose_png = None
    if dose_results:
        dose_png = make_dose_figure(dose_results, args, out_dir)
    # 保存剂量结果表
    if dose_results:
        drows = []
        for ndi, rs in dose_results.items():
            accs = [r["final_val"] for r in rs]
            ca = [r["dec"].get("corr_abs", float("nan")) for r in rs if r["dec"]]
            er = [r["dec"].get("eff_rank", float("nan")) for r in rs if r["dec"]]
            gp = [r["r2"].get("r2_gap", float("nan")) for r in rs if r["r2"]]
            drows.append({"ndi": ndi,
                          "acc_mean": round(float(np.mean(accs)), 4),
                          "corr_abs": round(float(np.nanmean(ca)), 4) if ca else float("nan"),
                          "eff_rank": round(float(np.nanmean(er)), 3) if er else float("nan"),
                          "r2_gap": round(float(np.nanmean(gp)), 3) if gp else float("nan")})
        pd.DataFrame(drows).to_csv(os.path.join(out_dir, "pem_dose.csv"), index=False, encoding="utf-8-sig")

    log(f"\n结果已保存:\n  {csv_p}\n" + "\n".join(f"  {p}" for p in pngs if p)
        + (f"\n  {dose_png}" if dose_png else ""), fh)
    fh.close()

    print("\n" + df[["config", "acc_mean", "acc_std", "converge_step",
                      "corr_abs", "eff_rank", "r2_gap"]].to_string(index=False))
    return results_by_cfg


if __name__ == "__main__":
    main()
