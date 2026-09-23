# -*- coding: utf-8 -*-
"""
ex3_synth.py -- 实验三 A 阶段：合成 ground-truth 上的「冗余耦合 vs 功能性耦合」
================================================================================

要回答的问题（paper.md §5.6 / §7.3）
------------------------------------
消融表 5.4 显示「关掉 PEM 后 |rho|_abs 反而更低」，但下游 Forgetting / ACC 变差。
论文的解释是：PEM 保留的相关性是**功能性耦合**，|rho|_abs 是无差别指标，把它误判为冗余。
本脚本在**耦合的功能性由构造给定**的合成数据上，做两件事：

  1. 指标校准：corr_useful / syn_share 能否把「有任务信息的耦合」与「无任务信息的耦合」分开；
  2. PEM 选择性：PEM 是否比无差别解耦（ZCA 白化）更多地剪掉无任务信息的耦合。

源信号结构（m = r = 8）
-----------------------
  (z0,z1)  rho=0.9 相关，标签权重 (+c,+c) -> REL：任务相关，冗余型（说同一件事）
  (z2,z3)  rho=0.9 相关，标签权重 (+c,-c) -> SYN：任务相关，协同型（差才携带信息）
  (z4,z5)  rho=0.9 相关，标签权重 (0,0)   -> IRR：任务无关，纯冗余耦合
  (z6,z7)  rho=0.9 相关，标签权重 (0,0)   -> IRR：任务无关，纯冗余耦合

c 归一化使 REL 与 SYN 对 logit 的贡献方差各为 1。观测 x = z + 0.05*noise。
**关键**：模型带一个辅助重建头（x_hat = h @ D），必须保留全部 8 个源方向，
否则 BP 自己就会把 IRR 维度投影掉，「PEM 是否剪冗余」这个问题将无从谈起
（真实 LLM 的语言建模损失正是这种重建压力）。

处理与对照
----------
  pem-off        neural_dynamics_iterations=0            （论文 w/o PEM）
  pem-1/3/10     低秩 PEM 迭代步数，剂量-反应
  pem-10-w05     lowrank_whiten_coef=0.5，强剂量
  hardwhite      对 h 做精确 ZCA 白化（无差别解耦上界，阴性对照）

用法
----
  python ex3_synth.py --seeds 0,1,2,3,4 --steps 400
  python ex3_synth.py --seeds 0 --steps 60 --quick      # smoke
"""

import os
import sys
import math
import time
import argparse
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aeloru_v3 import AeloruConfig, AeloruLayer

M = 8                      # 源维度 = 低秩维度
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
PAIR_TAG = {(0, 1): "REL", (2, 3): "SYN", (4, 5): "IRR", (6, 7): "IRR"}
FUNC_DIMS = [0, 1, 2, 3]
IRR_DIMS = [4, 5, 6, 7]

CONDITIONS = ["pem-off", "pem-1", "pem-3", "pem-10", "pem-10-w05", "hardwhite"]


# ============================================================================
# 数据
# ============================================================================

def make_data(m: int, n: int, rho: float, energy: str, seed: int,
              device: str, dtype: torch.dtype):
    """生成 (x, z, y)：z 的耦合结构与标签依赖关系由构造给定。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(n, m, generator=g, dtype=torch.float64)

    for a, b in PAIRS:
        z[:, b] = rho * z[:, a] + math.sqrt(max(1.0 - rho ** 2, 1e-12)) * z[:, b]

    scale = torch.ones(m, dtype=torch.float64)
    if energy == "mismatched":
        # IRR 维度方差放大 4 倍 -> 成为主方向，用于分离「选择性」与「谱依赖」
        scale[IRR_DIMS] = 2.0
    z = z * scale

    s = torch.diagonal(torch.cov(z.t()))                      # 各维方差
    var_sum = float(s[0] + s[1] + 2 * rho * math.sqrt(s[0] * s[1]))
    var_dif = float(s[2] + s[3] - 2 * rho * math.sqrt(s[2] * s[3]))
    c_rel = 1.0 / math.sqrt(max(var_sum, 1e-12))
    c_syn = 1.0 / math.sqrt(max(var_dif, 1e-12))

    w = torch.zeros(m, dtype=torch.float64)
    w[0] = w[1] = c_rel
    w[2] = c_syn
    w[3] = -c_syn

    logit = z @ w + 0.3 * torch.randn(n, generator=g, dtype=torch.float64)
    y = (logit > 0).long()
    x = (z + 0.05 * torch.randn(n, m, generator=g, dtype=torch.float64)).to(dtype)
    return x.to(device), z.to(device), y.to(device)


# ============================================================================
# 模型：AeloruLayer 的 h 通路 + 分类头 + 重建头
# ============================================================================

def build_layer(device: str, cond: str, lateral: bool) -> AeloruLayer:
    ndi = {"pem-off": 0, "pem-1": 1, "pem-3": 3,
           "pem-10": 10, "pem-10-w05": 10, "hardwhite": 0}[cond]
    coef = 0.5 if cond == "pem-10-w05" else 0.1

    cfg = AeloruConfig(
        r=M, lora_alpha=float(M),
        in_features=M, out_features=2,
        AMP_DTYPE=torch.float32, device=device,
        pem_space="lowrank",
        neural_dynamics_iterations=ndi,
        lowrank_whiten_coef=coef,
        lowrank_cond_max=5.0,
        lambda_lateral=0.9,
        lateral_lr=1e-5, lateral_decay=0.99,
        # 关掉与本问题无关的全部机制，隔离 PEM
        presumed_domain="none",
        homeostatic_enabled=False,
        whitening_interval=10 ** 9,
        merge_every=10 ** 9,
        wta_k_ratio=1.0,            # WTA 退化为恒等
        fisher_gamma=0.0,           # 门控恒为 1
        prototype_guard=0.0,
        hebbian_lr=0.0,
        sphere_lambda=0.0,
        ortho_lambda=0.0,
        logging_enabled=False,
        diagnostic_enabled=False,
    )
    layer = AeloruLayer(M, 2, cfg)
    with torch.no_grad():
        layer.W0.zero_()            # 无基座：本实验只看 h 通路
        layer.bias.zero_()
        layer.W0_norm_sq = torch.tensor(0.0)
        layer.lora_A.copy_(torch.eye(M, device=device) + 0.01 * torch.randn(M, M, device=device))
        layer.lora_B.copy_(0.01 * torch.randn(2, M, device=device))
        layer.m_x.fill_(1.0)
        layer.m_y.fill_(1.0)
    layer.m_x.requires_grad_(False)
    layer.m_y.requires_grad_(False)
    layer._use_lateral = lateral
    return layer


def zca_white(h: torch.Tensor) -> torch.Tensor:
    """批次内精确白化：无差别地去除**全部**相关（阴性对照）。"""
    mu = h.mean(0, keepdim=True)
    hc = h - mu
    C = (hc.t() @ hc) / max(hc.shape[0] - 1, 1)
    try:
        evals, evecs = torch.linalg.eigh(C)
        W = evecs @ torch.diag(evals.clamp(min=1e-10).rsqrt()) @ evecs.t()
    except RuntimeError:
        return h
    return (hc @ W) + mu


def forward_h(layer: AeloruLayer, x: torch.Tensor, cond: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    镜像 AeloruLayer.forward 的 h 通路（aeloru_v3.py L1053-1072），
    去掉 W0 / W_acc / 稳态 / 域约束等与本问题无关的部分。
    """
    p_in = layer.plasticity_in.to(device=x.device, dtype=x.dtype)
    p_out = layer.plasticity_out.to(device=x.device, dtype=x.dtype)
    A_eff = layer.lora_A * (layer.m_y * p_in).unsqueeze(0)
    h = F.linear(x, A_eff)

    if layer.training:
        if cond == "hardwhite":
            h = h + (zca_white(h) - h.detach())          # straight-through 保持梯度通路一致
        elif layer.cfg.neural_dynamics_iterations > 0:
            h = h + (layer._low_rank_neural_dynamics(h) - h.detach())

    if getattr(layer, "_use_lateral", False):
        lateral_w = layer.lateral_weights.detach().to(h.dtype)
        h = h - F.linear(h, lateral_w)

    B_eff = layer.lora_B * (layer.m_x * p_out).unsqueeze(1)
    logits = F.linear(h, B_eff) * (layer.cfg.lora_alpha / layer.cfg.r)
    return logits, h


@torch.no_grad()
def update_lateral(layer: AeloruLayer, h: torch.Tensor):
    """与 _update_lateral_weights 同源的纯 Hebbian 侧向更新（用 h 均值外积）。"""
    h_mean = h.detach().float().mean(0)
    dL = layer.cfg.lateral_lr * torch.ger(h_mean, h_mean)
    new = layer.lateral_weights * (1.0 - layer.cfg.lateral_decay) + dL.to(layer.lateral_weights.dtype)
    new.fill_diagonal_(0)
    layer.lateral_weights.copy_(new)


# ============================================================================
# 指标
# ============================================================================

def cov_of(H: torch.Tensor) -> torch.Tensor:
    Hc = H - H.mean(0, keepdim=True)
    return (Hc.t() @ Hc) / max(H.shape[0] - 1, 1)


def corr_from_cov(C: torch.Tensor) -> torch.Tensor:
    d = torch.diagonal(C).clamp(min=1e-12)
    return C / torch.sqrt(torch.outer(d, d))


def within_cov(H: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """类内（条件）协方差 Sigma_w = E[Cov(h | Y)]。"""
    N = H.shape[0]
    C = torch.zeros(H.shape[1], H.shape[1], dtype=H.dtype, device=H.device)
    for c in torch.unique(y):
        Hc = H[y == c]
        if Hc.shape[0] > 3:
            C = C + cov_of(Hc) * (Hc.shape[0] / N)
    return C


def task_info_and_pid(H: torch.Tensor, y: torch.Tensor, ridge: float = 1e-6):
    """
    高斯闭式的任务信息量与 PID 分解。

    I_i  = -0.5 log( Sigma_w,ii / Sigma_t,ii )
    I_ij = -0.5 log( det Sigma_w^[ij] / det Sigma_t^[ij] )
    Red_ij = min(I_i, I_j)        冗余型（两维度在说同一件事）
    Syn_ij = I_ij - max(I_i, I_j) 协同型（必须联合才产生信息）
    """
    H = H.double()
    r = H.shape[1]
    I = torch.eye(r, dtype=H.dtype, device=H.device)

    St = cov_of(H)
    Sw = within_cov(H, y)
    St = St + ridge * torch.diagonal(St).mean() * I
    Sw = Sw + ridge * torch.diagonal(Sw).mean() * I

    I_i = (-0.5 * torch.log(torch.diagonal(Sw).clamp(min=1e-12)
                            / torch.diagonal(St).clamp(min=1e-12))).clamp(min=0.0)

    I_ij = torch.zeros(r, r, dtype=H.dtype, device=H.device)
    for i in range(r):
        for j in range(i + 1, r):
            idx = torch.tensor([i, j], device=H.device)
            det_t = torch.det(St[idx][:, idx]).clamp(min=1e-18)
            det_w = torch.det(Sw[idx][:, idx]).clamp(min=1e-18)
            v = float(-0.5 * torch.log(det_w / det_t))
            I_ij[i, j] = I_ij[j, i] = v

    Red = torch.minimum(I_i[:, None].expand(-1, r), I_i[None, :].expand(r, -1))
    Syn = I_ij - torch.maximum(I_i[:, None].expand(-1, r), I_i[None, :].expand(r, -1))
    return I_i, I_ij, Red, Syn


def coupling_metrics(H: torch.Tensor, y: torch.Tensor) -> dict:
    """
    把 |rho| 拆成「被任务使用的部分」与「冗余部分」。

    **必须同时报绝对量与比值**：hardwhite 会把 corr_abs 压到 ~0，
    此时以 corr_abs 为分母的比值指标退化为噪声。判据用**保留率**
    （相对 pem-off 的比值），比值指标只作参考。
    """
    H = H.double()
    r = H.shape[1]
    R = corr_from_cov(cov_of(H))
    off = (R - torch.eye(r, dtype=H.dtype, device=H.device)).abs()
    I_i, I_ij, Red, Syn = task_info_and_pid(H, y)

    n_off = max(r * (r - 1), 1)
    Imax = float(I_ij.max().clamp(min=1e-12))
    It = (I_ij / Imax).clamp(0.0, 1.0)
    syn_mask = (Syn > Red).double()

    corr_abs = float(off.sum() / n_off)
    useful = float((off * It).sum() / n_off)
    useless = float((off * (1.0 - It)).sum() / n_off)
    syn_c = float((off * syn_mask).sum() / n_off)
    red_c = float((off * (1.0 - syn_mask)).sum() / n_off)
    return {
        "corr_abs": corr_abs,
        "useful_corr": useful,          # 绝对量：被任务使用的相关
        "useless_corr": useless,        # 绝对量：冗余耦合
        "syn_corr": syn_c,
        "red_corr": red_c,
        # 比值（corr_abs 足够大时才有意义）
        "corr_useful": useful / corr_abs if corr_abs > 1e-6 else float("nan"),
        "corr_useless": useless / corr_abs if corr_abs > 1e-6 else float("nan"),
        "syn_share": syn_c / corr_abs if corr_abs > 1e-6 else float("nan"),
        "I_max": Imax,
        "_R": R.detach(), "_I_ij": I_ij.detach(),
        "_Syn": Syn.detach(), "_Red": Red.detach(),
    }


def ridge_r2(Htr: torch.Tensor, Ztr: torch.Tensor,
             Hte: torch.Tensor, Zte: torch.Tensor, lam: float = 1e-3):
    """每个源维度从 h 线性可解码的程度 R^2。"""
    Htr, Ztr, Hte, Zte = Htr.double(), Ztr.double(), Hte.double(), Zte.double()
    r = Htr.shape[1]
    A = torch.linalg.solve(Htr.t() @ Htr + lam * torch.eye(r, dtype=Htr.dtype, device=Htr.device),
                           Htr.t() @ Ztr)
    pred = Hte @ A
    ss_res = ((Zte - pred) ** 2).sum(0)
    ss_tot = ((Zte - Zte.mean(0)) ** 2).sum(0).clamp(min=1e-12)
    return (1.0 - ss_res / ss_tot).clamp(min=-1.0, max=1.0)


def probe_acc(Htr, ytr, Hte, yte, steps: int = 300, lr: float = 0.05):
    """线性探针 h -> y；返回 (原样, ZCA 白化, 维度置换) 三种输入的准确率。"""
    Htr, ytr = Htr.double(), ytr.double()
    Hte = Hte.double()

    def fit_predict(Htr_v, Hte_v):
        W = torch.zeros(1, Htr_v.shape[1], dtype=torch.float64, device=Htr_v.device,
                        requires_grad=True)
        opt = torch.optim.LBFGS([W], max_iter=60, line_search_fn="strong_wolfe")
        yv = (ytr * 2 - 1).unsqueeze(1)

        def closure():
            opt.zero_grad()
            l = F.binary_cross_entropy_with_logits((Htr_v @ W.t()), (yv > 0).double())
            l.backward()
            return l
        opt.step(closure)
        with torch.no_grad():
            p = (torch.sigmoid(Hte_v @ W.t()).squeeze(1) > 0.5).long()
        return float((p == yte.long()).float().mean())

    out = {}
    out["acc_full"] = fit_predict(Htr, Hte)
    out["acc_white"] = fit_predict(zca_white(Htr), zca_white(Hte))
    g = torch.Generator(device="cpu").manual_seed(0)
    perm = [Htr[:, k][torch.randperm(Htr.shape[0], generator=g)] for k in range(Htr.shape[1])]
    perm_te = [Hte[:, k][torch.randperm(Hte.shape[0], generator=g)] for k in range(Hte.shape[1])]
    out["acc_perm"] = fit_predict(torch.stack(perm, 1), torch.stack(perm_te, 1))
    return out


# ============================================================================
# 训练与评估
# ============================================================================

def run_one(cond: str, energy: str, seed: int, args, device: str) -> dict:
    set_seed = seed
    torch.manual_seed(set_seed)
    np.random.seed(set_seed)

    x_tr, z_tr, y_tr = make_data(M, args.n_train, args.rho, energy, seed, device, torch.float32)
    x_te, z_te, y_te = make_data(M, args.n_eval, args.rho, energy, seed + 100000, device, torch.float32)

    layer = build_layer(device, cond, args.lateral).to(device)
    with torch.no_grad():
        D = torch.pinverse(layer.lora_A.detach().float().t()).clone()   # r x d_in 重建头
    D = nn.Parameter(D.to(device))
    if args.fix_a:
        layer.lora_A.requires_grad_(False)   # 开环：h 维度严格等于源维度，ground truth 无污染
    params = [p for p in (layer.lora_A, layer.lora_B, D) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)

    n, bs = x_tr.shape[0], args.bs
    layer.train()
    for step in range(args.steps):
        idx = torch.randint(0, n, (bs,), device=device)
        xb, yb = x_tr[idx], y_tr[idx]
        logits, h = forward_h(layer, xb, cond)
        loss_cls = F.cross_entropy(logits.float(), yb)
        loss_rec = F.mse_loss(h.float() @ D.float(), xb)
        loss = loss_cls + args.recon * loss_rec
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if getattr(layer, "_use_lateral", False):
            update_lateral(layer, h)

    # ---- 评估：train 模式下采集，保证 PEM 生效（与 ex2 的 h-mode=train 同理）----
    layer.train()
    with torch.no_grad():
        logits_te, H_te = forward_h(layer, x_te, cond)
        _, H_tr = forward_h(layer, x_tr[:args.n_eval], cond)
    acc = float(((logits_te.float().argmax(-1)) == y_te).float().mean())

    H_te = H_te.float()
    met = coupling_metrics(H_te, y_te)

    # ---- 维度对齐：源维度 j -> 与之最相关的 h 维度 ----
    Zc = z_te.double()
    Hc = H_te.double()
    Zs = (Zc - Zc.mean(0)) / Zc.std(0).clamp(min=1e-12)
    Hs = (Hc - Hc.mean(0)) / Hc.std(0).clamp(min=1e-12)
    Czh = (Zs.t() @ Hs) / max(Zs.shape[0] - 1, 1)          # m x r
    pi = Czh.abs().argmax(dim=1)
    align = float(Czh.abs().gather(1, pi[:, None]).squeeze(1).mean())
    injective = float(pi.unique().numel()) / M

    # ---- 三类 ground-truth 对上的实际相关、信息量与冗余合并 ----
    def _mn(v):
        return float(np.mean(v)) if v else float("nan")

    row = {"cond": cond, "energy": energy, "seed": seed, "acc": acc,
           "align": align, "injective": injective}
    for tag in ("REL", "SYN", "IRR"):
        rhos, infos, syns, cosines, merges = [], [], [], [], []
        for (a, b) in PAIRS:
            if PAIR_TAG[(a, b)] != tag:
                continue
            i, j = int(pi[a]), int(pi[b])
            # 冗余合并度：两个源维度在 h 空间中的表示方向余弦（不依赖 argmax，更稳健）
            cosines.append(float(F.cosine_similarity(Czh[a][None], Czh[b][None], dim=1).abs()))
            if i == j:
                merges.append(1.0)     # 两个源维度塌缩到同一 h 维度 = 最大冗余
                continue
            merges.append(0.0)
            rhos.append(float(met["_R"][i, j].abs()))
            infos.append(float(met["_I_ij"][i, j]))
            syns.append(float(met["_Syn"][i, j] - met["_Red"][i, j]))
        row[f"rho_{tag}"] = _mn(rhos)
        row[f"info_{tag}"] = _mn(infos)
        row[f"synred_{tag}"] = _mn(syns)
        row[f"merge_{tag}"] = _mn(merges)
        row[f"cos_{tag}"] = _mn(cosines)

    # ---- 指标自校验：直接在源信号 z 上算，IRR 对的信息量必须 ≈ 0 ----
    _, zij_gt, _, _ = task_info_and_pid(z_te.double(), y_te)
    for tag in ("REL", "SYN", "IRR"):
        row[f"zinfo_{tag}"] = _mn([float(zij_gt[a, b]) for (a, b) in PAIRS
                                   if PAIR_TAG[(a, b)] == tag])

    # ---- 可解码性：h 是否还保留各源方向 ----
    r2 = ridge_r2(H_tr, z_tr[:args.n_eval].double(), H_te.double(), z_te.double())
    row["r2_func"] = float(r2[FUNC_DIMS].mean())
    row["r2_irr"] = float(r2[IRR_DIMS].mean())
    row["r2_gap"] = row["r2_func"] - row["r2_irr"]

    if args.probe:
        row.update(probe_acc(H_tr, y_tr[:args.n_eval], H_te, y_te))

    row.update({k: v for k, v in met.items() if not k.startswith("_")})
    return row


# ============================================================================
# 汇总
# ============================================================================

def paired_t(a: List[float], b: List[float]) -> Tuple[float, float]:
    """配对 t 检验；无 scipy 时用正态近似（n 小，仅作方向性参考）。"""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    if n < 2 or np.allclose(d.std(ddof=1), 0.0):
        return float(d.mean()), float("nan")
    t = float(d.mean() / (d.std(ddof=1) / math.sqrt(n)))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), n - 1))
    except Exception:
        from statistics import NormalDist
        p = float(2 * NormalDist().cdf(-abs(t)))
    return float(d.mean()), p


def main():
    ap = argparse.ArgumentParser(description="ex3 A: 冗余耦合 vs 功能性耦合（合成 ground-truth）")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--conds", type=str, default=",".join(CONDITIONS))
    ap.add_argument("--energy", type=str, default="matched,mismatched",
                    help="matched=三类维度等方差；mismatched=IRR 方差 x4（分离选择性与谱依赖）")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--recon", type=float, default=1.0, help="重建损失权重（保留全部源方向的压力）")
    ap.add_argument("--rho", type=float, default=0.9, help="构造的耦合强度")
    ap.add_argument("--n-train", type=int, default=8192)
    ap.add_argument("--n-eval", type=int, default=2048)
    ap.add_argument("--lateral", action="store_true", help="同时启用 Hebbian 侧向连接")
    ap.add_argument("--fix-a", action="store_true",
                    help="冻结 lora_A（开环）：h 维度严格等于源维度，ground truth 不被维度混合污染")
    ap.add_argument("--probe", action="store_true", help="额外跑读出去除探针（较慢）")
    ap.add_argument("--quick", action="store_true", help="最小 smoke：1 配置 1 seed")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    if args.quick:
        args.seeds, args.conds, args.energy = "0", "pem-off,pem-10", "matched"
        args.steps, args.n_train, args.n_eval = 60, 2048, 1024

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    conds = [c for c in args.conds.split(",") if c.strip()]
    energies = [e for e in args.energy.split(",") if e.strip()]
    device = args.device

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       f"ex3_synth_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")

    rows: List[dict] = []
    t0 = time.time()
    for energy in energies:
        for cond in conds:
            for seed in seeds:
                r = run_one(cond, energy, seed, args, device)
                rows.append(r)
                print(f"  {energy:11s} {cond:12s} seed={seed}  "
                      f"corr_abs={r['corr_abs']:.4f}  useful={r['useful_corr']:.4f}  "
                      f"useless={r['useless_corr']:.4f}  "
                      f"rho_REL={r['rho_REL']:.3f} rho_SYN={r['rho_SYN']:.3f} "
                      f"rho_IRR={r['rho_IRR']:.3f}  merge_IRR={r['merge_IRR']:.2f}  "
                      f"zinfo_IRR={r['zinfo_IRR']:.3f}  acc={r['acc']:.3f}  "
                      f"align={r['align']:.2f}", flush=True)

    import csv
    cols = [k for k in rows[0].keys()]
    with open(os.path.join(out_dir, "per_run.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---- 汇总：按 energy x cond 聚合，并做相对 pem-off 的配对检验 ----
    metrics = ["corr_abs", "useful_corr", "useless_corr", "syn_corr", "red_corr",
               "rho_REL", "rho_SYN", "rho_IRR", "info_REL", "info_SYN", "info_IRR",
               "merge_REL", "merge_SYN", "merge_IRR", "cos_REL", "cos_SYN", "cos_IRR",
               "r2_func", "r2_irr", "r2_gap", "acc"]
    summ = []
    for energy in energies:
        base = {s: next(r for r in rows if r["cond"] == "pem-off"
                        and r["energy"] == energy and r["seed"] == s) for s in seeds}
        for cond in conds:
            sub = [r for r in rows if r["cond"] == cond and r["energy"] == energy]
            item = {"energy": energy, "cond": cond}
            for mk in metrics:
                v = [r[mk] for r in sub]
                item[mk] = float(np.mean(v))
                item[f"{mk}_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                if cond != "pem-off":
                    d, p = paired_t(v, [base[s][mk] for s in seeds])
                    item[f"d_{mk}"] = d
                    item[f"p_{mk}"] = p
                    # 保留率：相对 pem-off 的比值（判据以它为准，避免归一化退化）
                    b = float(np.mean([base[s][mk] for s in seeds]))
                    item[f"ret_{mk}"] = (float(np.mean(v)) / b) if abs(b) > 1e-9 else float("nan")
            # 选择性：功能性相关的保留率 减去 冗余相关的保留率
            if cond != "pem-off":
                item["selectivity"] = item.get("ret_useful_corr", float("nan")) \
                    - item.get("ret_useless_corr", float("nan"))
            summ.append(item)
    scols: List[str] = []
    for it in summ:
        for k in it.keys():
            if k not in scols:
                scols.append(k)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=scols)
        w.writeheader()
        w.writerows(summ)

    # ---- 判据输出 ----
    print("\n" + "=" * 78)
    print("判据检查（matched 组，相对 pem-off 的配对差）")
    print("=" * 78)
    for energy in energies:
        print(f"\n[{energy}]")
        off = next(i for i in summ if i["cond"] == "pem-off" and i["energy"] == energy)
        print(f"  [校准] 源信号 z 上：zinfo_REL={off.get('info_REL', float('nan')):.3f} / "
              f"IRR 基准应以 z 空间为准（per_run.csv 的 zinfo_* 列）；"
              f"h 空间：info_REL={off['info_REL']:.3f} info_SYN={off['info_SYN']:.3f} "
              f"info_IRR={off['info_IRR']:.3f}")
        for it in summ:
            if it["energy"] != energy:
                continue
            if it["cond"] == "pem-off":
                print(f"  {it['cond']:12s} corr_abs={it['corr_abs']:.4f}  "
                      f"useful={it['useful_corr']:.4f}  useless={it['useless_corr']:.4f}  "
                      f"merge_IRR={it['merge_IRR']:.2f}  acc={it['acc']:.3f}")
                continue
            print(f"  {it['cond']:12s} corr_abs={it['corr_abs']:.4f} "
                  f"(ret={it['ret_corr_abs']:.2f})  "
                  f"useful={it['useful_corr']:.4f} (ret={it['ret_useful_corr']:.2f})  "
                  f"useless={it['useless_corr']:.4f} (ret={it['ret_useless_corr']:.2f})  "
                  f"SEL={it['selectivity']:+.2f}  merge_IRR={it['merge_IRR']:.2f}  "
                  f"acc={it['acc']:.3f}")
        print("  判据：SEL = ret_useful - ret_useless；"
              "SEL>0 支持 O1(选择性)，SEL≈0 支持 O2(无差别)，"
              "matched 与 mismatched 结论不一致则判 O3(谱依赖)")

    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"\n总耗时 {time.time() - t0:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
