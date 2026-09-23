# -*- coding: utf-8 -*-
"""
特征解耦 (Feature Decoupling) 测试：Aeloru v3.1 vs LoRA
=======================================================

与 ex1.py（概念漂移 / 抗遗忘）的区别
----------------------------------
ex1 测的是**结果**：学完任务序列后还记得多少（ACC / BWT / Forgetting）。
ex2 测的是**成因**：Aeloru 声称的低秩特征解耦是否真的发生了，
以及解耦程度能否解释它的抗遗忘表现。

被检验的核心假设
----------------
Aeloru v3.1 在 r 维低秩子空间 h = whiten(x) @ A'^T 中布置了几套解耦机制：

    1. PEM 低秩神经动态   _low_rank_neural_dynamics  侧向抑制 + 精度加权去相关
    2. WTA 硬竞争         _wta_competition           top-k 稀疏，迫使特征分工
    3. SPHeRe 结构投影    get_structure_preservation_loss   S = A C_x A^T -> I
    4. 正交惩罚           get_ortho_penalty          DeltaW ⊥ W0
    5. 双一维 HGF 门控    _update_plasticity         维度级可塑性分配
    6. 输入白化           whiten_input               解除输入维度间的相关

**解耦假设**：若上述机制生效，则 h 的各维度应趋于统计独立（低相关、低总相关）、
能量均衡（高有效秩、低条件数）、且不同任务占用不同的维度子空间；
后者直接意味着任务间干扰小 -> 遗忘少。

度量体系
--------
在 r 维低秩特征 h 上（对每个注入层分别计算后取均值）：

| 指标            | 定义                                          | 方向 |
|-----------------|-----------------------------------------------|------|
| `corr_abs`      | 相关矩阵非对角元的平均绝对值                   | ↓ 好 |
| `corr_max`      | 非对角元最大绝对值（最坏冗余维度对）           | ↓ 好 |
| `TC`            | 高斯总相关 0.5*(Σ log σ_i² − log det Σ)        | ↓ 好 |
| `MI`            | 平均成对互信息 −0.5·log(1−ρ²)                  | ↓ 好 |
| `eff_rank`      | 有效秩 exp(H(λ/tr λ))，1..r                    | ↑ 好 |
| `PR`            | 参与率 (Σλ)²/Σλ²，1..r                         | ↑ 好 |
| `iso_err`       | ‖C/tr(C)·r − I‖_F（各向同性误差）              | ↓ 好 |
| `cond`          | 归一化协方差条件数 λmax/λmin                   | ↓ 好 |
| `sparsity`      | 被 WTA 稀疏化的特征比例                        | — 参考 |
| `dw_w0_cos`     | \|cos(vec ΔW, vec W0)\| 新知识↔预训练知识正交度 | ↓ 好 |
| `dw_eff_rank`   | ΔW 的有效秩（更新是否坍缩到单一方向）          | ↑ 好 |
| `task_overlap`  | 各任务「维度重要性向量」的平均余弦相似度        | ↓ 好 |
| `task_principal`| 各任务主方向的平均 \|cos\|（主角度）            | ↓ 好 |
| `task_jaccard`  | 各任务 top-k 维度集合的平均 Jaccard 重叠        | ↓ 好 |
| `task_selectivity` | 各维度对「任务身份」的敏感度（组间/总方差比） | ↑ 好 |

TC 与 MI 是 β-TCVAE 以来的**标准解耦指标**；eff_rank / PR / iso_err 刻画
「特征坍缩」；task_* 刻画「任务级解耦」，是解耦假设与抗遗忘之间的桥梁。

关于 `dw_w0_cos` 的解读提醒：q_proj 的 W0 是 896x896 = 802816 维，
两个随机方向在该空间中的期望 |cos| ≈ sqrt(2/(π·N)) ≈ 9e-4。
因此 |cos| 落在 1e-3 量级是「与基座正交」的**随机基线**，
只有当它显著高于该量级时才说明微调在覆写基座已有方向。

方法（消融设计）
----------------
以 `aeloru`（全开解耦）为基准，逐个关掉一套机制，看指标如何退化：

| 方法               | 相对 aeloru 的改动                         |
|--------------------|--------------------------------------------|
| `lora`             | peft LoRA 基线，无任何解耦机制             |
| `aeloru`           | 全开解耦（基准）                           |
| `aeloru-nopem`     | `neural_dynamics_iterations=0` 关 PEM      |
| `aeloru-nowta`     | `wta_k_ratio=1.0` 关 WTA 硬竞争            |
| `aeloru-nosphere`  | `sphere_lambda=0` 关 SPHeRe 结构投影       |
| `aeloru-noortho`   | `ortho_lambda=0` 关 ΔW⊥W0 正交约束         |
| `aeloru-nogate`    | `fisher_gamma=0, prototype_guard=0` 关门控 |
| `aeloru-nohebb`    | `hebbian_lr=0` 关 Hebbian 探索             |
| `aeloru-whiten`    | `whitening_interval=200` 开输入白化        |

关于 h 的采集模式（重要）
------------------------
`_low_rank_neural_dynamics` 与 `_wta_competition` 都以 `self.training` 为开关，
**推理期自动退化为恒等映射**。若在 eval 模式下采集 h，消融 `nopem`/`nowta`
将完全看不出差异。因此默认在 **train 模式下采集**（`--h-mode train`），
此时三套机制全部生效，采集到的才是真正驱动学习的特征。
采集过程会快照并还原被改写的在线统计量（mu_h / C_hh / running_var），
保证不污染训练动态。

用法
----
    python ex2.py                                  # 默认 6 方法 x 3 任务 x 120 步
    python ex2.py --methods lora,aeloru,aeloru-nopem --steps 60
    python ex2.py --tasks sst2,mrpc,qnli,rte,cola --steps 200 --layers 8

@author JYIMU
"""

import os
import sys
import time
import math
import random
import argparse
import warnings
import contextlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 离线优先：数据集与模型均已在本地缓存
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from aeloru_v3 import AeloruConfig, AeloruLayer, cross_modal_sleep_step

# ============================================================================
# 常量
# ============================================================================

MODEL_PATH = r"E:\Qelys\models\Qwen2.5-0.5B"
N_LAYERS_TOTAL = 24          # Qwen2.5-0.5B

# GLUE 全部为二分类，可共享分类头（与 ex1 同协议）
TASK_FIELDS = {
    "sst2": ("sentence",),
    "cola": ("sentence",),
    "mrpc": ("sentence1", "sentence2"),
    "rte":  ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
}
ALL_TASKS = list(TASK_FIELDS.keys())

# 消融配置：相对 aeloru 基准的覆盖项
ABLATIONS: Dict[str, dict] = {
    "aeloru":           {},
    "aeloru-nopem":     {"neural_dynamics_iterations": 0},
    "aeloru-nowta":     {"wta_k_ratio": 1.0},
    "aeloru-nosphere":  {"sphere_lambda": 0.0},
    "aeloru-noortho":   {"ortho_lambda": 0.0, "ortho_lambda_anchor": 0.0},
    "aeloru-nogate":    {"fisher_gamma": 0.0, "prototype_guard": 0.0},
    "aeloru-nohebb":    {"hebbian_lr": 0.0},
    "aeloru-whiten":    {"whitening_interval": 200},
}
METHODS = ["lora"] + list(ABLATIONS.keys())
DEFAULT_METHODS = ["lora", "aeloru", "aeloru-nopem", "aeloru-nowta",
                   "aeloru-nosphere", "aeloru-noortho"]

# 汇总表中要展示的解耦指标（方向：True=越大越好，False=越小越好，None=参考）
DECOL_METRICS = [
    ("corr_abs", False), ("corr_max", False), ("TC", False), ("MI", False),
    ("eff_rank", True),  ("PR", True),        ("iso_err", False), ("cond", False),
    ("sparsity", None),  ("dw_w0_cos", False), ("dw_eff_rank", True),
    ("task_overlap", False), ("task_principal", False), ("task_jaccard", False),
    ("task_selectivity", True),
]


# ============================================================================
# 工具
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg: str, fh=None) -> None:
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


# ============================================================================
# 数据（与 ex1 完全一致）
# ============================================================================

def build_task_dataset(tok, task: str, max_len: int, n_train: int, n_val: int, seed: int):
    """加载并预编码单个 GLUE 任务。"""
    raw = load_dataset("glue", task)
    fields = TASK_FIELDS[task]

    def tok_fn(batch):
        if len(fields) == 2:
            return tok(batch[fields[0]], batch[fields[1]],
                       truncation=True, max_length=max_len)
        return tok(batch[fields[0]], truncation=True, max_length=max_len)

    enc = raw.map(tok_fn, batched=True)
    keep = {"input_ids", "attention_mask", "label"}
    for split in list(enc.keys()):
        drop = [c for c in enc[split].column_names if c not in keep]
        enc[split] = enc[split].remove_columns(drop)
        enc[split] = enc[split].rename_column("label", "labels")
    enc.set_format("torch")

    n_cls = len(enc["train"].unique("labels"))
    assert n_cls == 2, f"{task} 期望二分类，实际 {n_cls} 类"

    train = enc["train"].shuffle(seed=seed)
    val = enc["validation"].shuffle(seed=seed)
    if n_train > 0:
        train = train.select(range(min(n_train, len(train))))
    if n_val > 0:
        val = val.select(range(min(n_val, len(val))))
    return train, val


# ============================================================================
# 模型构建
# ============================================================================

def inject_aeloru_selected(model, layer_indices: List[int], cfg: AeloruConfig,
                           target: str = "q_proj") -> int:
    """
    按层号选择性注入 AeloruLayer。

    不用 inject_aeloru：它对全部 24 层递归匹配，无法按层过滤；
    层数是本实验的关键变量（层数过多时 Aeloru 的稳态增益会压垮残差流）。
    """
    n = 0
    for idx in layer_indices:
        attn = model.model.layers[idx].self_attn
        orig = getattr(attn, target, None)
        if not isinstance(orig, nn.Linear):
            continue
        layer = AeloruLayer(orig.in_features, orig.out_features, cfg, original_linear=orig)
        # set_pretrained_weight 只搬运 buffer，必须在 inject 之后统一 .to(device)
        layer.set_pretrained_weight(orig.weight.data, getattr(orig, "bias", None))
        setattr(attn, target, layer)
        n += 1
    return n


def make_aeloru_cfg(method: str, device: str, lr: float, r: int, alpha: float) -> AeloruConfig:
    """
    构造 Aeloru 配置：安全底座 + 该消融的覆盖项。

    安全底座沿用 ex1 的结论——下列几项在预训练 Transformer 上会破坏表示，
    必须关掉，否则解耦测试测到的是「表示被破坏」而非「特征解耦」：
        presumed_domain="none"        域约束 clamp 会截断半数负激活
        homeostatic_enabled=False     稳态增益会把每层 RMS 拉向 1
        merge_every=1e9               本实验步数少，ReLoRA 合并只增加噪声
    """
    base = dict(
        r=r, lora_alpha=alpha, LoRA_lr=lr,
        pem_space="lowrank",              # PEM 只在 r 维子空间解耦，不碰主激活路径
        neural_dynamics_iterations=2,      # 开启低秩预测编码解耦
        presumed_domain="none",
        homeostatic_max_gain=1.0,
        homeostatic_enabled=False,
        whitening_interval=10 ** 9,
        merge_every=10 ** 9,
        logging_enabled=False,
        diagnostic_enabled=False,
        device=device,
    )
    base.update(ABLATIONS.get(method, {}))
    return AeloruConfig().llm_tuning(**base)


def build_model(method: str, n_layers: int, device: str, lr: float,
                n_labels: int = 2, pad_id: int = 0,
                r: int = 8, alpha: float = 16.0):
    """构建待训练模型；返回 (model, 可训练参数量)。"""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=n_labels, trust_remote_code=True)
    model.config.pad_token_id = pad_id

    if method == "lora":
        from peft import LoraConfig, get_peft_model, TaskType
        model = get_peft_model(model, LoraConfig(
            r=r, lora_alpha=alpha, target_modules=["q_proj"],
            lora_dropout=0.0, bias="none", task_type=TaskType.SEQ_CLS,
            layers_to_transform=list(range(N_LAYERS_TOTAL - n_layers, N_LAYERS_TOTAL)),
        ))
        model.to(device)
    elif method in ABLATIONS:
        cfg = make_aeloru_cfg(method, device, lr, r, alpha)
        inject_aeloru_selected(
            model, list(range(N_LAYERS_TOTAL - n_layers, N_LAYERS_TOTAL)), cfg)
        model.to(device)
        _freeze_base(model)
    else:
        raise ValueError(f"未知方法: {method}，可选: {METHODS}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, trainable


def _freeze_base(model) -> None:
    """冻结基座，仅训练 Aeloru 注入参数与分类头。"""
    aeloru_ids = set()
    for layer in model.modules():
        if isinstance(layer, AeloruLayer):
            for p in layer.get_trainable_params():
                aeloru_ids.add(id(p))
    for p in model.parameters():
        p.requires_grad_(id(p) in aeloru_ids)
    for p in model.score.parameters():
        p.requires_grad_(True)


# ============================================================================
# 统一的「被注入层」抽象：抹平 Aeloru 与 peft LoRA 的接口差异
# ============================================================================

class Entry:
    """一个被注入（或替换）的注意力投影层。"""

    __slots__ = ("kind", "module", "name", "adapter")

    def __init__(self, kind: str, module: nn.Module, name: str,
                 adapter: Optional[str] = None):
        self.kind = kind          # 'aeloru' | 'lora'
        self.module = module
        self.name = name
        self.adapter = adapter

    def __repr__(self):
        return f"<Entry {self.kind} {self.name}>"


def collect_entries(model) -> List[Entry]:
    """收集模型中所有被注入的层，按 Transformer 层号排序。"""
    from peft.tuners.lora import Linear as LoraLinear

    entries: List[Entry] = []
    for name, mod in model.named_modules():
        if isinstance(mod, AeloruLayer):
            entries.append(Entry("aeloru", mod, name))
        elif isinstance(mod, LoraLinear):
            adapters = list(getattr(mod, "lora_A", {}).keys())
            if adapters:
                entries.append(Entry("lora", mod, name, adapters[0]))

    def key(e: Entry):
        parts = e.name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                return (int(parts[i + 1]), e.name)
        return (10 ** 6, e.name)

    entries.sort(key=key)
    return entries


# ============================================================================
# 低秩特征 h 的提取
# ============================================================================

def h_from_entry(entry: Entry, x: torch.Tensor) -> torch.Tensor:
    """
    计算 r 维低秩特征 h。

    Aeloru 分支严格镜像 AeloruLayer.forward 的 h 通路（aeloru_v3.py L1039-1069）：

        x -> whiten_input -> h = x @ A'^T -> PEM 低秩神经动态 -> 减侧向 -> WTA

    之所以不能直接在 forward 里插桩取 h：forward 是黑盒，而 PEM/WTA 只在
    self.training 下生效，必须原样复现整条 h 通路才能测到解耦机制。

    peft 分支的 h 就是 x @ A^T（LoRA 的下投影激活），无门控、无竞争、无白化。
    """
    if entry.kind == "aeloru":
        lay = entry.module
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))

        # 白化按 forward 原式在 AMP dtype 下计算（mu_x / W_whiten 都是 bf16），
        # 之后立即升到 fp32：h 的相关矩阵对数值精度敏感，
        # bf16 只有 8 位尾数，会把协方差估计污染成噪声。
        x = lay.whiten_input(x.to(dtype=lay.W0.dtype)).float()
        p_in = lay.plasticity_in.float()
        effective_A = lay.lora_A.float() * (lay.m_y.float() * p_in).unsqueeze(0)
        h = F.linear(x, effective_A)

        if lay.training and lay.cfg.pem_space == "lowrank" \
                and lay.cfg.neural_dynamics_iterations > 0:
            h = h + (lay._low_rank_neural_dynamics(h) - h.detach())

        lateral_w = lay.lateral_weights.detach().clone().to(h.dtype)
        h = h - F.linear(h, lateral_w)
        h = lay._wta_competition(h)
        return h

    mod = entry.module
    A = mod.lora_A[entry.adapter].weight.float()
    if x.dim() > 2:
        x = x.reshape(-1, x.size(-1))
    return F.linear(x.float(), A)


@contextlib.contextmanager
def preserve_dynamics(entries: List[Entry]):
    """
    保护会被「额外一次前向」改写的在线统计量。

    提取 h 需要在正常训练之外多跑一次 h 通路，而
        _low_rank_neural_dynamics : 原地 mul_/add_ 改写 mu_h / C_hh
        forward 第 7 步           : 重新绑定 running_var
    若不还原，采集过程就会污染训练动态，使消融对比失真。
    """
    saved = []
    for e in entries:
        if e.kind != "aeloru":
            continue
        lay = e.module
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


class HCollector:
    """用 forward_pre_hook 捕获每个被注入层的输入 x，再据此算出 h。"""

    def __init__(self, entries: List[Entry]):
        self.entries = entries
        self._xs: Dict[int, List[torch.Tensor]] = {id(e.module): [] for e in entries}
        self.handles = []
        for e in entries:
            self.handles.append(e.module.register_forward_pre_hook(self._make(e)))

    def _make(self, entry: Entry):
        def hook(mod, inp):
            if inp and torch.is_tensor(inp[0]):
                self._xs[id(mod)].append(inp[0].detach())
        return hook

    def take(self) -> Dict[str, torch.Tensor]:
        """取出并清空本轮捕获的 x，返回 {entry_name: (N, d_in)}。"""
        out = {}
        for e in self.entries:
            buf = self._xs[id(e.module)]
            if buf:
                out[e.name] = torch.cat([t.reshape(-1, t.size(-1)) for t in buf], dim=0)
            buf.clear()
        return out

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


@torch.no_grad()
def collect_h_per_task(model, entries: List[Entry], loaders: Dict[str, DataLoader],
                       tasks: List[str], device: str, h_mode: str,
                       max_rows: int = 4096) -> Dict[str, List[torch.Tensor]]:
    """
    对每个任务采集各层的低秩特征 h。

    返回 {task: [H_layer0, H_layer1, ...]}，H 形状 (N, r)，float32。

    h_mode='train' 时 WTA / PEM 生效（默认，见模块文档）；
    h_mode='eval'  时二者退化为恒等映射，只能测到「推理期特征」。
    """
    was_training = model.training
    model.train() if h_mode == "train" else model.eval()

    coll = HCollector(entries)
    out: Dict[str, List[List[torch.Tensor]]] = {t: [[] for _ in entries] for t in tasks}
    try:
        with preserve_dynamics(entries):
            for t in tasks:
                for batch in loaders[t]:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    model(input_ids=batch["input_ids"],
                          attention_mask=batch["attention_mask"])
                    xs = coll.take()
                    if any(e.name not in xs for e in entries):
                        continue
                    for li, e in enumerate(entries):
                        h = h_from_entry(e, xs[e.name]).float()
                        out[t][li].append(h.cpu())
    finally:
        coll.remove()
        model.train(was_training)

    # 拼接 + 限流（协方差只需足够样本，全量反而拖慢）
    merged: Dict[str, List[torch.Tensor]] = {}
    for t in tasks:
        merged[t] = []
        for chunks in out[t]:
            if not chunks:
                merged[t].append(torch.zeros(0, 1))
                continue
            H = torch.cat(chunks, dim=0)
            if H.shape[0] > max_rows:
                idx = torch.randperm(H.shape[0])[:max_rows]
                H = H[idx]
            merged[t].append(H)
    return merged


# ============================================================================
# 解耦度量（核心）
# ============================================================================

def decoupling_metrics(H: torch.Tensor) -> dict:
    """
    在 r 维低秩特征 h 上计算一整套解耦指标。

    H: (N, r) 特征矩阵（N 个样本/token，r 个低秩维度）
    """
    H = H.float()
    N, r = H.shape
    if N < 4 or r < 2:
        return {}

    Hc = H - H.mean(dim=0, keepdim=True)
    C = (Hc.t() @ Hc) / max(N - 1, 1)                      # r x r 协方差
    tr = torch.diagonal(C).sum().clamp(min=1e-12)
    # 微小岭项：WTA 会把部分维度精确置零，使 C 奇异，slogdet 会失败
    C = C + (1e-6 * tr / r) * torch.eye(r, device=C.device, dtype=C.dtype)

    d = torch.diagonal(C).clamp(min=1e-12)
    d_inv_sqrt = torch.diag(1.0 / torch.sqrt(d))
    R = d_inv_sqrt @ C @ d_inv_sqrt                        # 相关矩阵
    I = torch.eye(r, device=C.device, dtype=C.dtype)
    off = R - I
    n_off = max(r * (r - 1), 1)

    corr_abs = float(off.abs().sum() / n_off)
    corr_max = float(off.abs().max())

    # 高斯总相关 TC = 0.5 * (Σ log σ_i² − log det Σ)
    # 多元高斯下 TC == 各维度与整体之间的冗余，是 β-TCVAE 的标准解耦度量。
    sign, logdet = torch.linalg.slogdet(C)
    if float(sign) <= 0 or not math.isfinite(float(logdet)):
        logdet = torch.log(d).sum()
    tc = max(0.0, float(0.5 * (torch.log(d).sum() - logdet)))

    # 成对互信息（高斯闭式）I(h_i;h_j) = −0.5 log(1 − ρ_ij²)
    rho2 = (R ** 2).clamp(max=1.0 - 1e-7) * (1.0 - I)
    mi = float((-0.5 * torch.log1p(-rho2)).sum() / n_off)

    # 谱特征
    ev = torch.linalg.eigvalsh(C).clamp(min=1e-12)
    p = ev / ev.sum()
    eff_rank = float(torch.exp(-(p * torch.log(p)).sum()))
    pr = float((ev.sum() ** 2) / (ev ** 2).sum())
    cond = float(ev.max() / ev.min())

    # 各向同性误差：归一化协方差与单位阵的偏差（SPHeRe 的直接优化目标）
    C_norm = C * (r / float(tr))
    iso_err = float((C_norm - I).pow(2).sum().sqrt())

    # WTA 稀疏度：被压到峰值 5% 以下的特征比例
    peak = H.abs().max().clamp(min=1e-12)
    sparsity = float((H.abs() < 0.05 * peak).float().mean())

    return {
        "corr_abs": corr_abs, "corr_max": corr_max, "TC": tc, "MI": mi,
        "eff_rank": eff_rank, "PR": pr, "cond": cond, "iso_err": iso_err,
        "sparsity": sparsity,
        "h_std": float(H.std()),
        # 下划线前缀 = 仅供画图复用的内部量，不进汇总表
        "_C": C, "_R": R, "_p": p.detach(),
    }


def delta_w_of(entry: Entry) -> Optional[torch.Tensor]:
    """
    取出该层相对于预训练权重 W0 的有效增量 ΔW，全程 fp32。

    **不能**用 `compute_weights() - W0` 反推：两者都是 bf16，ΔW 的量级远小于
    W0，相减得到的主要是 bf16 量化噪声（全秩随机矩阵），会把
    dw_eff_rank 抬到几十、把 |cos(ΔW, W0)| 压到 ~1e-4，结论完全失真。
    因此这里在 fp32 下按 compute_weights() 的合成顺序显式重建 ΔW：

        ΔW = (α/r)·B@A ⊙ (m_x⊗m_y) ⊙ (p_out⊗p_in)   [能量预算缩放] + Hebbian + W_acc
    """
    if entry.kind == "aeloru":
        lay = entry.module
        cfg = lay.cfg
        scale = cfg.lora_alpha / cfg.r
        A = lay.lora_A.float()                       # (r, d_in)
        B = lay.lora_B.float()                       # (d_out, r)

        dW = scale * (B @ A)
        dW = dW * lay.m_x.float().unsqueeze(1) * lay.m_y.float().unsqueeze(0)
        dW = dW * lay.plasticity_out.float().unsqueeze(1) \
                  * lay.plasticity_in.float().unsqueeze(0)

        # 能量预算：与 apply_energy_budget 一致（用未调制的 B@A 范数判定）
        w0_norm = torch.sqrt(lay.W0_norm_sq.to(dW.device).float())
        max_allowed = cfg.energy_eta * w0_norm
        dw_norm = lay._delta_w_norm_exact().float()
        if float(dw_norm) > float(max_allowed) and float(dw_norm) > 1e-8:
            dW = dW * (max_allowed / dw_norm)

        if lay._hebbian_pending_apply:
            dW = dW + (lay._hebbian_delta_B.float() @ lay._hebbian_delta_A.float())

        return (dW + lay._get_W_acc().float()).detach()

    mod = entry.module
    try:
        A = mod.lora_A[entry.adapter].weight.float()      # (r, d_in)
        B = mod.lora_B[entry.adapter].weight.float()      # (d_out, r)
        return (mod.scaling[entry.adapter] * (B @ A)).detach()
    except Exception:
        return None


def w0_of(entry: Entry) -> Optional[torch.Tensor]:
    if entry.kind == "aeloru":
        return entry.module.W0.float().detach()
    mod = entry.module
    try:
        return mod.base_layer.weight.float().detach()
    except Exception:
        return None


def delta_w_metrics(entries: List[Entry]) -> dict:
    """
    ΔW 侧的度量：
      dw_w0_cos   |cos(vec ΔW, vec W0)| —— 新知识与预训练知识的方向重叠。
                  越低说明微调写入的是与基座正交的新方向（解耦），
                  越高说明在覆写基座已有方向（易破坏旧能力）。
      dw_eff_rank ΔW 的有效秩 —— 更新是否坍缩到少数几个方向。
    """
    cos_l, eff_l, rel_l = [], [], []
    for e in entries:
        dW = delta_w_of(e)
        W0 = w0_of(e)
        if dW is None or W0 is None or dW.shape != W0.shape:
            continue
        a, b = dW.reshape(-1), W0.reshape(-1)
        na, nb = a.norm(), b.norm()
        if na < 1e-12 or nb < 1e-12:
            continue
        cos_l.append(float(torch.dot(a, b).abs() / (na * nb)))
        rel_l.append(float(na / nb))
        try:
            s = torch.linalg.svdvals(dW)
            p = (s ** 2)
            p = p / p.sum().clamp(min=1e-12)
            eff_l.append(float(torch.exp(-(p * torch.log(p + 1e-12)).sum())))
        except Exception:
            pass
    return {
        "dw_w0_cos": _mean(cos_l),
        "dw_eff_rank": _mean(eff_l),
        "dw_rel_norm": _mean(rel_l),
    }


def task_subspace_metrics(Hs: List[torch.Tensor], k: Optional[int] = None) -> dict:
    """
    任务级解耦：各任务是否占用**不同**的低秩维度子空间。

    这是「解耦 -> 抗遗忘」链条的关键一环：若任务 A、B 各自占用互不重叠的
    r 维子空间，则训练 B 时对 A 的写入干扰天然很小。

    Hs: 每个任务一个 (N_t, r) 特征矩阵
    """
    T = len(Hs)
    if T < 2:
        return {"task_overlap": float("nan"), "task_principal": float("nan"),
                "task_jaccard": float("nan"), "task_selectivity": float("nan")}
    r = Hs[0].shape[1]
    k = k if k is not None else max(1, r // 2)
    k = max(1, min(k, r))

    imps, pcs, tops = [], [], []
    for H in Hs:
        Hf = H.float()
        Hc = Hf - Hf.mean(dim=0, keepdim=True)
        v = Hc.var(dim=0, unbiased=False).clamp(min=1e-12)
        imps.append(v / v.sum())
        C = (Hc.t() @ Hc) / max(Hc.shape[0] - 1, 1)
        try:
            _, evec = torch.linalg.eigh(C)
            pcs.append(evec[:, -1])                      # 最大特征向量 = 主方向
        except Exception:
            pcs.append(torch.zeros(r))
        tops.append(set(torch.topk(v, k).indices.tolist()))

    cos_imp, cos_pc, jac = [], [], []
    for i in range(T):
        for j in range(i + 1, T):
            cos_imp.append(float(F.cosine_similarity(
                imps[i].unsqueeze(0), imps[j].unsqueeze(0)).item()))
            cos_pc.append(abs(float(torch.dot(pcs[i], pcs[j]))))
            jac.append(len(tops[i] & tops[j]) / float(k))

    # 任务选择性：每个维度在多大程度上编码「任务身份」。
    # 单因素方差分析的组间/总方差比
    #     sel_d = Var_tasks(mean_t h_d) / Var_total(h_d)
    # 与 task_overlap 互补：overlap 看的是「各任务是否用同一批维度」（会被
    # 预训练特征的共享统计量主导而普遍偏高），selectivity 看的是「单个维度
    # 是否对任务身份敏感」，后者才真正对应「任务因子被解耦到独立轴上」。
    means = torch.stack([H.float().mean(dim=0) for H in Hs])            # T x r
    grand = means.mean(dim=0)
    var_between = ((means - grand) ** 2).mean(dim=0)                    # r
    var_total = torch.stack(
        [H.float().var(dim=0, unbiased=False) for H in Hs]).mean(dim=0)  # r
    sel = (var_between / var_total.clamp(min=1e-12)).clamp(0.0, 1.0)

    return {
        "task_overlap": _mean(cos_imp),
        "task_principal": _mean(cos_pc),
        "task_jaccard": _mean(jac),
        "task_selectivity": float(sel.mean()),
    }


def aggregate_decoupling(H_per_task: Dict[str, List[torch.Tensor]],
                         tasks: List[str], entries: List[Entry]) -> dict:
    """
    汇总解耦度量：先对每个注入层单独算，再跨层取均值。

    这样热力图 / 谱才有跨层的统计意义；同时保留各任务的「维度重要性」
    用于画任务 x 维度占用图。
    """
    L = len(entries)
    if L == 0:
        return {}

    per_layer: List[List[dict]] = [[] for _ in range(L)]
    r_dim = None

    for t in tasks:
        Hs = H_per_task.get(t, [])
        for li in range(L):
            if li >= len(Hs) or Hs[li].numel() == 0 or Hs[li].shape[0] < 4:
                continue
            m = decoupling_metrics(Hs[li])
            if not m:
                continue
            per_layer[li].append(m)
            r_dim = m["_p"].shape[0]

    if r_dim is None:
        return {}

    # 任务维度重要性（跨层平均）
    task_imp = np.zeros((len(tasks), r_dim))
    for ti, t in enumerate(tasks):
        acc = np.zeros(r_dim)
        cnt = 0
        Hs = H_per_task.get(t, [])
        for li in range(L):
            if li < len(Hs) and Hs[li].numel() > 0 and Hs[li].shape[0] >= 4:
                Hf = Hs[li].float()
                v = Hf.var(dim=0, unbiased=False).numpy()
                s = v.sum()
                if s > 1e-12:
                    acc += v / s
                    cnt += 1
        task_imp[ti] = acc / max(cnt, 1)

    # 任务级解耦：跨层平均
    t_over, t_prin, t_jac, t_sel = [], [], [], []
    for li in range(L):
        Hs = [H_per_task[t][li] for t in tasks
              if li < len(H_per_task.get(t, [])) and H_per_task[t][li].shape[0] >= 4]
        if len(Hs) == len(tasks):
            ts = task_subspace_metrics(Hs)
            if math.isfinite(ts["task_overlap"]):
                t_over.append(ts["task_overlap"])
                t_prin.append(ts["task_principal"])
                t_jac.append(ts["task_jaccard"])
                t_sel.append(ts["task_selectivity"])

    keys = ["corr_abs", "corr_max", "TC", "MI", "eff_rank", "PR",
            "cond", "iso_err", "sparsity", "h_std"]
    flat = [m for pl in per_layer for m in pl]
    out = {k: float(np.mean([m[k] for m in flat])) for k in keys}
    out.update({
        "task_overlap": _mean(t_over),
        "task_principal": _mean(t_prin),
        "task_jaccard": _mean(t_jac),
        "task_selectivity": _mean(t_sel),
    })

    # 相关矩阵与谱：跨层跨任务平均，供画图。
    #
    # 相关矩阵必须「先取绝对值再平均」，不能「先平均再取绝对值」：
    # 不同层 / 不同任务的相关符号不一致，直接平均会让非对角元相互抵消，
    # 得到的相关矩阵虚假地接近单位阵（实测会把 0.356 低估成 0.154）。
    # 先取 |·| 后平均，热力图才与汇总表里的 corr_abs 自洽。
    mats = [m["_R"].abs() for m in flat]
    specs = [m["_p"] for m in flat]
    out["_corr_mat"] = torch.stack([x.cpu() for x in mats]).mean(0).numpy() if mats else None
    out["_spectrum"] = torch.stack([x.cpu() for x in specs]).mean(0).numpy() if specs else None
    out["_task_imp"] = task_imp
    return out


# ============================================================================
# 训练引擎
# ============================================================================

class ActivationCapture:
    """
    为每个 AeloruLayer 注册 forward hook，捕获其真实输入/输出。

    必须折叠为 2D：Transformer 隐藏状态是 (B, S, D)，
    而 AeloruLayer 的若干统计更新（侧向 / 波动率 / C_x）假定 2D，
    对 3D 做 mean(dim=0) 会得到 (S, D) 并触发维度不匹配。
    """

    def __init__(self, model):
        self.records: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.handles = []
        for layer in model.modules():
            if isinstance(layer, AeloruLayer):
                self.handles.append(layer.register_forward_hook(self._make_hook(layer)))

    def _make_hook(self, layer):
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


def batch_h_metrics(entries: List[Entry], capture: ActivationCapture) -> dict:
    """
    用当前训练 batch 的激活即时算一组解耦指标（训练过程中的轻量探针）。

    与最终度量用的是同一套公式，只是样本量只有当前 batch，
    用于观察解耦程度的**演化**而非给出结论性数值。
    """
    vals = []
    try:
        with preserve_dynamics(entries):
            for e in entries:
                if e.kind != "aeloru":
                    continue
                rec = capture.get(e.module)
                if rec is None:
                    continue
                h = h_from_entry(e, rec[0]).float()
                if h.shape[0] < 8:
                    continue
                m = decoupling_metrics(h)
                if m:
                    vals.append(m)
    except Exception:
        return {}
    if not vals:
        return {}
    return {k: float(np.mean([v[k] for v in vals]))
            for k in ["corr_abs", "TC", "eff_rank", "PR", "iso_err", "sparsity"]}


def _sample_mechanisms(entries: List[Entry], task: str, step: int, loss: float) -> dict:
    """采样 Aeloru v3.1 的机制指标（取所有注入层的均值）。"""
    layers = [e.module for e in entries if e.kind == "aeloru"]

    def avg(fn):
        vals = []
        for l in layers:
            try:
                v = fn(l)
                if v is not None and math.isfinite(v):
                    vals.append(v)
            except Exception:
                pass
        return float(np.mean(vals)) if vals else 0.0

    return {
        "task": task, "step": step, "loss": loss,
        "plasticity_out": avg(lambda l: l.plasticity_out.mean().item()),
        "plasticity_in":  avg(lambda l: l.plasticity_in.mean().item()),
        "global_direction": avg(lambda l: l.global_direction.item()),
        "structure_loss": avg(lambda l: l._last_structure_loss),
        "proto_out": avg(lambda l: l.proto_importance_out.mean().item()),
        "proto_in":  avg(lambda l: l.proto_importance_in.mean().item()),
        "bcm_cap_out": avg(lambda l: l._adaptive_saturation(True).mean().item()),
        "conflict": avg(lambda l: float(l.conflict_score.item())
                        if isinstance(l.conflict_score, torch.Tensor)
                        else float(l.conflict_score)),
        "volatility": avg(lambda l: float(torch.exp(l.omega.float()).item())),
        "state": layers[0].state.value if layers else "",
    }


def train_one_task(model, entries: List[Entry], capture: Optional[ActivationCapture],
                   loader: DataLoader, optimizer, device: str, steps: int,
                   task_name: str, mech_log: List[dict], dec_trace: List[dict],
                   global_step: int, ortho_lambda: float = 0.1,
                   sample_every: int = 10) -> Tuple[List[float], int, dict]:
    """在单个任务上训练若干步。返回 (loss 序列, 新的 global_step, 汇总统计)。"""
    lossf = nn.CrossEntropyLoss()
    aeloru_entries = [e for e in entries if e.kind == "aeloru"]
    it = iter(loader)
    losses = []
    ortho_trace = []
    agg = {"merge": 0, "states": {}}

    model.train()
    for s in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss_task = lossf(out.logits.float(), batch["labels"])
        loss = loss_task

        # Aeloru 的结构性正则：
        #   get_ortho_penalty() = DeltaW/W0 方向重叠惩罚 + SPHeRe 低秩结构保持损失
        # 不接入的话 SPHeRe 与正交约束完全不参与优化，_last_structure_loss 恒为 0。
        loss_ortho = 0.0
        if aeloru_entries:
            for e in aeloru_entries:
                loss_ortho = loss_ortho + e.module.get_ortho_penalty()
            loss = loss + ortho_lambda * loss_ortho

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        lv = loss.item()
        losses.append(lv)
        lo = float(loss_ortho.item()) if torch.is_tensor(loss_ortho) else 0.0
        ortho_trace.append(lo)

        # --- Aeloru 专有后处理 ---
        if aeloru_entries:
            for e in aeloru_entries:
                layer = e.module
                layer.observe_global_loss(lv)      # GHL：全局损失方向作神经调质
                if capture is not None:
                    rec = capture.get(layer)
                    if rec is not None:
                        merged = layer.post_step_update(
                            rec[0], rec[1], is_correct=True, merged=False, y_target=None)
                        if merged:
                            agg["merge"] += 1

        global_step += 1

        # --- 解耦指标即时探针 ---
        if capture is not None and (s % sample_every == 0 or s == steps - 1):
            dm = batch_h_metrics(aeloru_entries, capture)
            if dm:
                dm.update({"task": task_name, "step": s, "loss": lv})
                dec_trace.append(dm)

        if aeloru_entries and (s % 20 == 0 or s == steps - 1):
            row = _sample_mechanisms(entries, task_name, s, lv)
            row["loss_ortho"] = lo
            mech_log.append(row)

        for e in aeloru_entries:
            st = e.module.state.value
            agg["states"][st] = agg["states"].get(st, 0) + 1

    if capture is not None:
        capture.clear()
    agg["ortho"] = float(np.mean(ortho_trace)) if ortho_trace else 0.0
    return losses, global_step, agg


def trigger_sleep_consolidation(model) -> int:
    """任务边界显式触发 DLAM 睡眠巩固；返回发生离线重放的层数。"""
    from aeloru_v3 import AeloruModel
    layers = [m for m in model.modules() if isinstance(m, AeloruLayer)]
    if not layers:
        return 0
    cross_modal_sleep_step(AeloruModel(model))
    n = 0
    for layer in layers:
        try:
            layer._offline_replay()   # 谱滤波 + 弱连接修剪 + top-k 原型提取
            n += 1
        except Exception:
            pass
    return n


@torch.no_grad()
def evaluate(model, loader, device: str) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits
        correct += (logits.argmax(-1) == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    model.train()
    return correct / max(total, 1)


def compute_cl_metrics(R: np.ndarray, zero_shot: Optional[np.ndarray] = None,
                       chance: float = 0.5) -> Dict[str, float]:
    """
    R[i][j] = 任务 i 训练结束后，在任务 j (j<=i) 上的准确率（下三角有效）。
    注意 R 的上三角为 NaN，聚合前必须按列裁掉 NaN。
    """
    T = R.shape[0]
    final = R[T - 1]
    acc = float(np.mean(final))
    bwt = float(np.mean([R[T - 1, j] - R[j, j] for j in range(T)]))
    if zero_shot is not None:
        fwt = float(np.mean([zero_shot[j] - chance for j in range(1, T)]))
    else:
        fwt = float("nan")
    forget = float(np.mean([np.nanmax(R[j:, j]) - R[T - 1, j] for j in range(T)]))
    return {"ACC": acc, "BWT": bwt, "FWT": fwt, "Forgetting": forget}


# ============================================================================
# 单个方法的完整流程
# ============================================================================

def run_method(method: str, tasks: List[str], train_loaders, val_loaders, tok, args,
               out_dir: str, fh) -> dict:
    set_seed(args.seed)
    device = args.device
    T = len(tasks)

    log(f"\n{'='*78}\n方法: {method}\n{'='*78}", fh)

    model, n_trainable = build_model(
        method, args.layers, device, args.lr,
        n_labels=2, pad_id=tok.pad_token_id, r=args.r, alpha=args.alpha)

    entries = collect_entries(model)
    has_aeloru = any(e.kind == "aeloru" for e in entries)
    capture = ActivationCapture(model) if has_aeloru else None

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)

    log(f"  可训练参数: {n_trainable:,}   注入层: {len(entries)} "
        f"(aeloru={sum(1 for e in entries if e.kind == 'aeloru')})", fh)

    R = np.full((T, T), np.nan)
    zero_shot = np.zeros(T)
    mech_log: List[dict] = []
    dec_trace: List[dict] = []
    global_step = 0
    sleep_events = 0
    t_start = time.perf_counter()

    for i, task in enumerate(tasks):
        zero_shot[i] = evaluate(model, val_loaders[task], device)

        set_seed(args.seed + i)   # 每个任务独立的数据顺序，保证方法间可比
        losses, global_step, agg = train_one_task(
            model, entries, capture, train_loaders[task], optimizer,
            device, args.steps, task, mech_log, dec_trace, global_step,
            ortho_lambda=args.ortho_lambda, sample_every=args.sample_every)

        if has_aeloru:
            sleep_events += trigger_sleep_consolidation(model)

        for j in range(i + 1):
            acc = evaluate(model, val_loaders[tasks[j]], device)
            R[i, j] = acc
            log(f"  [任务{i+1}/{T} {task:5s}] loss {np.mean(losses[:20]):.3f}"
                f"->{np.mean(losses[-20:]):.3f} | 回测 {tasks[j]:5s} = {acc:.4f}", fh)

        log(f"    -> 零样本 {zero_shot[i]:.4f} | 已见均值 {np.nanmean(R[i, :i+1]):.4f}"
            f" | 睡眠 {sleep_events} | 合并 {agg['merge']} | ortho {agg.get('ortho', 0):.5f}", fh)

    # ---- 训练结束：正式的解耦度量 ----
    log("  采集低秩特征 h 并计算解耦度量 ...", fh)
    t_dec = time.perf_counter()
    H_per_task = collect_h_per_task(
        model, entries, val_loaders, tasks, device,
        h_mode=args.h_mode, max_rows=args.h_samples)
    dec = aggregate_decoupling(H_per_task, tasks, entries)
    dec.update(delta_w_metrics(entries))
    dec["_collect_s"] = time.perf_counter() - t_dec

    elapsed = time.perf_counter() - t_start
    metrics = compute_cl_metrics(R, zero_shot)

    # 落盘明细
    if mech_log:
        pd.DataFrame(mech_log).to_csv(os.path.join(out_dir, f"mech_{method}.csv"),
                                      index=False, encoding="utf-8-sig")
    if dec_trace:
        pd.DataFrame(dec_trace).to_csv(os.path.join(out_dir, f"dectrace_{method}.csv"),
                                       index=False, encoding="utf-8-sig")
    np.save(os.path.join(out_dir, f"R_{method}.npy"), R)
    if dec.get("_corr_mat") is not None:
        np.save(os.path.join(out_dir, f"corrmat_{method}.npy"), dec["_corr_mat"])
    if dec.get("_task_imp") is not None:
        np.save(os.path.join(out_dir, f"taskimp_{method}.npy"), dec["_task_imp"])

    log(f"  => ACC={metrics['ACC']:.4f}  BWT={metrics['BWT']:+.4f}  "
        f"Forgetting={metrics['Forgetting']:.4f}", fh)
    log(f"  => 解耦: corr_abs={dec.get('corr_abs', float('nan')):.4f}  "
        f"TC={dec.get('TC', float('nan')):.4f}  MI={dec.get('MI', float('nan')):.4f}  "
        f"eff_rank={dec.get('eff_rank', float('nan')):.3f}/{args.r}  "
        f"iso_err={dec.get('iso_err', float('nan')):.4f}", fh)
    log(f"  => 任务解耦: overlap={dec.get('task_overlap', float('nan')):.4f}  "
        f"principal={dec.get('task_principal', float('nan')):.4f}  "
        f"jaccard={dec.get('task_jaccard', float('nan')):.4f}  "
        f"| ΔW⊥W0 |cos|={dec.get('dw_w0_cos', float('nan')):.5f}", fh)

    if capture:
        capture.remove()
    del model, optimizer
    torch.cuda.empty_cache()

    return {"method": method, "R": R, "zero_shot": zero_shot, "metrics": metrics,
            "dec": dec, "mech": mech_log, "dec_trace": dec_trace,
            "n_trainable": n_trainable, "elapsed": elapsed, "sleep": sleep_events}


# ============================================================================
# 可视化
# ============================================================================

def _pick(results: List[dict], name: str) -> Optional[dict]:
    for r in results:
        if r["method"] == name:
            return r
    return None


def make_decoupling_figure(results: List[dict], tasks: List[str], args, out_dir: str) -> str:
    """主图 2x3：相关矩阵对比 / 谱 / 核心指标 / 有效秩 / ΔW 正交性。"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    r_dim = args.r

    # --- (0,0)(0,1) 相关矩阵热力图：LoRA vs Aeloru ---
    for ax, name in zip([axes[0][0], axes[0][1]], ["lora", "aeloru"]):
        rr = _pick(results, name)
        if rr is None or rr["dec"].get("_corr_mat") is None:
            ax.set_title(f"Feature correlation — {name}", fontsize=11)
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        # _corr_mat 已是「先取绝对值再跨层/跨任务平均」，直接用即可
        M = rr["dec"]["_corr_mat"]
        off = M - np.eye(r_dim)
        im = ax.imshow(M, cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
        ax.set_title(f"Mean |correlation| of h — {name}", fontsize=11)
        ax.set_xticks(range(r_dim)); ax.set_yticks(range(r_dim))
        ax.set_ylabel("low-rank dim")
        ax.set_xlabel(f"low-rank dim   mean|off-diag|={off.sum()/max(r_dim*(r_dim-1),1):.3f}",
                      fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

    # --- (0,2) 归一化协方差特征值谱 ---
    ax = axes[0][2]
    ax.axhline(1.0, ls="--", c="k", lw=1, label="ideal (isotropic)")
    for rr in results:
        sp = rr["dec"].get("_spectrum")
        if sp is None:
            continue
        sp = np.sort(np.asarray(sp))[::-1]
        ax.plot(range(1, len(sp) + 1), sp, marker="o", ms=3, label=rr["method"])
    ax.set_xlabel("eigenvalue index"); ax.set_ylabel("normalized eigenvalue")
    ax.set_title("Covariance spectrum of h (flat = decoupled)", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    # --- (1,0) 核心解耦指标（越低越好，按 lora 归一化）---
    ax = axes[1][0]
    keys = ["corr_abs", "TC", "MI", "iso_err"]
    x = np.arange(len(keys)); w = 0.8 / max(len(results), 1)
    base = _pick(results, "lora")
    for k, rr in enumerate(results):
        vals = []
        for kk in keys:
            v = rr["dec"].get(kk, np.nan)
            if base is not None and np.isfinite(v):
                bv = base["dec"].get(kk, np.nan)
                if np.isfinite(bv) and abs(bv) > 1e-12:
                    v = v / bv
            vals.append(v if np.isfinite(v) else 0.0)
        ax.bar(x + k * w, vals, w, label=rr["method"])
    ax.set_xticks(x + w * (len(results) - 1) / 2)
    ax.set_xticklabels(keys, fontsize=9)
    ax.axhline(1.0, c="r", ls=":", lw=1, label="lora baseline")
    ax.set_ylabel("relative to lora (lower = more decoupled)")
    ax.set_title("Core disentanglement metrics (lower = better)", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")

    # --- (1,1) 有效秩 / 参与率 ---
    ax = axes[1][1]
    keys2 = ["eff_rank", "PR"]
    x = np.arange(len(keys2)); w = 0.8 / max(len(results), 1)
    for k, rr in enumerate(results):
        vals = [rr["dec"].get(kk, np.nan) for kk in keys2]
        ax.bar(x + k * w, [0.0 if not np.isfinite(v) else v for v in vals], w,
               label=rr["method"])
    ax.set_xticks(x + w * (len(results) - 1) / 2); ax.set_xticklabels(keys2)
    ax.axhline(args.r, c="k", ls="--", lw=1, label=f"max (r={args.r})")
    ax.set_ylabel("value (higher = less collapsed)")
    ax.set_title("Effective rank / participation ratio (higher = better)", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")

    # --- (1,2) ΔW 与 W0 的正交性 + ΔW 有效秩 ---
    ax = axes[1][2]
    names = [rr["method"] for rr in results]
    cos_v = [rr["dec"].get("dw_w0_cos", np.nan) for rr in results]
    eff_v = [rr["dec"].get("dw_eff_rank", np.nan) for rr in results]
    xx = np.arange(len(names))
    ax.bar(xx - 0.2, [0.0 if not np.isfinite(v) else v for v in cos_v], 0.4,
           label="|cos(dW, W0)|  (lower = better)")
    ax2 = ax.twinx()
    ax2.bar(xx + 0.2, [0.0 if not np.isfinite(v) else v for v in eff_v], 0.4,
            color="orange", alpha=.75, label="dW eff. rank (higher = better)")
    ax.set_xticks(xx); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|cos|"); ax2.set_ylabel("eff. rank")
    ax.set_title("Delta W vs pretrained W0", fontsize=11)
    ax.legend(fontsize=7, loc="upper left"); ax2.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=.3, axis="y")

    plt.suptitle("Aeloru v3.1 Feature Decoupling Test on Qwen2.5-0.5B "
                 f"(r={args.r}, layers={args.layers}, h-mode={args.h_mode})", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, "decoupling.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def make_ablation_figure(results: List[dict], out_dir: str) -> Optional[str]:
    """
    消融归因图：每个方法相对 `aeloru`（全开）的指标变化。
    负向指标（corr/TC/MI/cond/iso/overlap）取负号，统一成「向上 = 更解耦」。
    """
    full = _pick(results, "aeloru")
    if full is None:
        return None
    others = [r for r in results if r["method"] != "aeloru"]
    if not others:
        return None

    keys = [("corr_abs", False), ("TC", False), ("MI", False), ("iso_err", False),
            ("cond", False), ("eff_rank", True), ("PR", True),
            ("dw_w0_cos", False), ("task_overlap", False), ("task_jaccard", False),
            ("task_selectivity", True)]
    keys = [(k, d) for k, d in keys if k in full["dec"]]
    if not keys:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    ax = axes[0]
    x = np.arange(len(keys)); w = 0.8 / max(len(others), 1)
    for k, rr in enumerate(others):
        deltas = []
        for kk, bigger in keys:
            a = full["dec"].get(kk, np.nan)
            b = rr["dec"].get(kk, np.nan)
            if not (np.isfinite(a) and np.isfinite(b)):
                deltas.append(0.0)
                continue
            base = abs(a) if abs(a) > 1e-12 else 1.0
            d = (b - a) / base
            deltas.append(d if bigger else -d)
        ax.bar(x + k * w, deltas, w, label=rr["method"])
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(x + w * (len(others) - 1) / 2)
    ax.set_xticklabels([k for k, _ in keys], rotation=30, ha="right")
    ax.set_ylabel("relative change vs aeloru (up = more decoupled)")
    ax.set_title("Ablation attribution: what each mechanism contributes", fontsize=12)
    ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")

    ax = axes[1]
    keys_cl = ["ACC", "BWT", "Forgetting"]
    x = np.arange(len(keys_cl)); w = 0.8 / max(len(others), 1)
    for k, rr in enumerate(others):
        deltas = [rr["metrics"][kk] - full["metrics"][kk] for kk in keys_cl]
        ax.bar(x + k * w, deltas, w, label=rr["method"])
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(x + w * (len(others) - 1) / 2); ax.set_xticklabels(keys_cl)
    ax.set_ylabel("delta vs aeloru")
    ax.set_title("CL metrics change (Forgetting up = worse)", fontsize=12)
    ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")

    plt.tight_layout()
    p = os.path.join(out_dir, "ablation.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def make_task_figure(results: List[dict], tasks: List[str], args, out_dir: str) -> Optional[str]:
    """任务 x 维度重要性热力图 + 任务子空间重叠度。"""
    valid = [r for r in results if r["dec"].get("_task_imp") is not None]
    if not valid:
        return None
    n = len(valid)
    ncols = min(n, 4)
    nrows = int(math.ceil(n / ncols))
    fig = plt.figure(figsize=(4.2 * ncols + 3.4, 3.4 * nrows))
    # 前 ncols 列放各方法的「任务 x 维度」热力图，最后一列整列跨行放重叠度柱状图
    gs = fig.add_gridspec(nrows, ncols + 1)

    for i, rr in enumerate(valid):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        M = rr["dec"]["_task_imp"]
        im = ax.imshow(M, cmap="viridis", aspect="auto")
        ax.set_yticks(range(len(tasks))); ax.set_yticklabels(tasks, fontsize=8)
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(range(M.shape[1]), fontsize=7)
        ax.set_title(f"{rr['method']}", fontsize=10)
        ax.set_xlabel("low-rank dim", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[:, ncols])
    keys = ["task_overlap", "task_principal", "task_jaccard"]
    x = np.arange(len(keys)); w = 0.8 / max(n, 1)
    for k, rr in enumerate(valid):
        vals = [rr["dec"].get(kk, np.nan) for kk in keys]
        ax.bar(x + k * w, [0.0 if not np.isfinite(v) else v for v in vals], w,
               label=rr["method"])
    ax.set_xticks(x + w * (n - 1) / 2); ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("overlap (lower = better)")
    ax.set_title("Task subspace overlap", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")

    plt.suptitle("Per-task occupancy of low-rank dimensions "
                 "(disjoint rows = tasks decoupled)", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, "task_decoupling.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def make_relation_figure(results: List[dict], args, out_dir: str) -> Optional[str]:
    """
    关系图：解耦程度 vs 抗遗忘。检验核心假设——
    解耦越强（TC / corr_abs / task_overlap 越低），遗忘是否越少。
    """
    pts = []
    for rr in results:
        tc = rr["dec"].get("TC", np.nan)
        ca = rr["dec"].get("corr_abs", np.nan)
        ov = rr["dec"].get("task_overlap", np.nan)
        fw = rr["metrics"]["Forgetting"]
        bw = rr["metrics"]["BWT"]
        if all(np.isfinite(v) for v in (tc, ca, ov, fw, bw)):
            pts.append((rr["method"], tc, ca, ov, fw, bw))
    if len(pts) < 2:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    specs = [("TC", 1), ("corr_abs", 2), ("task_overlap", 3)]

    for ax, (label, idx) in zip(axes, specs):
        xs = [p[idx] for p in pts]
        ys_f = [p[4] for p in pts]
        ys_b = [p[5] for p in pts]
        ax.scatter(xs, ys_f, s=60, c="tab:red", label="Forgetting (lower better)")
        ax.scatter(xs, ys_b, s=60, c="tab:blue", marker="^", label="BWT (higher better)")
        for p in pts:
            ax.annotate(p[0], (p[idx], p[4]), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        title = f"{label} vs forgetting"
        if len(xs) > 2:
            try:
                xs_sorted = np.linspace(min(xs), max(xs), 20)
                c1 = np.polyfit(xs, ys_f, 1)
                ax.plot(xs_sorted, np.polyval(c1, xs_sorted), "r--", lw=1, alpha=.7)
                c2 = np.polyfit(xs, ys_b, 1)
                ax.plot(xs_sorted, np.polyval(c2, xs_sorted), "b--", lw=1, alpha=.7)
                r_f = float(np.corrcoef(xs, ys_f)[0, 1])
                r_b = float(np.corrcoef(xs, ys_b)[0, 1])
                title += f"   r(F)={r_f:+.2f}  r(BWT)={r_b:+.2f}"
            except Exception:
                pass
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"{label} (lower = more decoupled)")
        ax.legend(fontsize=8); ax.grid(alpha=.3)

    plt.suptitle("Does decoupling explain anti-forgetting?", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, "decoupling_vs_forgetting.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


def make_evolution_figure(results: List[dict], out_dir: str) -> Optional[str]:
    """解耦指标在训练过程中的演化（竖虚线 = 任务切换点）。"""
    has = [r for r in results if r.get("dec_trace")]
    if not has:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    panels = [("corr_abs  (lower = more decoupled)", "corr_abs"),
              ("TC  (lower = more decoupled)", "TC"),
              ("effective rank  (higher = better)", "eff_rank"),
              ("iso_err  (lower = more decoupled)", "iso_err")]

    for ax, (title, col) in zip(axes.ravel(), panels):
        for rr in has:
            df = pd.DataFrame(rr["dec_trace"])
            if col not in df.columns:
                continue
            df["g"] = np.arange(len(df))
            ax.plot(df["g"].values, df[col].values, lw=1.1, label=rr["method"])
            if col == "corr_abs":
                for b in df.groupby("task", sort=False)["g"].max().values[:-1]:
                    ax.axvline(b, c="r", ls=":", lw=.8)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7); ax.grid(alpha=.3)
        ax.set_xlabel("sampling index")

    plt.suptitle("Decoupling dynamics during training (red dotted = task switch)",
                 fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, "evolution.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    return p


# ============================================================================
# 主流程
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Aeloru v3.1 特征解耦测试")
    ap.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    ap.add_argument("--tasks", type=str, default="sst2,mrpc,rte")
    ap.add_argument("--steps", type=int, default=120, help="每个任务的训练步数")
    ap.add_argument("--layers", type=int, default=4, help="注入的 Transformer 层数")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=8, help="低秩维度（解耦发生在这个子空间）")
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--ortho-lambda", type=float, default=0.1,
                    help="正交惩罚 + SPHeRe 结构保持损失的权重（Aeloru 专用）")
    ap.add_argument("--h-mode", type=str, default="train", choices=["train", "eval"],
                    help="采集 h 时 WTA/PEM 是否生效（见模块文档）")
    ap.add_argument("--h-samples", type=int, default=4096,
                    help="每个任务每层用于统计的 h 样本数上限")
    ap.add_argument("--sample-every", type=int, default=10,
                    help="训练过程中解耦探针的采样间隔")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"未知方法 {m}，可选: {METHODS}")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in ALL_TASKS:
            raise ValueError(f"未知任务 {t}，可选: {ALL_TASKS}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       f"ex2_decouple_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.txt")
    fh = open(log_path, "w", encoding="utf-8")

    log("=" * 78, fh)
    log("Aeloru v3.1 特征解耦测试 (Feature Decoupling)", fh)
    log(f"模型: {MODEL_PATH}", fh)
    log(f"任务序列: {' -> '.join(tasks)}", fh)
    log(f"方法: {methods}", fh)
    log(f"步数/任务={args.steps} 层数={args.layers} bs={args.bs} seq={args.seq} "
        f"lr={args.lr} r={args.r} alpha={args.alpha} seed={args.seed}", fh)
    log(f"h-mode={args.h_mode}  h-samples={args.h_samples}  "
        f"ortho_lambda={args.ortho_lambda}", fh)
    log("=" * 78, fh)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- 数据 ----
    train_sets, val_sets = {}, {}
    for t in tasks:
        tr, va = build_task_dataset(tok, t, args.seq, args.n_train, args.n_val, args.seed)
        train_sets[t], val_sets[t] = tr, va
        log(f"  数据 {t:5s}: train={len(tr):5d} val={len(va):4d}", fh)

    collate = DataCollatorWithPadding(tok)
    train_loaders = {t: DataLoader(train_sets[t], batch_size=args.bs, shuffle=True,
                                   collate_fn=collate, drop_last=True) for t in tasks}
    val_loaders = {t: DataLoader(val_sets[t], batch_size=32, shuffle=False,
                                 collate_fn=collate) for t in tasks}

    # ---- 运行 ----
    results = []
    for m in methods:
        r = run_method(m, tasks, train_loaders, val_loaders, tok, args, out_dir, fh)
        results.append(r)

    # ---- 汇总 ----
    log(f"\n{'='*78}\n汇总\n{'='*78}", fh)
    rows = []
    for r in results:
        row = {"method": r["method"],
               "ACC": round(r["metrics"]["ACC"], 4),
               "BWT": round(r["metrics"]["BWT"], 4),
               "FWT": round(r["metrics"]["FWT"], 4),
               "Forgetting": round(r["metrics"]["Forgetting"], 4)}
        for k, _ in DECOL_METRICS:
            v = r["dec"].get(k, float("nan"))
            row[k] = round(float(v), 5) if np.isfinite(v) else np.nan
        row["params"] = r["n_trainable"]
        row["seconds"] = round(r["elapsed"], 1)
        rows.append(row)
        log(f"  {r['method']:16s} ACC={r['metrics']['ACC']:.4f}  "
            f"BWT={r['metrics']['BWT']:+.4f}  Forg={r['metrics']['Forgetting']:.4f}  "
            f"| corr_abs={r['dec'].get('corr_abs', float('nan')):.4f}  "
            f"TC={r['dec'].get('TC', float('nan')):.4f}  "
            f"eff_rank={r['dec'].get('eff_rank', float('nan')):.3f}", fh)

    df = pd.DataFrame(rows)
    csv_p = os.path.join(out_dir, "summary.csv")
    df.to_csv(csv_p, index=False, encoding="utf-8-sig")

    # R 矩阵明细
    with open(os.path.join(out_dir, "R_matrix.txt"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"\n=== {r['method']} ===\n")
            f.write("        " + "".join(f"{t:>8s}" for t in tasks) + "\n")
            for i, t in enumerate(tasks):
                f.write(f"{t:>7s} " + "".join(
                    f"{r['R'][i, j]:8.4f}" if not np.isnan(r["R"][i, j]) else "     n/a"
                    for j in range(len(tasks))) + "\n")

    pngs = [make_decoupling_figure(results, tasks, args, out_dir),
            make_ablation_figure(results, out_dir),
            make_task_figure(results, tasks, args, out_dir),
            make_relation_figure(results, args, out_dir),
            make_evolution_figure(results, out_dir)]
    log(f"\n结果已保存:\n  {csv_p}\n" + "\n".join(f"  {p}" for p in pngs if p), fh)
    fh.close()

    show = ["method", "ACC", "BWT", "Forgetting"]
    print("\n" + df[show].to_string(index=False))
    print("\n--- 解耦度量 ---")
    print(df[["method"] + [k for k, _ in DECOL_METRICS]].to_string(index=False))
    return results


if __name__ == "__main__":
    main()
