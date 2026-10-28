
# -*- coding: utf-8 -*-
"""
概念漂移 (Concept Drift)长线 测试：Aeloru v4 vs LoRA
==================================================

用 Qwen2.5-0.5B 在 GLUE 任务序列上构造**任务级概念漂移**：

    SST2 -> MRPC -> QNLI -> RTE -> CoLA -> SST2 -> ...

每个任务训练固定步数后**立刻切换**到下一个任务（输入分布与标签语义同时突变），
且**永不回访**旧任务数据。每完成一个任务，回测**所有已见任务**，得到准确率矩阵
R[i][j]（任务 i 训练结束后，在任务 j 上的准确率）。

持续学习标准指标
----------------
- ACC        最终平均准确率            mean_j R[T][j]
- BWT        向后迁移(负值=遗忘)       mean_j (R[T][j] - R[j][j])
- FWT        向前迁移                  mean_j (R[j-1][j] - random_j)
- Forgetting 遗忘率                    mean_j (max_i R[i][j] - R[T][j])

方法对比
--------
- lora            : peft LoRA 基线（无抗遗忘机制）
- aeloru-safe     : Aeloru 论文 §4.1.3 推荐配置（PEFT 持续学习最优），
                    关闭对预训练表示有害的组件：
                    预测编码 ndi=0(关闭全输出空间 PEM) / 域约束 none /
                    稳态增益上限 2.0(按 llm_tuning 预设，非 1.0，且保持开启)
- aeloru-default  : Aeloru llm_tuning 出厂默认（低秩 PEM 开启、稳态增益上限 2.0）
- aeloru-fromscratch: 从零训练组（论文 §3.7.3 cognitive_model 预设），
                    冻结 Qwen 词嵌入前端 + 从零 Kaiming 初始化的 Aeloru MLP；
                    可训练参数预算对齐 aeloru-safe（≈66,304），W0 仅由 ReLoRA 合并更新；
                    按 Aeloru-fromscratch.md 第1档修复：ndi=0(关 PEM)、ortho_lambda=0.03、
                    lr=3e-4、merge_every=200、red_threshold=2.0、homeostatic_max_gain=2.0。

Aeloru v4 机制监控
--------------------
每步记录 plasticity_out/in（双一维 HGF 精度门控）、global_direction（GHL 全局方向）、
loss_structure（SPHeRe 结构保持）、认知状态、原型重要性（DLAM 原型记忆）、
睡眠触发次数，用于验证五项新增机制在漂移下是否按设计响应。

用法
----
    python ex1.py                                       # 全量：3 方法 x 5任务*(repeat)轮 x 250 步
    python ex1.py --methods lora,aeloru-safe --steps 100
    python ex1.py --methods aeloru-safe --layers 4 --seq 64 --bs 16
    python ex1.py --repeat 3 --runs 2                   # 长线重复 3 轮、完整实验跑 2 次
    python ex1.py --methods lora --seed 7 --device cpu  # 单方法/指定种子/CPU
    python ex1.py --seed 42,9589,6519,1412             # 论文 §4.1.3 四种子（默认）
    python ex1.py --methods aeloru-fromscratch --device cpu   # 单跑从零训练组
    # 长线抗遗忘推荐参数（ex1.1）：合并+余弦退火+逐轮衰减+加强睡眠
    python ex1_long_line.py --methods lora,aeloru-safe,aeloru-default \\
        --repeat 2 --steps 250 --merge-every 500 --lr-schedule 1 \\
        --lr-decay-per-round 0.8 --sleep-intensity 2 --fs-relora-every 250

@author JYIMU
"""

import os
import sys
import time
import math
import random
import argparse
import warnings
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

from aeloru_v4 import AeloruConfig, AeloruLayer, AeloruModel, cross_modal_sleep_step

# ============================================================================
# 常量
# ============================================================================

MODEL_PATH = r"E:\Qelys\models\Qwen2.5-0.5B"

# Qwen2.5-0.5B 的 Transformer 总层数（用于按层号注入/选择 LoRA 目标层）。
N_TOTAL_LAYERS = 24

# GLUE 五任务：全部为二分类，可共享分类头，
# 从而把「分类头漂移」与「骨干遗忘」一并计入遗忘指标（更严格的协议）。

# TASKS: List[Tuple[str, int]] = [
#     ("sst2", 2),
#     ("mrpc", 2),
#     ("qnli", 2),
#     ("rte",  2),
#     ("cola", 2),
# ]
TASKS: List[Tuple[str, int]] = [
    ("cola", 2),
    ("rte",  2),
    ("qnli", 2), 
    ("mrpc", 2), 
    ("sst2", 2),                          
]                               

# 各任务的字段组合不同，需分别处理
TASK_FIELDS = {
    "sst2": ("sentence",),
    "cola": ("sentence",),
    "mrpc": ("sentence1", "sentence2"),
    "rte":  ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
}

METHODS = ["lora", "aeloru-safe", "aeloru-default", "aeloru-fromscratch"]


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


# ============================================================================
# 数据：预编码全部任务，构造漂移数据流
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

    # 类别数校验（datasets>=3 的 Column 对象无 .tolist()，用 unique）
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

    不直接用 inject_aeloru，因为它对全部 24 层做递归匹配无法按层过滤；
    层数是本实验的关键变量（层数过多时 Aeloru 的稳态增益会压垮残差流）。
    """
    n = 0
    for idx in layer_indices:
        attn = model.model.layers[idx].self_attn
        orig = getattr(attn, target, None)
        if not isinstance(orig, nn.Linear):
            continue
        layer = AeloruLayer(orig.in_features, orig.out_features, cfg, original_linear=orig)
        # 关键：set_pretrained_weight 只搬运 buffer，不搬运新创建的 parameter，
        # 因此必须在 inject 之后统一 .to(device)。
        layer.set_pretrained_weight(orig.weight.data, getattr(orig, "bias", None))
        setattr(attn, target, layer)
        n += 1
    return n


def build_model(method: str, n_layers: int, device: str, lr: float,
                n_labels: int = 2, pad_id: int = 0,
                r: int = 8, alpha: float = 16.0, merge_every: int = 0,
                fs_relora_every: int = 0, fs_enable_sleep: int = 0):
    """构建待训练模型；返回 (model, 可训练参数量)。"""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=n_labels, trust_remote_code=True)
    model.config.pad_token_id = pad_id

    # 注入/选择靠近输出端的 n_layers 层（保留浅层预训练表示）。
    layer_indices = list(range(N_TOTAL_LAYERS - n_layers, N_TOTAL_LAYERS))

    if method == "lora":
        from peft import LoraConfig, get_peft_model, TaskType
        model = get_peft_model(model, LoraConfig(
            r=r, lora_alpha=alpha, target_modules=["q_proj"],
            lora_dropout=0.0, bias="none", task_type=TaskType.SEQ_CLS,
            layers_to_transform=layer_indices,
        ))
        model.to(device)

    elif method == "aeloru-safe":
        # 论文 §3.7.3 / §4.1.3 推荐配置（PEFT 持续学习最优）：
        #   neural_dynamics_iterations=0  -> 关闭全输出空间 PEM（预测编码会把输出放大/归零）
        #   presumed_domain="none"        -> 关闭硬域约束 clamp(0,1)（截断负激活）
        #   homeostatic_max_gain=2.0      -> 稳态增益按 llm_tuning 预设上限 2.0（非 1.0），且保持开启
        cfg = AeloruConfig.llm_tuning(
            r=r, lora_alpha=alpha, LoRA_lr=lr,
            neural_dynamics_iterations=0,
            presumed_domain="none",
            homeostatic_max_gain=2.0,
            whitening_interval=10 ** 9,
            merge_every=10 ** 9,
            logging_enabled=False,
            diagnostic_enabled=False,
            device=device,
        )
        if merge_every > 0:
            cfg.merge_every = merge_every
        inject_aeloru_selected(model, layer_indices, cfg)
        model.to(device)
        _freeze_base(model)

    elif method == "aeloru-default":
        # V4 llm_tuning 出厂默认（低秩 PEM 开启，pem_space="output"）：
        # domain="none"/gain=2.0 来自预设，ndi 由显式=2 覆盖；保留 ReLoRA 合并观察长期行为。
        # 保留 ReLoRA 合并以观察长期行为。
        cfg = AeloruConfig.llm_tuning(
            r=r, lora_alpha=alpha, LoRA_lr=lr,
            logging_enabled=False,
            diagnostic_enabled=False,
            device=device,
            homeostatic_enabled=True,
            pem_space="output",
            neural_dynamics_iterations=2,
        )
        if merge_every > 0:
            cfg.merge_every = merge_every
        inject_aeloru_selected(model, layer_indices, cfg)
        model.to(device)
        _freeze_base(model)

    elif method == "aeloru-fromscratch":
        # 从零训练组（论文 §3.7.3 cognitive_model 预设）：冻结 Qwen 词嵌入作前端
        # （不计入可训练参数），后接从零 Kaiming 初始化的 Aeloru MLP。W0 仅由周期性
        # ReLoRA 合并更新（不接收 BP 梯度）。可训练参数预算对齐 aeloru-safe（≈66,304）：
        # 4 层 896→896 的 LoRA(r=8)+Hi-DoRA + 分类头。
        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH, num_labels=n_labels, trust_remote_code=True)
        embed = base.model.embed_tokens
        del base
        cfg = make_fs_cfg(device, r, alpha, 3e-4)  # fromscratch 修复⑤：从零训练 lr 高于微调(1e-4)
        hid = embed.embedding_dim
        net = FromScratchNet([hid] + [hid] * n_layers + [n_labels], cfg, device)
        model = FromScratchWrapper(embed, net, device)
        # 保存 fromscratch 专属参数到 model 实例，供 run_method 读取
        model.fs_relora_every = fs_relora_every
        model.fs_enable_sleep = bool(fs_enable_sleep)
        model.to(device)

    else:
        raise ValueError(f"未知方法: {method}")

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
# 从零训练实验组（Aeloru cognitive_model 预设，论文 §3.7.3 最优配置）
# ----------------------------------------------------------------------------
# 设计：冻结 Qwen 词嵌入作前端特征提取（不计入可训练参数），后接从零 Kaiming
# 初始化的 Aeloru MLP（W0 仅由周期性 ReLoRA 合并更新，不接收 BP 梯度），可训练
# 参数预算与 aeloru-safe 对齐（≈66,304）：4 层 896→896 的 LoRA(r=8)+Hi-DoRA + 头。
# 这样可在同一 GLUE 概念漂移协议下，公平比较「从零训练的认知模型」与「PEFT 微调」。
# 参考 ex4_fromscratch.py / ex4_fromscratch_1.py 的 FromScratchNet + ReLoRA 方案。
# ============================================================================

def make_fs_cfg(device: str, r: int, alpha: float, lr: float) -> AeloruConfig:
    """从零训练（cognitive_model）预设：论文 §3.7.3 推荐的从零认知模型配置。"""
    fields = dict(
        pem_space="output",                 # 论文预设：全输出空间 PEM（ndi=0 时实际关闭）
        neural_dynamics_iterations=0,       # fromscratch 修复①：关 PEM（正反馈爆炸根源）
        presumed_domain="nnantisparse",     # 论文预设
        homeostatic_enabled=True,
        homeostatic_max_gain=2.0,           # fromscratch 修复②：放宽稳态增益上限，冷启动统计量不准
        whitening_interval=500,             # 论文预设
        wta_k_ratio=0.5,
        sphere_lambda=0.01,
        ortho_lambda=0.03,                  # fromscratch 修复③：弱正交（W0 未成熟时空满强度会摁死训练）
        red_threshold=0.5,                  # fromscratch 修复④：红温阈值降低，避免全程卡死 red 状态
        ortho_lambda_anchor=0.05,
        fisher_gamma=0.5,
        prototype_guard=0.5,
        hebbian_lr=1e-6,
        lowrank_whiten_coef=0.1,
        lowrank_cond_max=5.0,
        lambda_lateral=0.9,
        lateral_lr=1e-5,
        lateral_decay=0.99,
        merge_every=100,                
        min_steps_between_sleep=100,    
        sleep_condition_threshold=0.5,
        energy_eta=0.15,
        predictive_coding_rank=16,
        gamma_predictive=1.0,
        selective_pem = True,
        dynamic_rank= True,
        adaptive_threshold = True,
        logging_enabled=False,
        diagnostic_enabled=False,
        AMP_DTYPE=torch.float32,            # 从零训练强制 float32，避免 bf16/fp16 不稳定
    )
    fields.update(dict(r=r, lora_alpha=alpha, LoRA_lr=lr, device=device))
    return AeloruConfig(**fields)


class FromScratchNet(nn.Module):
    """从零训练网络：若干 AeloruLayer 隐藏层 + 线性输出头（镜像 ex4_fromscratch）。

    W0 为 ReLoRA 基底（Kaiming 初始化，requires_grad=False，仅由周期性合并更新）；
    LoRA 路径是唯一接收 BP 梯度的部分。每层后接 GELU 保证表达力。
    """

    def __init__(self, dims: List[int], cfg: AeloruConfig, device: str):
        super().__init__()
        self.dims = dims
        self.aeloru_layers: List[AeloruLayer] = nn.ModuleList()
        for i in range(len(dims) - 2):  # 最后一层是输出头
            lay = AeloruLayer(dims[i], dims[i + 1], cfg)
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
            x = F.gelu(x)
        return self.head(x)


class FromScratchWrapper(nn.Module):
    """把 FromScratchNet 包成与 transformers 模型同接口：接受 (input_ids, attention_mask)。

    冻结的 Qwen 词嵌入作前端（不计入可训练参数），对真实 token 做 masked mean-pool，
    再送入 FromScratchNet。forward 返回带 .logits 的对象，兼容 ex1 训练/评估代码。
    """

    do_sleep = False  # 从零组关闭 DLAM 睡眠

    def __init__(self, embed: nn.Embedding, net: FromScratchNet, device: str):
        super().__init__()
        self.embed = embed
        self.embed.weight.requires_grad_(False)
        self.net = net
        self.to(device)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)                       # (B, S, D)
        mask = attention_mask.unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        logits = self.net(x)
        return type("Out", (), {"logits": logits})()


def relora_merge_into_W0(layer: "AeloruLayer", opt) -> None:
    """ReLoRA 合并：把 LoRA ΔW 折入 W0（W0 唯一更新途径），重置适配器并清优化器动量。"""
    with torch.no_grad():
        layer._flush_hebbian()
        delta_w = layer.compute_delta_w()
        delta_w = layer.apply_hidora(delta_w)
        delta_w = layer.apply_plasticity_gate(delta_w)
        delta_w = layer.apply_energy_budget(delta_w)
        layer.W0.data.add_(delta_w.detach().to(layer.W0.dtype))
        layer._reset_adapters()
        layer.W0_norm_sq = torch.tensor(
            float((layer.W0.detach().float() ** 2).sum()), dtype=torch.float32)
    layer.step_counter = 0
    layer._explore_start_step = 0
    layer.clear_optimizer_state(opt)
    if layer.W0 in opt.state:
        del opt.state[layer.W0]


def _recondition_layer(lay):
    """给协方差/白化缓冲加对角抖动，缓解 eig/slogdet 病态（镜像 ex4）。"""
    for name in ("C_h", "C_hh", "C_x", "W_whiten"):
        buf = getattr(lay, name, None)
        if isinstance(buf, torch.Tensor) and buf.dim() == 2 and buf.shape[0] == buf.shape[1]:
            j = 1e-4 * buf.diagonal().abs().max().clamp(min=1e-6)
            buf.add_(j * torch.eye(buf.shape[0], device=buf.device, dtype=buf.dtype))


# ============================================================================
# 逐层输入捕获：为 post_step_update 提供真实激活
# ============================================================================

class ActivationCapture:
    """
    为每个 AeloruLayer 注册 forward hook，捕获其真实输入/输出。

    必须折叠为 2D：Transformer 隐藏状态是 (B, S, D)，
    而 AeloruLayer 的若干统计更新（lateral / volatility / C_x）假定 2D，
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


# ============================================================================
# 训练引擎
# ============================================================================

def train_one_task(model, aeloru_layers: List[AeloruLayer], capture: ActivationCapture,
                   loader: DataLoader, optimizer, device: str, steps: int,
                   method: str, mech_log: List[dict], task_name: str,
                   global_step: int, ortho_lambda: float = 0.1,
                   grad_clip: float = 1.0, relora_every: int = 0,
                   lr: float = 1e-4, lr_schedule: bool = False,
                   ortho_warmup_steps: int = 0) -> Tuple[List[float], int, dict]:
    """在单个任务上训练若干步。返回 (loss 序列, 新的 global_step, 汇总统计)。"""
    lossf = nn.CrossEntropyLoss()
    # 可训练参数列表只构建一次（梯度裁剪每步都用到）。
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    has_aeloru = bool(aeloru_layers)

    # 每任务余弦学习率退火：任务末期 lr 降到 10%，减少后期梯度扰动
    scheduler = None
    if lr_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=lr * 0.1
        )

    it = iter(loader)
    losses = []
    ortho_trace = []
    agg = {"sleep": 0, "merge": 0, "states": {}}

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

        # Aeloru 的结构性正则：
        #   get_ortho_penalty() = DeltaW/W0 方向重叠惩罚 + SPHeRe 低秩结构保持损失
        # 不接入的话 SPHeRe 与正交约束完全不参与优化，_last_structure_loss 恒为 0。
        loss_ortho = 0.0
        if has_aeloru:
            for layer in aeloru_layers:
                loss_ortho = loss_ortho + layer.get_ortho_penalty()

        # fromscratch 冷启动正交预热：前 ortho_warmup_steps 步 ortho_lambda 线性 0→目标值，
        # 缓解冷启动阶段正交惩罚量纲失真/早期震荡；PEFT 场景 ortho_warmup_steps=0 -> 系数恒为 1。
        ortho_scale = min(global_step / ortho_warmup_steps, 1.0) if ortho_warmup_steps > 0 else 1.0
        loss = loss_task + ortho_lambda * ortho_scale * loss_ortho if has_aeloru else loss_task

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        lv = loss.item()
        losses.append(lv)
        lo = float(loss_ortho.item()) if has_aeloru else 0.0
        ortho_trace.append(lo)

        # --- Aeloru 专有后处理（单次遍历完成 observe / post_step / 状态统计）---
        if has_aeloru:
            for layer in aeloru_layers:
                # GHL：把全局任务损失方向作为神经调质信号注入（只取方向，不传幅值）
                layer.observe_global_loss(lv)
                rec = capture.get(layer)
                if rec is not None:
                    try:
                        merged = layer.post_step_update(
                            rec[0], rec[1], is_correct=True, merged=False, y_target=None)
                    except RuntimeError:
                        # 协方差矩阵病态（eigh/slogdet 不收敛）：加对角抖动后重试一次
                        _recondition_layer(layer)
                        merged = layer.post_step_update(
                            rec[0], rec[1], is_correct=True, merged=False, y_target=None)
                    if merged:
                        agg["merge"] += 1
                agg["states"][layer.state.value] = agg["states"].get(layer.state.value, 0) + 1
            # ReLoRA：周期性把 LoRA ΔW 折入 W0（从零组 W0 唯一更新途径）
            if relora_every and (s + 1) % relora_every == 0:
                for layer in aeloru_layers:
                    relora_merge_into_W0(layer, optimizer)

        global_step += 1

        # --- 机制指标采样 ---
        if has_aeloru and (s % 10 == 0 or s == steps - 1):
            row = _sample_mechanisms(aeloru_layers, task_name, s, lv)
            row["loss_ortho"] = lo
            mech_log.append(row)

    if capture is not None:
        capture.clear()
    agg["ortho"] = float(np.mean(ortho_trace)) if ortho_trace else 0.0
    return losses, global_step, agg


def _sample_mechanisms(layers: List[AeloruLayer], task: str, step: int, loss: float) -> dict:
    """采样 Aeloru v4 的六项新增机制指标（取所有注入层的均值，单层单次遍历）。"""
    plast_out, plast_in, gdir, struct = [], [], [], []
    proto_out, proto_in, bcm_cap, conflict, vol = [], [], [], [], []

    for l in layers:
        try:
            po = l.plasticity_out.mean().item()
            pi = l.plasticity_in.mean().item()
            gd = l.global_direction.item()
            sl = l._last_structure_loss
            po2 = l.proto_importance_out.mean().item()
            pi2 = l.proto_importance_in.mean().item()
            bc = l._adaptive_saturation(True).mean().item()
            cf = float(l.conflict_score.item()) if isinstance(l.conflict_score, torch.Tensor) \
                else float(l.conflict_score)
            vo = float(torch.exp(l.omega.float()).item())
            if math.isfinite(po):  plast_out.append(po)
            if math.isfinite(pi):  plast_in.append(pi)
            if math.isfinite(gd):  gdir.append(gd)
            if math.isfinite(sl):  struct.append(sl)
            if math.isfinite(po2): proto_out.append(po2)
            if math.isfinite(pi2): proto_in.append(pi2)
            if math.isfinite(bc):  bcm_cap.append(bc)
            if math.isfinite(cf):  conflict.append(cf)
            if math.isfinite(vo):  vol.append(vo)
        except Exception:
            continue

    def _mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    return {
        "task": task,
        "step": step,
        "loss": loss,
        # 双一维 HGF 精度门控
        "plasticity_out": _mean(plast_out),
        "plasticity_in":  _mean(plast_in),
        # GHL 全局方向引导
        "global_direction": _mean(gdir),
        # SPHeRe 低秩结构保持损失
        "structure_loss": _mean(struct),
        # DLAM 原型记忆重要性
        "proto_out": _mean(proto_out),
        "proto_in":  _mean(proto_in),
        # BCM 自适应饱和上限
        "bcm_cap_out": _mean(bcm_cap),
        # 认知状态与冲突
        "conflict": _mean(conflict),
        "volatility": _mean(vol),
        "state": layers[0].state.value if layers else "",
    }


def trigger_sleep_consolidation(model, aeloru_layers: List[AeloruLayer]) -> int:
    """
    任务边界显式触发 DLAM 睡眠巩固：跨模态绑定 + 离线重放（谱滤波 + 原型记忆提取）。
    返回发生离线重放的层数。
    """
    if not aeloru_layers:
        return 0
    cross_modal_sleep_step(AeloruModel(model))
    n = 0
    for layer in aeloru_layers:
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


# ============================================================================
# 指标
# ============================================================================

def compute_cl_metrics(R: np.ndarray, zero_shot: Optional[np.ndarray] = None,
                       chance: float = 0.5) -> Dict[str, float]:
    """
    R[i][j]     = 任务 i 训练结束后，在任务 j (j<=i) 上的准确率（下三角有效）。
    zero_shot[j]= 训练任务 j 之前，模型在任务 j 上的零样本准确率。

    注意：R 的上三角为 NaN，任何聚合都必须先按列裁掉 NaN，
    否则 np.max/np.mean 会被 NaN 污染。
    """
    T = R.shape[0]
    final = R[T - 1]
    acc = float(np.mean(final))

    # BWT: 学完全部任务后，各任务相对其「刚学完时」的准确率变化（负值=遗忘）
    bwt = float(np.mean([R[T - 1, j] - R[j, j] for j in range(T)]))

    # FWT: 学回合 j 之前在该任务上的即时表现相对随机基线的提升。
    #   长线场景下重复任务的 zero_shot 实为"回访前表现"（反映中途遗忘），
    #   FWT 在重复任务上的解释力下降，结果仅供参考。
    if zero_shot is not None:
        fwt = float(np.mean([zero_shot[j] - chance for j in range(1, T)]))
    else:
        fwt = float("nan")

    # Forgetting: 各任务在「已见过之后」的历史峰值，与最终值的差
    forget = float(np.mean([np.nanmax(R[j:, j]) - R[T - 1, j] for j in range(T)]))

    return {"ACC": acc, "BWT": bwt, "FWT": fwt, "Forgetting": forget}


# ============================================================================
# 单个方法的完整漂移实验
# ============================================================================

def run_method(method: str, seq_labels, seq_tasks, val_loaders, train_loaders, tok, args,
               out_dir: str, fh) -> dict:
    set_seed(args.seed)
    device = args.device
    T = len(seq_tasks)

    log(f"\n{'='*78}\n方法: {method}\n{'='*78}", fh)

    model, n_trainable = build_model(
        method, args.layers, device, args.lr,
        n_labels=2, pad_id=tok.pad_token_id, r=args.r, alpha=args.alpha,
        merge_every=args.merge_every,
        fs_relora_every=args.fs_relora_every,
        fs_enable_sleep=args.fs_enable_sleep)

    aeloru_layers = [m for m in model.modules() if isinstance(m, AeloruLayer)]
    capture = ActivationCapture(model) if aeloru_layers else None

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)

    log(f"  可训练参数: {n_trainable:,}   注入 AeloruLayer: {len(aeloru_layers)}", fh)

    # 从零训练组：显式 ReLoRA 合并周期；fromscratch 忽略 --merge-every，
    # 只用专属 --fs-relora-every（默认 steps//2），与 PEFT 组的自动合并互不干扰
    relora_every = args.relora_every
    if method == "aeloru-fromscratch":
        if model.fs_relora_every > 0:
            relora_every = model.fs_relora_every
        else:
            relora_every = max(1, args.steps // 2)
        # 实验性开启睡眠（风险见 md：wrapper 非标准 transformers，可能报错）
        model.do_sleep = bool(model.fs_enable_sleep)

    # 每轮学习率衰减：一轮 = len(base_tasks) 个任务
    round_len = T // args.repeat if args.repeat > 0 else T

    R = np.full((T, T), np.nan)
    zero_shot = np.zeros(T)       # 训练每个任务「之前」的零样本准确率 -> 用于 FWT
    mech_log: List[dict] = []
    global_step = 0
    task_losses = {}
    sleep_events = 0
    t_start = time.perf_counter()

    for i in range(T):
        # 每轮学习率自动衰减：第二轮复习用小步长，避免冲掉旧记忆
        cur_lr = args.lr * (args.lr_decay_per_round ** (i // round_len))
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        # 训练前的零样本评估：衡量模型面对「尚未学习的新分布」时的即时表现
        zero_shot[i] = evaluate(model, val_loaders[seq_tasks[i]], device)

        set_seed(args.seed + i)   # 每个任务独立的数据顺序，保证方法间可比
        losses, global_step, agg = train_one_task(
            model, aeloru_layers, capture, train_loaders[seq_tasks[i]], optimizer,
            device, args.steps, method, mech_log, seq_labels[i], global_step,
            ortho_lambda=args.ortho_lambda, relora_every=relora_every,
            lr=cur_lr, lr_schedule=bool(args.lr_schedule),
            ortho_warmup_steps=50 if method == "aeloru-fromscratch" else 0)
        task_losses[seq_labels[i]] = losses

        # 任务边界：DLAM 睡眠巩固（模拟"睡眠"把当前任务压入原型记忆）
        # 从零训练组关闭睡眠（wrapper.do_sleep=False），避免对非 transformers 模型调用失败
        if aeloru_layers and getattr(model, "do_sleep", True):
            for _ in range(args.sleep_intensity):
                sleep_events += trigger_sleep_consolidation(model, aeloru_layers)

        # 回测所有已见任务
        for j in range(i + 1):
            acc = evaluate(model, val_loaders[seq_tasks[j]], device)
            R[i, j] = acc
            log(f"  [回合{i+1}/{T} {seq_labels[i]:10s}] 训练 {args.steps} 步"
                f" | loss {np.mean(losses[:20]):.3f}->{np.mean(losses[-20:]):.3f}"
                f" | 回测 {seq_labels[j]:10s} = {acc:.4f}", fh)

        log(f"    -> 零样本 {zero_shot[i]:.4f} | 已见任务均值 {np.nanmean(R[i, :i+1]):.4f}"
            f" | 睡眠 {sleep_events} | 合并 {agg['merge']} | ortho {agg.get('ortho', 0):.5f}"
            f" | 状态 {agg['states']}", fh)

    elapsed = time.perf_counter() - t_start
    metrics = compute_cl_metrics(R, zero_shot)

    # 保存机制指标
    if mech_log:
        pd.DataFrame(mech_log).to_csv(
            os.path.join(out_dir, f"mech_{method}.csv"), index=False, encoding="utf-8-sig")

    # 保存 R 矩阵
    np.save(os.path.join(out_dir, f"R_{method}.npy"), R)

    log(f"  => ACC={metrics['ACC']:.4f}  BWT={metrics['BWT']:+.4f}  "
        f"FWT={metrics['FWT']:+.4f}  Forgetting={metrics['Forgetting']:.4f}  "
        f"({elapsed:.1f}s)", fh)

    if capture:
        capture.remove()
    del model, optimizer
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return {"method": method, "R": R, "zero_shot": zero_shot, "metrics": metrics,
            "mech": mech_log, "n_trainable": n_trainable, "elapsed": elapsed,
            "sleep": sleep_events, "task_losses": task_losses}


# ============================================================================
# 可视化
# ============================================================================

def make_figures(results: List[dict], tasks: List[str], out_dir: str):
    """主图：准确率矩阵 / 最终准确率 / 遗忘动态 / CL 指标（固定 2x2）。"""
    T = len(tasks)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) 每个方法的准确率矩阵热力图（取最后一个）
    ax = axes[0][0]
    im = ax.imshow(results[-1]["R"], cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(T)); ax.set_xticklabels(tasks, rotation=30)
    ax.set_yticks(range(T)); ax.set_yticklabels(tasks)
    ax.set_title(f"Accuracy Matrix R[i][j] - {results[-1]['method']}")
    for i in range(T):
        for j in range(T):
            if not np.isnan(results[-1]["R"][i, j]):
                ax.text(j, i, f"{results[-1]['R'][i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax)

    # (0,1) 漂移曲线：每个方法在各任务上的「最终」准确率
    ax = axes[0][1]
    for r in results:
        ax.plot(range(1, T + 1), r["R"][T - 1], marker="o", label=r["method"])
    ax.axhline(0.5, ls="--", c="gray", lw=1, label="chance")
    ax.set_xticks(range(1, T + 1)); ax.set_xticklabels(tasks, rotation=30)
    ax.set_ylabel("Final Accuracy"); ax.set_title("Concept Drift: Final Accuracy per Task")
    ax.legend(); ax.grid(alpha=.3)

    # (1,0) 遗忘曲线：每个任务随后续任务推进的准确率演变
    ax = axes[1][0]
    for r in results:
        for j in range(T):
            ax.plot(range(j + 1, T + 1), r["R"][j:, j], marker=".", lw=1.2,
                    label=f"{r['method']} {tasks[j]}" if j == 0 else None)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_xlabel("Tasks Seen"); ax.set_ylabel("Accuracy")
    ax.set_title("Forgetting Dynamics (each line = one task)")
    ax.set_xticks(range(1, T + 1)); ax.grid(alpha=.3)

    # (1,1) 指标柱状图
    ax = axes[1][1]
    keys = ["ACC", "BWT", "FWT", "Forgetting"]
    x = np.arange(len(keys)); w = 0.8 / max(len(results), 1)
    for k, r in enumerate(results):
        vals = [r["metrics"][kk] for kk in keys]
        ax.bar(x + k * w, vals, w, label=r["method"])
    ax.set_xticks(x + w * (len(results) - 1) / 2); ax.set_xticklabels(keys)
    ax.axhline(0, c="k", lw=.8); ax.set_title("Continual Learning Metrics")
    ax.legend(); ax.grid(alpha=.3, axis="y")

    plt.suptitle("Aeloru v4 Concept Drift Test on Qwen2.5-0.5B (GLUE task sequence)", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, "concept_drift.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p


def make_mech_figure(results: List[dict], tasks: List[str], out_dir: str) -> Optional[str]:
    """
    机制图：Aeloru v4 六项新增机制在整个漂移过程中的演化（2x3）。
    竖虚线为任务切换点（概念漂移发生处）。
    """
    has = [r for r in results if r.get("mech")]
    if not has:
        return None
    r = has[0]
    df = pd.DataFrame(r["mech"])
    if df.empty:
        return None

    df["g"] = np.arange(len(df))
    # 任务边界：每个任务的最后一次采样位置
    bounds = df.groupby("task", sort=False)["g"].max().values

    panels = [
        ("双一维 HGF 精度门控", ["plasticity_out", "plasticity_in"]),
        ("GHL 全局方向", ["global_direction"]),
        ("SPHeRe 结构保持损失", ["structure_loss", "loss_ortho"]),
        ("DLAM 原型重要性", ["proto_out", "proto_in"]),
        ("BCM 自适应饱和上限", ["bcm_cap_out"]),
        ("波动率 / 冲突", ["volatility", "conflict"]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    flat = axes.ravel()
    for idx, (title, cols) in enumerate(panels):
        ax = flat[idx]
        for c in cols:
            if c in df.columns:
                ax.plot(df["g"].values, df[c].values, lw=1.2, label=c)
        for b in bounds[:-1]:
            ax.axvline(b, c="r", ls=":", lw=1)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        if idx >= 3:
            ax.set_xlabel("sampling index (10 steps)")

    # x 轴刻度标为任务名：每个任务的「边界」与「中心」位置都标注其名
    centers = []
    prev = -1
    for b in bounds:
        centers.append((prev + b) / 2)
        prev = b
    order = df.groupby("task", sort=False)["g"].max().index.tolist()
    xtick_positions = list(bounds) + centers
    xtick_labels = [t for t in order for _ in range(2)]  # 每任务重复两次：边界 + 中心
    for ax in flat:
        ax.set_xticks(xtick_positions)
    for ax in flat[3:]:
        ax.set_xticklabels(xtick_labels[:len(xtick_positions)], rotation=30)

    plt.suptitle(f"Aeloru v4 mechanism dynamics under concept drift "
                 f"({r['method']}, red dotted = task switch)", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, "mechanisms.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p


# ============================================================================
# 主流程
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Aeloru 概念漂移测试（论文 §4 协议）")
    ap.add_argument("--methods", type=str, default=",".join(METHODS))
    ap.add_argument("--steps", type=int, default=250, help="每个任务的训练步数（论文 §4.1.2: 250）")
    ap.add_argument("--layers", type=int, default=4, help="注入的 Transformer 层数（论文 §4.1.1: 4）")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4, help="学习率（论文 §4.1.3: 1e-4）")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--ortho-lambda", type=float, default=0.1,
                    help="正交惩罚 + SPHeRe 结构保持损失的权重（Aeloru 专用）")
    ap.add_argument("--relora-every", type=int, default=0,
                    help="从零训练组的 ReLoRA 合并周期(步)；0=按 steps//2 自动")
    ap.add_argument("--merge-every", type=int, default=0,
                    help="全局 ReLoRA 合并周期(步)，0=使用各配置默认值；长线推荐 500")
    ap.add_argument("--lr-schedule", type=int, default=1, choices=[0, 1],
                    help="是否启用每任务余弦学习率退火：1=开，0=关")
    ap.add_argument("--lr-decay-per-round", type=float, default=1.0,
                    help="每完成一轮任务序列后的学习率衰减系数，长线推荐 0.8")
    ap.add_argument("--sleep-intensity", type=int, default=1,
                    help="每任务后睡眠巩固强度(调用次数)，长线推荐 2~3")
    ap.add_argument("--fs-relora-every", type=int, default=0,
                    help="aeloru-fromscratch专属ReLoRA合并周期；0=自动取steps//2；长线建议250(每任务合并一次)")
    ap.add_argument("--fs-enable-sleep", type=int, default=0, choices=[0, 1],
                    help="是否给fromscratch打开睡眠巩固；0关闭(原默认)，1开启(实验性，可能报错)")
    ap.add_argument("--repeat", type=int, default=2,
                    help="长线循环轮数：把任务序列重复 N 次（同任务多次出现，测巩固/回访）")
    ap.add_argument("--runs", type=int, default=0,
                    help="运行 --seed 列表中的前 N 个种子；0=运行全部（默认）")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--seed", type=str, default="42,9589,6519,1412",
                    help="随机种子，逗号分隔（论文 §4.1.3 四种子 42/9589/6519/1412；"
                         "主口径为后三个新种子，42 单列）")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    # 解析种子列表（论文四种子）；--runs>0 时只取前 N 个
    seed_list = [int(s) for s in args.seed.split(",") if s.strip()]
    if args.runs > 0:
        seed_list = seed_list[:args.runs]
    seed_tag = args.seed.replace(",", "-")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"未知方法 {m}，可选: {METHODS}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       f"ex1_drift_{seed_tag}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.txt")
    fh = open(log_path, "w", encoding="utf-8")
    all_results = []
    for run, sd in enumerate(seed_list):
        args.seed = sd
        log("=" * 78, fh)
        log("Aeloru 概念漂移长线测试 (Concept Drift)", fh)
        log(f"模型: {MODEL_PATH}", fh)
        log(f"任务序列单元: {' -> '.join(t for t, _ in TASKS)}  (长线重复 x{args.repeat})", fh)
        log(f"方法: {methods}", fh)
        log(f"步数/任务={args.steps} 注入层数={args.layers} bs={args.bs} seq={args.seq} "
            f"lr={args.lr} r={args.r} alpha={args.alpha} seed={args.seed}", fh)
        log(f"输出: {out_dir}", fh)
        log(f"第{run+1}/{len(seed_list)}次实验（种子 {sd}）", fh)
        log("=" * 78, fh)

        tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # ---- 数据 ----
        base_tasks = [t for t, _ in TASKS]
        seq_tasks = []
        seq_labels = []
        for rep in range(args.repeat):
            for t in base_tasks:
                seq_tasks.append(t)
                seq_labels.append(f"{t}@{rep}")
        train_sets, val_sets = {}, {}
        for t in base_tasks:
            tr, va = build_task_dataset(tok, t, args.seq, args.n_train, args.n_val, args.seed)
            train_sets[t], val_sets[t] = tr, va
            log(f"  数据 {t:5s}: train={len(tr):5d} val={len(va):4d}", fh)

        collate = DataCollatorWithPadding(tok)
        train_loaders = {t: DataLoader(train_sets[t], batch_size=args.bs, shuffle=True,
                                       collate_fn=collate, drop_last=True) for t in base_tasks}
        val_loaders = {t: DataLoader(val_sets[t], batch_size=32, shuffle=False,
                                     collate_fn=collate) for t in base_tasks}

        # ---- 运行（每个种子独立子目录，避免 R/mech 文件互相覆盖）----
        run_out = os.path.join(out_dir, f"seed_{sd}")
        os.makedirs(run_out, exist_ok=True)
        results = []
        for m in methods:
            r = run_method(m, seq_labels, seq_tasks, val_loaders, train_loaders, tok, args, run_out, fh)
            results.append(r)
        all_results.append(results)

    # ---- 汇总（跨种子 mean ± std，对齐论文 §4.1.3 四种子口径）----
    log(f"\n{'='*78}\n汇总（{len(all_results)} 种子）\n{'='*78}", fh)
    methods_list = [r["method"] for r in all_results[0]]
    metric_keys = ["ACC", "BWT", "FWT", "Forgetting"]

    agg_rows = []
    mean_results = []          # 供绘图：各方法 R 矩阵/指标跨种子均值
    for k, m in enumerate(methods_list):
        per_seed = [all_results[s][k] for s in range(len(all_results))]
        row = {"method": m, "params": per_seed[0]["n_trainable"],
               "sleep": per_seed[0]["sleep"]}
        for key in metric_keys:
            vals = np.array([ps["metrics"][key] for ps in per_seed], dtype=float)
            row[f"{key}_mean"] = round(float(vals.mean()), 4)
            row[f"{key}_std"] = round(float(vals.std(ddof=0)), 4)
        agg_rows.append(row)
        mean_R = np.nanmean(np.stack([ps["R"] for ps in per_seed]), axis=0)
        mean_results.append({
            "method": m, "R": mean_R,
            "metrics": {key: row[f"{key}_mean"] for key in metric_keys},
            "mech": per_seed[0].get("mech"),
        })

    cols = (["method"]
            + [f"{k}_mean" for k in metric_keys]
            + [f"{k}_std" for k in metric_keys]
            + ["params", "sleep"])
    df = pd.DataFrame(agg_rows)[cols]
    csv_p = os.path.join(out_dir, "summary.csv")
    df.to_csv(csv_p, index=False, encoding="utf-8-sig")

    # R 矩阵明细（跨种子均值）
    with open(os.path.join(out_dir, "R_matrix.txt"), "w", encoding="utf-8") as f:
        for idx, r in enumerate(mean_results):
            f.write(f"\n=== {r['method']} (mean over {len(all_results)} seeds) ===\n")
            f.write("R[i][j]: 第 i 个任务训练后，在任务 j 上的平均准确率\n")
            f.write("        " + "".join(f"{t:>10s}" for t in seq_labels) + "\n")
            for i, t in enumerate(seq_labels):
                f.write(f"{t:>9s} " + "".join(
                    f"{r['R'][i, j]:8.4f}" if not np.isnan(r["R"][i, j]) else "     n/a"
                    for j in range(len(seq_labels))) + "\n")
            zs = np.nanmean(np.stack([all_results[s][idx]["zero_shot"]
                                      for s in range(len(all_results))]), axis=0)
            f.write("zero-shot: " + "".join(f"{v:8.4f}" for v in zs) + "\n")

    png = make_figures(mean_results, seq_labels, out_dir)
    png2 = make_mech_figure(mean_results, seq_labels, out_dir)
    log(f"\n结果已保存:\n  {csv_p}\n  {png}"
        + (f"\n  {png2}" if png2 else "") + f"\n  {log_path}", fh)

    # 控制台打印跨种子 mean ± std
    print("\n方法   " + "  ".join(f"{k:>14s}" for k in metric_keys) + "   (mean ± std)")
    for row in agg_rows:
        line = f"{row['method']:16s}"
        for k in metric_keys:
            line += f"  {row[f'{k}_mean']:+.4f}±{row[f'{k}_std']:.4f}"
        print(line)
    fh.close()
    return all_results


if __name__ == "__main__":
    main()