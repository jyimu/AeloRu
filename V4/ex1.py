# -*- coding: utf-8 -*-
"""
概念漂移 (Concept Drift) 测试：Aeloru v4 vs LoRA
==================================================

用 Qwen2.5-0.5B 在 GLUE 任务序列上构造**任务级概念漂移**：

    SST2 -> MRPC -> QNLI -> RTE -> CoLA

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
- aeloru-safe     : Aeloru v4，关闭对预训练表示有害的组件
                    (预测编码 ndi=0 / 域约束 none / 稳态增益上限 1.0)
- aeloru-default  : Aeloru v4 出厂默认配置（用于暴露默认配置的问题）

Aeloru v4 机制监控
--------------------
每步记录 plasticity_out/in（双一维 HGF 精度门控）、global_direction（GHL 全局方向）、
loss_structure（SPHeRe 结构保持）、认知状态、原型重要性（DLAM 原型记忆）、
睡眠触发次数，用于验证五项新增机制在漂移下是否按设计响应。

用法
----
    python ex1.py                          # 全量：3 方法 x 5 任务 x 250 步
    python ex1.py --methods lora,aeloru-safe --steps 100
    python ex1.py --methods aeloru-safe --layers 4 --seq 64 --bs 16

@author JYIMU
"""

import os
import sys
import time
import json
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

# GLUE 五任务：全部为二分类，可共享分类头，
# 从而把「分类头漂移」与「骨干遗忘」一并计入遗忘指标（更严格的协议）。
TASKS: List[Tuple[str, int]] = [
    ("sst2", 2),
    ("mrpc", 2),
    ("qnli", 2),
    ("rte",  2),
    ("cola", 2),
]

# 各任务的字段组合不同，需分别处理
TASK_FIELDS = {
    "sst2": ("sentence",),
    "cola": ("sentence",),
    "mrpc": ("sentence1", "sentence2"),
    "rte":  ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
}

METHODS = ["lora", "aeloru-safe", "aeloru-default"]


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
            layers_to_transform=list(range(24 - n_layers, 24)),
        ))
        model.to(device)

    elif method == "aeloru-safe":
        # 关闭三项对预训练表示有害的组件：
        #   neural_dynamics_iterations=0  -> 预测编码神经动态（实测会把输出放大 13x 或归零）
        #   presumed_domain="none"        -> 源信号域约束 clamp(0,1)（会截断半数负激活）
        #   homeostatic_max_gain=1.0      -> 稳态增益（把每层 RMS 拉向 s^(1/3)，多层累积压垮残差流）
        cfg = AeloruConfig.llm_tuning(
            r=r, lora_alpha=alpha, LoRA_lr=lr,
            neural_dynamics_iterations=0,
            presumed_domain="none",
            homeostatic_max_gain=1.0,
            whitening_interval=10 ** 9,
            merge_every=10 ** 9,
            logging_enabled=False,
            homeostatic_enabled=False,
            diagnostic_enabled=False,
            device=device,
        )
        inject_aeloru_selected(model, list(range(24 - n_layers, 24)), cfg)
        model.to(device)
        _freeze_base(model)

    elif method == "aeloru-default":
        # v4 出厂默认值（llm_tuning 预设）：neural_dynamics_iterations=0,
        # presumed_domain="none", homeostatic_max_gain=2.0；保留 ReLoRA 合并以观察长期行为。
        # 与 aeloru-safe 的区别仅在于开启稳态可塑性 (homeostatic_enabled=True)。
        cfg = AeloruConfig.llm_tuning(
            r=r, lora_alpha=alpha, LoRA_lr=lr,
            logging_enabled=False,
            diagnostic_enabled=False,
            device=device,
            homeostatic_enabled=True,
            pem_space="output"
        )
        inject_aeloru_selected(model, list(range(24 - n_layers, 24)), cfg)
        model.to(device)
        _freeze_base(model)

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
                   global_step: int, ortho_lambda: float = 0.1) -> Tuple[List[float], int, dict]:
    """在单个任务上训练若干步。返回 (loss 序列, 新的 global_step, 汇总统计)。"""
    lossf = nn.CrossEntropyLoss()
    it = iter(loader)
    losses = []
    ortho_trace = []
    agg = {"sleep": 0, "merge": 0, "states": {}}
    converge_hits = 0

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
        if aeloru_layers:
            for layer in aeloru_layers:
                loss_ortho = loss_ortho + layer.get_ortho_penalty()
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
        if aeloru_layers:
            for layer in aeloru_layers:
                # GHL：把全局任务损失方向作为神经调质信号注入（只取方向，不传幅值）
                layer.observe_global_loss(lv)
                rec = capture.get(layer)
                if rec is None:
                    continue
                merged = layer.post_step_update(
                    rec[0], rec[1], is_correct=True, merged=False, y_target=None)
                if merged:
                    agg["merge"] += 1

        global_step += 1

        # --- 机制指标采样 ---
        if aeloru_layers and (s % 10 == 0 or s == steps - 1):
            row = _sample_mechanisms(aeloru_layers, task_name, s, lv)
            row["loss_ortho"] = lo
            mech_log.append(row)

        # 统计认知状态分布
        if aeloru_layers:
            for layer in aeloru_layers:
                st = layer.state.value
                agg["states"][st] = agg["states"].get(st, 0) + 1

    if capture is not None:
        capture.clear()
    agg["ortho"] = float(np.mean(ortho_trace)) if ortho_trace else 0.0
    return losses, global_step, agg


def _sample_mechanisms(layers: List[AeloruLayer], task: str, step: int, loss: float) -> dict:
    """采样 Aeloru v4 的六项新增机制指标（取所有注入层的均值）。"""
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

    row = {
        "task": task,
        "step": step,
        "loss": loss,
        # 双一维 HGF 精度门控
        "plasticity_out": avg(lambda l: l.plasticity_out.mean().item()),
        "plasticity_in":  avg(lambda l: l.plasticity_in.mean().item()),
        # GHL 全局方向引导
        "global_direction": avg(lambda l: l.global_direction.item()),
        # SPHeRe 低秩结构保持损失
        "structure_loss": avg(lambda l: l._last_structure_loss),
        # DLAM 原型记忆重要性
        "proto_out": avg(lambda l: l.proto_importance_out.mean().item()),
        "proto_in":  avg(lambda l: l.proto_importance_in.mean().item()),
        # BCM 自适应饱和上限
        "bcm_cap_out": avg(lambda l: l._adaptive_saturation(True).mean().item()),
        # 认知状态与冲突
        "conflict": avg(lambda l: float(l.conflict_score.item())
                        if isinstance(l.conflict_score, torch.Tensor) else float(l.conflict_score)),
        "volatility": avg(lambda l: float(torch.exp(l.omega.float()).item())),
        "state": layers[0].state.value if layers else "",
    }
    return row


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

    # FWT: 学习任务 j 之前，模型在 j 上的零样本表现相对随机基线的提升
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

def run_method(method: str, tasks, val_loaders, train_loaders, tok, args,
               out_dir: str, fh) -> dict:
    set_seed(args.seed)
    device = args.device
    T = len(tasks)

    log(f"\n{'='*78}\n方法: {method}\n{'='*78}", fh)

    model, n_trainable = build_model(
        method, args.layers, device, args.lr,
        n_labels=2, pad_id=tok.pad_token_id, r=args.r, alpha=args.alpha)

    aeloru_layers = [m for m in model.modules() if isinstance(m, AeloruLayer)]
    capture = ActivationCapture(model) if aeloru_layers else None

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)

    log(f"  可训练参数: {n_trainable:,}   注入 AeloruLayer: {len(aeloru_layers)}", fh)

    R = np.full((T, T), np.nan)
    zero_shot = np.zeros(T)       # 训练每个任务「之前」的零样本准确率 -> 用于 FWT
    mech_log: List[dict] = []
    global_step = 0
    task_losses = {}
    sleep_events = 0
    t_start = time.perf_counter()

    for i, task in enumerate(tasks):
        # 训练前的零样本评估：衡量模型面对「尚未学习的新分布」时的即时表现
        zero_shot[i] = evaluate(model, val_loaders[task], device)

        set_seed(args.seed + i)   # 每个任务独立的数据顺序，保证方法间可比
        losses, global_step, agg = train_one_task(
            model, aeloru_layers, capture, train_loaders[task], optimizer,
            device, args.steps, method, mech_log, task, global_step,
            ortho_lambda=args.ortho_lambda)
        task_losses[task] = losses

        # 任务边界：DLAM 睡眠巩固（模拟"睡眠"把当前任务压入原型记忆）
        if aeloru_layers:
            sleep_events += trigger_sleep_consolidation(model, aeloru_layers)

        # 回测所有已见任务
        for j in range(i + 1):
            acc = evaluate(model, val_loaders[tasks[j]], device)
            R[i, j] = acc
            log(f"  [任务{i+1}/{T} {task:5s}] 训练 {args.steps} 步"
                f" | loss {np.mean(losses[:20]):.3f}->{np.mean(losses[-20:]):.3f}"
                f" | 回测 {tasks[j]:5s} = {acc:.4f}", fh)

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

    # x 轴刻度标为任务名
    centers = []
    prev = -1
    order = df.groupby("task", sort=False)["g"].max().index.tolist()
    for b in bounds:
        centers.append((prev + b) / 2)
        prev = b
    for ax in flat:
        ax.set_xticks(list(bounds) + [t for t in centers])
    for ax in flat[3:]:
        ax.set_xticklabels([t for t in order for _ in (0, 1)][:len(ax.get_xticks())], rotation=30)

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

def main(seed: int = 42):
    ap = argparse.ArgumentParser(description="Aeloru v4 概念漂移测试")
    ap.add_argument("--methods", type=str, default=",".join(METHODS))
    ap.add_argument("--steps", type=int, default=250, help="每个任务的训练步数")
    ap.add_argument("--layers", type=int, default=4, help="注入的 Transformer 层数")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--ortho-lambda", type=float, default=0.1,
                    help="正交惩罚 + SPHeRe 结构保持损失的权重（Aeloru 专用）")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=seed)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"未知方法 {m}，可选: {METHODS}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       f"ex1_drift_{args.seed}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.txt")
    fh = open(log_path, "w", encoding="utf-8")

    log("=" * 78, fh)
    log("Aeloru v4 概念漂移测试 (Concept Drift)", fh)
    log(f"模型: {MODEL_PATH}", fh)
    log(f"任务序列: {' -> '.join(t for t, _ in TASKS)}", fh)
    log(f"方法: {methods}", fh)
    log(f"步数/任务={args.steps} 注入层数={args.layers} bs={args.bs} seq={args.seq} "
        f"lr={args.lr} r={args.r} alpha={args.alpha} seed={args.seed}", fh)
    log(f"输出: {out_dir}", fh)
    log("=" * 78, fh)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- 数据 ----
    tasks = [t for t, _ in TASKS]
    train_sets, val_sets = {}, {}
    for t, _ in TASKS:
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
        r = run_method(m, tasks, val_loaders, train_loaders, tok, args, out_dir, fh)
        results.append(r)

    # ---- 汇总 ----
    log(f"\n{'='*78}\n汇总\n{'='*78}", fh)
    rows = []
    for r in results:
        mt = r["metrics"]
        rows.append({
            "method": r["method"],
            "ACC": round(mt["ACC"], 4),
            "BWT": round(mt["BWT"], 4),
            "FWT": round(mt["FWT"], 4),
            "Forgetting": round(mt["Forgetting"], 4),
            "params": r["n_trainable"],
            "sleep": r["sleep"],
            "seconds": round(r["elapsed"], 1),
        })
        log(f"  {r['method']:16s} ACC={mt['ACC']:.4f}  BWT={mt['BWT']:+.4f}  "
            f"FWT={mt['FWT']:+.4f}  Forgetting={mt['Forgetting']:.4f}", fh)

    df = pd.DataFrame(rows)
    csv_p = os.path.join(out_dir, "summary.csv")
    df.to_csv(csv_p, index=False, encoding="utf-8-sig")

    # R 矩阵明细
    with open(os.path.join(out_dir, "R_matrix.txt"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"\n=== {r['method']} ===\n")
            f.write("R[i][j]: 第 i 个任务训练后，在任务 j 上的准确率\n")
            f.write("        " + "".join(f"{t:>8s}" for t in tasks) + "\n")
            for i, t in enumerate(tasks):
                f.write(f"{t:>7s} " + "".join(
                    f"{r['R'][i, j]:8.4f}" if not np.isnan(r["R"][i, j]) else "     n/a"
                    for j in range(len(tasks))) + "\n")
            f.write("zero-shot: " + "".join(f"{v:8.4f}" for v in r["zero_shot"]) + "\n")

    png = make_figures(results, tasks, out_dir)
    png2 = make_mech_figure(results, tasks, out_dir)
    log(f"\n结果已保存:\n  {csv_p}\n  {png}"
        + (f"\n  {png2}" if png2 else "") + f"\n  {log_path}", fh)
    fh.close()

    print("\n" + df.to_string(index=False))
    return results


if __name__ == "__main__":
    if argparse.__name__ == "__main__":
        main()
    else:
        import random
        for seed in [random.randint(0, 9999) for _ in range(3)]:
            main(seed=seed)
