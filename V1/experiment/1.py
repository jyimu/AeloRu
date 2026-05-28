"""
Qelys Aeloru 高性能训练框架 (Refactored)
==========================================
优化点：
1. 梯度累积：有效 batch_size 放大 4x，减少 optimizer 同步
2. 评估安全化：不囤积 GPU logits，流式计算 accuracy
3. ReLoRA bug 修复：named_modules() 解包
4. Aeloru 双模式：Full(全功能) / Base(纯 LoRA 等价)
5. 零训练同步：进度条每 20 步更新一次，epoch 结束再 .item()
6. 实时性能面板：吞吐(samples/s)、显存峰值、单步延迟
"""

import os

import platform
IS_WINDOWS = platform.system() == "Windows"

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    # Windows 下彻底禁用 Triton 尝试
    if IS_WINDOWS:
        torch._dynamo.config.disable = True
        print("[System] Windows detected: torch.compile disabled (Triton unavailable)")
except Exception:
    pass

#换用中国镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#关闭SSL
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["NO_PROXY"] = "hf-mirror.com,cdn-lfs.hf-mirror.com"
for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(key, None)

import math
import time
import warnings
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from aeloru_layer import AeloruConfig, AeloruLayer, inject_aeloru

import urllib3
urllib3.disable_warnings()

import matplotlib
matplotlib.use('Agg')  # 无头环境安全
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

os.environ["PYTHONIOENCODING"] = "utf-8"

# ========================= CUDA 性能预设 =========================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 启用内存高效的注意力（Qwen2原生支持）
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# 只在真正需要时开 anomaly detection
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = "models/Qwen2.5-0.5B"

# ========================= 超参配置区 =========================
@dataclass
class TrainConfig:
    max_seq_length: int = 128
    batch_size: int = 8          # 单卡物理 batch
    grad_accum_steps: int = 4    # 梯度累积 → 有效 batch = 32
    num_epochs: int = 3
    lr: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42
    max_grad_norm: float = 1.0
    
    # 性能监控
    log_interval: int = 20       # 每 20 步更新进度条（避免 CUDA 同步）
    eval_every_n_epoch: int = 1  # 每几 epoch 评估一次
    
    # 系统
    num_workers: int = 0       # Windows 安全
    pin_memory: bool = True if torch.cuda.is_available() else False

TRAIN_CFG = TrainConfig()

AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
USE_AMP = True

torch.manual_seed(TRAIN_CFG.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(TRAIN_CFG.seed)

# ========================= 数据集配置 =========================
DATASET_CONFIGS = [
    {"name": "glue", "config": "mrpc", "text_cols": ["sentence1", "sentence2"], "label_col": "label", "num_labels": 2},
    {"name": "glue", "config": "rte",  "text_cols": ["sentence1", "sentence2"], "label_col": "label", "num_labels": 2},
    {"name": "glue", "config": "cola", "text_cols": ["sentence"], "label_col": "label", "num_labels": 2},
]

# LoRA 基参
LORA_R = 8
LORA_ALPHA = 4.0
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj"]


# ========================= 工具：安全评估指标 =========================
class StreamingAccuracy:
    """流式准确率计算，不囤积 GPU 张量"""
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.loss_sum = 0.0
        self.samples = 0
    
    @torch.no_grad()
    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor):
        preds = logits.argmax(dim=-1)
        self.correct += (preds == labels).sum().item()  # ← 只传一个标量
        self.total += labels.size(0)
        self.loss_sum += loss.item() * labels.size(0)
        self.samples += labels.size(0)
    
    def compute(self) -> Tuple[float, float]:
        acc = self.correct / max(self.total, 1)
        avg_loss = self.loss_sum / max(self.samples, 1)
        return acc, avg_loss


# ========================= ReLoRA 训练器（Bug 修复版）========================
class ReLoRATrainer:
    def __init__(self, model, merge_every: int = 1000):
        self.model = model
        self.merge_every = merge_every
        # 修复：named_modules() 返回 (name, module) 元组
        self.lora_layers = [
            module for _, module in model.named_modules()
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B')
        ]
    
    def maybe_merge_and_reinit(self, global_step: int):
        if global_step > 0 and global_step % self.merge_every == 0:
            print(f"[ReLoRA] Merging at step {global_step}...")
            self._merge_lora_weights()
            self._reset_lora_parameters()
    
    def _merge_lora_weights(self):
        with torch.no_grad():
            for module in self.lora_layers:
                scaling = getattr(module, 'scaling', module.lora_alpha / module.r)
                delta = (module.lora_B @ module.lora_A) * scaling
                # PEFT LoRA 的 base_layer 是原始 Linear
                if hasattr(module, 'base_layer'):
                    module.base_layer.weight.data.add_(delta)
                elif hasattr(module, 'weight'):
                    module.weight.data.add_(delta)
    
    def _reset_lora_parameters(self):
        with torch.no_grad():
            for module in self.lora_layers:
                nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                nn.init.zeros_(module.lora_B)


# ========================= 数据集加载 =========================
def load_and_preprocess_dataset(ds_cfg: Dict, tokenizer):
    cache_name = f"cached_{ds_cfg['config']}_{TRAIN_CFG.max_seq_length}_v2"
    cache_path = os.path.join("cache", cache_name)
    
    if os.path.exists(cache_path):
        print(f"[Cache] Loading from {cache_path}")
        from datasets import load_from_disk
        encoded = load_from_disk(cache_path)
        encoded.set_format("torch")
        return encoded

    raw = load_dataset(ds_cfg["name"], ds_cfg["config"])
    text_cols = ds_cfg["text_cols"]
    label_col = ds_cfg["label_col"]

    def tokenize_fn(examples):
        if len(text_cols) == 1:
            return tokenizer(
                examples[text_cols[0]], 
                truncation=True, 
                max_length=TRAIN_CFG.max_seq_length, 
                padding=False
            )
        return tokenizer(
            examples[text_cols[0]], 
            examples[text_cols[1]],
            truncation=True, 
            max_length=TRAIN_CFG.max_seq_length, 
            padding=False
        )

    encoded = raw.map(tokenize_fn, batched=True, remove_columns=text_cols)
    
    # 统一列名
    for split in list(encoded.keys()):
        if label_col in encoded[split].column_names and label_col != "labels":
            encoded[split] = encoded[split].rename_column(label_col, "labels")
        # 移除无用列
        keep = {"input_ids", "attention_mask", "labels"}
        to_remove = [c for c in encoded[split].column_names if c not in keep]
        if to_remove:
            encoded[split] = encoded[split].remove_columns(to_remove)
    
    encoded.set_format("torch")
    os.makedirs("cache", exist_ok=True)
    encoded.save_to_disk(cache_path)
    print(f"[Cache] Saved to {cache_path}")
    return encoded


# ========================= 模型工厂 =========================
def create_model(method: str, num_labels: int, pad_token_id: int):
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=num_labels, 
        trust_remote_code=True,
        torch_dtype=AMP_DTYPE,
        attn_implementation="sdpa"
    ).to(DEVICE)
    base.config.pad_token_id = pad_token_id

    if method == "LoRA":
        config = LoraConfig(
            r=LORA_R, lora_alpha=int(LORA_ALPHA), target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(base, config)

    elif method == "DoRA":
        config = LoraConfig(
            r=LORA_R, lora_alpha=int(LORA_ALPHA), target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.SEQ_CLS,
            use_dora=True,
        )
        model = get_peft_model(base, config)

    elif method == "ReLoRA":
        config = LoraConfig(
            r=LORA_R, lora_alpha=int(LORA_ALPHA), target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(base, config)

    elif method == "Aeloru-Base":
        cfg = AeloruConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, LoRA_lr=TRAIN_CFG.lr,
            verbose=False, device=str(DEVICE), AMP_DTYPE=AMP_DTYPE,
            use_hidora=False, use_relora=False, use_hebbian=False,
            use_fisher=False, use_hongwen=False,
            use_orthogonal_penalty=False, use_energy_budget=False,
            async_merge=False, fisher_async=False,
        )
        model = inject_aeloru(base, target_names=TARGET_MODULES, cfg=cfg)

    elif method == "Aeloru-Full":
        cfg = AeloruConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, LoRA_lr=TRAIN_CFG.lr,
            verbose=False,
            enable_cognitive_report=False,
            diagnostic_interval=10000,
            device=str(DEVICE), AMP_DTYPE=AMP_DTYPE,
            use_hidora=True, use_relora=True, use_hebbian=True,
            use_fisher=True, use_hongwen=True,
            use_orthogonal_penalty=True, use_energy_budget=True,
            fisher_mode='hierarchical',
            fisher_topk_ratio=0.2,
            fisher_compute_interval=2000,
            fisher_full_snapshot_interval=20000,
            fisher_quant_bits=8,
            fisher_async=True,
            fisher_bp16=True,
            fisher_gamma=2.0,
            merge_every=1000,
            merge_on_red=True,
            async_merge=True,
            acc_quant_bits=8,
            hebbian_lr=1e-6,
            hebbian_decay=0.99,
            hebbian_accum_steps=16,
            saturation_limit=5.0,
            red_threshold=0.65,
            snapshot_interval=50,
            anchor_converge=1e-4,
            solid_steps=200,
            use_grad_conflict=True,
            grad_conflict_window=50,
            grad_conflict_threshold=0.3,
            ortho_lambda=0.001,
            ortho_lambda_anchor=0.05,
            ortho_random_proj=4,
            energy_eta=0.30,
            energy_sample_ratio=0.1,
        )
        model = inject_aeloru(base, target_names=TARGET_MODULES, cfg=cfg)
    else:
        raise ValueError(f"Unknown method: {method}")

    # ========== Windows 安全：torch.compile 只在 Linux+Triton 可用时启用 ==========
    if hasattr(torch, 'compile') and not IS_WINDOWS:
        print(f"[Optimize] Attempting torch.compile for {method}")
        try:
            # 先做一次 dummy forward 触发编译，失败立即回退
            model.eval()
            with torch.no_grad():
                dummy_input = {
                    "input_ids": torch.zeros(2, 8, dtype=torch.long, device=DEVICE),
                    "attention_mask": torch.ones(2, 8, dtype=torch.long, device=DEVICE),
                }
                _ = model(**dummy_input)
            
            # 如果 dummy forward 成功，再正式 compile
            model = torch.compile(model, mode="reduce-overhead", dynamic=False, fullgraph=False)
            print(f"[Optimize] torch.compile SUCCESS (reduce-overhead)")
        except Exception as e:
            print(f"[Optimize] torch.compile FAILED: {e}")
            print(f"[Optimize] Fallback to eager mode")
            # 确保如果 compile 失败，model 还是原始未 compile 的版本
            # （dummy forward 已经修改了状态，但模型结构没变）
    else:
        if IS_WINDOWS:
            print(f"[Optimize] Skipping torch.compile on Windows")
        else:
            print(f"[Optimize] torch.compile not available")

    return model


# ========================= 训练引擎（梯度累积版）========================
def train_epoch(
    model, 
    train_loader, 
    optimizer, 
    scheduler, 
    scaler, 
    relora_trainer=None,
    aeloru_layers=None,
    aeloru_cache=None,
    need_hook=False,
):
    model.train()
    
    # 性能监控
    epoch_loss = torch.tensor(0.0, device=DEVICE)
    num_batches = 0
    step_times = []
    global_samples = 0
    
    pbar = tqdm(train_loader, desc="Train")
    optimizer.zero_grad()  # 梯度累积：在循环外先 zero
    
    for batch_idx, batch in enumerate(pbar):
        batch = {k: v.to(DEVICE, non_blocking=TRAIN_CFG.pin_memory) for k, v in batch.items()}
        
        t0 = time.perf_counter()
        
        # 前向
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=USE_AMP, dtype=AMP_DTYPE):
            outputs = model(**batch)
            loss = outputs.loss / TRAIN_CFG.grad_accum_steps  # 梯度累积：loss 缩放
        
        # Aeloru 正交惩罚（只在实际 step 时加，避免累积干扰）
        if aeloru_layers and batch_idx % TRAIN_CFG.grad_accum_steps == 0:
            with torch.no_grad():
                ortho_loss = sum(
                    layer.get_ortho_penalty() for layer in aeloru_layers
                ) / TRAIN_CFG.grad_accum_steps
            loss = loss + ortho_loss
        
        # 反向
        scaler.scale(loss).backward() if scaler else loss.backward()
        
        # 记录时间（不含梯度裁剪/优化器，因为不是每步都做）
        t1 = time.perf_counter()
        step_times.append(t1 - t0)
        
        # ========== 梯度累积结算 ==========
        is_accum_step = (batch_idx + 1) % TRAIN_CFG.grad_accum_steps == 0
        is_last_batch = (batch_idx + 1) == len(train_loader)
        
        if is_accum_step or is_last_batch:
            # 梯度裁剪
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG.max_grad_norm)
            
            # 优化器步进
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            # ReLoRA 合并检查
            if relora_trainer:
                relora_trainer.maybe_merge_and_reinit(batch_idx // TRAIN_CFG.grad_accum_steps)
            
            # Aeloru post_step_update（只在真正优化器步后执行）
            if aeloru_layers and need_hook:
                for layer in aeloru_layers:
                    lid = id(layer)
                    if lid in aeloru_cache:
                        x_c, y_c = aeloru_cache[lid]
                        layer.post_step_update(x_c, y_c, is_correct=True)
                        del aeloru_cache[lid]
        
        # 异步统计：GPU 累加，不 .item()
        with torch.no_grad():
            epoch_loss += loss.detach() * TRAIN_CFG.grad_accum_steps
        
        num_batches += 1
        global_samples += batch["input_ids"].size(0)
        
        # 低频率更新进度条（减少 CUDA 同步）
        if (batch_idx + 1) % TRAIN_CFG.log_interval == 0:
            avg_step_ms = np.mean(step_times[-TRAIN_CFG.log_interval:]) * 1000
            throughput = (TRAIN_CFG.log_interval * TRAIN_CFG.batch_size * TRAIN_CFG.grad_accum_steps) / sum(step_times[-TRAIN_CFG.log_interval:])
            pbar.set_postfix({
                "loss": f"{(epoch_loss / num_batches).item():.4f}",
                "ms/step": f"{avg_step_ms:.1f}",
                "samples/s": f"{throughput:.1f}",
            })
    
    # Epoch 结束：一次同步
    avg_loss = (epoch_loss / num_batches).item()
    avg_step_ms = np.mean(step_times) * 1000 if step_times else 0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    torch.cuda.reset_peak_memory_stats()
    
    return avg_loss, avg_step_ms, peak_mem


# ========================= 评估引擎（流式、不囤积 GPU）========================
@torch.no_grad()
def evaluate(model, eval_loader):
    model.eval()
    metric = StreamingAccuracy()
    
    for batch in tqdm(eval_loader, desc="Eval"):
        batch = {k: v.to(DEVICE, non_blocking=TRAIN_CFG.pin_memory) for k, v in batch.items()}
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', 
                                enabled=USE_AMP, dtype=AMP_DTYPE):
            outputs = model(**batch)
        
        metric.update(outputs.logits, batch["labels"], outputs.loss)
    
    acc, avg_loss = metric.compute()
    ppl = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    return {"accuracy": acc, "loss": avg_loss, "perplexity": ppl}


# ========================= 主训练流程 =========================
def train_and_evaluate(method: str, dataset_cfg: Dict, tokenizer, pad_token_id: int):
    print(f"\n{'='*70}")
    print(f"Method: {method:12s} | Dataset: {dataset_cfg['config'].upper()}")
    print(f"Effective Batch Size: {TRAIN_CFG.batch_size} × {TRAIN_CFG.grad_accum_steps} = "
          f"{TRAIN_CFG.batch_size * TRAIN_CFG.grad_accum_steps}")
    print(f"{'='*70}")

    encoded = load_and_preprocess_dataset(dataset_cfg, tokenizer)
    collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        encoded["train"],
        batch_size=TRAIN_CFG.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=TRAIN_CFG.num_workers,
        pin_memory=TRAIN_CFG.pin_memory,
    )
    
    val_key = "validation_matched" if dataset_cfg["config"] == "mnli" else "validation"
    eval_loader = DataLoader(
        encoded[val_key],
        batch_size=TRAIN_CFG.batch_size * 2,  # 评估可以用更大 batch（无梯度）
        collate_fn=collator,
        num_workers=TRAIN_CFG.num_workers,
        pin_memory=TRAIN_CFG.pin_memory,
    )

    model = create_model(method, dataset_cfg["num_labels"], pad_token_id)
    
    # 优化器：fused AdamW（比 8bit 更稳定，速度相当）
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=TRAIN_CFG.lr, weight_decay=TRAIN_CFG.weight_decay, fused=True
    )
    
    total_steps = math.ceil(len(train_loader) / TRAIN_CFG.grad_accum_steps) * TRAIN_CFG.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * TRAIN_CFG.warmup_ratio), 
        num_training_steps=total_steps
    )
    
    # 梯度缩放器（bf16 不需要，但保留接口）
    scaler = torch.cuda.amp.GradScaler() if AMP_DTYPE == torch.float16 else None

    # ReLoRA
    relora_trainer = ReLoRATrainer(model, merge_every=1000) if method == "ReLoRA" else None

    # Aeloru 专用设置
    aeloru_layers = []
    aeloru_hooks = []
    aeloru_cache = {}
    need_hook = False

    if method.startswith("Aeloru"):
        for layer in model.modules():
            if isinstance(layer, AeloruLayer):
                aeloru_layers.append(layer)
                layer.to(DEVICE)
                if layer.W0.device != DEVICE:
                    layer.W0 = layer.W0.to(DEVICE)
                if layer.bias.device != DEVICE:
                    layer.bias = layer.bias.to(DEVICE)
                if layer.original_linear is not None:
                    layer.original_linear.to(DEVICE)
        
        need_hook = any([
            layer.cfg.use_hebbian or layer.cfg.use_fisher or 
            layer.cfg.use_hongwen or layer.cfg.use_relora
            for layer in aeloru_layers
        ]) if aeloru_layers else False
        
        if need_hook:
            def forward_hook(mod, inp, out):
                if mod.training and isinstance(mod, AeloruLayer):
                    aeloru_cache[id(mod)] = (inp[0].detach(), out.detach())
            for layer in aeloru_layers:
                aeloru_hooks.append(layer.register_forward_hook(forward_hook))
        
        print(f"[Aeloru] {len(aeloru_layers)} layers | hooks={'ON' if need_hook else 'OFF'}")

    # 训练循环
    history = {"train_loss": [], "eval_loss": [], "eval_acc": [], "step_ms": [], "peak_mem_mb": []}
    best_acc = 0.0
    
    for epoch in range(TRAIN_CFG.num_epochs):
        train_loss, step_ms, peak_mem = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            relora_trainer, aeloru_layers, aeloru_cache, need_hook
        )
        
        eval_metrics = {"accuracy": 0.0, "loss": 999.0, "perplexity": float('inf')}
        if (epoch + 1) % TRAIN_CFG.eval_every_n_epoch == 0 or epoch == TRAIN_CFG.num_epochs - 1:
            eval_metrics = evaluate(model, eval_loader)
        
        history["train_loss"].append(train_loss)
        history["eval_loss"].append(eval_metrics["loss"])
        history["eval_acc"].append(eval_metrics["accuracy"])
        history["step_ms"].append(step_ms)
        history["peak_mem_mb"].append(peak_mem)
        
        tqdm.write(
            f"  Epoch {epoch+1}/{TRAIN_CFG.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Eval Acc: {eval_metrics['accuracy']:.4f} | "
            f"Speed: {step_ms:.1f}ms/step | Peak Mem: {peak_mem:.1f}MB"
        )
        
        if eval_metrics["accuracy"] > best_acc:
            best_acc = eval_metrics["accuracy"]

    # 清理
    for h in aeloru_hooks:
        h.remove()
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {"accuracy": best_acc, "perplexity": history["eval_loss"][-1]}, history


# ========================= 绘图 =========================
def plot_results(all_results: Dict):
    methods = list(all_results.keys())
    datasets = [cfg["config"].upper() for cfg in DATASET_CONFIGS]
    
    fig, axes = plt.subplots(len(datasets), 2, figsize=(16, 4.5 * len(datasets)))
    if len(datasets) == 1:
        axes = np.array([axes])
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    color_map = {m: colors[i] for i, m in enumerate(methods)}
    
    for idx, ds_name in enumerate(datasets):
        ax_loss = axes[idx, 0]
        ax_speed = axes[idx, 1]
        
        for method in methods:
            if ds_name not in all_results[method]:
                continue
            _, hist = all_results[method][ds_name]
            epochs = np.arange(1, len(hist["train_loss"]) + 1)
            
            ax_loss.plot(epochs, hist["train_loss"], '-o', color=color_map[method], label=f"{method} Train")
            ax_loss.plot(epochs, hist["eval_loss"], '--s', color=color_map[method], label=f"{method} Eval", alpha=0.7)
            
            ax_speed.plot(epochs, hist["step_ms"], '-^', color=color_map[method], label=f"{method} Latency")
        
        ax_loss.set_title(f"Loss - {ds_name}")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=7, ncol=2)
        ax_loss.grid(True, alpha=0.3)
        
        ax_speed.set_title(f"Speed - {ds_name}")
        ax_speed.set_xlabel("Epoch")
        ax_speed.set_ylabel("ms/step")
        ax_speed.legend(fontsize=7)
        ax_speed.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("performance_report.png", dpi=200, bbox_inches='tight')
    print("\n[Report] 性能图表已保存至 performance_report.png")


# ========================= 主入口 =========================
def run_experiment():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    
    print(f"[System] Device: {DEVICE} | AMP: {AMP_DTYPE} | Compile: {hasattr(torch, 'compile')}")
    print(f"[Config] Batch: {TRAIN_CFG.batch_size} | Accum: {TRAIN_CFG.grad_accum_steps} | "
          f"Effective: {TRAIN_CFG.batch_size * TRAIN_CFG.grad_accum_steps}")

    # 建议测试方法：全基线 + Aeloru 双模式
    methods = ["LoRA", "DoRA", "ReLoRA", "Aeloru-Base", "Aeloru-Full"]
    # methods = ["Aeloru-Full"]  # 单独调试时可只跑一个
    
    results = {}
    all_histories = {}

    for ds_cfg in DATASET_CONFIGS:
        ds_name = ds_cfg["config"].upper()
        for method in methods:
            try:
                metrics, history = train_and_evaluate(method, ds_cfg, tokenizer, pad_token_id)
                results.setdefault(method, {})[ds_name] = metrics
                all_histories.setdefault(method, {})[ds_name] = (metrics, history)
            except Exception as e:
                print(f"[ERROR] {method} on {ds_name}: {e}")
                import traceback
                traceback.print_exc()
                results.setdefault(method, {})[ds_name] = {"accuracy": 0.0, "perplexity": float('inf')}

    # 汇总表
    rows = []
    for method in methods:
        row = {"Method": method}
        accs = []
        for ds_cfg in DATASET_CONFIGS:
            ds_name = ds_cfg["config"].upper()
            m = results.get(method, {}).get(ds_name, {"accuracy": 0.0})
            row[f"{ds_name}_Acc"] = f"{m['accuracy']:.4f}"
            accs.append(m["accuracy"])
        row["Avg_Acc"] = f"{np.mean(accs):.4f}"
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80)
    print(df.to_string(index=False))
    df.to_csv("benchmark_results.csv", index=False, encoding="utf-8-sig")
    
    plot_results(all_histories)
    return df


if __name__ == "__main__":
    df = run_experiment()
    print(f"\n[Done] 预览:\n{df.head().to_string(index=False)}")