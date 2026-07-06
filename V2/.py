"""
PEFT Methods Comparison on GLUE Tasks
Methods: FT, LoRA, DoRA, Hi-DoRA, ReLoRA
Model: Qwen2.5-0.5B

改进：添加完善的显存释放与监控功能
"""

import os
# 换用中国镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["NO_PROXY"] = "hf-mirror.com,cdn-lfs.hf-mirror.com"
for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(key, None)

import math
import copy
import warnings
import gc
import traceback
import json
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

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

import urllib3
urllib3.disable_warnings()

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

os.environ["PYTHONIOENCODING"] = "utf-8"


# ============================================================================
# 全局配置
# ============================================================================
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_data")
os.makedirs(RESULTS_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = "models/Qwen2.5-0.5B"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
LR = 2e-4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 42

# 当前环境下，bf16 + GradScaler 会触发不兼容报错，因此默认关闭 AMP
# 以保证训练流程稳定运行。
AMP_DTYPE = torch.float32
USE_AMP = False


torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATASET_CONFIGS = [

    {"name": "glue", "config": "rte",  "text_cols": ["sentence1", "sentence2"], "label_col": "label", "num_labels": 2},

]


# ============================================================================
# 新增：显存管理工具模块
# ============================================================================

class GPUMemoryManager:
    """
    显存管理器：监控、释放、报告显存使用情况
    """
    def __init__(self, device: torch.device = None):
        self.device = device or DEVICE
        self.memory_log = []

    def get_memory_info(self) -> Dict[str, float]:
        """获取当前显存使用信息（MB）"""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}

        allocated = torch.cuda.memory_allocated(self.device) / 1024**2
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**2
        free = total - allocated

        return {
            "allocated": round(allocated, 2),
            "reserved": round(reserved, 2),
            "free": round(free, 2),
            "total": round(total, 2),
        }

    def log(self, tag: str = ""):
        """记录当前显存状态"""
        info = self.get_memory_info()
        self.memory_log.append({"tag": tag, **info})
        print(f"  [VRAM] {tag}: Alloc={info['allocated']:.1f}MB | "
              f"Reserved={info['reserved']:.1f}MB | Free={info['free']:.1f}MB")
        return info

    def print_summary(self):
        """打印显存使用摘要"""
        print("\n" + "="*60)
        print("显存使用摘要")
        print("="*60)
        for entry in self.memory_log:
            print(f"  {entry['tag']:<30s} | Alloc: {entry['allocated']:>8.1f}MB | "
                  f"Free: {entry['free']:>8.1f}MB")
        print("="*60)


def aggressive_memory_cleanup(
    model=None,
    optimizer=None,
    scheduler=None,
    relora_trainer=None,
    data_loaders=None,
    tensors=None,
    device=None,
    verbose: bool = True,
):
    """
    激进的显存清理函数

    清理内容：
    1. 模型权重（包括 PEFT adapter）
    2. 优化器状态（一阶/二阶矩）
    3. 调度器状态
    4. DataLoader worker 进程
    5. 累积的张量
    6. Python 垃圾回收
    7. CUDA 缓存清空

    Args:
        model: 要清理的模型
        optimizer: 要清理的优化器
        scheduler: 要清理的调度器
        relora_trainer: ReLoRA 训练器
        data_loaders: DataLoader 列表
        tensors: 要显式删除的张量列表
        device: 目标设备
        verbose: 是否打印日志
    """
    device = device or DEVICE

    if verbose:
        before = torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
        print(f"\n[Memory Cleanup] 开始清理... (Before: {before:.1f}MB)")

    # 1. 清理模型相关
    if model is not None:
        # 清理 PEFT adapter 参数
        if hasattr(model, 'peft_config'):
            for adapter_name in list(model.peft_config.keys()):
                model.delete_adapter(adapter_name)

        # 清理所有参数的梯度
        for param in model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad = None

        # 清理 buffer
        for buffer in model.buffers():
            if buffer is not None:
                buffer.detach_()

        model.cpu()  # 先移到 CPU
        del model
        if verbose:
            print("  ✓ Model cleared")

    # 2. 清理优化器状态
    if optimizer is not None:
        for group in optimizer.param_groups:
            for p in group.get("params", []):
                if p in optimizer.state:
                    state = optimizer.state[p]
                    for key in list(state.keys()):
                        if isinstance(state[key], torch.Tensor):
                            state[key] = state[key].cpu()
                        del state[key]
                    optimizer.state[p] = {}
        del optimizer
        if verbose:
            print("  ✓ Optimizer state cleared")

    # 3. 清理调度器
    if scheduler is not None:
        del scheduler
        if verbose:
            print("  ✓ Scheduler cleared")

    # 4. 清理 ReLoRA 训练器
    if relora_trainer is not None:
        relora_trainer.lora_layers.clear()
        del relora_trainer
        if verbose:
            print("  ✓ ReLoRA trainer cleared")

    # 5. 清理 DataLoader（关闭 worker 进程）
    if data_loaders is not None:
        for loader in data_loaders if isinstance(data_loaders, list) else [data_loaders]:
            if hasattr(loader, '_iterator') and loader._iterator is not None:
                if hasattr(loader._iterator, '_shutdown_workers'):
                    loader._iterator._shutdown_workers()
            del loader
        if verbose:
            print("  ✓ DataLoaders cleared")

    # 6. 清理指定张量
    if tensors is not None:
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                tensor.detach_()
                del tensor
        if verbose:
            print("  ✓ Specified tensors cleared")

    # 7. Python 垃圾回收
    gc.collect()
    if verbose:
        print("  ✓ Python GC completed")

    # 8. CUDA 缓存清空
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        if verbose:
            print("  ✓ CUDA cache emptied")

    if verbose:
        after = torch.cuda.memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
        freed = before - after
        print(f"[Memory Cleanup] 完成 | Freed: {freed:.1f}MB | After: {after:.1f}MB\n")


def oom_safe_train(func):
    """
    装饰器：捕获 OOM 异常并清理显存
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"\n⚠️  OOM 错误捕获: {e}")
                print("执行紧急显存清理...")
                aggressive_memory_cleanup(verbose=True)
                raise RuntimeError(f"训练因显存不足失败。建议：减小 BATCH_SIZE 或 MAX_SEQ_LENGTH。原始错误: {e}")
            else:
                raise
    return wrapper


# ============================================================================
# 1. 数据加载与预处理
# ============================================================================

def load_and_preprocess_data(dataset_cfg, tokenizer):
    """加载并预处理 GLUE 数据集"""
    ds = load_dataset(dataset_cfg["name"], dataset_cfg["config"])
    text_cols = dataset_cfg["text_cols"]
    label_col = dataset_cfg["label_col"]

    def tokenize_fn(examples):
        if len(text_cols) == 1:
            texts = examples[text_cols[0]]
        else:
            texts = list(zip(*[examples[c] for c in text_cols]))
        result = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=MAX_SEQ_LENGTH,
        )
        # 保留 label 列
        result[label_col] = examples[label_col]
        return result

    # 只移除原始文本列，保留 tokenize 后的列和 label 列
    cols_to_remove = [c for c in ds["train"].column_names if c not in [label_col]]
    ds = ds.map(tokenize_fn, batched=True, remove_columns=cols_to_remove)
    ds = ds.rename_column(label_col, "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


# ============================================================================
# 2. ReLoRA 实现
# ============================================================================

class ReLoRATrainer:
    """
    ReLoRA: High-Rank Training Through Low-Rank Updates
    核心思想：周期性将 LoRA 权重合并到基权重中，重置 LoRA 矩阵，
    利用 rank(A+B) <= rank(A) + rank(B) 实现高秩更新
    """
    def __init__(
        self,
        model,
        relora_steps: int = 500,
        warmup_steps: int = 100,
        reset_optimizer: bool = True,
        magnitude_pruning: float = 0.0,
    ):
        self.model = model
        self.relora_steps = relora_steps
        self.warmup_steps = warmup_steps
        self.reset_optimizer = reset_optimizer
        self.magnitude_pruning = magnitude_pruning
        self.step_count = 0
        self.merge_count = 0
        self.lora_layers = []
        self._find_lora_layers()

    def _find_lora_layers(self):
        """查找模型中所有的 LoRA 层"""
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                self.lora_layers.append((name, module))

    def should_reset(self) -> bool:
        """判断当前是否需要执行 merge-and-reset"""
        return self.step_count > 0 and self.step_count % self.relora_steps == 0

    def merge_and_reset(self, optimizer):
        """
        执行 ReLoRA 的核心操作：
        1. 将 LoRA 权重合并到基权重
        2. 重置 LoRA A/B 矩阵
        3. 可选：重置优化器状态
        """
        print(f"\n[ReLoRA] Merge & Reset at step {self.step_count} (merge #{self.merge_count + 1})")

        # 1. 合并 LoRA 到基权重
        for name, module in self.lora_layers:
            if hasattr(module, "merge"):
                module.merge()
            else:
                # 手动合并: W_base += scaling * B @ A
                lora_A = module.lora_A["default"] if isinstance(module.lora_A, nn.ModuleDict) else module.lora_A
                lora_B = module.lora_B["default"] if isinstance(module.lora_B, nn.ModuleDict) else module.lora_B
                scaling = module.scaling["default"] if isinstance(module.scaling, dict) else module.scaling

                delta = (lora_B.weight @ lora_A.weight) * scaling
                module.base_layer.weight.data += delta

        # 2. 重置 LoRA 矩阵
        for name, module in self.lora_layers:
            lora_A = module.lora_A["default"] if isinstance(module.lora_A, nn.ModuleDict) else module.lora_A
            lora_B = module.lora_B["default"] if isinstance(module.lora_B, nn.ModuleDict) else module.lora_B

            # A 用高斯初始化
            nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
            # B 用零初始化
            nn.init.zeros_(lora_B.weight)

        # 3. 重置优化器状态
        if self.reset_optimizer and optimizer is not None:
            self._reset_optimizer_state(optimizer)

        self.merge_count += 1
        print(f"[ReLoRA] Merge complete. Total merges: {self.merge_count}")

    def _reset_optimizer_state(self, optimizer):
        """重置 Adam 优化器的状态（一阶和二阶矩）"""
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p not in optimizer.state:
                    continue
                state = optimizer.state[p]

                # 幅度裁剪：将低幅度的优化器状态置零
                if self.magnitude_pruning > 0:
                    if "exp_avg" in state:
                        mask = state["exp_avg"].abs() > state["exp_avg"].abs().quantile(self.magnitude_pruning)
                        state["exp_avg"] *= mask
                    if "exp_avg_sq" in state:
                        mask = state["exp_avg_sq"].abs() > state["exp_avg_sq"].abs().quantile(self.magnitude_pruning)
                        state["exp_avg_sq"] *= mask
                else:
                    # 完全重置
                    if "exp_avg" in state:
                        state["exp_avg"].zero_()
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"].zero_()

        print(f"[ReLoRA] Optimizer state {'pruned' if self.magnitude_pruning > 0 else 'reset'}.")

    def step(self):
        """每步调用，计数器递增"""
        self.step_count += 1


# ============================================================================
# 3. 模型构建
# ============================================================================

def build_model(method: str, num_labels: int, r: int = 8, alpha: float = 16.0):
    """
    构建指定微调方法的模型

    Methods:
        ft:      Full Fine-Tuning
        lora:    LoRA (Low-Rank Adaptation)
        dora:    DoRA (Weight-Decomposed LoRA) - via PEFT
        hidora:  Hi-DoRA (Hidden Direction LoRA) - 自定义实现
        relora:  ReLoRA (Periodic Merge & Reset LoRA)
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        trust_remote_code=True,
    )
    # 关键修复：设置模型的 pad_token_id，否则 batch_size > 1 时会报错
    model.config.pad_token_id = tokenizer.pad_token_id

    if method == "ft":
        # Full FT: 全部可训练
        for param in model.parameters():
            param.requires_grad = True
        return model, None

    # PEFT 方法配置
    if method in ["lora", "dora", "relora"]:
        use_dora = (method == "dora")
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            use_dora=use_dora,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        if method == "relora":
            relora_trainer = ReLoRATrainer(
                model,
                relora_steps=500,
                warmup_steps=100,
                reset_optimizer=True,
                magnitude_pruning=0.0,
            )
            return model, relora_trainer
        return model, None

    elif method == "hidora":
        # Hi-DoRA: 基于 PEFT LoRA，但注入自定义 hidden scale
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            use_dora=False,
        )
        model = get_peft_model(model, lora_config)

        # 为每个 LoRA 层注入 hidden_scale
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # 获取 rank
                rank = list(module.lora_A.values())[0].weight.shape[0]
                module.hidden_scale = nn.Parameter(torch.ones(rank))

                # 关键修复：用闭包正确捕获当前 module，且不再调用已被替换的 forward
                def make_forward(orig_module):
                    def forward(x, *args, **kwargs):
                        if hasattr(orig_module, "base_layer"):
                            result = orig_module.base_layer(x, *args, **kwargs)
                        else:
                            result = x

                        for adapter_name, lora_A in orig_module.lora_A.items():
                            lora_B = orig_module.lora_B[adapter_name]
                            scaling = orig_module.scaling[adapter_name]

                            # ========== 关键修复：补上 PEFT 原版的 dtype 转换 ==========
                            previous_dtype = x.dtype
                            # 把 x 转成和 LoRA 权重一样的 dtype（通常是 float32）
                            x_lora = x.to(lora_A.weight.dtype)

                            x_after_A = lora_A(x_lora)
                            if hasattr(orig_module, "hidden_scale"):
                                x_after_A = x_after_A * orig_module.hidden_scale

                            # 计算完再转回原来的 dtype（BFloat16），加回 result
                            result += lora_B(x_after_A).to(previous_dtype) * scaling
                        return result
                    return forward

                module.forward = make_forward(module)

        model.print_trainable_parameters()
        return model, None


# ============================================================================
# 4. 训练函数（含显存监控与释放）
# ============================================================================

@oom_safe_train
def train_model(
    model,
    train_loader,
    eval_loader,
    method: str,
    relora_trainer: Optional[ReLoRATrainer] = None,
    epochs: int = NUM_EPOCHS,   
    lr: float = LR,
):
    """
    通用训练函数，支持 ReLoRA 的特殊逻辑

    改进：
    - 添加显存监控
    - 每 epoch 后清理中间变量
    - 训练结束后全面清理
    """
    memory_mgr = GPUMemoryManager()
    memory_mgr.log("训练开始前")

    # 获取可训练参数
    if method == "ft":
        params = model.parameters()
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY, fused=True)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and AMP_DTYPE == torch.float16)

    history = {
        "train_loss": [],
        "eval_acc": [],
        "eval_f1": [],
        "steps": [],
    }

    global_step = 0
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f"[{method.upper()}] Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # ReLoRA: 检查是否需要 merge-and-reset
            if relora_trainer is not None and relora_trainer.should_reset():
                relora_trainer.merge_and_reset(optimizer)
                # ReLoRA 重置后需要 LR warm-up
                for _ in range(relora_trainer.warmup_steps):
                    scheduler.step()

            optimizer.zero_grad()

            if USE_AMP:
                with autocast(enabled=True, dtype=AMP_DTYPE):
                    outputs = model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            scheduler.step()

            if relora_trainer is not None:
                relora_trainer.step()

            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "vram": f"{memory_mgr.get_memory_info()['allocated']:.0f}MB"
            })

            # 每 100 步评估一次
            if global_step % 100 == 0:
                eval_metrics = evaluate_model(model, eval_loader)
                history["train_loss"].append(epoch_loss / batch_count)
                history["eval_acc"].append(eval_metrics["accuracy"])
                history["eval_f1"].append(eval_metrics.get("f1", eval_metrics["accuracy"]))
                history["steps"].append(global_step)
                model.train()

        # Epoch 结束评估
        eval_metrics = evaluate_model(model, eval_loader)
        print(f"  Epoch {epoch+1} | Loss: {epoch_loss/batch_count:.4f} | "
              f"Acc: {eval_metrics['accuracy']:.4f} | F1: {eval_metrics.get('f1', 0):.4f}")

        if eval_metrics["accuracy"] > best_acc:
            best_acc = eval_metrics["accuracy"]

        # ===== 每 epoch 结束后清理中间变量 =====
        # 清理 epoch 级别的累积变量
        del epoch_loss, batch_count
        if 'outputs' in locals():
            del outputs
        if 'loss' in locals():
            del loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        memory_mgr.log(f"Epoch {epoch+1} 结束后")

    history["best_acc"] = best_acc

    # ===== 训练结束后全面清理 =====
    # 清理训练过程中创建的所有对象
    del optimizer, scheduler, scaler, pbar
    if 'params' in locals():
        del params
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    memory_mgr.log("训练完全结束后")

    # 打印显存摘要
    memory_mgr.print_summary()

    return history


@torch.no_grad()
def evaluate_model(model, eval_loader):
    """评估模型（含显存清理）"""
    model.eval()
    all_preds, all_labels = [], []

    for batch in eval_loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

        # 清理 batch 级别的显存
        del batch, outputs, preds

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    # 清理评估累积数据
    del all_preds, all_labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"accuracy": acc, "f1": f1}


# ============================================================================
# 5. 主实验（含显存释放）
# ============================================================================

def save_plot_data(all_results, names, colors, output_dir=None):
    """保存绘图所需的完整结果数据，便于后续重新绘图。"""
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "names": names,
        "colors": colors,
        "methods": list(names.keys()),
        "results": all_results,
    }

    data_path = os.path.join(output_dir, "plot_data.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    flat_rows = []
    for method, entries in all_results.items():
        for entry in entries:
            flat_rows.append({
                "dataset": entry["dataset"],
                "method": method,
                "method_name": names.get(method, method),
                "best_acc": float(entry["best_acc"]),
                "steps": entry["history"].get("steps", []),
                "train_loss": entry["history"].get("train_loss", []),
                "eval_acc": entry["history"].get("eval_acc", []),
                "eval_f1": entry["history"].get("eval_f1", []),
            })

    csv_path = os.path.join(output_dir, "plot_metrics.csv")
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 为每个 method/dataset 单独保存一份 JSON，便于后续按任务查看
    for method, entries in all_results.items():
        for entry in entries:
            detail_path = os.path.join(output_dir, f"{method}_{entry['dataset']}_history.json")
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump({
                    "dataset": entry["dataset"],
                    "method": method,
                    "method_name": names.get(method, method),
                    "best_acc": entry["best_acc"],
                    "history": entry["history"],
                }, f, ensure_ascii=False, indent=2)

    print(f"  ✅ 绘图数据已保存: {data_path} 和 {csv_path}")
    return data_path, csv_path


def load_saved_plot_data(output_dir=None):
    """从磁盘读取之前保存的绘图数据。"""
    output_dir = output_dir or RESULTS_DIR
    data_path = os.path.join(output_dir, "plot_data.json")
    if not os.path.exists(data_path):
        return None, None, None

    with open(data_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload.get("results", {}), payload.get("names", {}), payload.get("colors", {})


def run_experiment():
    """运行完整实验（含显存监控与释放）"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    methods = ["ft", "lora", "dora", "hidora", "relora"]
    # methods=["hidora"]
    names = {
        "ft": "Full FT",
        "lora": "LoRA",
        "dora": "DoRA",
        "hidora": "Hi-DoRA",
        "relora": "ReLoRA",
    }
    colors = {
        "ft": "#00ff5e",
        "lora": "#3498db",
        "dora": "#aa2ecc",
        "hidora": "#ff0000",
        "relora": "#f39c12",
    }

    all_results = {m: [] for m in methods}

    # 全局显存管理器
    global_memory_mgr = GPUMemoryManager()
    global_memory_mgr.log("实验开始前")

    for ds_cfg in DATASET_CONFIGS:
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_cfg['name'].upper()} - {ds_cfg['config'].upper()}")
        print(f"{'='*70}")

        ds = load_and_preprocess_data(ds_cfg, tokenizer)
        data_collator = DataCollatorWithPadding(tokenizer)

        train_loader = DataLoader(
            ds["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator
        )
        eval_loader = DataLoader(
            ds["validation"], batch_size=BATCH_SIZE, collate_fn=data_collator
        )

        global_memory_mgr.log(f"数据集 {ds_cfg['config']} 加载后")

        for method in methods:
            print(f"\n--- Method: {names[method]} ---")

            # 记录方法开始前的显存
            method_start_mem = global_memory_mgr.get_memory_info()["allocated"]

            model, relora_trainer = build_model(method, ds_cfg["num_labels"], r=8, alpha=16.0)
            model = model.to(DEVICE)

            global_memory_mgr.log(f"{names[method]} 模型加载后")

            history = train_model(
                model, train_loader, eval_loader,
                method=method,
                relora_trainer=relora_trainer,
                epochs=NUM_EPOCHS,
            )

            result = {
                "dataset": ds_cfg["config"],
                "method": method,
                "best_acc": history["best_acc"],
                "history": history,
            }
            all_results[method].append(result)
            save_plot_data(all_results, names, colors)

            print(f"  Best Accuracy: {history['best_acc']:.4f}")

            # ===== 自动保存当前方法结果到 CSV =====
            method_result_df = pd.DataFrame([{
                "dataset": ds_cfg["config"],
                "method": names[method],
                "method_key": method,
                "best_acc": round(history["best_acc"], 6),
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "r": 8,
                "alpha": 16.0,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            }])
            method_csv_path = f"result_{method}_{ds_cfg['config']}.csv"
            method_result_df.to_csv(method_csv_path, index=False, encoding="utf-8-sig")
            print(f"  ✅ 方法结果已保存: {method_csv_path}")

            # ===== 关键改进：使用 aggressive_memory_cleanup 替代简单的 del =====
            aggressive_memory_cleanup(
                model=model,
                relora_trainer=relora_trainer,
                verbose=True,
            )

            global_memory_mgr.log(f"{names[method]} 清理后")

            # 验证显存是否回到方法开始前的水平
            method_end_mem = global_memory_mgr.get_memory_info()["allocated"]
            mem_leak = method_end_mem - method_start_mem
            if mem_leak > 50:  # 超过 50MB 视为泄漏
                print(f"⚠️  警告：{names[method]} 存在显存泄漏 {mem_leak:.1f}MB！")

        # 数据集切换时清理 DataLoader
        aggressive_memory_cleanup(
            data_loaders=[train_loader, eval_loader],
            verbose=False,
        )
        del ds
        global_memory_mgr.log(f"数据集 {ds_cfg['config']} 完全清理后")

    # 汇总结果
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    summary_rows = []
    for method in methods:
        accs = [r["best_acc"] for r in all_results[method]]
        avg_acc = np.mean(accs)
        print(f"  {names[method]:<10s} | Avg Acc: {avg_acc:.4f} | Tasks: {accs}")
        summary_rows.append({"Method": names[method], "Avg Accuracy": f"{avg_acc:.4f}"})

    # 保存汇总结果（带时间戳）
    df = pd.DataFrame(summary_rows)
    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    summary_csv_path = f"peft_comparison_results_{timestamp_str}.csv"
    df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 汇总结果已保存: {summary_csv_path}")

    # 同时保留无时间戳版本便于脚本调用
    df.to_csv("peft_comparison_results.csv", index=False, encoding="utf-8-sig")
    print("✅ 汇总结果已保存: peft_comparison_results.csv")

    # 最终显存报告
    global_memory_mgr.log("实验完全结束后")
    global_memory_mgr.print_summary()

    save_plot_data(all_results, names, colors)
    return all_results, names, colors


# ============================================================================
# 6. 可视化
# ============================================================================

def plot_results(all_results=None, names=None, colors=None, output_dir=None):
    """绘制对比图表；如未传入结果则从磁盘加载。"""
    if all_results is None or names is None:
        all_results, names, colors = load_saved_plot_data(output_dir)

    if not all_results:
        raise ValueError("没有可用于绘图的数据，请先运行实验或确认 plot_data.json 已生成。")

    methods = list(names.keys())
    datasets = [r["dataset"] for r in all_results["lora"]]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))    

    # 1. 各数据集上的最佳精度对比
    ax1 = axes[0, 0]
    x = np.arange(len(datasets))
    width = 0.15
    for i, method in enumerate(methods):
        accs = [r["best_acc"] for r in all_results[method]]
        ax1.bar(x + i * width, accs, width, label=names[method], color=colors[method], alpha=0.8)
    ax1.set_xlabel("Dataset")
    ax1.set_ylabel("Best Accuracy")
    ax1.set_title("Best Accuracy per Dataset")
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. 平均精度对比
    ax2 = axes[0, 1]
    avg_accs = [np.mean([r["best_acc"] for r in all_results[m]]) for m in methods]
    bars = ax2.bar([names[m] for m in methods], avg_accs, color=[colors[m] for m in methods], alpha=0.8)
    ax2.set_ylabel("Average Accuracy")
    ax2.set_title("Average Accuracy Across All Datasets")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, acc in zip(bars, avg_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10)

    # 3. 训练曲线 (以第一个数据集为例)
    ax3 = axes[1, 0]
    for method in methods:
        history = all_results[method][0]["history"]
        if history["steps"]:
            ax3.plot(history["steps"], history["eval_acc"], "-o",
                    color=colors[method], label=names[method], markersize=3, alpha=0.7)
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Accuracy")
    ax3.set_title(f"Training Curves ({datasets[0]})")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Loss 曲线
    ax4 = axes[1, 1]
    for method in methods:
        history = all_results[method][0]["history"]
        if history["steps"]:
            # 用 eval_acc 的互补作为近似 loss 趋势
            ax4.plot(history["steps"], history["train_loss"], "-s",
                    color=colors[method], label=names[method], markersize=3, alpha=0.7)
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Training Loss")
    ax4.set_title(f"Training Loss ({datasets[0]})")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle("PEFT Methods Comparison on GLUE", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("peft_glue_comparison.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.show()
    print("\n✅ Chart saved to peft_glue_comparison.png")


# ============================================================================
# 7. 入口
# ============================================================================
if __name__ == "__main__":
    all_results, names, colors = run_experiment()
    plot_results(all_results, names, colors)