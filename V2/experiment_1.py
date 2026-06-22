# 导入 所需 库

import os
#换用中国镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#关闭SSL
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["NO_PROXY"] = "hf-mirror.com,cdn-lfs.hf-mirror.com"
for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(key, None)


import math
import copy
import warnings
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
from aeloru_layer import AeloruConfig, AeloruLayer, inject_aeloru, train_aeloru_step

import urllib3
urllib3.disable_warnings()

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

os.environ["PYTHONIOENCODING"] = "utf-8"


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = "models/Qwen2.5-0.5B"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
LR = 2e-4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 42

AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
USE_AMP = True

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATASET_CONFIGS = [
    {"name": "glue", "config": "sst2", "text_cols": ["sentence"], "label_col": "label", "num_labels": 2},
    {"name": "glue", "config": "mrpc", "text_cols": ["sentence1", "sentence2"], "label_col": "label", "num_labels": 2},
    {"name": "glue", "config": "qnli", "text_cols": ["question", "sentence"], "label_col": "label", "num_labels": 2},
    {"name": "glue", "config": "rte",  "text_cols": ["sentence1", "sentence2"], "label_col": "label", "num_labels": 2},
    {"name": "glue", "config": "cola", "text_cols": ["sentence"], "label_col": "label", "num_labels": 2},
]

LORA_R = 8
LORA_ALPHA = 4.0
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj"]


# ========================= ReLoRA =========================

class ReLoRATrainer:
    def __init__(self, model, merge_every: int = 1000, lr: float = LR):
        self.model = model
        self.merge_every = merge_every
        self.lr = lr

    def maybe_merge_and_reinit(self, global_step: int):
        if global_step > 0 and global_step % self.merge_every == 0:
            print(f"[ReLoRA] Merging at step {global_step}...")
            self._merge_lora_weights()
            self._reset_lora_parameters()

    def _merge_lora_weights(self):
        for _, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                scaling = getattr(module, 'scaling', module.lora_alpha / module.r)
                delta = (module.lora_B @ module.lora_A) * scaling
                module.base_layer.weight.data.add_(delta.t())

    def _reset_lora_parameters(self):
        for _, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                nn.init.zeros_(module.lora_B)


# ========================= 数据集 =========================

def load_and_preprocess_dataset(ds_cfg: Dict, tokenizer):
    raw = load_dataset(ds_cfg["name"], ds_cfg["config"])
    text_cols = ds_cfg["text_cols"]
    label_col = ds_cfg["label_col"]

    def tokenize_fn(examples):
        if len(text_cols) == 1:
            return tokenizer(examples[text_cols[0]], truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)
        else:
            return tokenizer(examples[text_cols[0]], examples[text_cols[1]],
                           truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)

    encoded = raw.map(tokenize_fn, batched=True)
    cols_to_remove = list(text_cols)
    for split in encoded.keys():
        for col in list(encoded[split].column_names):
            if col not in ["input_ids", "attention_mask", label_col]:
                cols_to_remove.append(col)
    cols_to_remove = list(set(cols_to_remove))
    encoded = encoded.remove_columns(cols_to_remove)
    if label_col != "labels":
        encoded = encoded.rename_column(label_col, "labels")
    encoded.set_format("torch")
    return encoded


# ========================= 评估 =========================

def compute_metrics(eval_preds, eval_loss: float) -> Dict[str, float]:
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    ppl = math.exp(eval_loss) if eval_loss < 10 else float('inf')
    return {"accuracy": acc, "perplexity": ppl}


def create_peft_model(method: str, num_labels: int, pad_token_id: int):
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, trust_remote_code=True,
    ).to(DEVICE)
    base.config.pad_token_id = pad_token_id

    if method == "LoRA":
        config = LoraConfig(
            r=LORA_R, lora_alpha=int(LORA_ALPHA), target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(base, config)

    elif method == "DoRA":
        try:
            config = LoraConfig(
                r=LORA_R, lora_alpha=int(LORA_ALPHA), target_modules=TARGET_MODULES,
                lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.SEQ_CLS,
                use_dora=True,
            )
            model = get_peft_model(base, config)
        except Exception as e:
            print(f"[Warn] DoRA fallback: {e}")
            config = LoraConfig(
                r=LORA_R, lora_alpha=int(LORA_ALPHA), target_modules=TARGET_MODULES,
                lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.SEQ_CLS,
            )
            model = get_peft_model(base, config)
            for _, module in model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'base_layer'):
                    module.magnitude = nn.Parameter(module.base_layer.weight.norm(p=2, dim=1).detach().clone())

    elif method == "ReLoRA":
        config = LoraConfig(
            r=LORA_R, lora_alpha=int(LORA_ALPHA), target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(base, config)
    elif method == "Aeloru":
        # ========== 【关键修复】新版 Aeloru 配置 ==========
        cfg = AeloruConfig(
            # --- 基础维度 ---
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            LoRA_lr=LR,
            
            # --- 核心功能开关(按需启用)---
            use_hidora=True,               # Hi-DoRA 幅度调制
            use_relora=True,               # ReLoRA 合并重置
            use_hebbian=True,              # Hebbian 在线学习
            use_fisher=False,               # Fisher 认知掩码
            use_hongwen=True,              # Hong Wen 状态机
            use_orthogonal_penalty=True,   # 正交惩罚
            use_energy_budget=True,        # 能量预算
            
            # --- 新增功能(v2.0)---
            use_lateral_connection=True,           # PEM 侧向连接
            use_homeostatic_plasticity=True,       # PEM 稳态可塑性
            use_hgf_fisher=True,                   # HGF 闭式 Fisher
            use_volatility_coupling=True,          # HGF 波动率耦合
            use_dlam_sleep=True,                   # DLAM 睡眠
            sleep_condition_threshold=500.0,       # 条件数触发阈值（提升以避免频繁误触发）
            min_steps_between_sleep=1000,          # 两次睡眠间最小步数（拉到千步级，降低开销）

            
            # --- 预测编码三方向 ---
            use_predictive_coding=True,            # 预测编码神经动态
            use_source_domain_constraint=True,     # 源信号域约束
            presumed_domain="nnantisparse",        # 非负反稀疏域
            use_online_covariance=True,            # 在线协方差
            use_whitening=True,                    # 动态白化
            
            # --- HGF 闭式更新(实验性，可选)---
            use_hgf_closed_form=True,             # 默认关闭，使用标准autograd更稳定
            hgf_loss_mode="ce",                    # CE 模式
            hgf_label_smoothing=0.0,
            
            # --- 训练控制 ---
            hebbian_before_backprop=True,         # 标准顺序：先BP后Hebbian
            hebbian_accum_steps=4,                 # Hebbian累积步数
            fisher_mode="off",                      # 关闭 Fisher分层策略 使用HGF
            fisher_topk_ratio=0.2,
            merge_every=1000,                      # ReLoRA合并周期
            red_threshold=0.65,                    # 红温阈值
            verbose=False,                         # 关闭详细日志
            diagnostic_interval=100000,              # 诊断间隔
            enable_cognitive_report=False,         # 关闭认知报告避免同步
            USE_AMP=USE_AMP,
            AMP_DTYPE=AMP_DTYPE,
            device=str(DEVICE),
        )
        model = inject_aeloru(base, target_names=TARGET_MODULES, cfg=cfg)

    else:
        raise ValueError(f"Unknown method: {method}")

    return model


# ========================= 训练 =========================

def train_and_evaluate(method: str, dataset_cfg: Dict, tokenizer, pad_token_id: int):
    print(f"\n{'='*60}")
    print(f"Method: {method:8s} | Dataset: {dataset_cfg['config'].upper()}")
    print(f"{'='*60}")

    encoded = load_and_preprocess_dataset(dataset_cfg, tokenizer)
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(encoded["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_key = "validation_matched" if dataset_cfg["config"] == "mnli" else "validation"
    eval_loader = DataLoader(encoded[val_key], batch_size=BATCH_SIZE, collate_fn=collator)

    model = create_peft_model(method, dataset_cfg["num_labels"], pad_token_id)
    model.to(device=DEVICE)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps
    )

    relora_trainer = ReLoRATrainer(model, merge_every=1000, lr=LR) if method == "ReLoRA" else None

    # ========== 【关键修复】Aeloru 层收集与优化器构建 ==========
    aeloru_layers = []
    if method == "Aeloru":
        aeloru_layers = [m for m in model.modules() if isinstance(m, AeloruLayer)]
        # 为每个Aeloru层构建独立优化器(用于train_aeloru_step)
        aeloru_optimizers = {}
        for layer in aeloru_layers:
            layer_params = layer.get_trainable_params()
            if layer_params:
                aeloru_optimizers[id(layer)] = torch.optim.AdamW(
                    layer_params, lr=layer.cfg.LoRA_lr, weight_decay=WEIGHT_DECAY
                )
        print(f"[Aeloru] 注入 {len(aeloru_layers)} 层，构建 {len(aeloru_optimizers)} 个优化器")

    epoch_train_losses = []
    step_train_losses = []  # 每步训练 loss，用于绘制更密集的曲线
    epoch_eval_losses = []
    global_step = 0
    best_eval_acc = 0.0
    final_metrics = {}

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for batch in pbar:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            if method == "Aeloru":
                # ========== 【关键修复】Aeloru 专用训练步 ==========
                # 标准transformers模型前向：获取hidden states和logits
                with autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
                    outputs = model(**batch)
                    loss = outputs.loss
                
                # 正交惩罚(所有Aeloru层累加)
                ortho_loss = torch.tensor(0.0, device=DEVICE)
                for layer in aeloru_layers:
                    ortho_loss = ortho_loss + layer.get_ortho_penalty()
                
                total_loss = loss + ortho_loss
                
                # 标准反向传播
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                
                # ========== 【关键修复】Aeloru post_step_update ==========
                # 获取中间层输出用于Hebbian更新
                # 注意：对于分类任务，我们使用最后一层隐藏状态
                with torch.no_grad():
                    # 重新前向获取中间表示(用于Hebbian)
                    base_outputs = model.base_model(**batch, output_hidden_states=True) if hasattr(model, 'base_model') else model(**batch, output_hidden_states=True)
                    if getattr(base_outputs, 'hidden_states', None) is not None:
                        hidden_states = base_outputs.hidden_states[-1]  # 最后一层
                    else:
                        hidden_states = base_outputs.last_hidden_state if hasattr(base_outputs, 'last_hidden_state') else base_outputs[0]
                    
                    # 对每个Aeloru层执行post_step_update
                    # 由于无法精确知道每层的输入，我们使用近似：
                    # 选择 in_features == out_features == hidden_size 的层（通常是 q_proj），
                    # 用最后一层隐藏状态的均值作为其输入/输出代理。若不存在匹配层，
                    # 则对最后一层使用基座权重投影到其输出空间，避免维度不匹配。
                    if aeloru_layers:
                        hidden_size = hidden_states.size(-1)
                        h_mean = hidden_states.mean(dim=1)  # (batch, hidden_size)
                        
                        target_layer = None
                        for layer in reversed(aeloru_layers):
                            if layer.in_features == hidden_size and layer.out_features == hidden_size:
                                target_layer = layer
                                break
                        
                        if target_layer is None:
                            target_layer = aeloru_layers[-1]
                            # 对 v_proj 等维度不匹配的层，用基座权重投影到输出空间
                            with torch.no_grad():
                                W_base = target_layer.W0 + target_layer._get_W_acc()
                                y_approx = F.linear(h_mean.to(W_base.dtype), W_base)
                        else:
                            y_approx = h_mean
                        
                        # 使用loss作为reward信号(loss越小越正确)
                        is_correct = loss.item() < 1.0
                        target_layer.post_step_update(
                            h_mean, y_approx,  # 近似输入输出
                            is_correct=is_correct,
                            y_target=None
                        )
                
            else:
                # 标准方法(LoRA/DoRA/ReLoRA)
                optimizer.zero_grad()
                with autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
                    outputs = model(**batch)
                    loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()

            if relora_trainer:
                relora_trainer.maybe_merge_and_reinit(global_step)

            global_step += 1
            epoch_loss += loss.item()
            step_train_losses.append(loss.item())  # 每步记录训练 loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)

        # 评估
        model.eval()
        all_logits, all_labels = [], []
        eval_loss_sum = 0.0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                with autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
                    outputs = model(**batch)
                eval_loss_sum += outputs.loss.item() * batch["input_ids"].size(0)
                all_logits.append(outputs.logits.float().cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())

        eval_loss = eval_loss_sum / len(eval_loader.dataset)
        epoch_eval_losses.append(eval_loss)

        metrics = compute_metrics((np.concatenate(all_logits), np.concatenate(all_labels)), eval_loss)
        print(f"  Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
              f"Eval Loss: {eval_loss:.4f} | Acc: {metrics['accuracy']:.4f} | PPL: {metrics['perplexity']:.2f}")

        if metrics["accuracy"] > best_eval_acc:
            best_eval_acc = metrics["accuracy"]
            final_metrics = metrics.copy()

    del model, optimizer
    if method == "Aeloru":
        del aeloru_optimizers
    torch.cuda.empty_cache()
    return final_metrics, step_train_losses, epoch_eval_losses


# ========================= 绘图 =========================

def plot_all_loss_curves(all_results: Dict):
    methods = ["LoRA", "DoRA", "ReLoRA", "Aeloru"]
    datasets = [cfg["config"].upper() for cfg in DATASET_CONFIGS]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(14, 4 * len(datasets)))
    if len(datasets) == 1:
        axes = [axes]
    colors = {"LoRA": "#1f77b4", "DoRA": "#ff7f0e", "ReLoRA": "#2ca02c", "Aeloru": "#d62728"}

    for idx, ds_name in enumerate(datasets):
        ax = axes[idx]
        for method in methods:
            if ds_name in all_results.get(method, {}):
                _, train_losses, eval_losses = all_results[method][ds_name]
                num_steps = len(train_losses)
                num_epochs = len(eval_losses)
                steps = list(range(1, num_steps + 1))
                # 训练曲线：每步一个点，更密集
                ax.plot(steps, train_losses, color=colors[method], linestyle='-', alpha=0.6, label=f"{method} Train (step)")
                # 评估曲线：每个 epoch 结束时的步数位置画点
                if num_epochs > 0 and num_steps > 0:
                    steps_per_epoch = max(1, num_steps // num_epochs)
                    eval_steps = [min(steps_per_epoch * i, num_steps - 1) for i in range(1, num_epochs + 1)]
                    ax.plot(eval_steps, eval_losses, color=colors[method], linestyle='--', marker='s', markersize=4, label=f"{method} Eval (epoch)")
        ax.set_title(f"Loss Curves - {ds_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("loss_curves_all_datasets.png", dpi=150, bbox_inches='tight')
    print("\nLoss 曲线图已保存至 loss_curves_all_datasets.png")
    plt.show()


# ========================= 主入口 =========================

def run_experiment():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    print(f"[Tokenizer] pad_token: {tokenizer.pad_token}, pad_token_id: {pad_token_id}")
    print(f"[AMP] dtype={AMP_DTYPE}, enabled={USE_AMP}")

    # methods = ["LoRA", "DoRA", "ReLoRA", "Aeloru"]
    methods = ["Aeloru"]  # 调试时只跑Aeloru
    results = defaultdict(dict)
    all_loss_records = defaultdict(dict)

    for ds_cfg in DATASET_CONFIGS:
        ds_name = ds_cfg["config"].upper()
        # 预计算该数据集的总步数，用于异常时填充 step-level loss
        encoded = load_and_preprocess_dataset(ds_cfg, tokenizer)
        collator = DataCollatorWithPadding(tokenizer)
        train_loader = DataLoader(encoded["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
        total_steps = len(train_loader) * NUM_EPOCHS

        for method in methods:
            try:
                metrics, train_losses, eval_losses = train_and_evaluate(method, ds_cfg, tokenizer, pad_token_id)
                results[method][ds_name] = metrics
                all_loss_records[method][ds_name] = (metrics, train_losses, eval_losses)
            except Exception as e:
                print(f"[ERROR] {method} on {ds_name}: {e}")
                import traceback
                traceback.print_exc()
                results[method][ds_name] = {"accuracy": 0.0, "perplexity": float('inf')}
                all_loss_records[method][ds_name] = (
                    {"accuracy": 0.0, "perplexity": float('inf')},
                    [float('inf')] * total_steps, [float('inf')] * NUM_EPOCHS,
                )

    rows = []
    for method in methods:
        row = {"Method": method}
        accs, ppls = [], []
        for ds_cfg in DATASET_CONFIGS:
            ds_name = ds_cfg["config"].upper()
            m = results[method].get(ds_name, {"accuracy": 0.0, "perplexity": float('inf')})
            row[f"{ds_name}_Acc"] = f"{m['accuracy']:.4f}"
            row[f"{ds_name}_PPL"] = f"{m['perplexity']:.2f}"
            accs.append(m["accuracy"])
            ppls.append(m["perplexity"] if m["perplexity"] != float('inf') else 100)
        row["Avg_Acc"] = f"{np.mean(accs):.4f}"
        row["Avg_PPL"] = f"{np.mean(ppls):.2f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "="*80)
    print("实验结果汇总表")
    print("="*80)
    print(df.to_string(index=False))
    df.to_csv("peft_comparison_results.csv", index=False, encoding="utf-8-sig")
    print("\n结果已保存至 peft_comparison_results.csv")
    plot_all_loss_curves(all_loss_records)
    return df


if __name__ == "__main__":
    df = run_experiment()