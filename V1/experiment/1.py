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
import torch.amp

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

# 获取当前脚本所在目录的上层目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # E:\Qelys\AeloRu\V1\experiment
parent_dir = os.path.dirname(current_dir)                 # E:\Qelys\AeloRu\V1

# 将上层目录添加到模块搜索路径
sys.path.append(parent_dir)
print(sys.path)
from aeloru_layer import AeloruConfig , AeloruLayer , inject_aeloru , test_aeloru

import urllib3
urllib3.disable_warnings()

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

os.environ["PYTHONIOENCODING"] = "utf-8"

import warnings
warnings.filterwarnings("ignore")

# 开启所有CUDA优化
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 启用内存高效的注意力（Qwen2原生支持）
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

# 禁用不必要的调试功能
torch.set_grad_enabled(True)
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = "models/Qwen2.5-0.5B"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 8 
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
# DATASET_CONFIGS = [
#     {"name": "glue", "config": "mrpc", "text_cols": ["sentence1", "sentence2"], "label_col": "label", "num_labels": 2},
# ]
LORA_R = 8 
LORA_ALPHA = 4.0
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "v_proj"]


# ========================= ReLoRA 训练包装 =========================

class ReLoRATrainer:
    def __init__(self, model, merge_every: int = 1000, lr: float = LR):
        self.model = model
        self.merge_every = merge_every
        self.step_counter = 0
        self.lr = lr

    def maybe_merge_and_reinit(self, global_step: int):
        if global_step > 0 and global_step % self.merge_every == 0:
            print(f"[ReLoRA] Merging at step {global_step}...")
            self._merge_lora_weights()
            self._reset_lora_parameters()

    def _merge_lora_weights(self):
        for module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                scaling = getattr(module, 'scaling', module.lora_alpha / module.r)
                delta = (module.lora_B @ module.lora_A) * scaling
                module.base_layer.weight.data.add_(delta.t())

    def _reset_lora_parameters(self):
        for module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                nn.init.zeros_(module.lora_B)


# ========================= 数据集加载与预处理 =========================

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


# ========================= 训练与评估 =========================

def compute_metrics(eval_preds, eval_loss: float) -> Dict[str, float]:
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    ppl = math.exp(eval_loss) if eval_loss < 10 else float('inf')
    return {"accuracy": acc, "perplexity": ppl}


def create_peft_model(method: str, num_labels: int, pad_token_id: int):
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, trust_remote_code=True,torch_dtype=AMP_DTYPE
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
            for module in model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'base_layer'):
                    module.magnitude = nn.Parameter(module.base_layer.weight.norm(p=2, dim=1).detach().clone())

    elif method == "ReLoRA":
        config = LoraConfig(
            r=LORA_R, lora_alpha=int(LORA_ALPHA), target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(base, config)

    elif method == "Aeloru":
        cfg = AeloruConfig(
            r = LORA_R,
            lora_alpha = LORA_ALPHA,
            LoRA_lr = LR,
            verbose = False,

            # === 关键：关闭不兼容compile的异步功能 ===
            use_hidora = True,
            use_relora = True,
            use_hebbian = True,
            use_fisher = True,
            use_hongwen = False,  # 初期先关闭，稳定后再开
            use_orthogonal_penalty = True,
            use_energy_budget = True,

            # 关闭所有异步操作，让compile能完全优化
            async_merge = False,
            fisher_async = False,

            # 降低计算频率，减少开销
            fisher_compute_interval=2000,
            fisher_full_snapshot_interval=20000,
            hebbian_accum_steps=16,  # 从8改成16，减少更新次数
            ortho_random_proj=4,     # 从8改成4，计算量减半
            )
        model = inject_aeloru(base, target_names=TARGET_MODULES, cfg=cfg)
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model, backend="eager", fullgraph=False)  
    

        print("[Aeloru] torch.compile enabled")
    else:
        raise ValueError(f"Unknown method: {method}")

    return model


def train_and_evaluate(method: str, dataset_cfg: Dict, tokenizer, pad_token_id: int):
    print(f"\n{'='*60}")
    print(f"Method: {method:8s} | Dataset: {dataset_cfg['config'].upper()}")
    print(f"{'='*60}")

    encoded = load_and_preprocess_dataset(dataset_cfg, tokenizer)
    collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        encoded["train"],
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collator,
        num_workers=2,          # Windows上最多设4，2是最稳定的
        pin_memory=True,        # 数据直接加载到GPU显存
        prefetch_factor=2,      # 提前预加载2个batch
        persistent_workers=True # 避免每次epoch重新创建worker
        )
    
    val_key = "validation_matched" if dataset_cfg["config"] == "mnli" else "validation"
    eval_loader = DataLoader(
        encoded[val_key], 
        batch_size=BATCH_SIZE, 
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
        )

    model = create_peft_model(method, dataset_cfg["num_labels"], pad_token_id)
    model.to(DEVICE)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps
    )

    relora_trainer = ReLoRATrainer(model, merge_every=1000, lr=LR) if method == "ReLoRA" else None

    # ===== Aeloru 专用设置 =====
    aeloru_layers = []
    aeloru_hooks = []
    aeloru_cache = {}

    if method == "Aeloru":
        for layer in model.modules():
            if isinstance(layer, AeloruLayer):
                aeloru_layers.append(layer)

        # 注册 forward hook 捕获每层输入输出，用于 post_step_update
        # 替代原来的循环注册hook
    aeloru_cache = {}
    def forward_hook(mod, inp, out):
        if mod.training and isinstance(mod, AeloruLayer):
            aeloru_cache[id(mod)] = (inp[0].detach(), out.detach())

    for layer in aeloru_layers:
        handle = layer.register_forward_hook(forward_hook)
        aeloru_hooks.append(handle)

    print(f"[Aeloru] Found {len(aeloru_layers)} injected layers")

    epoch_train_losses = []
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
            optimizer.zero_grad()

            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with torch.amp.autocast(device_type=device_type, enabled=USE_AMP, dtype=AMP_DTYPE):
                outputs = model(**batch)
                loss = outputs.loss

                # Aeloru 正交惩罚（新版方法名 get_ortho_penalty）
                if method == "Aeloru" and aeloru_layers:
                    ortho_loss = sum(
                        layer.get_ortho_penalty() for layer in aeloru_layers
                        if hasattr(layer, 'get_ortho_penalty')
                    )
                    loss = loss + ortho_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()

            # ===== Aeloru 后处理（必须在 optimizer.step() 之后）=====
            if method == "Aeloru" and aeloru_layers:
                any_merged = False
                for layer in aeloru_layers:
                    lid = id(layer)
                    if lid in aeloru_cache:
                        x_c, y_c = aeloru_cache[lid]
                        merged = layer.post_step_update(x_c, y_c, is_correct=True)
                        if merged:
                            any_merged = True
                        del aeloru_cache[lid]
                if any_merged:
                    optimizer.state.clear()
                    print(f"  [Aeloru] Optimizer state cleared after merge")

            if relora_trainer:
                relora_trainer.maybe_merge_and_reinit(global_step)

            global_step += 1
            epoch_loss += loss.item()
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

    # 清理 hooks
    for h in aeloru_hooks:
        h.remove()

    del model, optimizer
    torch.cuda.empty_cache()
    return final_metrics, epoch_train_losses, epoch_eval_losses


# ========================= 绘图 =========================

def plot_all_loss_curves(all_results: Dict):
    methods = ["Aeloru","LoRA", "DoRA", "ReLoRA"]
    datasets = [cfg["config"].upper() for cfg in DATASET_CONFIGS]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 4 * len(datasets)))
    if len(datasets) == 1:
        axes = [axes]
    colors = {"LoRA": "#1f77b4", "DoRA": "#ff7f0e", "ReLoRA": "#2ca02c", "Aeloru": "#d62728"}

    for idx, ds_name in enumerate(datasets):
        ax = axes[idx]
        for method in methods:
            if ds_name in all_results.get(method, {}):
                _, train_losses, eval_losses = all_results[method][ds_name]
                epochs = list(range(1, len(train_losses) + 1))
                ax.plot(epochs, train_losses, color=colors[method], linestyle='-', marker='o', label=f"{method} Train", alpha=0.8)
                ax.plot(epochs, eval_losses, color=colors[method], linestyle='--', marker='s', label=f"{method} Eval", alpha=0.8)
        ax.set_title(f"Loss Curves - {ds_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)

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
    methods = ["Aeloru"]
    results = defaultdict(dict)
    all_loss_records = defaultdict(dict)

    for ds_cfg in DATASET_CONFIGS:
        ds_name = ds_cfg["config"].upper()
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
                    [float('inf')] * NUM_EPOCHS, [float('inf')] * NUM_EPOCHS,
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
    print(f"部分展示{df[:3].to_string(index=False)}")
