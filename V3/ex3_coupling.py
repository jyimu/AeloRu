# -*- coding: utf-8 -*-
"""
ex3_coupling.py -- 实验三 B 阶段：真实 LLM 上区分「冗余耦合」与「功能性耦合」
==============================================================================

承接 ex3_synth.py（合成 ground-truth）。阶段 A 给出的结论：

  * PEM 自身（开环、冻结 A）**没有**任务选择性（SEL ≈ 0）；
  * 选择性只在闭环下涌现：PEM 施加无差别去相关压力 -> BP 只修复对任务有用的相关；
  * 该选择性依赖能量结构：冗余耦合若占据主方向，选择性会**反转**。

本脚本把同一套指标放回论文的真实场景（Qwen2.5-0.5B + GLUE 漂移流），回答：
  1. 论文表 5.4 中「w/o PEM 的 |rho|_abs 更低」是否可以重新归因：
     即 PEM 开时 corr_abs 更高，但 **useless_corr / coup_red 更低**；
  2. 剂量—反应：coup_red 是否随 ndi / whiten_coef 单调下降；
  3. 重做图 5.5：coup_red 对 Forgetting 的解释力是否强于 corr_abs。

新增的三件事（相对 ex2.py）
---------------------------
  1. **标签对齐的 h 采集**：ex2 只做 token 级 (N, r) 统计，无法算条件协方差；
     这里用 attention_mask 做 masked-mean 池化，得到样本级 (h, y)。
  2. **Fisher 加权 Γ**：用 register_full_backward_hook 取层输出梯度 g_y，
     由 delta = (alpha/r) * h @ B_eff^T 反解 g_h = (alpha/r) * g_y @ B_eff，
     得到 F = E[g_h g_h^T] 与 Γ_ij = F_ij / sqrt(F_ii F_jj)。
     （不能用线性探针的梯度：二分类下它退化为秩 1，|Γ| ≡ 1。）
  3. **读出去除探针**：acc_full / acc_white / acc_perm，行为层面验证耦合是否有用。

用法
----
  python ex3_coupling.py --steps 120 --tasks sst2,mrpc,rte
  python ex3_coupling.py --steps 5 --tasks sst2 --methods aeloru-nopem,aeloru --probe  # smoke
"""

import os
import sys
import time
import math
import argparse
import csv
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ex2 import (                                  # 复用 ex2 的数据 / 注入 / 采集基建
    MODEL_PATH, ALL_TASKS, TASK_FIELDS,
    build_task_dataset, build_model, collect_entries,
    inject_aeloru_selected, h_from_entry, preserve_dynamics,
    evaluate, compute_cl_metrics, set_seed, log, _freeze_base,
)
from aeloru_v3 import AeloruConfig, AeloruLayer
from ex3_synth import coupling_metrics, task_info_and_pid, probe_acc

METHODS = ["lora", "aeloru-nopem", "aeloru", "aeloru-pem8", "aeloru-pem8-w05"]
CFG_OVERRIDE = {
    "aeloru-nopem":     {"neural_dynamics_iterations": 0},
    "aeloru":           {"neural_dynamics_iterations": 2},
    "aeloru-pem8":      {"neural_dynamics_iterations": 8},
    "aeloru-pem8-w05":  {"neural_dynamics_iterations": 8, "lowrank_whiten_coef": 0.5},
}
PEM_DOSE = {"lora": 0.0, "aeloru-nopem": 0.0, "aeloru": 1.0,
            "aeloru-pem8": 2.0, "aeloru-pem8-w05": 3.0}   # 仅用于画图排序


# ============================================================================
# 模型
# ============================================================================

def make_cfg(method: str, device: str, lr: float, r: int, alpha: float) -> AeloruConfig:
    """与 ex2.make_aeloru_cfg 相同的安全底座 + 本实验的 PEM 剂量覆盖。"""
    base = dict(
        r=r, lora_alpha=alpha, LoRA_lr=lr,
        pem_space="lowrank",
        presumed_domain="none",
        homeostatic_max_gain=1.0,
        homeostatic_enabled=False,
        whitening_interval=10 ** 9,
        merge_every=10 ** 9,
        logging_enabled=False,
        diagnostic_enabled=False,
        device=device,
    )
    base.update(CFG_OVERRIDE[method])
    return AeloruConfig().llm_tuning(**base)


def build_model_ex3(method: str, n_layers: int, device: str, lr: float,
                    n_labels: int, pad_id: int, r: int, alpha: float):
    from transformers import AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model, TaskType

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=n_labels, trust_remote_code=True)
    model.config.pad_token_id = pad_id

    if method == "lora":
        model = get_peft_model(model, LoraConfig(
            r=r, lora_alpha=alpha, target_modules=["q_proj"],
            lora_dropout=0.0, bias="none", task_type=TaskType.SEQ_CLS,
            layers_to_transform=list(range(24 - n_layers, 24)),
        ))
        model.to(device)
    else:
        cfg = make_cfg(method, device, lr, r, alpha)
        inject_aeloru_selected(model, list(range(24 - n_layers, 24)), cfg)
        model.to(device)
        _freeze_base(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, trainable


# ============================================================================
# 采集：样本级 (h, g_h, y)
# ============================================================================

class XGCapture:
    """捕获被注入层的输入 x 与输出梯度 g_y。"""

    def __init__(self, entries):
        self.entries = entries
        self.xs: Dict[int, torch.Tensor] = {}
        self.gs: Dict[int, torch.Tensor] = {}
        self.handles = []
        for e in entries:
            self.handles.append(e.module.register_forward_pre_hook(self._mk_pre()))
            self.handles.append(e.module.register_full_backward_hook(self._mk_bw()))

    def _mk_pre(self):
        def hook(mod, inp):
            if inp and torch.is_tensor(inp[0]):
                self.xs[id(mod)] = inp[0].detach()
        return hook

    def _mk_bw(self):
        def hook(mod, gin, gout):
            if gout and torch.is_tensor(gout[0]):
                self.gs[id(mod)] = gout[0].detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def grad_h_of(entry, g_y: torch.Tensor) -> torch.Tensor:
    """由层输出梯度反解 ∂L/∂h：delta = c * h @ B_eff^T  =>  g_h = c * g_y @ B_eff。"""
    if entry.kind == "aeloru":
        lay = entry.module
        with torch.no_grad():
            B_eff = lay.lora_B.float() * (lay.m_x.float() * lay.plasticity_out.float()).unsqueeze(1)
            c = lay.cfg.lora_alpha / lay.cfg.r
        return c * (g_y.float().reshape(-1, g_y.size(-1)) @ B_eff)
    mod = entry.module
    with torch.no_grad():
        Bm = mod.lora_B[entry.adapter].weight.float()
        sc = float(mod.scaling[entry.adapter])
    return sc * (g_y.float().reshape(-1, g_y.size(-1)) @ Bm)


def collect_coupling(model, entries, loader, device: str, max_batches: int = 8):
    """
    返回 (H, G, y)：样本级特征 (N, r)、任务梯度 (N, r)、标签 (N,)。
    采集过程中的一次额外前反不更新参数；在线统计量由 preserve_dynamics 还原。
    """
    lossf = nn.CrossEntropyLoss()
    was = model.training
    model.train()
    cap = XGCapture(entries)
    Hs: Dict[str, List[torch.Tensor]] = {e.name: [] for e in entries}
    Gs: Dict[str, List[torch.Tensor]] = {e.name: [] for e in entries}
    Ys: List[torch.Tensor] = []
    try:
        with preserve_dynamics(entries):
            for i, batch in enumerate(loader):
                if i >= max_batches:
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                model.zero_grad(set_to_none=True)
                out = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])
                loss = lossf(out.logits.float(), batch["labels"])
                loss.backward()

                mask = batch["attention_mask"].float()          # (B, S)
                for e in entries:
                    x = cap.xs.get(id(e.module))
                    g = cap.gs.get(id(e.module))
                    if x is None or g is None:
                        continue
                    h = h_from_entry(e, x).float()
                    gh = grad_h_of(e, g)
                    B_, S_ = x.shape[0], x.shape[1]
                    if h.shape[0] != B_ * S_:
                        continue
                    hv = h.view(B_, S_, -1)
                    gv = gh.view(B_, S_, -1)
                    denom = mask.sum(1, keepdim=True).clamp(min=1.0)
                    hm = (hv * mask.unsqueeze(-1)).sum(1) / denom
                    gm = (gv * mask.unsqueeze(-1)).sum(1) / denom
                    Hs[e.name].append(hm.detach().cpu())
                    Gs[e.name].append(gm.detach().cpu())
                Ys.append(batch["labels"].detach().cpu())
                model.zero_grad(set_to_none=True)
    finally:
        cap.remove()
        model.zero_grad(set_to_none=True)
        model.train(was)
    if not Ys:
        return None
    return (torch.cat(Ys, 0),
            {k: torch.cat(v, 0) for k, v in Hs.items() if v},
            {k: torch.cat(v, 0) for k, v in Gs.items() if v})


def fisher_gamma(G: torch.Tensor) -> torch.Tensor:
    """Γ = 归一化 Fisher：Γ_ij = F_ij / sqrt(F_ii F_jj)。"""
    G = G.double()
    Fm = (G.t() @ G) / max(G.shape[0] - 1, 1)
    d = torch.diagonal(Fm).clamp(min=1e-12)
    return Fm / torch.sqrt(torch.outer(d, d))


def layer_coupling_metrics(H: torch.Tensor, G: torch.Tensor, y: torch.Tensor) -> dict:
    """单层：无差别指标 + 任务信息分解 + Fisher 加权分解。"""
    out = coupling_metrics(H, y)
    H = H.double()
    r = H.shape[1]
    I = torch.eye(r, dtype=H.dtype, device=H.device)
    R = torch.corrcoef(H.t()) if H.shape[0] > 2 else I
    off = (R - I).abs().clamp(max=1.0)

    Gam = fisher_gamma(G).abs().clamp(max=1.0)
    mask = 1.0 - I
    n_off = max(r * (r - 1), 1)

    # |Γ| 的绝对量级在高维下普遍很小（随机向量近乎正交），直接用作权重会把
    # 所有耦合都判成冗余。改为**相对协同度**：Γ̃ = |Γ| / mean_{i≠j}|Γ|，
    # 衡量「这对维度的任务梯度协同是否高于平均」。
    gbar = float((Gam * mask).sum() / n_off)
    Gtil = (Gam / max(gbar, 1e-9)).clamp(max=1.0) * mask

    # 相关但梯度不协同 = 冗余耦合；相关且协同 = 功能性耦合
    out["coup_red"] = float((off * (1.0 - Gtil)).sum() / n_off)
    out["coup_func"] = float((off * Gtil).sum() / n_off)
    out["gamma_mean"] = gbar
    out["gamma_max"] = float((Gam * mask).max())
    out["corr_abs2"] = float(off.sum() / n_off)
    tot = float(off.sum().clamp(min=1e-12))
    out["coup_red_share"] = out["coup_red"] / tot if tot > 1e-9 else float("nan")
    out["coup_red_ratio"] = (out["coup_red"] / out["coup_func"]
                             if out.get("coup_func", 0.0) > 1e-9 else float("nan"))
    return {k: v for k, v in out.items() if not k.startswith("_")}


# ============================================================================
# 主流程
# ============================================================================

def run_method(method: str, tasks, train_loaders, val_loaders, args, device, fh=None):
    from transformers import DataCollatorWithPadding, AutoTokenizer  # 仅用于类型一致
    import torch.optim as optim

    set_seed(args.seed)
    model, n_params = build_model_ex3(
        method, args.layers, device, args.lr, 2, 0, args.r, args.alpha)
    entries = collect_entries(model)
    aeloru_entries = [e for e in entries if e.kind == "aeloru"]

    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(params, lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    T = len(tasks)
    R = np.full((T, T), np.nan)
    rows: List[dict] = []
    global_step = 0

    from ex2 import ActivationCapture, train_one_task, trigger_sleep_consolidation

    for i, t in enumerate(tasks):
        capture = ActivationCapture(model) if aeloru_entries else None
        _, global_step, _ = train_one_task(
            model, entries, capture, train_loaders[t], opt, device, args.steps,
            t, [], [], global_step, ortho_lambda=args.ortho_lambda,
            sample_every=max(args.steps, 1))
        if capture is not None:
            capture.remove()
        if aeloru_entries:
            trigger_sleep_consolidation(model)

        # --- 回测全部已见任务 ---
        for j in range(i + 1):
            R[i, j] = evaluate(model, val_loaders[tasks[j]], device)

        # --- 耦合指标（在该任务自己的验证集上，样本级对齐） ---
        got = collect_coupling(model, entries, val_loaders[t], device, args.collect_batches)
        if got is not None:
            y, Hd, Gd = got
            accs_full, accs_white = [], []
            agg: Dict[str, List[float]] = {}
            for name, H in Hd.items():
                G = Gd.get(name)
                if G is None or H.shape[0] < 32:
                    continue
                m = layer_coupling_metrics(H, G, y)
                for k, v in m.items():
                    agg.setdefault(k, []).append(v)
                if args.probe:
                    ntr = max(H.shape[0] // 2, 32)
                    pa = probe_acc(H[:ntr], y[:ntr], H[ntr:], y[ntr:])
                    for k, v in pa.items():
                        agg.setdefault(k, []).append(v)
            item = {"method": method, "task": t, "boundary": i,
                    "params": n_params, "n_layers": len(entries)}
            for k, v in agg.items():
                item[k] = float(np.mean(v))
            rows.append(item)
            log(f"    [{method}] after {t}: " + "  ".join(
                f"{k}={item[k]:.4f}" for k in ("corr_abs", "useful_corr",
                                               "useless_corr", "coup_red",
                                               "coup_func", "gamma_mean")
                if k in item), fh)

    cl = compute_cl_metrics(R)
    return {"method": method, "params": n_params, "R": R, "cl": cl, "rows": rows}


def plot(results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 流末（最后一个任务边界）的平均指标
    last = {}
    for res in results:
        rows = [r for r in res["rows"] if r["boundary"] == max(
            (x["boundary"] for x in res["rows"]), default=0)]
        if rows:
            last[res["method"]] = {k: float(np.mean([r[k] for r in rows if k in r]))
                                   for k in rows[0] if isinstance(rows[0][k], float)}

    names = [m for m in METHODS if m in last]
    if not names:
        return None
    xs = [PEM_DOSE.get(m, 0) for m in names]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for k, title in (("corr_abs", "(a) corr_abs (paper metric)"),
                     ("useless_corr", "(b) useless_corr (task-blind)"),
                     ("coup_red", "(c) coup_red (Fisher-weighted)")):
        i = ("corr_abs", "useless_corr", "coup_red").index(k)
        vals = [last[m].get(k, float("nan")) for m in names]
        ax[i].plot(xs, vals, "o-")
        for x, v, n in zip(xs, vals, names):
            ax[i].annotate(n.split("-")[-1], (x, v), fontsize=7,
                           xytext=(3, 3), textcoords="offset points")
        ax[i].set_xlabel("PEM dose (0=nopem)")
        ax[i].set_title(title, fontsize=10)
        ax[i].grid(alpha=.3)

    # 冗余指标 vs Forgetting 的解释力
    fig2, ax2 = plt.subplots(1, 2, figsize=(9, 4))
    for j, key in enumerate(("corr_abs", "coup_red")):
        xx, yy, labs = [], [], []
        for m in names:
            xx.append(last[m].get(key, float("nan")))
            yy.append([r["cl"]["Forgetting"] for r in results if r["method"] == m][0])
            labs.append(m)
        xx, yy = np.array(xx), np.array(yy)
        ok = np.isfinite(xx) & np.isfinite(yy)
        ax2[j].scatter(xx[ok], yy[ok])
        for x, y_, l in zip(xx[ok], yy[ok], np.array(labs)[ok]):
            ax2[j].annotate(l.split("-")[-1], (x, y_), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
        if ok.sum() >= 3:
            c = np.corrcoef(xx[ok], yy[ok])[0, 1]
            ax2[j].set_title(f"{key} vs Forgetting  r={c:.2f}, R2={c ** 2:.2f}", fontsize=10)
        ax2[j].set_xlabel(key); ax2[j].set_ylabel("Forgetting")
        ax2[j].grid(alpha=.3)

    p1 = os.path.join(out_dir, "fig_dose.png")
    p2 = os.path.join(out_dir, "fig_forget.png")
    fig.tight_layout(); fig.savefig(p1, dpi=150); plt.close(fig)
    fig2.tight_layout(); fig2.savefig(p2, dpi=150); plt.close(fig2)
    return p1, p2


def main():
    ap = argparse.ArgumentParser(description="ex3 B: LLM 上的冗余/功能性耦合分解")
    ap.add_argument("--methods", type=str, default=",".join(METHODS))
    ap.add_argument("--tasks", type=str, default="sst2,mrpc,rte")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--ortho-lambda", type=float, default=0.1)
    ap.add_argument("--collect-batches", type=int, default=8,
                    help="每个任务边界用于耦合统计的 batch 数")
    ap.add_argument("--probe", action="store_true", help="额外跑读出去除探针")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer, DataCollatorWithPadding

    methods = [m for m in args.methods.split(",") if m]
    tasks = [t for t in args.tasks.split(",") if t]
    device = args.device

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       f"ex3_coupling_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    fh = open(os.path.join(out_dir, "log.txt"), "w", encoding="utf-8")
    log(f"ex3 B: coupling decomposition | tasks={tasks} | steps={args.steps} "
        f"| r={args.r} | layers={args.layers}", fh)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_loaders, val_loaders = {}, {}
    for t in tasks:
        tr, va = build_task_dataset(tok, t, args.seq, args.n_train, args.n_val, args.seed)
        coll = DataCollatorWithPadding(tok)
        train_loaders[t] = DataLoader(tr, batch_size=args.bs, shuffle=True,
                                      collate_fn=coll, drop_last=True)
        val_loaders[t] = DataLoader(va, batch_size=32, shuffle=False,
                                    collate_fn=coll)

    results = []
    t0 = time.time()
    for m in methods:
        log(f"[{m}] start", fh)
        res = run_method(m, tasks, train_loaders, val_loaders, args, device, fh)
        results.append(res)
        log(f"[{m}] ACC={res['cl']['ACC']:.4f} BWT={res['cl']['BWT']:+.4f} "
            f"Forgetting={res['cl']['Forgetting']:.4f} params={res['params']}", fh)

    # ---- 输出 ----
    all_rows = [r for res in results for r in res["rows"]]
    if all_rows:
        cols: List[str] = []
        for r in all_rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(os.path.join(out_dir, "coupling.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(all_rows)

    summ = []
    for res in results:
        rows = res["rows"]
        item = {"method": res["method"], "params": res["params"]}
        item.update(res["cl"])
        if rows:
            last_b = max(r["boundary"] for r in rows)
            sub = [r for r in rows if r["boundary"] == last_b]
            for k in sub[0]:
                if isinstance(sub[0][k], float):
                    item[k] = float(np.mean([r[k] for r in sub if k in r]))
        summ.append(item)
    if summ:
        cols = []
        for it in summ:
            for k in it:
                if k not in cols:
                    cols.append(k)
        with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(summ)

    for res in results:
        np.savetxt(os.path.join(out_dir, f"R_{res['method']}.csv"), res["R"], delimiter=",")

    try:
        plot(results, out_dir)
    except Exception as e:
        log(f"[warn] plot failed: {e}", fh)

    print("\n" + "=" * 78)
    print("summary (last task boundary)")
    for it in summ:
        print(f"  {it['method']:16s} ACC={it.get('ACC', float('nan')):.4f} "
              f"Forget={it.get('Forgetting', float('nan')):.4f} "
              f"corr_abs={it.get('corr_abs', float('nan')):.4f} "
              f"useless={it.get('useless_corr', float('nan')):.4f} "
              f"useful={it.get('useful_corr', float('nan')):.4f} "
              f"coup_red={it.get('coup_red', float('nan')):.4f}")
    print("=" * 78)
    print(f"总耗时 {time.time() - t0:.1f}s -> {out_dir}")
    fh.close()


if __name__ == "__main__":
    main()
