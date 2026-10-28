# -*- coding: utf-8 -*-
"""
smoke_test_v4.py -- V4 新机制冒烟测试
================================================================================
覆盖 V4_main_way.md 勾选的改动：
  二.1  SPHeRe 拆成两个独立消融开关 (nosphere_ortho / nosphere_struct)
  二.2  正交惩罚随任务相似度自适应 (ortho_adapt)
  二.3  状态机阈值自适应 (adaptive_threshold)
  二.4  ReLoRA 合并时 Hebbian 痕迹按 W_acc 增长同步缩放 (hebb_trace_sync)
  三.5  任务选择性 PEM (selective_pem)
  三.6  动态秩分配 (dynamic_rank / r_dim / _expand_rank / rebind_optimizer)
  三.7  睡眠巩固单独量化 (nosleep)
  五    fromscratch 预热 (warmup_steps)

运行（conda Qelys 环境，CPU）：
  conda run -n Qelys python e:/Qelys/AeloRu/V4/smoke_test_v4.py
"""
import os
import sys
import math
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aeloru_v4 import AeloruConfig, AeloruLayer, train_aeloru_step  # noqa: E402

DEVICE = "cuda"
torch.manual_seed(0)
np.random.seed(0)

RNG = torch.Generator().manual_seed(1234)


# ============================================================================
# 数据与基础配置
# ============================================================================
def make_dataset(n, d_in, d_out, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d_in, generator=g)
    A = torch.randn(d_out, d_in, generator=g) * 0.7
    Y = X @ A.t() + 0.05 * torch.randn(n, d_out, generator=g)
    return X, Y


def base_cfg(**overrides):
    cfg = AeloruConfig(
        r=8, lora_alpha=8.0,
        in_features=16, out_features=8,
        AMP_DTYPE=torch.float32, device=DEVICE,
        presumed_domain="none",
        homeostatic_enabled=False,
        whitening_interval=10 ** 9,
        merge_every=10 ** 9,
        neural_dynamics_iterations=0,   # 默认 10 会在极小合成数据上发散，冒烟测试关闭
        wta_k_ratio=1.0,
        fisher_gamma=0.0,
        hebbian_lr=0.0,
        ortho_lambda=0.1,
        sphere_lambda=0.5,
        logging_enabled=False, diagnostic_enabled=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run_training(cfg, steps=120, d_in=16, d_out=8, n=1024, bs=128, lr=5e-3, pretrain_w0=False):
    layer = AeloruLayer(d_in, d_out, cfg).to(DEVICE)
    layer.train()
    if pretrain_w0:
        # 注入一个非零基座权重，使正交惩罚 (DeltaW vs W0) 在该测试中有意义
        with torch.no_grad():
            layer.W0.normal_(0.0, 0.2)
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    X, Y = make_dataset(n, d_in, d_out)
    Xtr, Ytr = X.to(DEVICE), Y.to(DEVICE)
    loss_fn = lambda a, b: F.mse_loss(a.float(), b.float())
    log = []
    for _ in range(steps):
        idx = torch.randint(0, n, (bs,), generator=RNG)
        xb, yb = Xtr[idx], Ytr[idx]
        _, metrics = train_aeloru_step(layer, xb, yb, opt, loss_fn=loss_fn)
        log.append(metrics)
    return layer, opt, log, (Xtr, Ytr)


# ============================================================================
# 测试集合
# ============================================================================
def test_baseline():
    cfg = base_cfg()
    layer, _, log, (Xtr, Ytr) = run_training(cfg, steps=120)
    first = log[0]["loss_task"]
    last = log[-1]["loss_task"]
    if not (last < first * 0.95):
        return False, f"task loss 未下降: {first:.4f} -> {last:.4f}"
    if math.isnan(last) or math.isinf(last):
        return False, f"loss 异常: {last}"
    return True, f"baseline task loss {first:.4f} -> {last:.4f}"


def test_nosphere_struct():
    cfg = base_cfg(nosphere_struct=True)
    layer, _, log, _ = run_training(cfg, steps=5, pretrain_w0=True)
    m = log[-1]
    if abs(m["loss_structure"]) > 1e-6:
        return False, f"nosphere_struct 下 structure loss 应=0, 实际 {m['loss_structure']}"
    if m["loss_ortho"] <= 0:
        return False, "nosphere_struct 下应仍有正交项 (loss_ortho>0)"
    return True, "nosphere_struct: 结构项关闭, 正交项保留"


def test_nosphere_ortho():
    cfg = base_cfg(nosphere_ortho=True)
    layer, _, log, _ = run_training(cfg, steps=5)
    m = log[-1]
    if abs(m["loss_ortho"] - m["loss_structure"]) > 1e-5:
        return False, (f"nosphere_ortho 下 loss_ortho 应等于 structure loss: "
                       f"{m['loss_ortho']} vs {m['loss_structure']}")
    if m["loss_structure"] <= 0:
        return False, "nosphere_ortho 下应仍有结构项 (loss_structure>0)"
    return True, "nosphere_ortho: 正交项关闭, 结构项保留"


def test_warmup():
    cfg = base_cfg(warmup_steps=50, ortho_lambda=1.0, sphere_lambda=1.0)
    layer = AeloruLayer(16, 8, cfg).to(DEVICE)
    # 制造一个非零的正交/结构惩罚，便于观察 warmup 缩放
    with torch.no_grad():
        layer.W0.fill_(1.0)
        layer.lora_B.fill_(1.0)
    # 1) 直接验证 _warmup_factor
    layer._global_step = 0
    if abs(layer._warmup_factor() - 0.0) > 1e-6:
        return False, f"global_step=0 时 warmup 因子应=0, 实际 {layer._warmup_factor()}"
    layer._global_step = 25
    if abs(layer._warmup_factor() - 0.5) > 1e-6:
        return False, f"global_step=25 时 warmup 因子应=0.5, 实际 {layer._warmup_factor()}"
    layer._global_step = 100
    if abs(layer._warmup_factor() - 1.0) > 1e-6:
        return False, f"global_step>=50 时 warmup 因子应=1, 实际 {layer._warmup_factor()}"
    # 2) 验证 get_ortho_penalty 随 global_step 从 0 升到目标值
    layer._global_step = 0
    p0 = layer.get_ortho_penalty().item()
    layer._global_step = 100
    p1 = layer.get_ortho_penalty().item()
    if not (abs(p0) < 1e-6):
        return False, f"warmup 第0步惩罚应≈0, 实际 {p0}"
    if not (p1 > p0 + 1e-3):
        return False, f"warmup 后期惩罚应>0并升到目标值: p0={p0}, p1={p1}"
    return True, f"warmup: 因子 0→0.5→1.0, 惩罚 {p0:.4f}→{p1:.4f}"


def test_ortho_adapt_factor():
    cfg = base_cfg(ortho_adapt=True)
    layer = AeloruLayer(16, 8, cfg)
    # 注入一个原型：只激活第 0 个低秩维度
    with torch.no_grad():
        layer.stored_prototypes.zero_()
        layer.stored_prototypes[0, 0] = 1.0
        # 当前任务方向也集中在第 0 维 -> 高相似度
        layer.lora_A.zero_()
        layer.lora_A[0, 0] = 1.0
    f_high = layer._ortho_similarity_factor()
    # 当前任务方向与第 1 维 -> 与原型正交 -> 低相似度
    with torch.no_grad():
        layer.lora_A.zero_()
        layer.lora_A[1, 0] = 1.0
    f_low = layer._ortho_similarity_factor()
    lo, hi = cfg.ortho_adapt_min, cfg.ortho_adapt_max
    if not (lo - 1e-6 <= f_high <= hi + 1e-6 and lo - 1e-6 <= f_low <= hi + 1e-6):
        return False, f"factor 超出 [{lo},{hi}]: high={f_high}, low={f_low}"
    if not (f_high > f_low):
        return False, f"与原型更相似应得到更强的缩放: high={f_high} > low={f_low}?"
    return True, f"ortho_adapt 因子随相似度变化: {f_low:.3f} -> {f_high:.3f}"


def test_selective_pem_mask():
    cfg = base_cfg(selective_pem=True)
    layer = AeloruLayer(16, 8, cfg)
    with torch.no_grad():
        layer.lateral_weights.zero_()
        layer.lateral_weights[0, 0] = 1.0  # 高重叠维度 (0) 间连接 -> 应保留
        layer.lateral_weights[0, 1] = 1.0  # 涉及低重叠维度 (1) 的连接 -> 应清零
        layer.stored_prototypes.zero_()
        layer.stored_prototypes[0, 0] = 1.0  # 原型只覆盖第 0 维
        layer.lora_A.zero_()
        layer.lora_A[0, 0] = 1.0  # 当前任务方向集中在第 0 维
    mask = layer._selective_pem_mask(layer.lateral_weights.detach().clone())
    # 维度 1 与当前任务原型无重叠 -> 该侧连接应被屏蔽 (mask[0,1] == 0)
    if mask[0, 1].item() != 0.0:
        return False, f"不相关维度的侧向连接应被清零, mask[0,1]={mask[0,1].item()}"
    # 高重叠维度 (0,0) 间的侧向连接应保留 (=1.0, 因 lateral[0,0]=1)
    if abs(mask[0, 0].item()) < 1e-9:
        return False, "高重叠维度 (0,0) 的侧向连接应保留"
    return True, "selective_pem: 只保留与当前任务原型高重叠维度的侧向连接"


def test_hebb_trace_sync():
    # hebb_trace_sync=False -> 合并时痕迹 *0.5
    cfg_off = base_cfg(hebb_trace_sync=False)
    layer_off = AeloruLayer(16, 8, cfg_off)
    layer_off.train()
    with torch.no_grad():
        layer_off.hebb_trace_A.fill_(2.0)
        layer_off.hebb_trace_B.fill_(2.0)
    norm_before = layer_off.hebb_trace_A.norm().item()
    layer_off.merge_and_reset()
    norm_after = layer_off.hebb_trace_A.norm().item()
    if not abs(norm_after - 0.5 * norm_before) < 1e-3:
        return False, (f"hebb_trace_sync=False 应 *0.5: before={norm_before}, "
                       f"after={norm_after}")
    # hebb_trace_sync=True -> 按 W_acc 增长缩放（delta 很小 -> 近似不变）
    cfg_on = base_cfg(hebb_trace_sync=True)
    layer_on = AeloruLayer(16, 8, cfg_on)
    layer_on.train()
    with torch.no_grad():
        layer_on.hebb_trace_A.fill_(2.0)
        layer_on.hebb_trace_B.fill_(2.0)
    nb = layer_on.hebb_trace_A.norm().item()
    layer_on.merge_and_reset()
    na = layer_on.hebb_trace_A.norm().item()
    if not (math.isfinite(na) and na > 0):
        return False, f"hebb_trace_sync=True 缩放异常: {na}"
    return True, f"hebb_trace_sync: off→×0.5 ({norm_before:.2f}→{norm_after:.2f}), on→按增长缩放 ({nb:.2f}→{na:.2f})"


def test_dynamic_rank_expand():
    cfg = base_cfg(dynamic_rank=True)
    layer = AeloruLayer(16, 8, cfg).to(DEVICE)
    layer.train()
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
    old_r = layer.r_dim
    new_r = 12
    layer._expand_rank(new_r)
    if layer.r_dim != new_r:
        return False, f"r_dim 未更新: {layer.r_dim}"
    if layer.lora_A.shape != (new_r, 16) or layer.lora_B.shape != (8, new_r):
        return False, f"lora 形状未扩展: {layer.lora_A.shape}, {layer.lora_B.shape}"
    if layer.hebb_trace_A.shape != (new_r, 16) or layer.lateral_weights.shape != (new_r, new_r):
        return False, "r 维缓冲未扩展"
    # Hebbian 累加/影子缓冲也应随之扩展（避免动态秩+Hebbian 形状不匹配）
    if layer._hebbian_acc_B.shape != (8, new_r) or layer._hebbian_acc_A.shape != (new_r, 16):
        return False, "Hebbian 累加缓冲未扩展"
    # 跑一步训练，触发 rebind_optimizer，确保优化器能作用到新参数
    X, Y = make_dataset(256, 16, 8)
    xb, yb = X[:32].to(DEVICE), Y[:32].to(DEVICE)
    loss_fn = lambda a, b: F.mse_loss(a.float(), b.float())
    before = layer.lora_B.data[0, 0].clone()
    _, _ = train_aeloru_step(layer, xb, yb, opt, loss_fn=loss_fn)
    after = layer.lora_B.data[0, 0]
    if torch.equal(before, after):
        return False, "rebind 后新参数未收到梯度更新"
    return True, f"dynamic_rank: r {old_r}->{new_r}, 缓冲/优化器同步成功"


def test_nosleep():
    cfg = base_cfg(nosleep=True, hebbian_lr=1e-3, sleep_condition_threshold=1.0,
                   min_steps_between_sleep=3)
    layer = AeloruLayer(16, 8, cfg)
    layer.train()
    layer.stored_patterns.fill_(100)  # 绕过数量门槛
    with torch.no_grad():
        layer.hebb_trace_A.fill_(1.0)
    for _ in range(20):
        if layer.should_sleep():
            return False, "nosleep=True 时 should_sleep 应始终返回 False"
    return True, "nosleep: 睡眠被关闭, should_sleep 恒为 False"


def test_sleep_and_prototypes():
    """确定性验证 DLAM 睡眠触发 + 原型提取（nosleep 为 V4 开关，睡眠本身为既有机制）。"""
    cfg = base_cfg(
        hebbian_lr=1e-3,
        prototype_top_k=4,
        sleep_condition_threshold=1.0,
        min_steps_between_sleep=1,
    )
    layer = AeloruLayer(16, 8, cfg)
    layer.train()
    with torch.no_grad():
        layer.stored_patterns.fill_(100)         # 绕过数量门槛
        layer.step_counter = 100                 # 满足最小间隔门槛
        layer._last_sleep_step = 0
        # 构造秩亏的 hebb_trace_A -> 条件数很大 -> 超过阈值
        A = torch.zeros_like(layer.hebb_trace_A)
        A[0, :] = 1.0
        A[1, 0] = 0.01
        layer.hebb_trace_A.copy_(A)
        layer.hebb_trace_B.fill_(0.1)
    slept = layer.should_sleep()
    if not slept:
        return False, "should_sleep 未触发（条件数应>阈值）"
    # 走睡眠管线（post_step_update 内部在 should_sleep 为真时调用这两个）
    layer._offline_replay()
    layer._extract_prototypes()
    proto_sum = layer.stored_prototypes.abs().sum().item()
    if proto_sum <= 0:
        return False, "睡眠后未提取到 DLAM 原型 (stored_prototypes 全 0)"
    return True, f"sleep+prototype: should_sleep=True, 提取原型 (Σ|proto|={proto_sum:.3f})"


def test_adaptive_threshold_no_crash():
    """确定性验证二.3：阶段阈值快照 = 上一阶段冲突分布 75% 分位。"""
    import torch as _t
    cfg = base_cfg(adaptive_threshold=True)
    layer = AeloruLayer(16, 8, cfg)
    # _snapshot_phase_thresholds 要求历史样本 >= 20，构造 24 个样本
    hist = [0.1 + 0.7 * i / 23 for i in range(24)]
    layer._conflict_hist = list(hist)
    layer._snapshot_phase_thresholds()
    if layer._phase_red_threshold is None:
        return False, "adaptive_threshold 下 _phase_red_threshold 未被快照"
    expected = float(_t.quantile(_t.tensor(hist, dtype=_t.float32), 0.75))
    if abs(layer._phase_red_threshold - expected) > 1e-5:
        return False, f"快照阈值应=75%分位 {expected}, 实际 {layer._phase_red_threshold}"
    if abs(layer._effective_red_threshold() - expected) > 1e-5:
        return False, "adaptive_threshold 下 _effective_red_threshold 应返回快照值"
    return True, f"adaptive_threshold: 快照红温阈值={layer._phase_red_threshold:.3f} (=75%分位)"


# ============================================================================
# 运行
# ============================================================================
def main():
    tests = [
        ("baseline", test_baseline),
        ("nosphere_struct (二.1)", test_nosphere_struct),
        ("nosphere_ortho (二.1)", test_nosphere_ortho),
        ("warmup (五)", test_warmup),
        ("ortho_adapt (二.2)", test_ortho_adapt_factor),
        ("selective_pem (三.5)", test_selective_pem_mask),
        ("hebb_trace_sync (二.4)", test_hebb_trace_sync),
        ("dynamic_rank (三.6)", test_dynamic_rank_expand),
        ("nosleep (三.7)", test_nosleep),
        ("sleep+prototype", test_sleep_and_prototypes),
        ("adaptive_threshold (二.3)", test_adaptive_threshold_no_crash),
    ]
    print("=" * 72)
    print("Aeloru V4 冒烟测试  (device=%s)" % DEVICE)
    print("=" * 72)
    passed, failed = [], []
    t0 = time.time()
    for name, fn in tests:
        try:
            ok, msg = fn()
        except Exception as e:  # noqa: BLE001
            ok, msg = False, "EXCEPTION: " + str(e) + "\n" + traceback.format_exc()
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name:26s} {msg}")
        (passed if ok else failed).append(name)
    dt = time.time() - t0
    print("-" * 72)
    print(f"通过 {len(passed)}/{len(tests)}  耗时 {dt:.1f}s")
    if failed:
        print("失败: " + ", ".join(failed))
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
