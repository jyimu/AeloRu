from aeloru_v3 import AeloruConfig, AeloruLayer, train_aeloru_step
import torch


def benchmark_aeloru():
    """Aeloru性能基准测试 v2.0"""
    import time
    import platform

    print("=" * 70)
    print("Aeloru Fisher-Hierarchical 基准测试 v2.0")
    print(f"平台: {'Windows (WDDM)' if platform.system() == 'Windows' else 'Linux'}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_dim, out_dim, batch = 896, 151000, 2  # 以Qwen2.5-0.5B为例
    # in_dim, out_dim, batch = 128, 128, 4
    steps = 100

    # Windows 下禁用异步 Stream(WDDM 驱动假异步)
    cfg = AeloruConfig()
    layer = AeloruLayer(in_dim, out_dim, cfg).to(device)
    layer.set_pretrained_weight(torch.randn(out_dim, in_dim, device=device) * 0.02)
    layer.train()

    optimizer = torch.optim.AdamW(layer.get_trainable_params(), lr=1e-3, fused=True)
    x = torch.randn(batch, in_dim, device=device, dtype=AeloruConfig.AMP_DTYPE)
    y = torch.randn(batch, out_dim, device=device, dtype=AeloruConfig.AMP_DTYPE)
    
    if platform.system() != 'Windows':
        # Linux下编译优化，减少 kernel launch overhead
        layer = torch.compile(layer, mode="reduce-overhead")

        # 预热：跑几个空batch让它编译
        print("[Aeloru] 正在编译模型...")
        with torch.no_grad():
            for _ in range(10):
                dummy_input = {
                    "input_ids": torch.randint(0, 151643, (16, 128), device="cuda"),
                    "attention_mask": torch.ones(16, 128, device="cuda"),
                    "labels": torch.randint(0, 2, (16,), device="cuda")
                }
                layer(**dummy_input)
        print("[Aeloru] 模型编译完成")

    # 预热
    for _ in range(10):
        train_aeloru_step(layer, x, y, optimizer)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 测试前向传播延迟
    t0 = time.perf_counter()
    for _ in range(steps):
        _ = layer(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    fwd_time = (time.perf_counter() - t0) / steps * 1000  # ms

    # 测试训练步延迟
    t0 = time.perf_counter()
    for _ in range(steps):
        train_aeloru_step(layer, x, y, optimizer)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    step_time = (time.perf_counter() - t0) / steps * 1000  # ms

    # 获取峰值显存
    mem_mb = 0
    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()

    # 输出性能指标
    print(f"\n设备: {device}")
    print(f"维度: in={in_dim}, out={out_dim}, batch={batch}, r={cfg.r}")
    print(f"前向传播延迟: {fwd_time:.2f} ms")
    print(f"训练步延迟: {step_time:.2f} ms")
    print(f"峰值显存: {mem_mb:.2f} MB")
    # 验证 eval/train 一致性
    layer.eval()
    with torch.no_grad():
        out1 = layer(x)
    layer.train()
    with torch.no_grad():
        out2 = layer(x)
    diff = (out1 - out2).abs().max().item()
    print(f"\nEval/Train 一致性: {diff:.2e}")

if __name__ == "__main__":
    benchmark_aeloru()
