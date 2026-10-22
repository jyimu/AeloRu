针对长线任务序列效果不好的核心问题（低秩空间饱和、固定学习率冲击大、睡眠巩固不足、缺少逐轮退火），我给你整理了完整的修改方案，按位置改完就能显著改善长线遗忘。

## 一、核心修改点总览
| 问题 | 修改 | 预期收益 |
|---|---|---|
| 低秩增量长期堆积饱和 | 支持全局 ReLoRA 定期合并，safe 配置也能开启 | 释放低秩容量，避免后期增量挤压旧任务 |
| 固定学习率冲击大 | 每任务余弦学习率退火，任务末期 lr 降到 10% | 减少后期梯度扰动，收敛更稳 |
| 睡眠巩固太弱 | 睡眠强度可调，长线可叠加多次 | 强化原型记忆固化，提升旧任务保留 |
| 重复轮次步长不变 | 每完成一轮任务序列，lr 自动衰减 | 第二轮复习用小步长，不冲掉旧记忆 |
| 协方差抖动过大 | 修正 `_recondition_layer` 抖动系数 | 减少数值噪声，机制更稳定 |

---

## 二、具体修改位置和代码

### 1. 新增命令行参数（argparse 部分）
在 `--ortho-lambda` 参数后面插入 4 个新参数：

```python
    ap.add_argument("--merge-every", type=int, default=0,
                    help="全局 ReLoRA 合并周期（步），0=使用各配置默认值；长线推荐 500")
    ap.add_argument("--lr-schedule", type=int, default=1, choices=[0, 1],
                    help="是否启用每任务余弦学习率退火：1=开，0=关")
    ap.add_argument("--lr-decay-per-round", type=float, default=1.0,
                    help="每完成一轮任务序列后的学习率衰减系数，长线推荐 0.8")
    ap.add_argument("--sleep-intensity", type=int, default=1,
                    help="每任务后睡眠巩固强度（调用次数），长线推荐 2~3")
```

顺手删掉重复的 `import random`（第 27 行附近重复导入了一次）。

### 2. 修正协方差抖动（数值稳定性）
找到 `_recondition_layer` 函数，把抖动系数从 `1e-2` 降到 `1e-4`，避免污染真实统计：

```python
            j = 1e-4 * buf.diagonal().abs().max().clamp(min=1e-6)
```

### 3. build_model 支持全局覆盖合并周期
在 **三个 aeloru 配置块**（aeloru-safe / aeloru-default / aeloru-fromscratch）的 `inject_aeloru_selected(...)` 之前，各加一行：

```python
        if args.merge_every > 0:
            cfg.merge_every = args.merge_every
```

> 这样就可以通过命令行统一控制所有方法的合并周期，不用再改每个配置的默认值。

### 4. train_one_task 增加余弦学习率调度
#### 第一步：修改函数签名，加上 lr 和调度开关
```python
def train_one_task(model, aeloru_layers: List[AeloruLayer], capture: ActivationCapture,
                   loader: DataLoader, optimizer, device: str, steps: int,
                   method: str, mech_log: List[dict], task_name: str,
                   global_step: int, ortho_lambda: float = 0.1,
                   grad_clip: float = 1.0, relora_every: int = 0,
                   lr: float = 1e-4, lr_schedule: bool = False) -> Tuple[List[float], int, dict]:
```

#### 第二步：optimizer 后创建调度器
在 `optimizer = torch.optim.AdamW(...)` 后面插入：

```python
    scheduler = None
    if lr_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=lr * 0.1
        )
```

#### 第三步：optimizer.step() 后更新调度
在 `optimizer.step()` 后面插入：

```python
        if scheduler is not None:
            scheduler.step()
```

### 5. 调用处传入参数
找到 `run_method` 里调用 `train_one_task` 的地方，补充传入 lr 和调度开关：

```python
        losses, global_step, agg = train_one_task(
            model, aeloru_layers, capture, train_loaders[seq_tasks[i]], optimizer,
            device, args.steps, method, mech_log, seq_labels[i], global_step,
            ortho_lambda=args.ortho_lambda, relora_every=relora_every,
            lr=args.lr, lr_schedule=bool(args.lr_schedule))
```

### 6. 每轮学习率自动衰减
在 `run_method` 的任务循环外层，维护当前学习率，每轮衰减。
找到 `for rep in range(args.repeat):` 循环，在循环开头加入：

```python
        current_lr = args.lr * (args.lr_decay_per_round ** rep)
        for g in optimizer.param_groups:
            g['lr'] = current_lr
```

> 注意：这一段要放在**每轮任务开始训练之前**，确保第二轮开始时 lr 已经降下来了。

### 7. 睡眠巩固强度可调
找到任务边界调用睡眠的地方：

```python
            sleep_events += trigger_sleep_consolidation(model, aeloru_layers)
```

改成循环调用，强度由参数控制：

```python
            for _ in range(args.sleep_intensity):
                sleep_events += trigger_sleep_consolidation(model, aeloru_layers)
```

---

## 三、推荐长线运行参数
改完之后用这组参数跑，长线遗忘会有明显改善：

```bash
python ex1_long_line.py \
  --methods lora,aeloru-safe,aeloru-default \
  --repeat 2 \
  --steps 250 \
  --r 8 --alpha 16 \
  --merge-every 500 \
  --lr-schedule 1 \
  --lr-decay-per-round 0.8 \
  --sleep-intensity 2
```

---

## 四、预期变化
1. **safe 配置追近 default 的遗忘表现**：之前 safe 差很多主要是因为没合并，开启定期合并后差距会大幅缩小。
2. **第二轮任务的遗忘斜率明显变缓**：lr 衰减 + 合并释放空间，第二轮复习不会再猛烈冲击旧任务。
3. **已见任务均值更平稳**：不会出现第一轮上升、第二轮跳水的情况。
4. **训练 loss 后期更平滑**：余弦退火避免了尾部梯度震荡。

如果跑出来效果还想再优化，可以再把 `--merge-every` 调到 250（每任务合并一次），或者把 `--sleep-intensity` 加到 3，进一步加强记忆固化。