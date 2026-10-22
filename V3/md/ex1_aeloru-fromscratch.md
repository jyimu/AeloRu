# aeloru‑fromscratch 专项说明 & 修改
`aeloru‑fromscratch` 和另外两个 PEFT 模式（aeloru‑safe / aeloru‑default）底层逻辑完全不一样，很多之前的改动**不能直接照搬**，直接套会崩、loss爆炸。

## 核心原生约束（源码自带）
1. `FromScratchWrapper.do_sleep = False`：**默认关闭 DLAM sleep 离线重放**。
> 原因：它不是标准 transformers 模型，`cross_modal_sleep_step(AeloruModel(model))` 的封装对这个自定义 wrapper 兼容性差，容易维度报错、buffer错乱，原论文设定从零认知模型不靠睡眠巩固，靠**高频 ReLoRA 合并**做记忆固化。
2. `W0` **不接收BP梯度**，唯一更新途径只有 `relora_merge_into_W0()`；LoRA适配器是唯一被反向传播更新的部分。
3. cfg 默认 `merge_every=1e9`，不做自动合并；代码里会自动设置 `relora_every = max(1, args.steps // 2)`，靠**显式周期调用relora_merge_into_W0**。
4. 强制 float32，不能bf16，数值更敏感。
5. 它自己有一套完整PEM/ndi=5、homeostatic_max_gain=5.0，和PEFT组参数空间不一样。

> 重点：
> - ✅ **学习率调度（余弦退火、每轮lr衰减）完全可以用，对fromscratch有效**
> - ❌ **`--merge‑every`全局cfg.merge_every 对 fromscratch无效！！** fromscratch不走cfg.merge_every自动合并，只用代码里的`relora_every`显式调用`relora_merge_into_W0`。
> - ❌ **sleep‑intensity 对 fromscratch 默认无效，因为 do_sleep=False，会直接跳过trigger_sleep_consolidation**，强行打开有风险。

---

## 需要追加的修改（针对fromscratch）
### 1. 命令行新增 fromscratch专属参数
在你之前新增的4个参数后面追加：
```python
    ap.add_argument("--fs-relora-every", type=int, default=0,
                    help="aeloru‑fromscratch专属ReLoRA合并周期；0=自动取steps//2；长线建议250(每任务合并一次)")
    ap.add_argument("--fs-enable-sleep", type=int, default=0, choices=[0,1],
                    help="是否给fromscratch打开睡眠巩固；0关闭(原默认)，1开启（实验性，可能报错）")
```

### 2. build_model，aeloru‑fromscratch分支读取fs_relora_every
原来fromscratch分支不会理会`--merge‑every`，保持这个行为；但把fs_relora_every保存下来，后面run_method读取。
> 因为build_model返回只有(model, trainable_num)，我们改个小技巧：把参数挂在model实例上。

找到`elif method == "aeloru-fromscratch":`内部，`model = FromScratchWrapper(embed, net, device)`之后添加两行：
```python
        model = FromScratchWrapper(embed, net, device)
        # 保存fromscratch专属参数到model实例
        model.fs_relora_every = args.fs_relora_every
        model.fs_enable_sleep = bool(args.fs_enable_sleep)
        model.to(device)
```

### 3. run_method 内部，run_method 拿到relora_every的逻辑做修正
原来代码：
```python
    # 从零训练组：显式 ReLoRA 合并周期；未指定时按步数自动折半
    relora_every = args.relora_every
    if method == "aeloru-fromscratch" and relora_every <= 0:
        relora_every = max(1, args.steps // 2)
```
**替换成下面这段**，优先读取挂在model上的`fs_relora_every`：
```python
    relora_every = args.relora_every
    if method == "aeloru-fromscratch":
        # fromscratch优先使用fs‑relora‑every
        if model.fs_relora_every > 0:
            relora_every = model.fs_relora_every
        else:
            relora_every = max(1, args.steps // 2)
        # 实验性开启sleep
        if model.fs_enable_sleep:
            model.do_sleep = True
        else:
            model.do_sleep = False
```

> 这样：
> - `aeloru‑safe/default`：使用全局`--merge‑every`修改cfg.merge_every自动合并
> - `aeloru‑fromscratch`：使用`--fs‑relora‑every`，走手动relora_merge_into_W0，两套互不干扰。

### 4. 学习率部分（无需改动）
- 余弦lr‑schedule、每轮`lr‑decay‑per‑round`对fromscratch完全生效，因为优化器只更新LoRA适配器，调度器作用于AdamW参数组。
> 注意：fromscratch对lr更敏感，不建议lr大于1e‑4；长线repeat场景建议`--lr‑decay‑per‑round 0.7~0.8`。

### 5. sleep说明（实验性）
如果设置`--fs‑enable‑sleep 1`，就会把`model.do_sleep=True`，那么之前写的`sleep‑intensity`循环就会对fromscratch生效。
⚠️风险提示：
> FromScratchWrapper不是transformers标准model，`cross_modal_sleep_step(AeloruModel(model))`内部做模块遍历，**有概率触发维度错误、buffer key缺失**；如果只是做论文基线对比，**不建议打开fs‑enable‑sleep，保持0，依靠高频ReLoRA合并做记忆巩固**。

---

## 完整推荐运行命令（包含fromscratch）
```bash
python ex1_long_line.py \
  --methods lora,aeloru‑safe,aeloru‑default,aeloru‑fromscratch \
  --repeat 2 \
  --steps 250 \
  --r 8 --alpha 16 \
  --merge‑every 500 \
  --lr‑schedule 1 \
  --lr‑decay‑per‑round 0.8 \
  --sleep‑intensity 2 \
  --fs‑relora‑every 250
```
- `--merge‑every 500`：作用 aeloru‑safe / aeloru‑default
- `--fs‑relora‑every 250`：**aeloru‑fromscratch每250步（刚好一个任务训练完）执行一次ReLoRA合并，把LoRA增量折叠进W0，释放低秩容量，对抗长线漂移遗忘**
- fromscratch保持`--fs‑enable‑sleep 0`（默认关闭睡眠）

---

## fromscratch长线常见现象与调参建议
|现象|原因|调参|
|---|---|---|
|第二轮ACC暴跌、遗忘飙升|ReLoRA合并周期太长，LoRA增量堆积饱和|减小`--fs‑relora‑every`到200‑250，不要大于500|
|训练loss震荡、偶尔nan|数值敏感；合并后Adam动量没有完全清空|保持float32；grad_clip=1.0不要调大；不要开fs‑enable‑sleep|
|训练loss降不下去|合并太频繁，刚学到的知识被重置|fs‑relora‑every不要小于125|
|比aeloru‑safe整体acc低|fromscratch是从零认知网络，不是基座微调，表达能力天然和PEFT不同|不要直接绝对值对比，看BWT/Forgetting相对指标|

> 论文原始设定中fromscratch不靠sleep，核心手段就是**高频ReLoRA合并**，这是它对抗概念漂移最重要的杠杆。

## 额外小坑提醒
1. 不要给fromscratch设置`--merge‑every`指望生效，这个参数修改的是AeloruConfig.merge_every，fromscratch代码完全忽略这个配置项，只用显式调用`relora_merge_into_W0`。
2. 如果打开`fs‑enable‑sleep=1`遇到报错，直接关掉，不要调试这个路径；这个wrapper不是为sleep设计的。
3. fromscratch的可训练参数量要和PEFT组对齐，代码已经处理，打印日志可以核对`n_trainable`。

如果你需要，我可以把**全部整合好的完整diff片段**给你，可以直接复制替换对应代码块。