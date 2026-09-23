
## 一、稳态增益量纲修正（最高优先级，1行代码）
### 问题根源
`running_var` 维护的是**输出激活的二阶矩（方差）**，单位是“平方”；而稳态增益的物理意义是「激活强度归一化」，应该和**标准差**成反比，而不是方差。

原代码 `gain = 1 / running_var` 相当于做了 1/σ² 的缩放，多层叠加之后，输出RMS会被压缩到原来的1/3次方量级，层数越多压缩越狠——这就是24层全注入直接学不动的根本原因。

现在的safe配置靠 `homeostatic_max_gain=1.0` 强行把增益上限卡死在1，相当于“绕过了这个bug”，但没有真正发挥稳态归一化的作用。

### 具体修改
找到 `forward` 函数里的稳态增益计算部分：
```python
# 原代码
homeostatic_gain = 1.0 / (self.running_var + 1e-5)
```
替换为：
```python
# 修正后：与标准差成反比，量纲匹配
homeostatic_gain = 1.0 / torch.sqrt(self.running_var + self.cfg.precision_eps)
```

### 配套调整
量纲修正后增益整体会变大，建议把 `homeostatic_max_gain` 默认值从 5.0 下调到 **2.0**（LLM微调场景），避免过度缩放；safe配置里的1.0限制可以放开到1.5，稳态归一化的效果会真正发挥出来，准确率预计能再涨0.5~1个百分点。

## 二、PEM低秩空间改造（可选增强，小改提升大）
### 核心思路
原来的预测编码神经动态是在**完整输出维度（d_out）**上做迭代修正，会直接扰动预训练模型已经收敛的激活分布，所以在LLM上会破坏表征。

把神经动态挪到**r维低秩特征空间h**里做，只在低秩子空间内做特征解耦和侧抑制，完全不碰主激活路径——既保留PEM的特征解耦作用，又几乎不影响预训练表征。

### 具体修改
1. **位置**：`forward` 函数中，计算出 `h = F.linear(x, effective_A)` 之后，做侧向抑制和WTA之前插入。
2. **改造方式**：不在完整y上调用`run_predictive_neural_dynamics`，改为在r维h空间做轻量神经动态。

可以新增一个简化版的低秩神经动态函数，或者直接复用现有逻辑，把输入换成h：
```python
# 在 h 空间做预测编码动态（只解耦特征，不碰主输出）
if self.training and self.cfg.neural_dynamics_iterations > 0:
    h = h + (self._low_rank_neural_dynamics(h) - h.detach())
```
其中 `_low_rank_neural_dynamics` 就是把原来的run_predictive_neural_dynamics简化：输入是r维，不用算yke（基准设为0），只做侧向抑制和精度加权，计算量只有O(r²)，几乎可以忽略。

### 收益
- 彻底解决PEM破坏预训练激活的问题，不用再完全关闭ndi；
- 低秩特征的解耦效果更强，有效秩进一步提升；
- 计算开销比全输出版小两个数量级。

## 三、LowRankPrecisionMatrix初始化修复（冷启动稳定性）
### 问题根源
`LowRankPrecisionMatrix` 初始化时 `B_basis` 是全零矩阵，导致第一次更新前，精度矩阵退化为 `C_y ≈ I / epsilon`，侧向项 `C_y_inv_y - y` 近似等于 `y/epsilon - y`，直接把输出放大几十倍——这就是PEM初期输出爆炸的核心原因之一。

### 具体修改
在 `set_pretrained_weight` 函数的末尾，初始化完lora_B之后，给B_basis赋初值：
```python
# 在 set_pretrained_weight 内添加
with torch.no_grad():
    # 用初始低秩B作为精度矩阵的初始基，避免冷启动爆炸
    init_basis = self.lora_B.data.float().clone()
    rank_pad = self.cfg.predictive_coding_rank - init_basis.shape[1]
    if rank_pad > 0:
        init_basis = F.pad(init_basis, (0, rank_pad))
    elif rank_pad < 0:
        init_basis = init_basis[:, :self.cfg.predictive_coding_rank]
    self.lowrank_precision.B_basis.copy_(init_basis)
```

### 效果
PEM冷启动阶段的输出不会再出现数量级爆炸，训练初期loss起点更正常，不需要从零开始慢慢涨基。

## 四、默认配置拆分（易用性优化）
### 方案
在 `AeloruConfig` 类里新增两个类方法，直接返回场景化预设，用户不用手动调十几个参数，避免踩坑。

```python
@classmethod
def llm_tuning(cls, **kwargs):
    """
    预训练LLM微调安全预设：
    - 关闭全输出PEM神经动态（改用低秩版可自行开启）
    - 关闭硬域约束
    - 稳态增益修正量纲并限制上限
    - 适合直接注入现有大模型做微调
    """
    defaults = dict(
        neural_dynamics_iterations=0,
        presumed_domain="none",
        homeostatic_max_gain=2.0,
        whitening_interval=10**9,  # 预训练模型已有LayerNorm，无需频繁白化
    )
    defaults.update(kwargs)
    return cls(**defaults)

@classmethod
def cognitive_model(cls, **kwargs):
    """
    认知模型完整预设：
    - 全开PEM预测编码、域约束、稳态可塑性
    - 适合从零训练独立认知模型（Qelys/Lorys）
    """
    defaults = dict(
        neural_dynamics_iterations=5,
        presumed_domain="nnantisparse",
        homeostatic_max_gain=5.0,
        whitening_interval=500,
    )
    defaults.update(kwargs)
    return cls(**defaults)
```

### 使用方式
```python
# 微调大模型
cfg = AeloruConfig.llm_tuning(r=8, alpha=16)
# 训练自己的认知模型
cfg = AeloruConfig.cognitive_model(r=16, alpha=32)
```
对外发布默认推荐用 `llm_tuning`，新手不会踩坑；自己做Qelys的时候用完整预设。

## 五、其他发现的潜在问题与修复
### 1. 冲突检测缓冲未接入数据，HongWen状态机自动触发未生效
**问题**：`_hgf_delta_buffer` 只定义了追加函数，但整个代码里没有地方调用 `_append_hgf_delta`，缓冲区永远为空。导致 `_detect_and_transition` 里的冲突检测永远不触发，HongWen状态机目前只能靠固定步数切换，动态冲突检测功能没有真正跑起来。

同理，`check_anchor_convergence` 也依赖这个缓冲区，目前也处于未生效状态。

**修复方法**：在 `post_step_update` 末尾追加一行，把Hebbian更新幅度作为冲突信号喂入缓冲区：
```python
# 在 post_step_update 末尾添加
delta_mag = float(self._hebbian_acc_B.abs().mean().item() + self._hebbian_acc_A.abs().mean().item())
self._append_hgf_delta(delta_mag)
```
修复后状态机才能真正根据训练波动自动切换红温/探索/固化，整套认知节律才是活的。

### 2. batch_forward_lora 批量优化存在双重缩放风险
**问题**：`batch_forward_lora` 里的scale乘了 `plasticity_out.mean() * plasticity_in.mean()`，而正常forward里是逐元素乘门控。如果后续启用批量前向钩子，会出现“全局均值缩放 + 逐元素缩放”的双重增益，等效学习率偏高。
**现状**：目前钩子只是注册了，结果没有被forward使用，所以不影响当前实验。
**后续修复**：启用批量优化的时候，把forward里的门控移除，只在batch_forward里统一做缩放，保持一致。

### 3. 输入白化默认间隔对预训练模型过频
`whitening_interval=1000` 对于预训练LLM来说太频繁了，预训练模型本身已经有LayerNorm，输入白化属于重复归一化，反而可能破坏表征。safe配置里已经设成了1e9（关闭），建议默认配置也调大，或者默认关闭。
