
## 一、核心推导

### 1.1 符号定义

| 符号 | 维度 | 含义 |
|------|------|------|
| $z$ | $(N, C)$ | **logits**（最终线性输出，不含激活） |
| $p$ | $(N, C)$ | 概率输出（Softmax 或 Sigmoid） |
| $y$ | $(N, C)$ | 目标（one-hot / 概率分布 / 0-1） |
| $\delta$ | $(N, C)$ | **损失对 logits 的梯度** $\partial \mathcal{L}/\partial z$ |
| $h'$ | $(N, r)$ | ReLU 后的低秩特征 |

### 1.2 Softmax + 交叉熵（CE）

**Sum Loss**（用于梯度推导的基准形式）：
$$
\mathcal{L}_{\text{sum}} = -\sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(p_{i,c}), \quad p_{i,c} = \frac{e^{z_{i,c}}}{\sum_{k}e^{z_{i,k}}}
$$

利用 Softmax 导数性质 $\frac{\partial p_{i,c}}{\partial z_{i,k}} = p_{i,c}(\delta_{ck} - p_{i,k})$：

$$
\frac{\partial \mathcal{L}_{\text{sum}}}{\partial z_{i,k}} = -\sum_c y_{i,c} \frac{1}{p_{i,c}} \cdot p_{i,c}(\delta_{ck} - p_{i,k}) = p_{i,k} - y_{i,k}
$$

因此：
$$
\boxed{\delta = p - y \quad \text{（不除以 } N\text{）}}
$$

若使用 **Mean Loss**（代码中默认，用于监控）：
$$
\mathcal{L}_{\text{mean}} = \frac{1}{N}\mathcal{L}_{\text{sum}}, \quad \frac{\partial \mathcal{L}_{\text{mean}}}{\partial z} = \frac{1}{N}(p - y)
$$

**修正后的工程约定**：代码中 `delta` 取 **sum-gradient 形式**（不除 $N$），与 MSE 版本的 `error = y_pred - y_target` 保持范式一致。Loss 值仍计算 mean（除以 $N$）以保持与 PyTorch 默认行为一致，batch size 的尺度由学习率 $lr$ 吸收。

### 1.3 Sigmoid + 二元交叉熵（BCE）

$$
\mathcal{L} = -\sum_{i}\left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right], \quad p_i = \sigma(z_i)
$$

直接求导：
$$
\frac{\partial \mathcal{L}}{\partial z_i} = -\left[\frac{y_i}{p_i}p_i(1-p_i) + \frac{1-y_i}{1-p_i}(-p_i(1-p_i))\right] = p_i - y_i
$$

同样得到：
$$
\boxed{\delta = \sigma(z) - y \quad \text{（不除以 } N\text{）}}
$$

### 1.4 链式法则后半段（与 MSE 完全一致）

无论 MSE 还是 CE/BCE，一旦得到 $\delta = \partial \mathcal{L}/\partial z$，后续通过低秩适配器的梯度传播链完全相同：

$$
\begin{aligned}
\text{grad}_B &= \delta^\top h' \cdot \frac{\alpha}{r} \quad &&(C, r) \\
\text{grad}_A &= \left[(\delta B) \odot \mathbb{1}_{h>0}\right]^\top x \cdot \frac{\alpha}{r} \quad &&(r, d_{\text{in}})
\end{aligned}
$$

---

## 二、修正后的完整代码

### 2.1 配置类新增字段（`AeloruConfig` 中）

```python
# --- HGF 闭式一步更新（实验性）---
use_hgf_closed_form: bool = False    # 实验性：用闭式梯度替代 autograd
hgf_loss_mode: str = "ce"           # "ce" (Softmax交叉熵) | "bce" (Sigmoid二分类)
hgf_label_smoothing: float = 0.0     # 标签平滑系数（仅CE模式）
```

### 2.2 替换 `hgf_closed_form_update` 方法

```python
def hgf_closed_form_update(self, x: torch.Tensor, y_target: torch.Tensor, loss_mode: str = "ce") -> torch.Tensor:
    """
    HGF 闭式一步更新 —— 交叉熵版本（Softmax CE / Sigmoid BCE）。
    
    推导要点：
    1. 严格定义 δ = ∂L/∂z（损失对 logits 的梯度）。
    2. Softmax + CE:  δ = softmax(z) - y_target。
    3. Sigmoid + BCE: δ = sigmoid(z) - y_target。
    4. 链式法则后半段与 MSE 完全一致：grad_B = δ^T @ h' * (α/r)。
    
    【修正】v2.1: δ 不再预先除以 batch size N，保持与 MSE 版本的 error 一致。
    损失值 loss 仍使用 mean（除以 N）以保持与 PyTorch 默认行为一致，
    梯度更新采用 sum 形式，由学习率 lr 吸收 batch size 尺度。
    
    Args:
        x:        输入 (batch, in_features)
        y_target: 目标 
                  - CE 模式: (batch,) Long 类别索引，或 (batch, C) one-hot/概率
                  - BCE 模式: (batch, C) float，值域 [0, 1]
        loss_mode: "ce" | "bce"
    
    Returns:
        loss: 标量损失值（mean）
    """
    if not self.cfg.use_hgf_closed_form:
        raise RuntimeError("hgf_closed_form_update 仅在 use_hgf_closed_form=True 时可用")
    
    if x.dtype != self.lora_A.dtype:
        x = x.to(self.lora_A.dtype)
    
    with torch.no_grad():
        # ========== 1. 前向（与原版完全一致）==========
        h = F.linear(x, self.lora_A)  # (batch, r)
        
        # 侧向抑制
        if self.cfg.use_lateral_connection and self.lateral_weights is not None:
            h = h - F.linear(h, self.lateral_weights)
        
        h_relu = F.relu(h)  # (batch, r)
        
        # Hi-DoRA 调制
        if self.cfg.use_hidora and self.m_x is not None and self.m_y is not None:
            effective_B = self.lora_B * self.m_x.unsqueeze(1)
            y_pred = F.linear(h_relu, effective_B)
        else:
            y_pred = F.linear(h_relu, self.lora_B)
        
        scale = self.cfg.lora_alpha / self.cfg.r
        y_pred = y_pred * scale
        
        # 加上冻结基座
        y_base = F.linear(x, self.W0, self.bias)
        if self.cfg.use_relora:
            y_base = y_base + F.linear(x, self._get_W_acc())
        y_pred = y_pred + y_base  # (batch, out_features)
        
        N = y_pred.size(0)
        
        # ========== 2. 损失与误差项 δ（核心修正区）==========
        if loss_mode == "ce":
            # Softmax 交叉熵
            if y_target.dtype == torch.long:
                # y_target 是 (batch,) 的类别索引
                loss = F.cross_entropy(y_pred, y_target)  # mean（内部已除N）
                probs = F.softmax(y_pred, dim=-1)
                y_onehot = F.one_hot(y_target, num_classes=y_pred.size(-1)).float()
                delta = probs - y_onehot  # (batch, C)，不除N
            else:
                # y_target 是 (batch, C) 的概率分布 / one-hot
                probs = F.softmax(y_pred, dim=-1)
                probs = probs.clamp(min=1e-7, max=1.0)
                loss = -(y_target * torch.log(probs)).sum() / N  # mean
                delta = probs - y_target  # 不除N
            
            # 标签平滑（可选）
            if getattr(self.cfg, 'hgf_label_smoothing', 0.0) > 0.0:
                eps = self.cfg.hgf_label_smoothing
                delta = delta * (1.0 - eps)
                
        elif loss_mode == "bce":
            # Sigmoid 二分类交叉熵（数值稳定版）
            y_target_float = y_target.float() if y_target.dtype != torch.float32 else y_target
            loss = F.binary_cross_entropy_with_logits(y_pred, y_target_float)  # mean
            probs = torch.sigmoid(y_pred)
            delta = probs - y_target_float  # 不除N
            
        else:
            raise ValueError(f"不支持的 loss_mode: {loss_mode}，仅支持 'ce' 或 'bce'")
        
        # ========== 3. HGF 冲突信号（使用 |δ| 均值）==========
        conflict_signal = delta.abs().mean().item()
        self._append_hgf_conflict(conflict_signal)
        
        # ========== 4. 闭式梯度（δ 不除N，与 MSE 的 error 保持一致）==========
        # grad_B: (out_features, r) = (out_features, batch) @ (batch, r)
        grad_B = torch.mm(delta.t(), h_relu) * scale
        
        # grad_A: (r, in_features)
        B_for_grad = effective_B if (self.cfg.use_hidora and self.m_x is not None) else self.lora_B
        grad_h = torch.mm(delta, B_for_grad) * (h > 0).float() * scale  # (batch, r)
        grad_h = grad_h.to(self.cfg.AMP_DTYPE)
        grad_A = torch.mm(grad_h.t(), x)  # (r, in_features)
        
        # ========== 5. 手动参数更新 ==========
        lr = self._get_effective_lora_lr()
        self.lora_B.data.sub_(lr * grad_B)
        self.lora_A.data.sub_(lr * grad_A)
        
        # ========== 6. Hi-DoRA 幅度向量 ==========
        if self.cfg.use_hidora and self.m_x is not None and self.m_y is not None:
            self.m_x.data.sub_(lr * grad_B.norm(dim=1) * 1e-3)
            self.m_y.data.sub_(lr * grad_A.norm(dim=0) * 1e-3)
        
        # ========== 7. 记录参数更新幅度 ==========
        delta_norm = (lr * grad_B).norm().item() + (lr * grad_A).norm().item()
        self._append_hgf_delta(delta_norm)
        
        return loss
```

### 2.3 `train_aeloru_step` 衔接修改

```python
# --- 0. HGF 闭式路径（修正后）---
if layer.cfg.use_hgf_closed_form:
    loss_mode = getattr(layer.cfg, 'hgf_loss_mode', 'ce')
    loss = layer.hgf_closed_form_update(x, y_target, loss_mode=loss_mode)
    
    # Hebbian 和状态机仍需通过 post_step_update 处理
    with torch.no_grad():
        y_pred = layer(x)  # 基于更新后参数的重新前向
    layer.post_step_update(x, y_pred.detach(), is_correct=reward_signal, y_target=y_target)
    
    metrics = {
        'state': layer.state.value,
        'loss_total': loss.item(),
        'grad_norm': 0.0,
        'anchor_converged': False,
        'relora_merged': False,
        'hebbian_order': 'hgf_closed_form',
        'hebbian_pending': layer._hebbian_pending_apply if layer.cfg.use_hebbian else False,
        'hgf_closed_form': True,
        'hgf_loss_mode': loss_mode,
    }
    return loss, metrics
```

---

## 三、关键修正对照表

| 项目 | 原实现（错误） | 修正后 |
|------|---------------|--------|
| **CE 的 `delta`** | `(softmax(y) - target) / N` | `softmax(y) - target`（不除N） |
| **BCE 的 `delta`** | 未实现 | `sigmoid(y) - target`（不除N） |
| **损失函数** | MSE 仅 | `F.cross_entropy` / `F.binary_cross_entropy_with_logits` |
| **标签类型** | 仅 float | 支持 `Long` 索引（CE）和 `Float` 0-1（BCE） |
| **δ 的严格定义** | 未明确 | 明确为 **损失对 logits 的梯度** $\partial \mathcal{L}/\partial z$ |

**注**：由于 `delta` 不再除 $N$，实际梯度为 **sum-gradient** 形式。若需与原始 MSE 代码（`grad_B` 处除 $N$）保持学习率数值一致，请将学习率缩小为原来的 $1/N$（$N$ 为 batch size），或在 `grad_B`/`grad_A` 计算后手动补 `/ N`（但 `delta` 本身仍应保持不除N）。