

## HGF 闭式更新：MSE → 交叉熵 推导总结

### 一、问题设定

**模型结构**（低秩适配器）：
```
x → h = xA^T → h' = ReLU(h) → z = h'B^T·(α/r) → y = y_base + z
```
其中 $y_{base} = x(W_0+W_{acc})^T + b$ 为冻结基座输出。

**目标**：将损失函数从 MSE 替换为 Softmax 交叉熵（CE）或 Sigmoid BCE，同时保持**闭式梯度更新**（无 autograd）。

---

### 二、核心洞察

> **损失函数的差异仅体现在输出层对 logits 的局部梯度 $\delta = \partial\mathcal{L}/\partial z$，低秩适配器内部的反向传播链完全不变。**

因此，只需替换 $\delta$ 的计算，$grad_B$ 和 $grad_A$ 的公式形式保持不变。

---

### 三、MSE 版本（基准）

**损失**：
$$\mathcal{L}_{MSE} = \frac{1}{N}\sum_{i=1}^N \|z_i - y_i^{target}\|^2$$

**误差项**（损失对 logits 的梯度）：
$$\delta_{MSE} = \frac{2}{N}(z - y^{target}) \approx z - y^{target}$$

（常数 2 和 $1/N$ 吸收进学习率）

**闭式梯度**：
$$\text{grad}_B = \delta^T h' \cdot \frac{\alpha}{r}, \quad \text{grad}_A = [(\delta B) \odot \mathbb{1}_{h>0}]^T x \cdot \frac{\alpha}{r}$$

---

### 四、Softmax + 交叉熵（CE）版本

**损失**（mean 形式）：
$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^N \sum_{c=1}^C y_{i,c}^{target} \log(p_{i,c}), \quad p_{i,c} = \frac{e^{z_{i,c}}}{\sum_k e^{z_{i,k}}}$$

**关键推导**（利用 Softmax 导数性质）：

Softmax 概率对 logits 的 Jacobian：
$$\frac{\partial p_{i,c}}{\partial z_{i,k}} = p_{i,c}(\delta_{ck} - p_{i,k})$$

损失对 logits 的梯度：
$$\frac{\partial \mathcal{L}_{CE}}{\partial z_{i,k}} = -\frac{1}{N}\sum_c y_{i,c}\frac{1}{p_{i,c}} \cdot p_{i,c}(\delta_{ck}-p_{i,k}) = \frac{1}{N}(p_{i,k} - y_{i,k})$$

**工程约定**（闭式更新采用 sum-gradient）：
$$\boxed{\delta_{CE} = p - y^{target} \quad \text{（不除以 } N\text{）}}$$

**闭式梯度**（形式与 MSE 完全一致）：
$$\text{grad}_B = \delta_{CE}^T h' \cdot \frac{\alpha}{r}, \quad \text{grad}_A = [(\delta_{CE} B) \odot \mathbb{1}_{h>0}]^T x \cdot \frac{\alpha}{r}$$

---

### 五、Sigmoid + BCE 版本

**损失**：
$$\mathcal{L}_{BCE} = -\frac{1}{N}\sum_i \left[y_i\log(p_i) + (1-y_i)\log(1-p_i)\right], \quad p_i = \sigma(z_i)$$

**误差项**：
$$\frac{\partial \mathcal{L}_{BCE}}{\partial z_i} = \frac{1}{N}(p_i - y_i)$$

**工程约定**：
$$\boxed{\delta_{BCE} = \sigma(z) - y^{target} \quad \text{（不除以 } N\text{）}}$$

---

### 六、修正对照表

| 项目  | 修正后 |
|:---|:---|
| **$\delta_{CE}$ 计算**  | `softmax(z) - y`（不除N） |
| **$\delta_{BCE}$ 计算** | `sigmoid(z) - y`（不除N） |
| **损失值** | `F.cross_entropy` / `F.binary_cross_entropy_with_logits`（mean） |
| **梯度更新**  | sum-gradient，由 `lr` 吸收 batch size |
| **标签支持** | CE: `Long` 索引或概率分布；BCE: `[0,1]` float |

---

### 七、统一公式

$$\boxed{
\begin{aligned}
&\text{通用结构: } \delta = \text{activation}(z) - y^{target} \\
&\text{grad}_B = \delta^T h' \cdot \frac{\alpha}{r} \\
&\text{grad}_A = [(\delta B) \odot \mathbb{1}_{h>0}]^T x \cdot \frac{\alpha}{r}
\end{aligned}
}$$

| 损失模式 | activation(z) | 目标格式 |
|:---|:---|:---|
| MSE | $z$（恒等） | float |
| CE | $\text{softmax}(z)$ | `Long` 索引 或 one-hot |
| BCE | $\sigma(z)$ | `[0,1]` float |

**关键结论**：从 MSE 切换到交叉熵，仅需将线性残差 $z - y$ 替换为概率残差 $p - y$，低秩适配器的闭式梯度传播链无需任何修改。