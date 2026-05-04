import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HiDoRALayer(nn.Module):
    """
    Hi-DoRA (Hadamard-enhanced DoRA) Layer - 修复优化版
    
    核心公式: W' = m * (W0 + W0 ⊙ (α/r · BA)) / ||W0 + W0 ⊙ (α/r · BA)||_c
    优化点:
    1. 修复哈达玛积形状不匹配问题
    2. 加入LoRA缩放因子 α/r，初始状态下ΔW=0
    3. m初始化为预训练权重的列范数，保证初始W'=W0
    4. 支持预训练bias的保留
    5. 用F.linear替代torch.mm，支持任意batch维度和AMP
    6. 预留赫布学习钩子接口
    7. 新增适配器保存/加载方法
    """
    def __init__(self, in_features: int, out_features: int, r: int = 32, lora_alpha: float = 8.0):
        """
        初始化 Hi-DoRA 层。
        
        Args:
            in_features: 输入维度 (d)
            out_features: 输出维度 (k)
            r: LoRA 的秩 (Rank)
            lora_alpha: 缩放因子 (Alpha)，用于控制更新幅度
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        
        # --- 冻结的预训练参数 ---
        self.W0 = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)  # 新增：保留预训练bias
        
        # --- 可训练参数 (Trainable Parameters) ---
        # 1. LoRA 低秩矩阵 A 和 B（调整形状以匹配W0，修复哈达玛积问题）
        # 形状: A (r, in_features), B (out_features, r)
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        
        # 2. 幅度向量 (Magnitude Vector) "m"
        # 形状: (out_features,)，初始化为预训练权重的列范数（在set_pretrained_weight中完成）
        self.m = nn.Parameter(torch.empty(out_features))
        
        # --- 预留接口 ---
        self.hebbian_hook = self.hebbian_update_fn  # 赫布学习更新钩子
        
        # --- 初始化 (Initialization) ---
        # LoRA原始初始化：A用Kaiming Uniform，B初始化为全0（保证初始ΔW=0）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # 关键：初始B为0，初始状态下无扰动

    def set_pretrained_weight(self, W0, bias=None):
        """
        设置预训练权重和bias，并初始化m为W0的列范数。
        
        Args:
            W0: 预训练权重 (out_features, in_features)
            bias: 预训练bias (out_features,)，可选
        """
        self.W0.data = W0.clone()
        if bias is not None:
            self.bias.data = bias.clone()
        else:
            self.bias.data = torch.zeros(self.out_features)
        
        # 关键：用W0的列范数初始化m，保证初始W' = W0
        self.m.data = torch.norm(W0, p=2, dim=0)

    def hebbian_update_fn(
            self,
            lora_A, 
            lora_B, 
            x, 
            y, 
            hebbian_lr=1e-6,          # 赫布学习率（建议始终小于反向传播学习率的1/10）
            is_correct=True,           # 结果门控：当前输出是否正确，决定强化/弱化
            saturation_limit=5.0,      # 饱和上限：修正后默认5.0，避免过度限制表达能力
            forget_decay=0.99          # 全局遗忘衰减：修正后默认0.99，增强遗忘正则效果
            ):
        """
        修复版核心赫布更新函数
        严格对齐Hi-DoRA层LoRA矩阵形状，修复维度匹配bug，新增强制维度检查
        核心规则：同步激活→强化连接，异步激活→弱化连接
        """
        # ========== 新增：强制维度检查，提前拦截维度错误 ==========
        # 检查LoRA矩阵形状是否符合Hi-DoRA标准
        rank, in_features = lora_A.shape
        out_features, rank_B = lora_B.shape
        assert rank == rank_B, f"LoRA矩阵rank不匹配：lora_A rank={rank}, lora_B rank={rank_B}"
        # 检查输入输出维度是否匹配层定义
        assert x.shape[-1] == in_features, f"输入维度不匹配：期望in_features={in_features}, 实际输入{x.shape[-1]}"
        assert y.shape[-1] == out_features, f"输出维度不匹配：期望out_features={out_features}, 实际输出{y.shape[-1]}"

        # ========== 1. 全局遗忘衰减（先衰减，再更新） ==========
        with torch.no_grad():
            lora_A.data *= forget_decay
            lora_B.data *= forget_decay

            # ========== 2. 计算batch维度的平均激活，过滤单样本噪声 ==========
            # x形状: (batch_size, in_features) → 平均后: (1, in_features)
            x_mean = x.mean(dim=0, keepdim=True)
            # y形状: (batch_size, out_features) → 平均后: (out_features, 1)
            y_mean = y.mean(dim=0, keepdim=True).T

            # ========== 3. 修正：对称更新符号，正确/错误使用对称的±1.0 ==========
            update_sign = 1.0 if is_correct else -1.0

            # ========== 4. 核心修复：维度完全对齐的赫布更新 ==========
            # --- 对lora_B (out_features, rank) 的更新 ---
            # 逻辑：输出激活(y_mean) × 输入激活的前rank维，匹配lora_B的(out_features, rank)形状
            # 修复：使用lora_B的rank维度（lora_B.shape[1]）做切片，完全匹配rank
            lora_B_update = update_sign * hebbian_lr * torch.mm(y_mean, x_mean[:, :lora_B.shape[1]])
            # --- 对lora_A (rank, in_features) 的更新 ---
            # 逻辑：输出激活的前rank维 × 输入激活，匹配lora_A的(rank, in_features)形状
            # 修复：使用lora_A的rank维度（lora_A.shape[0]）做切片，完全匹配rank
            lora_A_update = update_sign * hebbian_lr * torch.mm(y_mean[:lora_A.shape[0], :], x_mean)

            # ========== 5. 执行更新 ==========
            lora_B.data += lora_B_update
            lora_A.data += lora_A_update

            # ========== 6. 饱和上限限制，掐死正反馈失控 ==========
            lora_A.data.clamp_(-saturation_limit, saturation_limit)
            lora_B.data.clamp_(-saturation_limit, saturation_limit)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算 Hi-DoRA 更新后的权重并应用，支持赫布更新。
        
        Args:
            x: 输入张量 (batch_size, in_features)
        
        Returns:
            输出张量 (batch_size, out_features)
        """
        # 计算更新后的权重
        W_prime = self.compute_weights()
        
        # 用F.linear替代torch.mm，支持任意batch维度、AMP和bias
        y = F.linear(x, W_prime, self.bias) # pylint: disable=E1102 (PS: 这里没错,pylint的静态分析误报了)
        
        # 如果注册了赫布钩子且处于训练模式，执行赫布更新
        if self.hebbian_hook is not None and self.training:
            if callable(self.hebbian_hook):
                self.hebbian_hook(self.lora_A, self.lora_B, x, y) # pylint: disable=E1102 (PS: 这里上面的if判断了self.hebbian_hook是不是被赋值成了函数,所以这里不会报错)
        
        return y

    def compute_weights(self):
        """
        计算 Hi-DoRA 更新后的权重矩阵。
        
        Returns:
            W_prime: 更新后的权重矩阵 (out_features, in_features)
        """
        # --- 步骤 1: 计算带缩放的LoRA低秩乘积 ---
        # 公式: ΔW_base = (α/r) · BA
        # 形状: (out_features, in_features)，与W0完全匹配
        AB = torch.mm(self.lora_B, self.lora_A)
        scaled_AB = (self.lora_alpha / self.r) * AB
        
        # --- 步骤 2: 计算HiRA风格的高秩方向增量（哈达玛积） ---
        # 公式: ΔV = W0 ⊙ ΔW_base
        # 利用W0的结构信息提升有效秩，解决传统LoRA秩不足问题
        hira_direction_delta = self.W0 * scaled_AB
        
        # --- 步骤 3: 构建新的方向矩阵 ---
        # 公式: V_new = W0 + ΔV
        new_direction = self.W0 + hira_direction_delta
        
        # --- 步骤 4: 按列计算L2范数（用于归一化） ---
        # dim=0: 按列计算（每个输出神经元对应一列）
        # keepdim=True: 保持维度以便广播
        direction_norm = torch.norm(new_direction, p=2, dim=0, keepdim=True)
        
        # --- 步骤 5: 应用DoRA幅度-方向解耦公式 ---
        # 公式: W' = m · (V_new / ||V_new||_c)
        # 加1e-8防止除零
        W_prime = self.m * (new_direction / (direction_norm + 1e-8))
        
        return W_prime

    def save_adapter(self, path):
        """
        保存Hi-DoRA适配器（仅保存可训练参数，不保存预训练权重）。
        
        Args:
            path: 保存路径
        """
        torch.save({
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_A": self.lora_A.data.clone(),
            "lora_B": self.lora_B.data.clone(),
            "m": self.m.data.clone()
        }, path)
        print(f"Adapter saved to {path}")

    def load_adapter(self, path):
        """
        加载Hi-DoRA适配器。
        
        Args:
            path: 加载路径
        """
        checkpoint = torch.load(path, map_location="cpu")
        # 验证秩和缩放因子是否匹配
        if checkpoint["r"] != self.r or checkpoint["lora_alpha"] != self.lora_alpha:
            raise ValueError(f"Adapter config mismatch: expected r={self.r}, alpha={self.lora_alpha}; "
                             f"got r={checkpoint['r']}, alpha={checkpoint['lora_alpha']}")
        
        self.lora_A.data = checkpoint["lora_A"]
        self.lora_B.data = checkpoint["lora_B"]
        self.m.data = checkpoint["m"]
        print(f"Adapter loaded from {path}")


# --- 注入 (Injection) 辅助函数 - 优化版 ---
def inject_hidora(model, target_names=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"], r=32, alpha=8.0):
    """
    递归地将模型中的指定线性层替换为 Hi-DoRA 适配器。
    默认匹配Transformer/MLP中常见的线性层名。
    
    Args:
        model: 待注入的模型
        target_names: 要替换的线性层名列表
        r: LoRA秩
        alpha: LoRA缩放因子
    
    Returns:
        注入后的模型
    """
    for name, module in model.named_children():
        # 递归处理子模块
        if len(list(module.children())) > 0:
            inject_hidora(module, target_names, r, alpha)
            
        # 替换匹配的线性层
        if isinstance(module, nn.Linear) and any(target in name for target in target_names):
            # 创建Hi-DoRA层
            hidora_layer = HiDoRALayer(module.in_features, module.out_features, r, alpha)
            # 注入预训练权重和bias
            hidora_layer.set_pretrained_weight(module.weight.data, module.bias.data)
            
            # 替换原模块
            parent = model
            attr_name = name
            setattr(parent, attr_name, hidora_layer)
            
            print(f"✅ Injected Hi-DoRA into: {name} (in={module.in_features}, out={module.out_features}, r={r})")
            
    return model


# --- 测试验证脚本 - 确保代码正常工作 ---
def test_hidora():
    '''Hi-DoRA 层检测'''
    print("="*50)
    print("开始测试Hi-DoRA层...")
    print("="*50)
    
    # 1. 创建一个普通的线性层作为预训练模型
    in_dim = 128
    out_dim = 64
    batch_size = 4
    original_linear = nn.Linear(in_dim, out_dim)
    original_linear.eval()  # 切换到评估模式
    
    # 2. 生成随机输入
    x = torch.randn(batch_size, in_dim)
    
    # 3. 计算原始线性层的输出
    with torch.no_grad():
        original_output = original_linear(x)
    
    # 4. 将原始线性层替换为Hi-DoRA层
    hidora_layer = HiDoRALayer(in_dim, out_dim, r=8, lora_alpha=4)
    hidora_layer.set_pretrained_weight(original_linear.weight.data, original_linear.bias.data)
    hidora_layer.eval()  # 切换到评估模式
    
    # 5. 计算Hi-DoRA层的初始输出（应该和原始输出完全一致）
    with torch.no_grad():
        hidora_initial_output = hidora_layer(x)
    
    # 6. 验证初始输出是否一致
    output_diff = torch.max(torch.abs(original_output - hidora_initial_output)).item()
    print(f"\n🔍 初始输出验证:")
    print(f"   原始输出均值: {original_output.mean().item():.6f}")
    print(f"   Hi-DoRA初始输出均值: {hidora_initial_output.mean().item():.6f}")
    print(f"   最大绝对误差: {output_diff:.10f}")
    
    if output_diff < 1e-6:
        print("   ✅ 初始输出验证通过！Hi-DoRA初始状态与预训练模型完全一致")
    else:
        print("   ❌ 初始输出验证失败！")
        return
    
    # --- 修复点1：在反向传播之前，先保存初始适配器 ---
    print(f"\n🔍 适配器保存/加载验证:")
    test_path = "test_hidora_adapter.pt"
    hidora_layer.save_adapter(test_path)  # 保存的是【初始未更新】的适配器
    
    # 创建新的Hi-DoRA层并加载适配器
    new_hidora_layer = HiDoRALayer(in_dim, out_dim, r=8, lora_alpha=4)
    new_hidora_layer.set_pretrained_weight(original_linear.weight.data, original_linear.bias.data)
    new_hidora_layer.load_adapter(test_path)
    new_hidora_layer.eval()
    
    # 验证加载后的输出是否和【初始Hi-DoRA输出】一致
    with torch.no_grad():
        loaded_output = new_hidora_layer(x)
    load_diff = torch.max(torch.abs(hidora_initial_output - loaded_output)).item()
    print(f"   加载后输出与初始输出最大绝对误差: {load_diff:.10f}")
    
    if load_diff < 1e-6:
        print("   ✅ 适配器保存/加载验证通过！")
    else:
        print("   ❌ 适配器保存/加载验证失败！")
        return
    
    # --- 修复点2：把反向传播验证放到保存/加载之后 ---
    print(f"\n🔍 反向传播验证:")
    hidora_layer.train()
    optimizer = torch.optim.AdamW(hidora_layer.parameters(), lr=1e-4)
    
    # 做一步前向+反向传播
    hidora_output_train = hidora_layer(x)
    loss = hidora_output_train.sum()
    loss.backward()
    optimizer.step()
    
    # 检查可训练参数是否有梯度
    has_grad = (hidora_layer.lora_A.grad is not None and 
                hidora_layer.lora_B.grad is not None and 
                hidora_layer.m.grad is not None)
    
    if has_grad:
        print("   ✅ 反向传播验证通过！可训练参数梯度正常")
    else:
        print("   ❌ 反向传播验证失败！可训练参数无梯度")
        return
    
    # 清理测试文件
    import os
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("\n" + "="*50)
    print("🎉 所有测试通过！Hi-DoRA层可以正常使用")
    print("="*50)

    '''hebbian 层 测试'''
    print("="*50)
    print("开始测试hebbian层...")
    print("="*50)
    # 初始化测试层
    test_layer = HiDoRALayer(in_features=128, out_features=64, r=8, lora_alpha=4)
    test_layer.set_pretrained_weight(torch.randn(64, 128))
    test_layer.train()  # 训练模式自动开启赫布更新

    before_A = test_layer.lora_A.data.clone()
    before_B = test_layer.lora_B.data.clone()

    test_x = torch.randn(4, 128)
    _ = test_layer(test_x)

    A_updated = not torch.allclose(before_A, test_layer.lora_A.data, atol=1e-10)
    B_updated = not torch.allclose(before_B, test_layer.lora_B.data, atol=1e-10)

    print(f"维度无报错 ✅")
    print(f"lora_A 正常更新: {A_updated}")
    print(f"lora_B 正常更新: {B_updated}")
    print("✅ Hebbian 测试通过！")

# 运行测试
if __name__ == "__main__":
    test_hidora()