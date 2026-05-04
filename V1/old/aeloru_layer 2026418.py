import torch
import torch.nn as nn
import math

class HiDoRALayer(nn.Module):
    """
    Hi-DoRA (Hadamard-enhanced DoRA) Layer
    
    结合了 HiRA 的高秩特性 (利用哈达玛积) 和 DoRA 的权重分解特性 (幅度/方向解耦)。
    公式: W' = m * (V + W0 * (A @ B)) / ||V + W0 * (A @ B)||
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
        
        # 冻结的预训练权重
        self.W0 = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        
        # --- 可训练参数 (Trainable Parameters) ---
        
        # 1. LoRA 低秩矩阵 A 和 B
        # 用于生成方向更新量 (Direction Update)
        # 形状: A (in_features, r), B (r, out_features)
        self.lora_A = nn.Parameter(torch.empty(in_features, r))
        self.lora_B = nn.Parameter(torch.empty(r, out_features))
        
        # 2. 幅度向量 (Magnitude Vector) "m"
        # 这是一个可学习的缩放因子，用于控制整个权重的尺度。
        # 形状: (out_features,) 或 (1, out_features)
        # 初始化为 1，保证初始输出与预训练权重一致。
        self.m = nn.Parameter(torch.ones(out_features))
        
        # --- 初始化 (Initialization) ---
        # 参考 LoRA 的 Kaiming Uniform 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
        # m 初始化为 1，或者可以初始化为预训练权重的范数
        # nn.init.ones_(self.m) 

    def set_pretrained_weight(self, W0):
        self.W0.data = W0.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算 Hi-DoRA 更新后的权重矩阵并应用。
        
        Args:
            x: 输入张量
        
        Returns:
            输出张量
        """
        # 计算更新后的权重
        W_prime = self.compute_weight()
        
        # 应用权重
        return torch.mm(x, W_prime.T)  # 假设 W_prime 是 (out_features, in_features)

    def compute_weight(self):
        """
        计算 Hi-DoRA 更新后的权重矩阵。
        
        Returns:
            W_prime: 更新后的权重矩阵 (Decomposed Weight)
        """
        # --- 步骤 1: 计算 HiRA 风格的方向增量 (Delta Direction) ---
        # 利用哈达玛积 (Hadamard Product) 提升有效秩
        # 公式: Delta_V = W0 * (A @ B)
        # 解释: 这里利用 W0 的结构信息与低秩矩阵的乘积进行逐元素相乘，
        #      从而产生一个高秩的方向扰动，解决了传统 LoRA 秩不足的问题。
        AB = torch.mm(self.lora_A, self.lora_B) # 形状: (in_features, out_features)
        hira_direction_delta = self.W0 * AB # 哈达玛积 (Element-wise multiplication)
        
        # --- 步骤 2: 构建新的方向矩阵 (New Direction Matrix) ---
        # 公式: V_new = W0 + Delta_V
        # 解释: 将基础方向 (W0) 与计算出的高秩增量相加，
        #      得到待归一化的新方向。
        new_direction = self.W0 + hira_direction_delta
        
        # --- 步骤 3: 计算方向矩阵的范数 (Norm for Normalization) ---
        # 公式: ||V_new||_c (按列计算 L2 范数)
        # 解释: 为了将方向矩阵归一化为单位向量，需要计算其长度。
        #      dim=0 表示按列计算 (即对每个输出神经元计算)。
        #      keepdim=True 保证维度一致以便后续广播 (Broadcasting)。
        direction_norm = torch.norm(new_direction, p=2, dim=0, keepdim=True)
        
        # --- 步骤 4: 应用 DoRA 分解公式 (Apply DoRA Formula) ---
        # 公式: W' = m * (V_new / ||V_new||)
        # 解释: 
        #   1. (V_new / ||V_new||): 将新方向归一化为单位向量 (Unit Vector)。
        #   2. m * (...): 利用可学习的幅度参数 m 对单位方向向量进行缩放。
        # 这样就实现了幅度 (Magnitude m) 和方向 (Normalized V_new) 的完全解耦。
        W_prime = self.m * (new_direction / (direction_norm + 1e-8)) # 加 1e-8 防止除零
        
        return W_prime

# --- 注入 (Injection) 辅助函数 ---
def inject_hidora(model, target_module_name="mlp", r=32, alpha=8.0):
    """
    递归地将模型中的指定线性层替换为 Hi-DoRA 适配器。
    注意：这通常需要配合一个包装类 (Wrapper) 使用，或者修改模型的 forward 函数。
    这里提供一个简单的逻辑示意。
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            inject_hidora(module, target_module_name, r, alpha)
            
        if isinstance(module, nn.Linear) and target_module_name in name:
            # 获取原始权重
            W0 = module.weight.data
            
            # 创建 Hi-DoRA 层
            hidora_layer = HiDoRALayer(module.in_features, module.out_features, r, alpha)
            hidora_layer.set_pretrained_weight(W0)
            
            # 替换模块
            parent = model
            attr_name = name
            setattr(parent, attr_name, hidora_layer)
            
            print(f"Injected Hi-DoRA into {name}")
            
    return model