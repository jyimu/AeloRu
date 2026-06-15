#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aeloru 单步训练性能分析器 (PyTorch Profiler Chrome Trace)
============================================================

功能：
1. 使用 PyTorch Profiler 记录单步训练（forward + backward + post_step_update）
2. 自动解析 Chrome Trace JSON，计算各模块用时占比
3. 生成可视化报告（表格 + 饼图）
4. 支持 CPU-only 和 CUDA 两种模式

核心流程:
1. warmup: 预热指定步数，稳定 GPU 状态
2. profile: 使用 torch.profiler 采集 Chrome Trace
3. parse: 解析 traceEvents，按模块分类聚合耗时
4. report: 生成文本报告（含瓶颈分析与优化建议）
5. chart: 生成饼图与水平条形图

输出：
- {output_dir}/profiler_step_0.json  (Chrome Trace)
- {output_dir}/profiler_report.txt   (文本报告)
- {output_dir}/profiler_pie.png      (可视化图表)

Author: JYIMU (Qelys Project)
"""

import os
import sys
import json
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

# =============================================================================
# 可选依赖：matplotlib（图表生成）
# =============================================================================
try:
    import matplotlib
    import matplotlib.pyplot as plt
    _HAS_MPL = True
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except ImportError:
    _HAS_MPL = False

# =============================================================================
# 导入 Aeloru 核心模块
# =============================================================================

# 【兼容】支持从同级目录或上传路径导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from aeloru_layer import AeloruLayer, AeloruConfig, train_aeloru_step
except ImportError:
    sys.path.insert(0, "/mnt/agents/upload")
    from aeloru_layer import AeloruLayer, AeloruConfig, train_aeloru_step


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class ProfilerConfig:
    """
    Profiler 分析配置类

    所有维度、功能开关与输出路径均可独立调整，便于消融实验。

    Args:
        # --- 维度设置 ---
        in_features: 输入特征维度（模拟 Qwen2.5-0.5B 的某一层）
        out_features: 输出特征维度（词表大小）
        r: LoRA 秩
        lora_alpha: LoRA 缩放因子
        batch_size: 训练批次大小
        seq_len: 序列长度

        # --- 设备 ---
        device: 计算设备，"cuda" 或 "cpu"

        # --- Profiler 设置 ---
        warmup_steps: 预热步数（稳定 GPU 状态）
        profile_steps: 正式 profiling 步数

        # --- Aeloru 功能开关 ---
        use_hidora: 是否启用 Hi-DoRA 幅度调制
        use_relora: 是否启用 ReLoRA 合并重置
        use_hebbian: 是否启用 Hebbian 在线学习
        use_fisher: 是否启用 Fisher 认知掩码
        use_hongwen: 是否启用 Hong Wen 状态机
        use_orthogonal_penalty: 是否启用正交惩罚损失
        use_energy_budget: 是否启用能量预算硬约束
        use_lateral_connection: 是否启用 PEM 自适应侧向连接
        use_homeostatic_plasticity: 是否启用 PEM 稳态可塑性
        use_volatility_coupling: 是否启用 HGF 波动率耦合
        use_dlam_sleep: 是否启用 DLAM 谱滤波睡眠
        use_hgf_fisher: 是否启用 HGF 闭式 Fisher
        use_predictive_coding: 是否启用预测编码神经动态
        use_source_domain_constraint: 是否启用源信号域约束
        use_online_covariance: 是否启用在线协方差估计
        use_whitening: 是否启用输入动态白化
        use_hgf_closed_form: 是否启用 HGF 闭式一步更新

        # --- 输出路径 ---
        output_dir: 输出目录
        trace_filename: Chrome Trace 文件名
        report_filename: 文本报告文件名
        chart_filename: 可视化图表文件名
    """

    # --- 维度设置 ---
    in_features: int = 896
    out_features: int = 151000
    r: int = 8
    lora_alpha: float = 4.0
    batch_size: int = 2
    seq_len: int = 1

    # --- 设备 ---
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # --- Profiler 设置 ---
    warmup_steps: int = 10
    profile_steps: int = 10

    # --- Aeloru 功能开关 ---
    use_hidora: bool = True
    use_relora: bool = True
    use_hebbian: bool = True
    use_fisher: bool = False
    use_hongwen: bool = True
    use_orthogonal_penalty: bool = True
    use_energy_budget: bool = True
    use_lateral_connection: bool = True
    use_homeostatic_plasticity: bool = True
    use_volatility_coupling: bool = True
    use_dlam_sleep: bool = True
    use_hgf_fisher: bool = True
    use_predictive_coding: bool = True
    use_source_domain_constraint: bool = True
    use_online_covariance: bool = True
    use_whitening: bool = True
    use_hgf_closed_form: bool = True

    # --- 输出路径 ---
    # 默认使用当前工作目录下的子目录，确保跨平台（Windows/Linux）可用
    output_dir: str = field(default_factory=lambda: os.path.join(os.getcwd(), "profiler_output"))
    trace_filename: str = "profiler_step_0.json"
    report_filename: str = "profiler_report.txt"
    chart_filename: str = "profiler_pie.png"


# =============================================================================
# 核心分析类
# =============================================================================

class AeloruProfiler:
    """
    Aeloru 单步训练性能分析器

    分析维度：
    - Forward: 前向传播各子模块（线性变换、LoRA、侧向抑制、稳态增益等）
    - Backward: 反向传播与梯度计算
    - Optimizer: 优化器参数更新
    - Post-Step: Hebbian + Hong Wen 状态机 + ReLoRA 合并 + DLAM 睡眠
    - Overhead: Profiler 与数据搬运等额外开销

    Args:
        cfg: ProfilerConfig 配置对象

    Attributes:
        cfg: 配置对象
        device: torch.device 实例
        layer: AeloruLayer 实例
        optimizer: torch.optim.Optimizer 实例
    """

    def __init__(self, cfg: ProfilerConfig) -> None:
        """
        初始化分析器。

        流程：
        1. 保存配置与设备
        2. 调用 _build_model 构建 Aeloru 层与优化器
        """
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.layer: Optional[AeloruLayer] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        self._build_model()

    # -------------------------------------------------------------------------
    # 模型构建
    # -------------------------------------------------------------------------

    def _build_model(self) -> None:
        """
        构建 Aeloru 层与优化器。

        流程：
        1. 将 ProfilerConfig 映射为 AeloruConfig
        2. 实例化 AeloruLayer 并迁移到目标设备
        3. 初始化预训练权重（W0 + bias）
        4. 构建 AdamW 优化器
        """
        aeloru_cfg = AeloruConfig(
            in_features=self.cfg.in_features,
            out_features=self.cfg.out_features,
            r=self.cfg.r,
            lora_alpha=self.cfg.lora_alpha,
            # 功能开关映射
            use_hidora=self.cfg.use_hidora,
            use_relora=self.cfg.use_relora,
            use_hebbian=self.cfg.use_hebbian,
            use_fisher=self.cfg.use_fisher,
            use_hongwen=self.cfg.use_hongwen,
            use_orthogonal_penalty=self.cfg.use_orthogonal_penalty,
            use_energy_budget=self.cfg.use_energy_budget,
            use_lateral_connection=self.cfg.use_lateral_connection,
            use_homeostatic_plasticity=self.cfg.use_homeostatic_plasticity,
            use_volatility_coupling=self.cfg.use_volatility_coupling,
            use_dlam_sleep=self.cfg.use_dlam_sleep,
            use_hgf_fisher=self.cfg.use_hgf_fisher,
            use_predictive_coding=self.cfg.use_predictive_coding,
            use_source_domain_constraint=self.cfg.use_source_domain_constraint,
            use_online_covariance=self.cfg.use_online_covariance,
            use_whitening=self.cfg.use_whitening,
            use_hgf_closed_form=self.cfg.use_hgf_closed_form,
            # 性能优化（关闭 verbose 与诊断，避免 .item() 同步）
            verbose=False,
            enable_cognitive_report=False,
            diagnostic_interval=100000,
            fisher_async=False,
            async_merge=False,
            USE_AMP=True,
            device=str(self.device),
        )

        self.layer = AeloruLayer(
            self.cfg.in_features,
            self.cfg.out_features,
            aeloru_cfg
        ).to(self.device)

        # 初始化预训练权重
        W0 = torch.randn(self.cfg.out_features, self.cfg.in_features, device=self.device) * 0.02
        bias = torch.zeros(self.cfg.out_features, device=self.device)
        self.layer.set_pretrained_weight(W0, bias)
        self.layer.train()

        # 优化器（fused=True 加速 CUDA）
        self.optimizer = torch.optim.AdamW(
            self.layer.get_trainable_params(),
            lr=1e-3,
            fused=True
        )

        print("[Profiler] 模型构建完成")
        print(f"  设备: {self.device}")
        print(f"  维度: {self.cfg.in_features} -> {self.cfg.out_features}, r={self.cfg.r}")
        print(f"  批次: {self.cfg.batch_size} x {self.cfg.seq_len}")
        print(f"  功能: Hi-DoRA={self.cfg.use_hidora}, ReLoRA={self.cfg.use_relora}, "
              f"Hebbian={self.cfg.use_hebbian}, Fisher={self.cfg.use_fisher}")

    # -------------------------------------------------------------------------
    # 路径与权限检查
    # -------------------------------------------------------------------------

    def _ensure_output_dir(self) -> None:
        """
        确保输出目录存在且具有写入权限。

        流程：
        1. 若目录不存在则递归创建
        2. 尝试写入临时文件验证权限
        3. 将路径转换为绝对路径，避免相对路径歧义
        """
        abs_dir = os.path.abspath(self.cfg.output_dir)
        self.cfg.output_dir = abs_dir

        try:
            os.makedirs(abs_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"[Profiler] 无法创建输出目录: {abs_dir}\n"
                f"  原因: {e}\n"
                f"  请检查磁盘空间、路径长度（Windows 限制 260 字符）或权限设置。"
            ) from e

        test_file = os.path.join(abs_dir, ".write_test")
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("test")
            os.remove(test_file)
        except OSError as e:
            raise RuntimeError(
                f"[Profiler] 输出目录无写入权限: {abs_dir}\n"
                f"  原因: {e}\n"
                f"  请以管理员身份运行或更换 output_dir 到用户目录下。"
            ) from e

        print(f"[Profiler] 输出目录已就绪: {abs_dir}")

    @staticmethod
    def _verify_file_exists(file_path: str, operation: str) -> None:
        """
        验证文件是否存在，若不存在则抛出清晰的错误信息。

        Args:
            file_path: 待验证的文件路径
            operation: 当前操作描述（用于错误信息）
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"[Profiler] {operation} 失败: 未找到文件 '{file_path}'\n"
                f"  可能原因:\n"
                f"    1. 保存步骤被跳过或发生异常\n"
                f"    2. output_dir 被修改导致保存与读取路径不一致\n"
                f"    3. 文件被外部程序删除\n"
                f"  请检查前序步骤是否成功执行。"
            )

    # -------------------------------------------------------------------------
    # 数据生成
    # -------------------------------------------------------------------------

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成模拟训练数据。

        Returns:
            x: 输入张量 (batch_size, seq_len, in_features)
            y_target: 目标类别索引 (batch_size, seq_len)，用于 CrossEntropyLoss
        """
        x = torch.randn(
            self.cfg.batch_size, self.cfg.seq_len, self.cfg.in_features,
            device=self.device, dtype=torch.float32
        )
        y_target = torch.randint(
            0, self.cfg.out_features,
            (self.cfg.batch_size, self.cfg.seq_len),
            device=self.device
        )
        return x, y_target

    # -------------------------------------------------------------------------
    # Profiler 执行
    # -------------------------------------------------------------------------

    def run_profiler(self) -> str:
        """
        运行 PyTorch Profiler 并保存 Chrome Trace。

        流程：
        1. 预热：运行若干空 step 稳定 GPU 状态
        2. 正式 profiling：使用 record_function 标记各阶段
        3. 导出 Chrome Trace JSON
        4. 打印 profiler 摘要（按 CPU / CUDA 时间排序）

        Returns:
            trace_path: Chrome Trace 文件绝对路径
        """
        # 确保输出目录存在且可写
        self._ensure_output_dir()
        trace_path = os.path.join(self.cfg.output_dir, self.cfg.trace_filename)

        # --- 预热 ---
        print(f"\n[Profiler] 预热 {self.cfg.warmup_steps} 步...")
        for _ in range(self.cfg.warmup_steps):
            x = torch.randn(
                self.cfg.batch_size, self.cfg.in_features,
                device=self.device, dtype=AeloruConfig.AMP_DTYPE
            )
            y = torch.randn(
                self.cfg.batch_size, self.cfg.out_features,
                device=self.device, dtype=AeloruConfig.AMP_DTYPE
            )
            train_aeloru_step(self.layer, x, y, self.optimizer)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # --- 正式 profiling ---
        print(f"[Profiler] 开始 profiling {self.cfg.profile_steps} 步...")

        activities = [ProfilerActivity.CPU]
        if self.device.type == 'cuda':
            activities.append(ProfilerActivity.CUDA)

        # 损失函数只需创建一次
        loss_fn = nn.CrossEntropyLoss()

        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            for step in range(self.cfg.profile_steps):
                with record_function("full_step"):
                    # 生成输入与目标（目标为整数类别索引）
                    x = torch.randn(
                        self.cfg.batch_size, self.cfg.in_features,
                        device=self.device, dtype=AeloruConfig.AMP_DTYPE
                    )
                    y_target = torch.randint(
                        0, self.cfg.out_features,
                        (self.cfg.batch_size,),
                        device=self.device
                    )

                    with record_function("forward"):
                        y_pred = self.layer(x)

                    with record_function("loss_compute"):
                        loss = loss_fn(
                            y_pred.view(-1, self.cfg.out_features),
                            y_target.view(-1)
                        )
                        loss_ortho = self.layer.get_ortho_penalty()
                        loss_total = loss + loss_ortho

                    with record_function("backward"):
                        self.optimizer.zero_grad()
                        loss_total.backward()

                    with record_function("optimizer_step"):
                        self.optimizer.step()

                    with record_function("post_step_update"):
                        self.layer.post_step_update(
                            x, y_pred.detach(),
                            is_correct=True,
                            y_target=y_target
                        )

        # --- 导出与摘要 ---
        prof.export_chrome_trace(trace_path)
        self._verify_file_exists(trace_path, "Chrome Trace 导出")
        print(f"[Profiler] Chrome Trace 已保存: {trace_path}")

        print("\n" + "=" * 70)
        print("PyTorch Profiler 摘要 (按 CPU 时间排序)")
        print("=" * 70)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

        if self.device.type == 'cuda':
            print("\n" + "=" * 70)
            print("PyTorch Profiler 摘要 (按 CUDA 时间排序)")
            print("=" * 70)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        return trace_path

    # -------------------------------------------------------------------------
    # Trace 解析
    # -------------------------------------------------------------------------

    def parse_trace(self, trace_path: str) -> Dict[str, float]:
        """
        解析 Chrome Trace JSON，按模块分类聚合耗时。

        Args:
            trace_path: Chrome Trace JSON 文件路径

        Returns:
            timing_dict: {模块名: 耗时(ms), ...}
        """
        self._verify_file_exists(trace_path, "Trace 解析")
        with open(trace_path, 'r', encoding='utf-8') as f:
            trace_data = json.load(f)

        events = trace_data.get('traceEvents', [])
        durations = defaultdict(float)

        for event in events:
            if event.get('ph') == 'X':  # Complete event
                name = event.get('name', 'unknown')
                dur_us = event.get('dur', 0)
                category = self._categorize_event(name)
                durations[category] += dur_us / 1000.0  # us -> ms

        return dict(durations)

    def _categorize_event(self, name: str) -> str:
        """
        将 profiler 事件名分类到标准模块。

        分类规则：
        - forward: 前向传播相关 kernel
        - backward: 反向传播与 autograd
        - loss_compute: 损失函数与惩罚项
        - optimizer_step: 优化器更新
        - post_step_update: Hebbian / 状态机 / 合并 / 睡眠等
        - data_movement: CPU-GPU 数据搬运
        - profiler_overhead: Profiler 自身开销
        - other: 未分类事件

        Args:
            name: profiler 事件名

        Returns:
            category: 分类后的模块名
        """
        name_lower = name.lower()

        # Post-step 优先级最高（避免被 forward/optimizer 误分类）
        if any(k in name_lower for k in [
            'post_step', 'hebbian', 'fisher', 'merge',
            'sleep', 'lateral', 'volatility', 'homeostatic',
            'hongwen', 'detect', 'transition'
        ]):
            return 'post_step_update'

        # Forward
        if any(k in name_lower for k in ['forward', 'linear', 'matmul', 'mm(', 'einsum']):
            return 'forward'

        # Backward
        if any(k in name_lower for k in ['backward', 'autograd', 'grad']):
            return 'backward'

        # Loss
        if any(k in name_lower for k in ['loss', 'crossentropy', 'mse', 'penalty']):
            return 'loss_compute'

        # Optimizer
        if any(k in name_lower for k in ['optimizer', 'adam', 'sgd', 'step']):
            return 'optimizer_step'

        # Data movement
        if any(k in name_lower for k in ['to(', 'copy_', 'cuda', 'cpu', 'transfer']):
            return 'data_movement'

        # Python overhead
        if any(k in name_lower for k in ['full_step', 'profiler', 'record_function']):
            return 'profiler_overhead'

        return 'other'

    # -------------------------------------------------------------------------
    # 数据预处理（报告与图表共享）
    # -------------------------------------------------------------------------

    @staticmethod
    def _prepare_display_data(
        timing_dict: Dict[str, float],
        min_pct: float = 0.01
    ) -> Tuple[Dict[str, float], float]:
        """
        预处理耗时数据，用于报告与图表的统一展示。

        流程：
        1. 过滤掉占比 < min_pct 的项
        2. 将过滤掉的部分合并为 "other"
        3. 保证报告与饼图/条形图数据完全一致

        Args:
            timing_dict: 原始耗时字典
            min_pct: 最小占比阈值（默认 1%），低于此值的项合并到 other

        Returns:
            filtered: 处理后的耗时字典（含可能的 "other" 项）
            total: 总耗时
        """
        total = sum(timing_dict.values())
        if total <= 0:
            return dict(timing_dict), total

        filtered = {k: v for k, v in timing_dict.items() if v / total > min_pct}
        other_time = total - sum(filtered.values())
        if other_time > 0:
            filtered['other'] = other_time
        return filtered, total

    # -------------------------------------------------------------------------
    # 报告生成
    # -------------------------------------------------------------------------

    def generate_report(self, timing_dict: Dict[str, float]) -> str:
        """
        生成文本性能分析报告。

        报告内容：
        1. 运行环境与配置摘要
        2. 各模块耗时分析（含柱状图，与饼图数据一致）
        3. 关键洞察（瓶颈识别、F/B 比例、Post-step 开销）
        4. 针对性优化建议

        Args:
            timing_dict: parse_trace 返回的耗时字典

        Returns:
            report_path: 文本报告文件绝对路径
        """
        report_path = os.path.join(self.cfg.output_dir, self.cfg.report_filename)

        # 使用与图表一致的过滤逻辑
        display_data, total_time = self._prepare_display_data(timing_dict, min_pct=0.01)
        sorted_items = sorted(display_data.items(), key=lambda x: x[1], reverse=True)

        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("Aeloru 单步训练性能分析报告")
        lines.append("=" * 70)
        lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"设备: {self.cfg.device}")
        lines.append(f"维度: {self.cfg.in_features} -> {self.cfg.out_features}")
        lines.append(f"批次: {self.cfg.batch_size} x {self.cfg.seq_len}")
        lines.append("")

        # --- 功能配置 ---
        lines.append("功能配置:")
        features = [
            ('Hi-DoRA', self.cfg.use_hidora),
            ('ReLoRA', self.cfg.use_relora),
            ('Hebbian', self.cfg.use_hebbian),
            ('Fisher', self.cfg.use_fisher),
            ('Hong Wen', self.cfg.use_hongwen),
            ('正交惩罚', self.cfg.use_orthogonal_penalty),
            ('能量预算', self.cfg.use_energy_budget),
            ('侧向连接', self.cfg.use_lateral_connection),
            ('稳态可塑性', self.cfg.use_homeostatic_plasticity),
            ('波动率耦合', self.cfg.use_volatility_coupling),
            ('DLAM睡眠', self.cfg.use_dlam_sleep),
            ('HGF Fisher', self.cfg.use_hgf_fisher),
            ('预测编码', self.cfg.use_predictive_coding),
            ('域约束', self.cfg.use_source_domain_constraint),
            ('在线协方差', self.cfg.use_online_covariance),
            ('白化', self.cfg.use_whitening),
        ]
        for name, enabled in features:
            status = 'ON' if enabled else 'OFF'
            lines.append(f"  [{status}] {name}")
        lines.append("")

        # --- 耗时分析 ---
        lines.append("=" * 70)
        lines.append("各模块耗时分析（占比 < 1% 的项已合并到 'other'）")
        lines.append("=" * 70)
        lines.append(f"{'模块':<25s} {'耗时(ms)':>12s} {'占比(%)':>10s} {'柱状图':>20s}")
        lines.append("-" * 70)

        for module, duration in sorted_items:
            pct = duration / total_time * 100 if total_time > 0 else 0
            bar_len = int(pct / 2)
            bar = '█' * bar_len
            lines.append(f"{module:<25s} {duration:>12.3f} {pct:>10.2f} {bar}")

        lines.append("-" * 70)
        lines.append(f"{'总计':<25s} {total_time:>12.3f} {100.0:>10.2f}")
        lines.append("")

        # --- 关键洞察 ---
        lines.append("=" * 70)
        lines.append("关键洞察")
        lines.append("=" * 70)

        max_module = ""
        max_time = 0.0
        if sorted_items:
            max_module, max_time = sorted_items[0]
            max_pct = max_time / total_time * 100
            lines.append(f"• 最大瓶颈: {max_module} ({max_pct:.1f}%)")

        # Forward vs Backward 比例
        fwd_time = timing_dict.get('forward', 0)
        bwd_time = timing_dict.get('backward', 0)
        if bwd_time > 0:
            ratio = fwd_time / bwd_time
            lines.append(f"• Forward/Backward 比例: {ratio:.2f}x")

        # Post-step 开销
        post_time = timing_dict.get('post_step_update', 0)
        post_pct = post_time / total_time * 100 if total_time > 0 else 0
        if post_time > 0:
            lines.append(f"• Post-step 开销: {post_pct:.1f}% (Hebbian + 状态机 + 合并)")

        # Optimizer 开销
        opt_time = timing_dict.get('optimizer_step', 0)
        opt_pct = opt_time / total_time * 100 if total_time > 0 else 0
        if opt_time > 0:
            lines.append(f"• Optimizer 开销: {opt_pct:.1f}%")

        # 数据搬运
        data_time = timing_dict.get('data_movement', 0)
        data_pct = data_time / total_time * 100 if total_time > 0 else 0
        if data_time > 0:
            lines.append(f"• 数据搬运开销: {data_pct:.1f}% (CPU-GPU 同步)")

        lines.append("")

        # --- 优化建议 ---
        lines.append("=" * 70)
        lines.append("优化建议")
        lines.append("=" * 70)

        if max_module == 'backward':
            lines.append("• Backward 是瓶颈，考虑:")
            lines.append("  - 启用 gradient checkpointing")
            lines.append("  - 减少 batch size 或序列长度")
            lines.append("  - 使用更高效的 kernel (如 FlashAttention)")

        if max_module == 'forward':
            lines.append("• Forward 是瓶颈，考虑:")
            lines.append("  - 启用 batch_forward_lora 批量优化")
            lines.append("  - 减少预测编码迭代次数")
            lines.append("  - 简化侧向连接计算")

        if post_pct > 20:
            lines.append("• Post-step 开销过高，考虑:")
            lines.append("  - 增大 hebbian_accum_steps 减少更新频率")
            lines.append("  - 关闭非必要的状态机检测")
            lines.append("  - 降低 Fisher 计算频率")

        if data_pct > 10:
            lines.append("• 数据搬运开销高，考虑:")
            lines.append("  - 使用 pin_memory + non_blocking=True")
            lines.append("  - 减少 CPU-GPU 同步点")

        lines.append("")
        lines.append("=" * 70)

        report_text = "\n".join(lines)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        self._verify_file_exists(report_path, "报告保存")
        print(f"[Profiler] 报告已保存: {report_path}")
        return report_path

    # -------------------------------------------------------------------------
    # 图表生成
    # -------------------------------------------------------------------------

    def generate_chart(self, timing_dict: Dict[str, float]) -> Optional[str]:
        """
        生成可视化图表（饼图 + 水平条形图）。

        Args:
            timing_dict: parse_trace 返回的耗时字典

        Returns:
            chart_path: 图表文件绝对路径；若 matplotlib 未安装则返回 None
        """
        if not _HAS_MPL:
            warnings.warn("[Profiler] matplotlib 未安装，跳过图表生成")
            return None

        matplotlib.use('Agg')
        chart_path = os.path.join(self.cfg.output_dir, self.cfg.chart_filename)

        # 使用与报告一致的过滤逻辑
        filtered, total = self._prepare_display_data(timing_dict, min_pct=0.01)

        labels = list(filtered.keys())
        sizes = list(filtered.values())
        colors = plt.cm.Set3(range(len(labels)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 饼图
        wedges, texts, autotexts = ax1.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90,
            textprops={'fontsize': 10}
        )
        ax1.set_title('Aeloru 单步训练耗时占比', fontsize=14, fontweight='bold')

        # 水平条形图（更精确）
        sorted_items = sorted(filtered.items(), key=lambda x: x[1])
        y_pos = range(len(sorted_items))
        vals = [v for _, v in sorted_items]
        labs = [k for k, _ in sorted_items]

        bars = ax2.barh(y_pos, vals, color=plt.cm.Set3(range(len(vals))))
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labs, fontsize=9)
        ax2.set_xlabel('耗时 (ms)', fontsize=11)
        ax2.set_title('各模块耗时对比（占比 < 1% 已合并到 other）',
                      fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        for i, (_, val) in enumerate(zip(bars, vals)):
            ax2.text(val + max(vals) * 0.01, i, f'{val:.2f}ms',
                     va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        self._verify_file_exists(chart_path, "图表保存")
        print(f"[Profiler] 图表已保存: {chart_path}")
        return chart_path

    # -------------------------------------------------------------------------
    # 完整分析流程
    # -------------------------------------------------------------------------

    def run_full_analysis(self) -> Dict[str, Any]:
        """
        运行完整分析流程：profile → parse → report → chart。

        Returns:
            results: {
                'trace': Chrome Trace 路径,
                'report': 文本报告路径,
                'chart': 图表路径（可能为 None）,
                'timing': 耗时字典
            }
        """
        print("\n" + "=" * 70)
        print("Aeloru 单步训练性能分析")
        print("=" * 70)

        # 1. 运行 profiler
        trace_path = self.run_profiler()

        # 2. 解析 trace
        print("\n[Profiler] 解析 Chrome Trace...")
        timing_dict = self.parse_trace(trace_path)

        print("\n各模块耗时 (ms):")
        for module, duration in sorted(timing_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"  {module:<25s}: {duration:>10.3f} ms")

        # 3. 生成报告
        report_path = self.generate_report(timing_dict)

        # 4. 生成图表
        chart_path = self.generate_chart(timing_dict)

        print("\n" + "=" * 70)
        print("分析完成!")
        print("=" * 70)
        print(f"Chrome Trace: {trace_path}")
        print(f"  → 用 Chrome 浏览器打开 chrome://tracing，加载此文件")
        print(f"文本报告: {report_path}")
        print(f"可视化图表: {chart_path}")

        return {
            'trace': trace_path,
            'report': report_path,
            'chart': chart_path,
            'timing': timing_dict
        }


# =============================================================================
# 高级分析：Chrome Trace 原始事件深度解析
# =============================================================================

def analyze_raw_trace(trace_path: str, top_k: int = 30) -> Dict[str, Any]:
    """
    深度分析 Chrome Trace 原始事件。

    输出：
    - 总事件数与 Complete 事件数
    - Top-K 最耗时操作（按单次 duration）
    - 按操作名聚合的 Top-K 耗时（按累计 duration）

    Args:
        trace_path: Chrome Trace JSON 路径
        top_k: 显示前 K 个最耗时操作

    Returns:
        分析结果字典，包含 total_events、complete_events、top_events、name_aggregation
    """
    # 读取前验证文件存在性
    if not os.path.isfile(trace_path):
        raise FileNotFoundError(
            f"[Profiler] 原始 Trace 深度分析失败: 未找到文件 '{trace_path}'\n"
            f"  请确认前序 profiling 步骤已成功保存 Chrome Trace。"
        )
    with open(trace_path, 'r', encoding='utf-8') as f:
        trace_data = json.load(f)

    events = trace_data.get('traceEvents', [])

    # 收集所有 Complete 事件
    complete_events = []
    for event in events:
        if event.get('ph') == 'X':
            complete_events.append({
                'name': event.get('name', 'unknown'),
                'dur_us': event.get('dur', 0),
                'tid': event.get('tid', 0),
                'pid': event.get('pid', 0),
                'args': event.get('args', {})
            })

    # 按单次耗时排序
    complete_events.sort(key=lambda x: x['dur_us'], reverse=True)

    print(f"\n[RawTrace] 总事件数: {len(events)}")
    print(f"[RawTrace] Complete 事件数: {len(complete_events)}")
    print(f"\n[RawTrace] Top {top_k} 最耗时操作:")
    print("-" * 80)
    print(f"{'排名':<6s} {'操作名':<40s} {'耗时(ms)':>12s} {'线程ID':>10s}")
    print("-" * 80)

    for i, event in enumerate(complete_events[:top_k], 1):
        dur_ms = event['dur_us'] / 1000.0
        print(f"{i:<6d} {event['name']:<40s} {dur_ms:>12.3f} {event['tid'] if event['tid'] is str else str(event['tid']):>10s}")

    # 按操作名聚合
    name_agg = defaultdict(float)
    for event in complete_events:
        name_agg[event['name']] += event['dur_us'] / 1000.0

    print(f"\n[RawTrace] 按操作名聚合 (Top {top_k}):")
    print("-" * 80)
    sorted_agg = sorted(name_agg.items(), key=lambda x: x[1], reverse=True)
    for name, dur in sorted_agg[:top_k]:
        print(f"  {name:<50s}: {dur:>10.3f} ms")

    return {
        'total_events': len(events),
        'complete_events': len(complete_events),
        'top_events': complete_events[:top_k],
        'name_aggregation': dict(sorted_agg[:top_k])
    }


# =============================================================================
# 主入口
# =============================================================================

def main() -> None:
    """
    主函数。

    流程：
    1. 加载 ProfilerConfig
    2. 设备回退（CUDA 不可用时切 CPU）
    3. 创建 AeloruProfiler 并运行完整分析
    4. 输出深度原始 Trace 分析
    """
    cfg = ProfilerConfig()

    # 设备回退
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA 不可用，切换到 CPU 模式")
        cfg.device = "cpu"

    # 创建分析器并运行
    profiler = AeloruProfiler(cfg)
    results = profiler.run_full_analysis()

    # 深度分析原始 trace
    print("\n" + "=" * 70)
    print("深度原始 Trace 分析")
    print("=" * 70)
    analyze_raw_trace(results['trace'], top_k=30)

    print("\n[Profiler] 全部完成!")


if __name__ == "__main__":
    main()
