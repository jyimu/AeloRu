#!/usr/bin/env python3
# aeloru_logger.py
# AeLoRu 统一日志模块

import os
import sys
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import torch

# ==================== 全局配置 ====================
device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择：cuda 如果可用，否则 cpu

# 日志根目录
LOG_ROOT = r"AeLoRu\experiment\log"
os.makedirs(LOG_ROOT, exist_ok=True)

# ==================== 统一日志类 ====================
class AeLoRuLogger:
    """
    AeLoRu 实验统一日志系统
    所有实验代码共享同一日志目录，便于后续分析
    """
    
    _instance = None  # 单例模式
    
    def __new__(cls, experiment_name: str = "default"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, experiment_name: str = "default"):
        if self._initialized:
            return
            
        self.experiment_name = experiment_name
        self.start_time = time.time()
        self.step_count = 0
        
        # 创建时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{experiment_name}_{self.timestamp}"
        
        # 日志文件路径
        self.log_file = os.path.join(LOG_ROOT, f"{self.session_id}.log")
        self.json_file = os.path.join(LOG_ROOT, f"{self.session_id}.json")
        
        # 初始化日志文件
        self._init_log_file()
        
        # 数据存储
        self.metrics_history: List[Dict[str, Any]] = []
        self.config: Dict[str, Any] = {}
        
        self._initialized = True
        self.info(f"Logger initialized: {self.session_id}")
        self.info(f"Log file: {self.log_file}")
        self.info(f"JSON file: {self.json_file}")
    
    def _init_log_file(self):
        """初始化日志文件头部"""
        header = f"""{'='*60}
AeLoRu Experiment Log
{'='*60}
Session ID: {self.session_id}
Experiment: {self.experiment_name}
Start Time: {datetime.now().isoformat()}
Device: {device}
PyTorch Version: {torch.__version__}
{'='*60}

"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(header)
    

    
    def _format_message(self, message: str, level: str = "INFO", step: Optional[int] = None) -> str:
        """格式化日志消息"""
        elapsed = time.time() - self.start_time
        step_str = f"[Step {step}]" if step is not None else f"[Step {self.step_count}]"
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{timestamp}] [{elapsed:10.3f}s] {step_str} [{level:8s}] {message}"
    
    def log(self, message: str, level: str = "INFO", step: Optional[int] = None):
        """写入日志"""
        formatted = self._format_message(message, level, step)
        print(formatted)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(formatted + "\n")
            f.flush()  # 立即写入磁盘
    
    # 快捷方法
    def debug(self, message: str, step: Optional[int] = None):
        self.log(message, "DEBUG", step)
    
    def info(self, message: str, step: Optional[int] = None):
        self.log(message, "INFO", step)
    
    def warning(self, message: str, step: Optional[int] = None):
        self.log(message, "WARNING", step)
    
    def error(self, message: str, step: Optional[int] = None):
        self.log(message, "ERROR", step)
    
    def success(self, message: str, step: Optional[int] = None):
        self.log(message, "SUCCESS", step)
    
    # ==================== 指标记录 ====================
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """记录数值指标"""
        s = step if step is not None else self.step_count
        self.metrics_history.append({
            'step': s,
            'timestamp': time.time(),
            'name': name,
            'value': float(value)
        })
        self.info(f"Metric [{name}]: {value:.6f}", s)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """批量记录指标"""
        s = step if step is not None else self.step_count
        for name, value in metrics.items():
            self.log_metric(name, value, s)
    
    def set_config(self, config: Dict[str, Any]):
        """记录实验配置"""
        self.config.update(config)
        self.info(f"Config updated: {json.dumps(config, indent=2)}")
    
    def step(self):
        """递增步骤计数器"""
        self.step_count += 1
        return self.step_count
    
    # ==================== 数据保存 ====================
    def save_result(self, data: Dict[str, Any], filename: Optional[str] = None):
        """保存结构化结果到JSON"""
        filepath = filename or self.json_file
        
        result = {
            'session_id': self.session_id,
            'experiment_name': self.experiment_name,
            'config': self.config,
            'metrics_history': self.metrics_history,
            'final_result': data,
            'summary': {
                'total_steps': self.step_count,
                'total_time': time.time() - self.start_time,
                'device': device,
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self.success(f"Results saved to {filepath}")
        return filepath
    
    def finalize(self, final_data: Optional[Dict[str, Any]] = None):
        """结束实验，保存所有数据"""
        self.info("Finalizing experiment...")
        
        if final_data:
            self.save_result(final_data)
        else:
            self.save_result({})
        
        # 写入结束标记
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Experiment End: {datetime.now().isoformat()}\n")
            f.write(f"Total Time: {time.time() - self.start_time:.2f}s\n")
            f.write(f"Total Steps: {self.step_count}\n")
            f.write(f"{'='*60}\n")
        
        self._initialized = False
        AeLoRuLogger._instance = None


# ==================== 便捷函数 ====================
def get_logger(experiment_name: str = "default") -> AeLoRuLogger:
    """获取日志器实例（全局统一入口）"""
    return AeLoRuLogger(experiment_name)


# ==================== 测试 ====================
if __name__ == "__main__":
    # 测试日志功能
    logger = get_logger("test_logger")
    
    logger.set_config({
        "model": "Qwen2.5-0.5B",
        "lr": 1e-4,
        "batch_size": 4
    })
    
    for i in range(5):
        logger.step()
        logger.log_metric("loss", 1.0 / (i + 1))
        logger.log_metric("accuracy", 0.5 + i * 0.1)
        time.sleep(0.1)
    
    logger.finalize({
        "final_accuracy": 0.95,
        "status": "success"
    })
    
    print(f"\nTest complete! Check {LOG_ROOT} for logs.")