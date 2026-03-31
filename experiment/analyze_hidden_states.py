#!/usr/bin/env python3
# analyze_hidden_states.py
# 纯 PyTorch 实现，分析隐藏状态语义聚类

import os
import sys
import json
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

# ==================== 配置 ====================
device = "cuda"  # 自动选择：cuda 如果可用，否则 cpu

# 路径设置
LOG_DIR = r"AeLoRu\experiment\log"
os.makedirs(LOG_DIR, exist_ok=True)

# 模型路径
MODEL_PATH = "models/Qwen2.5-1.5B"

# 日志文件
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"hidden_states_analysis_{timestamp}.log")
RESULT_FILE = os.path.join(LOG_DIR, f"hidden_states_result_{timestamp}.json")

# ==================== 日志工具 ====================
class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.start_time = time.time()
        # 创建文件并写入头部
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== AeLoRu Hidden States Analysis ===\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Model: {MODEL_PATH}\n")
            f.write(f"Device: {device if device else 'auto'}\n")
            f.write(f"{'='*50}\n\n")
    
    def log(self, message, level="INFO"):
        elapsed = time.time() - self.start_time
        log_line = f"[{elapsed:8.2f}s] [{level}] {message}"
        print(log_line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_line + "\n")
    
    def save_result(self, data):
        with open(RESULT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.log(f"Results saved to {RESULT_FILE}")

logger = Logger(LOG_FILE)

# ==================== PyTorch PCA 实现 ====================
class TorchPCA:
    """纯 PyTorch 实现的 PCA"""
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance_ratio = None
    
    def fit_transform(self, X):
        """
        X: [n_samples, n_features] torch.Tensor
        """
        n_samples, n_features = X.shape
        
        # 中心化
        self.mean = X.mean(dim=0)
        X_centered = X - self.mean
        
        # SVD
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        
        # 取前n_components个主成分
        self.components = Vt[:self.n_components]  # [n_components, n_features]
        
        # 计算解释方差比例
        explained_variance = (S ** 2) / (n_samples - 1)
        total_variance = explained_variance.sum()
        self.explained_variance_ratio = (explained_variance[:self.n_components] / total_variance).cpu().numpy()
        
        # 投影
        X_transformed = X_centered @ self.components.T  # [n_samples, n_components]
        
        return X_transformed

# ==================== 主程序 ====================
def main():
    logger.log("Starting hidden states analysis...")
    
    # 设置设备
    if not device:
        auto_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        auto_device = device
    torch_device = torch.device(auto_device)
    logger.log(f"Using device: {auto_device}")
    
    # 加载模型
    logger.log(f"Loading model from {MODEL_PATH}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            device_map="cpu",  # 先加载到CPU避免OOM
            local_files_only=True,
            trust_remote_code=True,
            output_hidden_states=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True
        )
        logger.log("Model loaded successfully")
    except Exception as e:
        logger.log(f"Failed to load model: {str(e)}", "ERROR")
        return
    
    # 准备语义对比句子
    sentences = {
        'animal': ["The dog is running", "A cat sleeps", "Birds fly high", "Fish swim deep"],
        'number': ["One plus two", "Three minus one", "Count to five", "Seven is prime"],
        'emotion': ["I feel happy", "She is sad", "They are angry", "We are excited"],
        'code': ["def function():", "return value", "class Object:", "import module"],
    }
    
    logger.log(f"Prepared {sum(len(v) for v in sentences.values())} sentences in {len(sentences)} categories")
    
    # 收集隐藏状态
    all_hidden = []
    all_labels = []
    all_sentences = []
    
    model.eval()
    with torch.no_grad():
        for category, sents in sentences.items():
            for sent in sents:
                logger.log(f"Processing: [{category}] '{sent}'")
                
                inputs = tokenizer(sent, return_tensors="pt")
                
                # 前向传播
                outputs = model(**inputs, output_hidden_states=True)
                
                # 取最后一层最后位置的隐藏状态 [hidden_dim]
                # hidden_states是tuple，每层一个tensor
                last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
                final_state = last_hidden[0, -1, :]  # [hidden_dim]
                
                all_hidden.append(final_state)
                all_labels.append(category)
                all_sentences.append(sent)
    
    # 堆叠成tensor
    hidden_tensor = torch.stack(all_hidden)  # [n_samples, hidden_dim]
    logger.log(f"Collected hidden states: {hidden_tensor.shape}")
    
    # PCA降维（纯PyTorch）
    logger.log("Running PCA (PyTorch implementation)...")
    pca = TorchPCA(n_components=2)
    hidden_2d = pca.fit_transform(hidden_tensor)
    
    logger.log(f"PCA explained variance: PC1={pca.explained_variance_ratio[0]:.2%}, PC2={pca.explained_variance_ratio[1]:.2%}")
    
    # 转换为numpy用于matplotlib
    hidden_2d_np = hidden_2d.cpu().numpy()
    
    # 可视化
    logger.log("Generating visualization...")
    colors = {
        'animal': '#FF6B6B',   # 红
        'number': '#4ECDC4',   # 青
        'emotion': '#45B7D1',  # 蓝
        'code': '#96CEB4'      # 绿
    }
    
    plt.figure(figsize=(12, 9))
    
    for category in sentences.keys():
        # 找到该类别的索引
        indices = [i for i, l in enumerate(all_labels) if l == category]
        x = hidden_2d_np[indices, 0]
        y = hidden_2d_np[indices, 1]
        
        plt.scatter(x, y, c=colors[category], label=category, 
                   alpha=0.7, s=150, edgecolors='white', linewidth=1)
    
    plt.legend(fontsize=12, loc='best')
    plt.title("AeLoRu: Hidden State Semantic Clustering\n(PyTorch PCA on Qwen2.5-1.5B)", fontsize=14)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio[0]:.1%} variance)", fontsize=12)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio[1]:.1%} variance)", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    plot_file = os.path.join(LOG_DIR, f"semantic_clusters_{timestamp}.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    logger.log(f"Plot saved to {plot_file}")
    
    # 分析聚类质量（纯PyTorch计算类内/类间距离）
    logger.log("Analyzing clustering quality...")
    
    # 计算每个类别的中心
    category_centers = {}
    for cat in sentences.keys():
        mask = torch.tensor([l == cat for l in all_labels])
        cat_points = hidden_2d[mask]  # [n_points, 2]
        center = cat_points.mean(dim=0)
        category_centers[cat] = center
    
    # 计算类内距离（平均到中心的距离）
    intra_distances = {}
    for cat in sentences.keys():
        mask = torch.tensor([l == cat for l in all_labels])
        cat_points = hidden_2d[mask]
        center = category_centers[cat]
        distances = torch.norm(cat_points - center, dim=1)
        intra_distances[cat] = distances.mean().item()
    
    # 计算类间距离（中心之间的距离）
    inter_distances = []
    cats = list(sentences.keys())
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            dist = torch.norm(category_centers[cats[i]] - category_centers[cats[j]]).item()
            inter_distances.append((cats[i], cats[j], dist))
    
    # 保存结果
    result_data = {
        'timestamp': datetime.now().isoformat(),
        'model': MODEL_PATH,
        'device': auto_device,
        'n_samples': len(all_hidden),
        'hidden_dim': hidden_tensor.shape[1],
        'pca_explained_variance': {
            'PC1': float(pca.explained_variance_ratio[0]),
            'PC2': float(pca.explained_variance_ratio[1])
        },
        'clustering_analysis': {
            'intra_class_distances': {k: float(v) for k, v in intra_distances.items()},
            'inter_class_distances': [
                {'cat1': c1, 'cat2': c2, 'distance': float(d)} 
                for c1, c2, d in inter_distances
            ],
            'category_centers': {
                k: v.cpu().numpy().tolist() 
                for k, v in category_centers.items()
            }
        },
        'samples': [
            {'sentence': s, 'label': l, 'pca_coords': coords.tolist()}
            for s, l, coords in zip(all_sentences, all_labels, hidden_2d)
        ]
    }
    
    logger.save_result(result_data)
    
    # 打印关键结论
    logger.log("="*50)
    logger.log("ANALYSIS SUMMARY")
    logger.log("="*50)
    
    avg_intra = np.mean(list(intra_distances.values()))
    avg_inter = np.mean([d for _, _, d in inter_distances])
    ratio = avg_inter / avg_intra if avg_intra > 0 else float('inf')
    
    logger.log(f"Average intra-class distance: {avg_intra:.3f}")
    logger.log(f"Average inter-class distance: {avg_inter:.3f}")
    logger.log(f"Inter/Intra ratio: {ratio:.3f}")
    
    if ratio > 2.0:
        logger.log("✓ Strong clustering: hidden states capture semantic structure!", "SUCCESS")
    elif ratio > 1.5:
        logger.log("~ Moderate clustering: some semantic structure present", "INFO")
    else:
        logger.log("✗ Weak clustering: may need larger model or different analysis", "WARNING")
    
    logger.log("Analysis complete!")

if __name__ == "__main__":
    main()