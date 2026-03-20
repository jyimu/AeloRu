#!/usr/bin/env python3
# analyze_hidden_states_v2.py
# 使用 AeLoRu 统一日志的隐藏状态分析

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

# 添加父目录到路径（假设 logger 在同级目录）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aeloru_logger import get_logger, device, LOG_ROOT



# ==================== 配置 ====================
MODEL_PATH = "./models/Qwen2.5-1.5B-Instruct"
EXPERIMENT_NAME = "hidden_states_analysis"

# ==================== PyTorch PCA ====================
class TorchPCA:
    """纯 PyTorch PCA"""
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance_ratio = None
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        n_samples, n_features = X.shape
        self.mean = X.mean(dim=0)
        X_centered = X - self.mean
        
        # SVD
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        self.components = Vt[:self.n_components]
        
        # 解释方差
        explained_variance = (S ** 2) / (n_samples - 1)
        total_variance = explained_variance.sum()
        self.explained_variance_ratio = (
            explained_variance[:self.n_components] / total_variance
        ).cpu().numpy()
        
        return X_centered @ self.components.T

# ==================== 主程序 ====================
def main():
    # 获取统一日志器
    logger = get_logger(EXPERIMENT_NAME)
    
    # 设置设备
    auto_device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(auto_device)
    logger.set_config({
        "model_path": MODEL_PATH,
        "device": auto_device,
        "model_size": "1.5B",
        "analysis_type": "hidden_states_semantic"
    })
    
    # 加载模型
    logger.info("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            device_map="cpu",
            local_files_only=True,
            trust_remote_code=True,
            output_hidden_states=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True
        )
        logger.success("Model loaded")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.finalize({"status": "failed", "error": str(e)})
        return
    
    # 准备数据
    sentences = {
        'animal': ["The dog is running", "A cat sleeps", "Birds fly high"],
        'number': ["One plus two", "Three minus one", "Count to five"],
        'emotion': ["I feel happy", "She is sad", "They are angry"],
        'code': ["def function():", "return value", "class Object:"],
    }
    
    logger.info(f"Prepared {sum(len(v) for v in sentences.values())} sentences")
    
    # 收集隐藏状态
    all_hidden = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for category, sents in sentences.items():
            for sent in sents:
                logger.step()
                logger.debug(f"Processing: [{category}] '{sent}'")
                
                inputs = tokenizer(sent, return_tensors="pt")
                outputs = model(**inputs, output_hidden_states=True)
                
                # 最后一层最后位置
                final_state = outputs.hidden_states[-1][0, -1, :]
                all_hidden.append(final_state)
                all_labels.append(category)
    
    # PCA
    logger.info("Running PCA...")
    hidden_tensor = torch.stack(all_hidden)
    pca = TorchPCA(n_components=2)
    hidden_2d = pca.fit_transform(hidden_tensor)
    
    logger.log_metrics({
        "pca_variance_pc1": pca.explained_variance_ratio[0],
        "pca_variance_pc2": pca.explained_variance_ratio[1],
        "n_samples": len(all_hidden),
        "hidden_dim": hidden_tensor.shape[1]
    })
    
    # 可视化
    logger.info("Generating visualization...")
    colors = {
        'animal': '#FF6B6B',
        'number': '#4ECDC4',
        'emotion': '#45B7D1',
        'code': '#96CEB4'
    }
    
    plt.figure(figsize=(12, 9))
    hidden_2d_np = hidden_2d.cpu().numpy()
    
    for category in sentences.keys():
        indices = [i for i, l in enumerate(all_labels) if l == category]
        x = hidden_2d_np[indices, 0]
        y = hidden_2d_np[indices, 1]
        plt.scatter(x, y, c=colors[category], label=category, 
                   alpha=0.7, s=150, edgecolors='white', linewidth=1)
    
    plt.legend(fontsize=12)
    plt.title(f"AeLoRu: {EXPERIMENT_NAME}\n({MODEL_PATH})")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio[1]:.1%})")
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(LOG_ROOT, f"{logger.session_id}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.success(f"Plot saved: {plot_path}")
    
    # 聚类分析
    logger.info("Analyzing clustering...")
    category_centers = {}
    for cat in sentences.keys():
        mask = torch.tensor([l == cat for l in all_labels])
        cat_points = hidden_2d[mask]
        category_centers[cat] = cat_points.mean(dim=0)
    
    # 计算距离
    intra_dists = {}
    for cat in sentences.keys():
        mask = torch.tensor([l == cat for l in all_labels])
        points = hidden_2d[mask]
        center = category_centers[cat]
        intra_dists[cat] = torch.norm(points - center, dim=1).mean().item()
    
    inter_dists = []
    cats = list(sentences.keys())
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            d = torch.norm(category_centers[cats[i]] - category_centers[cats[j]]).item()
            inter_dists.append((cats[i], cats[j], d))
    
    avg_intra = np.mean(list(intra_dists.values()))
    avg_inter = np.mean([d for _, _, d in inter_dists])
    ratio = avg_inter / avg_intra if avg_intra > 0 else 0
    
    logger.log_metrics({
        "avg_intra_distance": avg_intra,
        "avg_inter_distance": avg_inter,
        "cluster_ratio": ratio
    })
    
    # 结论
    if ratio > 2.0:
        logger.success("Strong semantic clustering detected!")
    elif ratio > 1.5:
        logger.info("Moderate clustering present")
    else:
        logger.warning("Weak clustering - may need larger model")
    
    # 保存最终结果
    logger.finalize({
        "clustering_quality": {
            "intra_distances": intra_dists,
            "inter_distances": [{"pair": f"{a}-{b}", "distance": d} for a, b, d in inter_dists],
            "ratio": ratio
        },
        "visualization": plot_path,
        "status": "success"
    })

if __name__ == "__main__":
    main()