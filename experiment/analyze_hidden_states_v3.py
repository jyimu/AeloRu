#!/usr/bin/env python3
# analyze_hidden_states_v3.py
# 改进版：多样本 + 多策略 + 多层分析

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import json


# ==================== 配置 ====================
MODEL_PATH = "models/Qwen2.5-1.5B"
EXPERIMENT_NAME = "hidden_states_analysis_v3"



# 扩充后的样本（每类20句）
with open("AeloRu\\experiment\\tool\\SENTENCES_V3.json", "r") as f:
    SENTENCES_V3 = json.load(f)

# ==================== PyTorch PCA ====================
class TorchPCA:
    """纯 PyTorch PCA 支持更多主成分"""
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance_ratio = None
        self.cumulative_variance = None
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        n_samples, n_features = X.shape
        self.mean = X.mean(dim=0)
        X_centered = X - self.mean
        
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        self.components = Vt[:self.n_components]
        
        explained_variance = (S ** 2) / (n_samples - 1)
        total_variance = explained_variance.sum()
        self.explained_variance_ratio = (
            explained_variance[:self.n_components] / total_variance
        ).cpu().numpy()
        self.cumulative_variance = np.cumsum(self.explained_variance_ratio)
        
        return X_centered @ self.components.T


# ==================== 隐藏状态提取策略 ====================
def extract_hidden_state(outputs, strategy="last_token", layers="last"):
    """
    多种隐藏状态提取策略
    
    Args:
        outputs: model output
        strategy: "last_token" | "mean_token" | "max_token" | "first_token"
        layers: "last" | "mean" | "concat_3" | "layer_X"
    """
    hidden_states = outputs.hidden_states  # (num_layers+1, batch, seq_len, dim)
    
    # 选择层
    if layers == "last":
        hidden = hidden_states[-1]  # (batch, seq_len, dim)
    elif layers == "mean":
        hidden = torch.stack(hidden_states[1:]).mean(dim=0)  # 跳过embedding
    elif layers == "concat_3":
        last_three = torch.cat([hidden_states[-i] for i in [1,2,3]], dim=-1)
        hidden = last_three
    elif isinstance(layers, int):
        hidden = hidden_states[layers]
    else:
        hidden = hidden_states[-1]
    
    # 选择token
    if strategy == "last_token":
        return hidden[0, -1, :]
    elif strategy == "mean_token":
        return hidden[0].mean(dim=0)
    elif strategy == "max_token":
        return hidden[0].max(dim=0)[0]
    elif strategy == "first_token":
        return hidden[0, 0, :]
    else:
        return hidden[0, -1, :]


# ==================== 聚类评估指标 ====================
def calculate_clustering_metrics(hidden_2d, labels):
    """计算多种聚类评估指标"""
    categories = list(set(labels))
    
    # 1. 类内距离
    intra_dists = {}
    for cat in categories:
        mask = torch.tensor([l == cat for l in labels])
        points = hidden_2d[mask]
        center = points.mean(dim=0)
        intra_dists[cat] = torch.norm(points - center, dim=1).mean().item()
    
    # 2. 类间距离
    inter_dists = []
    for cat1, cat2 in combinations(categories, 2):
        mask1 = torch.tensor([l == cat1 for l in labels])
        mask2 = torch.tensor([l == cat2 for l in labels])
        center1 = hidden_2d[mask1].mean(dim=0)
        center2 = hidden_2d[mask2].mean(dim=0)
        d = torch.norm(center1 - center2).item()
        inter_dists.append((cat1, cat2, d))
    
    # 3. 轮廓系数近似计算
    silhouette_scores = []
    for i, label in enumerate(labels):
        same_mask = torch.tensor([l == label for l in labels])
        diff_mask = ~same_mask
        
        if same_mask.sum() > 1 and diff_mask.sum() > 0:
            a = torch.norm(hidden_2d[i] - hidden_2d[same_mask], dim=1).mean().item()
            b = torch.norm(hidden_2d[i] - hidden_2d[diff_mask], dim=1).mean().item()
            if max(a, b) > 0:
                silhouette_scores.append((b - a) / max(a, b))
    
    avg_silhouette = np.mean(silhouette_scores) if silhouette_scores else 0
    
    # 4. 分离度比率
    avg_intra = np.mean(list(intra_dists.values()))
    avg_inter = np.mean([d for _, _, d in inter_dists])
    ratio = avg_inter / avg_intra if avg_intra > 0 else 0
    
    return {
        'intra_distances': intra_dists,
        'inter_distances': inter_dists,
        'avg_intra': avg_intra,
        'avg_inter': avg_inter,
        'separation_ratio': ratio,
        'silhouette_score': avg_silhouette
    }


# ==================== 可视化 ====================
def plot_clustering(hidden_2d, labels, pca, save_path, title_suffix=""):
    """生成聚类可视化图"""
    colors = {
        'animal': '#FF6B6B',
        'number': '#4ECDC4',
        'emotion': '#45B7D1',
        'code': '#96CEB4'
    }
    
    plt.figure(figsize=(14, 10))
    hidden_2d_np = hidden_2d.cpu().numpy()
    
    for category in set(labels):
        indices = [i for i, l in enumerate(labels) if l == category]
        x = hidden_2d_np[indices, 0]
        y = hidden_2d_np[indices, 1]
        plt.scatter(x, y, c=colors.get(category, '#999999'), 
                   label=f"{category} (n={len(indices)})", 
                   alpha=0.6, s=120, edgecolors='white', linewidth=1.5)
    
    # 添加类别中心
    for category in set(labels):
        mask = torch.tensor([l == category for l in labels])
        center = hidden_2d[mask].mean(dim=0).cpu().numpy()
        plt.scatter(center[0], center[1], c='black', marker='X', 
                   s=300, edgecolors='yellow', linewidth=2, label=f"{category}_center")
    
    plt.legend(fontsize=10, loc='best')
    plt.title(f"AeLoRu: Hidden State Clustering {title_suffix}\n"
             f"PC1 ({pca.explained_variance_ratio[0]:.1%}) | "
             f"PC2 ({pca.explained_variance_ratio[1]:.1%}) | "
             f"Cumulative ({pca.cumulative_variance[1]:.1%})", fontsize=12)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio[1]:.1%} variance)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.2)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.2)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


# ==================== 主程序 ====================
def main():
    logger = get_logger(EXPERIMENT_NAME)
    
    auto_device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(auto_device)
    
    logger.set_config({
        "model_path": MODEL_PATH,
        "device": auto_device,
        "model_size": "1.5B",
        "analysis_type": "hidden_states_semantic_v3",
        "n_samples_per_class": 20,
        "extraction_strategies": ["last_token", "mean_token"],
        "layer_strategies": ["last", "mean"]
    })
    
    # 加载模型
    logger.info("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            device_map=device,
            local_files_only=True,
            trust_remote_code=True,
            output_hidden_states=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True,
        )
        logger.success("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.finalize({"status": "failed", "error": str(e)})
        return
    
    # 实验配置列表
    experiments = [
        {"strategy": "last_token", "layers": "last", "suffix": "Last_Token_Last_Layer"},
        {"strategy": "mean_token", "layers": "last", "suffix": "Mean_Token_Last_Layer"},
        {"strategy": "last_token", "layers": "mean", "suffix": "Last_Token_All_Layers_Mean"},
        {"strategy": "mean_token", "layers": "mean", "suffix": "Mean_Token_All_Layers_Mean"},
    ]
    
    results = {}
    model.eval()
    
    with torch.no_grad():
        for exp_config in experiments:
            exp_name = exp_config["suffix"]
            logger.info(f"\n{'='*50}")
            logger.info(f"Running experiment: {exp_name}")
            logger.info(f"{'='*50}")
            
            all_hidden = []
            all_labels = []
            
            # 收集隐藏状态
            for category, sents in SENTENCES_V3.items():
                for sent in sents:
                    logger.step()
                    
                    inputs = tokenizer(sent, return_tensors="pt").to(torch_device)
                    outputs = model(**inputs, output_hidden_states=True)
                    
                    hidden = extract_hidden_state(
                        outputs, 
                        strategy=exp_config["strategy"],
                        layers=exp_config["layers"]
                    )
                    all_hidden.append(hidden)
                    all_labels.append(category)
            
            # PCA
            hidden_tensor = torch.stack(all_hidden)
            pca = TorchPCA(n_components=10)  # 计算10个主成分
            hidden_2d = pca.fit_transform(hidden_tensor)
            
            # 记录PCA信息
            logger.log_metrics({
                f"pca_variance_pc1_{exp_name}": pca.explained_variance_ratio[0],
                f"pca_variance_pc2_{exp_name}": pca.explained_variance_ratio[1],
                f"pca_cumulative_{exp_name}": pca.cumulative_variance[1],
                f"pca_cumulative_10_{exp_name}": pca.cumulative_variance[-1]
            })
            
            # 聚类分析
            metrics = calculate_clustering_metrics(hidden_2d, all_labels)
            
            logger.log_metrics({
                f"avg_intra_{exp_name}": metrics['avg_intra'],
                f"avg_inter_{exp_name}": metrics['avg_inter'],
                f"separation_ratio_{exp_name}": metrics['separation_ratio'],
                f"silhouette_{exp_name}": metrics['silhouette_score']
            })
            
            # 可视化
            plot_path = os.path.join(LOG_ROOT, f"{logger.session_id}_{exp_name}.png")
            plot_clustering(hidden_2d, all_labels, pca, plot_path, f"-{exp_name}")
            logger.success(f"Plot saved: {plot_path}")
            
            # 保存结果
            results[exp_name] = {
                "metrics": metrics,
                "pca_variance": pca.explained_variance_ratio.tolist(),
                "cumulative_variance": pca.cumulative_variance.tolist(),
                "plot_path": plot_path
            }
            
            # 质量评估
            if metrics['separation_ratio'] > 2.5:
                logger.success(f"Excellent clustering! Ratio: {metrics['separation_ratio']:.3f}")
            elif metrics['separation_ratio'] > 1.8:
                logger.info(f"Good clustering. Ratio: {metrics['separation_ratio']:.3f}")
            elif metrics['separation_ratio'] > 1.2:
                logger.warning(f"Moderate clustering. Ratio: {metrics['separation_ratio']:.3f}")
            else:
                logger.warning(f"Weak clustering. Ratio: {metrics['separation_ratio']:.3f}")
    
    # 比较各策略
    logger.info("\n" + "="*60)
    logger.info("STRATEGY COMPARISON SUMMARY")
    logger.info("="*60)
    
    best_strategy = max(results.keys(), 
                       key=lambda k: results[k]['metrics']['separation_ratio'])
    
    for exp_name, data in results.items():
        ratio = data['metrics']['separation_ratio']
        silhouette = data['metrics']['silhouette_score']
        marker = "⭐ BEST" if exp_name == best_strategy else ""
        logger.info(f"{exp_name}: Ratio={ratio:.3f}, Silhouette={silhouette:.3f} {marker}")
    
    # 保存最终结果
    logger.finalize({
        "status": "success",
        "best_strategy": best_strategy,
        "all_results": results,
        "comparison": {
            exp_name: {
                "separation_ratio": data['metrics']['separation_ratio'],
                "silhouette_score": data['metrics']['silhouette_score'],
                "cumulative_variance_10pc": data['cumulative_variance'][9]
            }
            for exp_name, data in results.items()
        }
    })
    
    logger.info(f"\nExperiment complete! Best strategy: {best_strategy}")


if __name__ == "__main__":
    main()