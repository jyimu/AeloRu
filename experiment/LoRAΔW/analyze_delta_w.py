import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine
import json
from datetime import datetime
import os

# ==================== ΔW 提取 ====================
class DeltaWExtractor:
    """提取 LoRA 的 ΔW = A × B"""
    
    def __init__(self, model):
        self.model = model
        self.delta_w_dict = {}
        
    def extract_all_layers(self):
        """提取所有层的 ΔW"""
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # 获取 A 和 B 矩阵
                A = module.lora_A.default.weight.data  # (r, in_features)
                B = module.lora_B.default.weight.data  # (out_features, r)
                
                # 计算 ΔW = B × A (注意维度)
                delta_w = B @ A  # (out_features, in_features)
                
                self.delta_w_dict[name] = {
                    'delta_w': delta_w.cpu(),
                    'A': A.cpu(),
                    'B': B.cpu(),
                    'shape': delta_w.shape
                }
        
        print(f"✅ 已提取 {len(self.delta_w_dict)} 个 LoRA 层的 ΔW")
        return self.delta_w_dict
    
    def save_to_file(self, output_path):
        """保存 ΔW 到文件"""
        save_data = {}
        for layer_name, data in self.delta_w_dict.items():
            save_data[layer_name] = {
                'delta_w': data['delta_w'].numpy().tolist(),
                'shape': data['shape']
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ ΔW 已保存至：{output_path}")

# ==================== 语义探测 ====================
class SemanticProbe:
    """用语义向量探测 ΔW 的语义响应"""
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        with open("AeloRu\experiment\\tool\WOLD.json", "r", encoding="utf-8") as f:
             self.semantic_categories = json.load(f)

    def get_category_embeddings(self):
        """获取各类别的词嵌入"""
        category_embeddings = {}
        device = next(self.model.parameters()).device  # 获取模型设备
        
        for category, words in self.semantic_categories.items():
            embeddings = []
            for word in words:
                inputs = self.tokenizer(word, return_tensors='pt')
                input_ids = inputs['input_ids'].to(device)  # 移动到模型设备
                with torch.no_grad():
                    emb = self.model.get_input_embeddings()(input_ids)
                    embeddings.append(emb.mean(dim=1).squeeze().cpu().numpy())
            
            category_embeddings[category] = np.mean(embeddings, axis=0)
        
        return category_embeddings
    
    def probe_delta_w(self, delta_w_dict, category_embeddings):
        """探测 ΔW 对各类别的响应"""
        results = {}
        
        for layer_name, data in delta_w_dict.items():
            delta_w = data['delta_w'].numpy()
            layer_results = {}
            
            for category, emb in category_embeddings.items():
                # 计算 ΔW 与语义向量的响应
                # 方法：将语义向量通过 ΔW 变换，看激活强度
                response = np.abs(np.dot(delta_w, emb))
                layer_results[category] = {
                    'mean': float(np.mean(response)),
                    'std': float(np.std(response)),
                    'max': float(np.max(response)),
                    'min': float(np.min(response))
                }
            
            results[layer_name] = layer_results
        
        return results

# ==================== 相似度分析 ====================
class SemanticSimilarityAnalyzer:
    """分析 ΔW 变化与语义相似性的关系"""
    
    def __init__(self, delta_w_dict):
        self.delta_w_dict = delta_w_dict
    
    def flatten_delta_w(self):
        """将 ΔW 展平为向量"""
        vectors = {}
        for layer_name, data in self.delta_w_dict.items():
            vectors[layer_name] = data['delta_w'].numpy().flatten()
        return vectors
    
    def compute_similarity_matrix(self, vectors):
        """计算层间 ΔW 相似度矩阵"""
        layer_names = list(vectors.keys())
        n = len(layer_names)
        similarity_matrix = np.zeros((n, n))
        
        for i, name_i in enumerate(layer_names):
            for j, name_j in enumerate(layer_names):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # 检查向量长度是否相同
                    if len(vectors[name_i]) != len(vectors[name_j]):
                        similarity_matrix[i, j] = 0.0  # 或 np.nan，如果需要标记为无效
                    else:
                        # 余弦相似度
                        sim = 1 - cosine(vectors[name_i], vectors[name_j])
                        similarity_matrix[i, j] = sim
        
        return similarity_matrix, layer_names
    
    def analyze_semantic_clustering(self, vectors, category_labels):
        """分析 ΔW 是否按语义类别聚类"""
        # 准备数据
        X = np.array([vectors[name] for name in vectors.keys()])
        
        # PCA 降维
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # KMeans 聚类
        n_clusters = len(set(category_labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # 计算轮廓系数
        if len(set(clusters)) > 1:
            silhouette = silhouette_score(X, clusters)
        else:
            silhouette = 0.0
        
        return {
            'pca_coordinates': X_pca,
            'clusters': clusters,
            'silhouette_score': silhouette,
            'variance_explained': pca.explained_variance_ratio_.sum()
        }

# ==================== 可视化 ====================
class DeltaWVisualizer:
    """ΔW 可视化工具"""
    
    def __init__(self, output_dir="./delta_w_viz"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def plot_singular_values(self, delta_w_dict, top_n=4):
        """绘制奇异值分布"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (layer_name, data) in enumerate(list(delta_w_dict.items())[:top_n]):
            delta_w = data['delta_w'].numpy()
            U, S, Vt = np.linalg.svd(delta_w)
            
            axes[idx].plot(S[:50], 'o-', linewidth=2)
            axes[idx].set_title(f'{layer_name}\nTop 50 Singular Values')
            axes[idx].set_xlabel('Rank')
            axes[idx].set_ylabel('Singular Value')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = f"{self.output_dir}/svd_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ 奇异值图保存：{save_path}")
    
    def plot_semantic_heatmap(self, probe_results):
        """绘制语义响应热力图"""
        # 准备数据
        layers = list(probe_results.keys())[:20]  # 取前 20 层
        categories = list(next(iter(probe_results.values())).keys()) if probe_results else []
        
        data_matrix = []
        for layer in layers:
            row = [probe_results[layer][cat]['mean'] for cat in categories]
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(data_matrix, 
                   xticklabels=categories,
                   yticklabels=[f"L{i}" for i in range(len(layers))],
                   cmap='YlOrRd',
                   annot=True,
                   fmt='.3f')
        
        plt.title('ΔW Semantic Response Heatmap')
        plt.xlabel('Semantic Category')
        plt.ylabel('LoRA Layer')
        
        save_path = f"{self.output_dir}/semantic_heatmap_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ 热力图保存：{save_path}")
    
    def plot_category_stats(self, probe_results):
        """绘制类别统计图"""
        categories = list(next(iter(probe_results.values())).keys()) if probe_results else []
        means = []
        stds = []
        
        for cat in categories:
            cat_means = [probe_results[layer][cat]['mean'] 
                        for layer in probe_results.keys()]
            means.append(np.mean(cat_means))
            stds.append(np.std(cat_means))
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, means, yerr=stds, capsize=5)
        
        # 添加数值标签
        for bar, mean in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{mean:.3f}', ha='center', va='bottom')
        
        plt.title('ΔW Response by Semantic Category')
        plt.xlabel('Category')
        plt.ylabel('Mean Activation')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        save_path = f"{self.output_dir}/category_stats_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ 统计图保存：{save_path}")
    
    def plot_pca_clustering(self, clustering_results, category_labels):
        """绘制 PCA 聚类图"""
        plt.figure(figsize=(10, 8))
        
        colors = {'animal': 'red', 'emotion': 'blue', 'number': 'green', 
                 'code': 'purple'}
        
        for category in set(category_labels):
            mask = [label == category for label in category_labels]
            coords = clustering_results['pca_coordinates'][mask]
            plt.scatter(coords[:, 0], coords[:, 1], 
                       label=category, c=colors.get(category, 'gray'),
                       alpha=0.6, s=100)
        
        plt.title(f'ΔW PCA Clustering\nSilhouette: {clustering_results["silhouette_score"]:.3f}')
        plt.xlabel(f'PC1')
        plt.ylabel(f'PC2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = f"{self.output_dir}/pca_clustering_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ 聚类图保存：{save_path}")
    
    def plot_similarity_matrix(self, similarity_matrix, layer_names):
        """绘制相似度矩阵"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, 
                   xticklabels=[f"L{i}" for i in range(len(layer_names))],
                   yticklabels=[f"L{i}" for i in range(len(layer_names))],
                   cmap='coolwarm',
                   vmin=0, vmax=1)
        
        plt.title('Inter-Layer ΔW Cosine Similarity')
        plt.xlabel('Layer')
        plt.ylabel('Layer')
        
        save_path = f"{self.output_dir}/similarity_matrix_{self.timestamp}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"✅ 相似度矩阵保存：{save_path}")

# ==================== 主分析流程 ====================
def run_delta_w_analysis(model, tokenizer, output_dir="./AeloRu/experiment/LoRA/delta_w_analysis"):
    """运行完整的 ΔW 分析流程"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("🚀 开始 ΔW 语义分析")
    print("=" * 60)
    
    # 1. 提取 ΔW
    print("\n[1/5] 提取 ΔW...")
    extractor = DeltaWExtractor(model)
    delta_w_dict = extractor.extract_all_layers()
    extractor.save_to_file(f"{output_dir}/delta_w_{timestamp}.json")
    
    # 2. 语义探测
    print("\n[2/5] 语义探测...")
    probe = SemanticProbe(tokenizer, model)
    category_embeddings = probe.get_category_embeddings()
    probe_results = probe.probe_delta_w(delta_w_dict, category_embeddings)
    
    with open(f"{output_dir}/probe_results_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(probe_results, f, ensure_ascii=False, indent=2)
    
    # 3. 相似度分析
    print("\n[3/5] 相似度分析...")
    sim_analyzer = SemanticSimilarityAnalyzer(delta_w_dict)
    vectors = sim_analyzer.flatten_delta_w()
    sim_matrix, layer_names = sim_analyzer.compute_similarity_matrix(vectors)
    
    # 4. 可视化
    print("\n[4/5] 生成可视化...")
    viz = DeltaWVisualizer(output_dir)
    viz.plot_singular_values(delta_w_dict)
    viz.plot_semantic_heatmap(probe_results)
    viz.plot_category_stats(probe_results)
    viz.plot_similarity_matrix(sim_matrix, layer_names)
    
    # 5. 生成报告
    print("\n[5/5] 生成分析报告...")
    report = generate_analysis_report(probe_results, sim_matrix, 
                                      layer_names, output_dir, timestamp)
    
    print("\n" + "=" * 60)
    print("✅ ΔW 分析完成!")
    print("=" * 60)
    
    return report

def generate_analysis_report(probe_results, sim_matrix, layer_names, output_dir, timestamp):
    """生成分析报告"""
    # 统计各类别响应
    categories = ['animal', 'emotion', 'number', 'code']
    stats = {}
    
    for cat in categories:
        cat_means = [probe_results[layer][cat]['mean'] 
                    for layer in probe_results.keys()]
        stats[cat] = {
            'mean': float(np.mean(cat_means)),
            'std': float(np.std(cat_means)),
            'min': float(np.min(cat_means)),
            'max': float(np.max(cat_means))
        }
    
    # 相似度统计
    sim_values = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
    
    report = {
        'timestamp': timestamp,
        'total_layers': len(layer_names),
        'category_stats': stats,
        'similarity_stats': {
            'mean': float(np.mean(sim_values)),
            'std': float(np.std(sim_values)),
            'min': float(np.min(sim_values)),
            'max': float(np.max(sim_values))
        },
        'files_generated': [
            f"delta_w_{timestamp}.json",
            f"probe_results_{timestamp}.json",
            f"svd_{timestamp}.png",
            f"semantic_heatmap_{timestamp}.png",
            f"category_stats_{timestamp}.png",
            f"similarity_matrix_{timestamp}.png"
        ]
    }
    
    with open(f"{output_dir}/analysis_report_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 报告保存：{output_dir}/analysis_report_{timestamp}.json")
    return report

# ==================== 入口 ====================
if __name__ == "__main__":
    from train_lora import load_lora_model, LoRAConfig
    
    config = LoRAConfig()
    model, tokenizer = load_lora_model(config, checkpoint_path="./AeloRu/experiment/LoRA/lora_checkpoints/final")
    report = run_delta_w_analysis(model, tokenizer)
    
    print("\n📊 分析摘要:")
    print(f"   总层数：{report['total_layers']}")
    print(f"   类别响应均值：{report['category_stats']}")
    print(f"   层间相似度均值：{report['similarity_stats']['mean']:.3f}")