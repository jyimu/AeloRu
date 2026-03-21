import torch
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import json
import glob
import os

class SemanticSimilarityVerifier:
    """验证 ΔW 变化与语句内容相似性的关系"""
    
    def __init__(self, model, tokenizer, delta_w_dict):
        self.model = model
        self.tokenizer = tokenizer
        self.delta_w_dict = delta_w_dict
    
    def get_text_embedding(self, text):
        """获取文本的嵌入表示"""
        inputs = self.tokenizer(text, return_tensors='pt',
                               padding=True, truncation=True, max_length=64)
        device = next(self.model.parameters()).device
        input_ids = inputs['input_ids'].to(device)
        with torch.no_grad():
            outputs = self.model.get_input_embeddings()(input_ids)
            embedding = outputs.mean(dim=1).squeeze().cpu().numpy()
        return embedding
    
    def compute_delta_w_response(self, text_embedding):
        """计算文本嵌入通过 ΔW 的响应"""
        responses = {}
        
        for layer_name, data in self.delta_w_dict.items():
            delta_w = torch.tensor(data['delta_w']).cpu().numpy()
            # 计算响应：ΔW × embedding
            response = np.dot(delta_w, text_embedding)
            responses[layer_name] = {
                'response': response,
                'magnitude': np.linalg.norm(response)
            }
        
        return responses
    
    def compute_delta_w_similarity(self, resp1, resp2, method='full_vector'):
        """
        计算 ΔW 响应相似度
        
        method: 'full_vector' | 'layer_weighted' | 'magnitude'
        """
        if method == 'full_vector':
            # 使用完整响应向量（推荐）
            vec1 = np.concatenate([resp1[layer]['response'] for layer in sorted(resp1.keys())])
            vec2 = np.concatenate([resp2[layer]['response'] for layer in sorted(resp2.keys())])
            return 1 - cosine(vec1, vec2)
        
        elif method == 'layer_weighted':
            # 分层加权
            layers = sorted(resp1.keys())
            n_layers = len(layers)
            sims = []
            weights = []
            for i, layer in enumerate(layers):
                vec1 = resp1[layer]['response']
                vec2 = resp2[layer]['response']
                sim = 1 - cosine(vec1, vec2)
                weight = (i + 1) / n_layers
                sims.append(sim)
                weights.append(weight)
            return np.average(sims, weights=weights)
        
        else:  # magnitude
            # 原来的模长方法（不推荐）
            mag1 = np.array([resp1[layer]['magnitude'] for layer in sorted(resp1.keys())])
            mag2 = np.array([resp2[layer]['magnitude'] for layer in sorted(resp2.keys())])
            return 1 - cosine(mag1, mag2)
    
    def verify_similarity_correlation(self, text_pairs, method='full_vector'):
        """
        验证：语义相似的文本，其 ΔW 响应也相似
        
        text_pairs: [(text1, text2, is_similar), ...]
        is_similar: True/False 表示是否语义相似
        """
        results = []
        
        for text1, text2, is_similar in text_pairs:
            # 获取文本嵌入
            emb1 = self.get_text_embedding(text1)
            emb2 = self.get_text_embedding(text2)
            
            # 计算文本语义相似度
            text_similarity = 1 - cosine(emb1, emb2)
            
            # 计算 ΔW 响应
            resp1 = self.compute_delta_w_response(emb1)
            resp2 = self.compute_delta_w_response(emb2)
            
            # 计算 ΔW 响应相似度（使用新方法）
            delta_w_similarity = self.compute_delta_w_similarity(resp1, resp2, method)
            
            results.append({
                'text1': text1,
                'text2': text2,
                'is_similar': bool(is_similar),
                'text_similarity': float(text_similarity),
                'delta_w_similarity': float(delta_w_similarity),
                'match': bool((is_similar and delta_w_similarity > 0.5) or 
                             (not is_similar and delta_w_similarity < 0.5))
            })
        
        # 计算整体准确率
        accuracy = sum([r['match'] for r in results]) / len(results)
        
        return {
            'pairs': results,
            'accuracy': accuracy,
            'correlation': self._compute_correlation(results),
            'method': method
        }
    
    def _compute_correlation(self, results):
        """计算文本相似度与 ΔW 响应相似度的相关系数"""
        text_sims = [r['text_similarity'] for r in results]
        delta_sims = [r['delta_w_similarity'] for r in results]
        
        correlation = np.corrcoef(text_sims, delta_sims)[0, 1]
        return float(correlation)

# ==================== 测试用例 ====================
def create_test_pairs():
    """创建测试文本对"""
    return [
        ("开心", "高兴", True),
        ("美丽", "漂亮", True),
        ("今天心情很好", "我感到非常开心", True),
        ("猫", "小猫", True),
        ("狗", "小狗", True),
        ("计算 123+456", "算一下 123+456", True),
        ("def hello():", "def hello()  ", True),
        ("我喜欢编程", "我热爱写代码", True),
        ("你好", "您好", True),
        
        # ===== 真正的不相似对 (不同概念) =====
        ("猫", "狗", False),  # 不同动物
        ("开心", "悲伤", False),  # 反义词
        ("猫", "def hello():", False),  # 不同领域
        ("计算 123+456", "3.14 是圆周率", False),  # 不同数学概念
        ("今天心情很好", "狮子是草原之王", False),
        ("import torch", "我感到悲伤", False),
        ("老鼠", "100", False),
        ("小猫", "happy", False),
        ("仓鼠", "sad", False),
        ("狗", "def world():", False),
    ]
# ==================== 主验证流程 ====================
def run_similarity_verification(model, tokenizer, delta_w_dict, method='full_vector'):
    """运行语义相似性验证"""
    print("=" * 60)
    print("🔍 开始 ΔW 语义相似性验证")
    print("=" * 60)
    print(f"📋 相似度计算方法：{method}")
    
    verifier = SemanticSimilarityVerifier(model, tokenizer, delta_w_dict)
    test_pairs = create_test_pairs()
    
    results = verifier.verify_similarity_correlation(test_pairs, method)
    
    print(f"\n✅ 验证完成!")
    print(f"   测试对数：{len(test_pairs)}")
    print(f"   匹配准确率：{results['accuracy']:.2%}")
    print(f"   相关系数：{results['correlation']:.3f}")
    
    # 统计相似对和不相似对的分别准确率
    similar_pairs = [r for r in results['pairs'] if r['is_similar']]
    dissimilar_pairs = [r for r in results['pairs'] if not r['is_similar']]
    
    similar_acc = sum([r['match'] for r in similar_pairs]) / len(similar_pairs) if similar_pairs else 0
    dissimilar_acc = sum([r['match'] for r in dissimilar_pairs]) / len(dissimilar_pairs) if dissimilar_pairs else 0
    
    print(f"   相似对准确率：{similar_acc:.2%} ({sum([r['match'] for r in similar_pairs])}/{len(similar_pairs)})")
    print(f"   不相似对准确率：{dissimilar_acc:.2%} ({sum([r['match'] for r in dissimilar_pairs])}/{len(dissimilar_pairs)})")
    
    # 保存结果
    with open('./similarity_verification_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 结果保存：./similarity_verification_results.json")
    
    return results

# ==================== 入口 ====================
if __name__ == "__main__":
    from train_lora import load_lora_model, LoRAConfig

    # 加载模型
    config = LoRAConfig()
    model, tokenizer = load_lora_model(config, checkpoint_path="./AeloRu/experiment/LoRA/lora_checkpoints/final")

    # 加载所有 delta_w 文件
    pattern = './AeloRu/experiment/LoRA/delta_w_analysis/delta_w_*.json'
    json_paths = sorted(glob.glob(pattern))
    if not json_paths:
        raise FileNotFoundError(f"未找到任何文件：{pattern}")

    delta_w_dict = {}
    for p in json_paths:
        print(f"加载 {p} ...")
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        delta_w_dict.update(data)
    print(f"共加载 {len(json_paths)} 个文件，合并层数 {len(delta_w_dict)}")

    # 运行验证（三种方法对比）
    print("\n" + "=" * 60)
    print("📊 对比不同相似度计算方法")
    print("=" * 60)
    
    methods = ['magnitude', 'full_vector', 'layer_weighted']
    all_results = {}
    
    for method in methods:
        print(f"\n🔹 方法：{method}")
        results = run_similarity_verification(model, tokenizer, delta_w_dict, method)
        all_results[method] = {
            'accuracy': results['accuracy'],
            'correlation': results['correlation']
        }
    
    # 汇总对比
    print("\n" + "=" * 60)
    print("📈 方法对比汇总")
    print("=" * 60)
    print(f"{'方法':<15} {'准确率':<10} {'相关系数':<10}")
    print("-" * 35)
    for method, metrics in all_results.items():
        print(f"{method:<15} {metrics['accuracy']:.2%}      {metrics['correlation']:.3f}")