# analyze_full_vector_errors.py
# 专门分析 full_vector 方法错误分类的题目

import torch
import numpy as np
from scipy.spatial.distance import cosine
import json
import glob
import os
from datetime import datetime

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
    
    def verify_with_details(self, text_pairs, method='full_vector', threshold=0.5):
        """
        验证并返回详细结果（包含错误分析）
        """
        results = []
        errors = []
        
        for text1, text2, is_similar in text_pairs:
            # 获取文本嵌入
            emb1 = self.get_text_embedding(text1)
            emb2 = self.get_text_embedding(text2)
            
            # 计算文本语义相似度
            text_similarity = 1 - cosine(emb1, emb2)
            
            # 计算 ΔW 响应
            resp1 = self.compute_delta_w_response(emb1)
            resp2 = self.compute_delta_w_response(emb2)
            
            # 计算 ΔW 响应相似度
            delta_w_similarity = self.compute_delta_w_similarity(resp1, resp2, method)
            
            # 预测
            predicted_similar = delta_w_similarity > threshold
            match = (predicted_similar == is_similar)
            
            result = {
                'text1': text1,
                'text2': text2,
                'is_similar': bool(is_similar),
                'text_similarity': float(text_similarity),
                'delta_w_similarity': float(delta_w_similarity),
                'predicted': bool(predicted_similar),
                'match': bool(match),
                'threshold': threshold
            }
            results.append(result)
            
            if not match:
                errors.append(result)
        
        # 计算统计
        accuracy = sum([r['match'] for r in results]) / len(results)
        similar_pairs = [r for r in results if r['is_similar']]
        dissimilar_pairs = [r for r in results if not r['is_similar']]
        similar_acc = sum([r['match'] for r in similar_pairs]) / len(similar_pairs) if similar_pairs else 0
        dissimilar_acc = sum([r['match'] for r in dissimilar_pairs]) / len(dissimilar_pairs) if dissimilar_pairs else 0
        
        return {
            'method': method,
            'threshold': threshold,
            'timestamp': datetime.now().isoformat(),
            'total_pairs': len(results),
            'correct': sum([r['match'] for r in results]),
            'errors': len(errors),
            'accuracy': accuracy,
            'similar_accuracy': similar_acc,
            'dissimilar_accuracy': dissimilar_acc,
            'pairs': results,
            'error_details': errors
        }


# ==================== 测试用例 ====================
def create_test_pairs():
    """创建测试文本对"""
    return [
        # ===== 相似对 =====
        ("开心", "高兴", True),
        ("美丽", "漂亮", True),
        ("今天心情很好", "我感到非常开心", True),
        ("猫", "小猫", True),
        ("狗", "小狗", True),
        ("计算 123+456", "算一下 123+456", True),
        ("def hello():", "def hello()  ", True),
        ("我喜欢编程", "我热爱写代码", True),
        ("你好", "您好", True),
        
        # ===== 不相似对 =====
        ("猫", "狗", False),
        ("开心", "悲伤", False),
        ("猫", "def hello():", False),
        ("计算 123+456", "3.14 是圆周率", False),
        ("今天心情很好", "狮子是草原之王", False),
        ("import torch", "我感到悲伤", False),
        ("老鼠", "100", False),
        ("小猫", "happy", False),
        ("仓鼠", "sad", False),
        ("狗", "def world():", False),
    ]


# ==================== 错误分析输出 ====================
def print_error_analysis(results):
    """打印详细的错误分析"""
    print("\n" + "=" * 70)
    print("🔍 full_vector 错误分析")
    print("=" * 70)
    
    print(f"\n📊 总体统计:")
    print(f"   总对数：{results['total_pairs']}")
    print(f"   正确：{results['correct']}")
    print(f"   错误：{results['errors']}")
    print(f"   准确率：{results['accuracy']:.2%}")
    print(f"   相似对准确率：{results['similar_accuracy']:.2%}")
    print(f"   不相似对准确率：{results['dissimilar_accuracy']:.2%}")
    print(f"   阈值：{results['threshold']}")
    
    print(f"\n❌ 错误分类的题目 ({results['errors']} 道):\n")
    
    for i, error in enumerate(results['error_details'], 1):
        label = "相似" if error['is_similar'] else "不相似"
        pred = "相似" if error['predicted'] else "不相似"
        gap = abs(error['delta_w_similarity'] - error['threshold'])
        
        print(f"{'='*70}")
        print(f"错误 #{i}")
        print(f"{'='*70}")
        print(f"文本对：\"{error['text1']}\" ↔ \"{error['text2']}\"")
        print(f"标注：{label} | 预测：{pred}")
        print(f"ΔW 相似度：{error['delta_w_similarity']:.6f}")
        print(f"文本相似度：{error['text_similarity']:.6f}")
        print(f"阈值：{error['threshold']}")
        print(f"与阈值差距：{gap:.6f}")
        
        # 错误类型分析
        if error['is_similar'] and not error['predicted']:
            print(f"错误类型：❌ 漏报 (False Negative)")
            print(f"说明：实际相似但被判定为不相似")
            print(f"建议：ΔW 相似度偏低，可能需要降低阈值或检查文本对")
        else:
            print(f"错误类型：❌ 误报 (False Positive)")
            print(f"说明：实际不相似但被判定为相似")
            print(f"建议：ΔW 相似度偏高，可能需要提高阈值或检查标注")
        print()
    
    # 边界案例分析
    print("=" * 70)
    print("📈 边界案例分析 (ΔW 相似度在 0.4-0.6 之间)")
    print("=" * 70)
    boundary_cases = [p for p in results['pairs'] if 0.4 <= p['delta_w_similarity'] <= 0.6]
    if boundary_cases:
        for p in boundary_cases:
            status = "✓" if p['match'] else "✗"
            label = "相似" if p['is_similar'] else "不相似"
            print(f"{status} \"{p['text1']}\" ↔ \"{p['text2']}\" | 标注:{label} | ΔW:{p['delta_w_similarity']:.4f}")
    else:
        print("无边界案例")
    print()
    
    # 相似度分布
    print("=" * 70)
    print("📊 ΔW 相似度分布")
    print("=" * 70)
    similar_sims = [p['delta_w_similarity'] for p in results['pairs'] if p['is_similar']]
    dissimilar_sims = [p['delta_w_similarity'] for p in results['pairs'] if not p['is_similar']]
    
    print(f"相似对 ΔW 相似度:")
    print(f"   均值：{np.mean(similar_sims):.4f}")
    print(f"   中位数：{np.median(similar_sims):.4f}")
    print(f"   最小：{np.min(similar_sims):.4f}")
    print(f"   最大：{np.max(similar_sims):.4f}")
    print(f"   标准差：{np.std(similar_sims):.4f}")
    
    print(f"\n不相似对 ΔW 相似度:")
    print(f"   均值：{np.mean(dissimilar_sims):.4f}")
    print(f"   中位数：{np.median(dissimilar_sims):.4f}")
    print(f"   最小：{np.min(dissimilar_sims):.4f}")
    print(f"   最大：{np.max(dissimilar_sims):.4f}")
    print(f"   标准差：{np.std(dissimilar_sims):.4f}")
    print()


# ==================== 主程序 ====================
if __name__ == "__main__":
    from train_lora import load_lora_model, LoRAConfig
    
    print("=" * 70)
    print("🚀 开始 full_vector 错误分析")
    print("=" * 70)
    
    # 加载模型
    print("\n📦 加载模型...")
    config = LoRAConfig()
    model, tokenizer = load_lora_model(
        config, 
        checkpoint_path="./AeloRu/experiment/LoRA/lora_checkpoints/final"
    )
    print("✅ 模型加载完成")
    
    # 加载 delta_w
    print("\n📦 加载 ΔW 数据...")
    pattern = './AeloRu/experiment/LoRA/delta_w_analysis/delta_w_*.json'
    json_paths = sorted(glob.glob(pattern))
    if not json_paths:
        raise FileNotFoundError(f"未找到任何文件：{pattern}")
    
    delta_w_dict = {}
    for p in json_paths:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        delta_w_dict.update(data)
    print(f"✅ 加载 {len(json_paths)} 个文件，共 {len(delta_w_dict)} 层")
    
    # 创建验证器
    verifier = SemanticSimilarityVerifier(model, tokenizer, delta_w_dict)
    test_pairs = create_test_pairs()
    
    # 运行验证（可以调整阈值）
    threshold = 0.5  # 可以改为 0.45 试试
    print(f"\n🔍 使用阈值：{threshold}")
    results = verifier.verify_with_details(test_pairs, method='full_vector', threshold=threshold)
    
    # 打印错误分析
    print_error_analysis(results)
    
    # 保存详细结果
    output_file = 'full_vector_error_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 详细结果已保存：{output_file}")
    
    # 保存错误题目列表（方便查看）
    errors_only = {
        'total_errors': results['errors'],
        'accuracy': results['accuracy'],
        'errors': results['error_details']
    }
    with open('full_vector_errors_only.json', 'w', encoding='utf-8') as f:
        json.dump(errors_only, f, ensure_ascii=False, indent=2)
    print(f"✅ 错误题目已保存：full_vector_errors_only.json")
    
    print("\n" + "=" * 70)
    print("✅ 分析完成!")
    print("=" * 70)