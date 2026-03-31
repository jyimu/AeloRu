#!/usr/bin/env python3
# quick_svd_semantic_check_local.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def main():
    # 本地模型路径
    model_path = "./models/Qwen2.5-1.5B"  # 请替换为你的本地模型路径
    
    print(f"Loading model from: {model_path}")
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        print(f"Error: Path {model_path} not found!")
        print("Please check the model path.")
        return
    
    # 加载本地模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # 小模型用float32更稳定
        device_map="cuda",           # 避免GPU内存问题
        local_files_only=True,      # 强制使用本地文件
        trust_remote_code=True      # Qwen模型需要
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    
    print(f"Model loaded: {model.config.model_type}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Vocab size: {model.config.vocab_size}")
    
    # 取输出层权重（lm_head）
    # Qwen2.5结构: model.lm_head.weight
    lm_head = model.lm_head.weight  # [vocab_size, hidden_dim]
    print(f"\nLM head shape: {lm_head.shape}")
    
    # 如果显存不够，只取前1000个奇异值
    max_rank = min(1000, min(lm_head.shape))
    print(f"Running SVD (rank={max_rank})...")
    
    # 使用float32避免精度问题
    lm_head_float = lm_head.float()
    
    # 截断SVD（更快）
    U, S, V = torch.svd(lm_head_float)
    
    # 只分析前20个
    num_vectors = min(20, len(S))
    
    print(f"\n{'='*50}")
    print("=== Top Singular Vectors Analysis ===")
    print(f"{'='*50}")
    
    results = []
    
    for i in range(num_vectors):
        # U[:, i] 是左奇异向量，直接对应vocab分布
        u_i = U[:, i]  # [vocab_size]
        
        # 取绝对值最大的15个token
        top_values, top_indices = torch.topk(u_i.abs(), k=15)
        
        tokens = []
        for idx, val in zip(top_indices, top_values):
            token = tokenizer.decode([idx.item()])
            # 清理特殊字符，方便阅读
            token_clean = repr(token).strip("'")
            tokens.append((token_clean, val.item()))
        
        # 打印
        print(f"\n--- Vector {i:2d} (σ={S[i].item():.4f}) ---")
        token_str = " | ".join([f"{t}({v:.2f})" for t, v in tokens[:10]])
        print(f"  {token_str}")
        
        # 保存结构化结果
        results.append({
            'index': i,
            'singular_value': S[i].item(),
            'top_tokens': [t for t, _ in tokens],
            'top_values': [v for _, v in tokens]
        })
    
    # 保存到文件
    output_file = ".\\AeloRu\\experiment\\log\\svd_decode_1.5B.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Analysis date: 2026-03-18\n")
        f.write(f"{'='*60}\n\n")
        
        for r in results:
            f.write(f"Vector {r['index']:2d} (σ={r['singular_value']:.4f})\n")
            f.write(f"  Tokens: {', '.join(r['top_tokens'])}\n")
            f.write(f"  Values: {[f'{v:.3f}' for v in r['top_values']]}\n\n")
    
    print(f"\n{'='*50}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*50}")
    
    # 简单统计：检查是否有明显模式
    print("\n=== Quick Pattern Check ===")
    all_tokens = [set(r['top_tokens']) for r in results[:5]]
    
    # 检查前5个向量是否有重叠（如果有，可能是语法词）
    common_tokens = set.intersection(*all_tokens)
    if common_tokens:
        print(f"Common tokens across top vectors: {common_tokens}")
        print("  → Likely grammatical/function words")
    
    print("\nNext step: Manually inspect svd_decode_0.5B.txt for semantic clusters!")

if __name__ == "__main__":
    main()