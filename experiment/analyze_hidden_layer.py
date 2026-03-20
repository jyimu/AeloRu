#!/usr/bin/env python3
# analyze_hidden_layer.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def main():
    model_path = "models/Qwen2.5-0.5B"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        local_files_only=True,
        trust_remote_code=True,
        output_hidden_states=True  # 关键：获取隐藏层
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    
    # 测试：输入一些有意义的句子，看隐藏层激活
    test_sentences = [
        "The dog is running in the park",  # 动物/动作
        "One plus two equals three",         # 数学
        "I feel happy today",                # 情感
        "The function returns a value",      # 编程
    ]
    
    print("Analyzing hidden layer representations...")
    print("="*60)
    
    for sent in test_sentences:
        inputs = tokenizer(sent, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
            # 取最后一层的 hidden states [batch, seq_len, hidden_dim]
            last_hidden = outputs.hidden_states[-1]  # [1, seq_len, 896]
            
            # 对每个 token 的 hidden vector 分析
            for i, token_id in enumerate(inputs['input_ids'][0]):
                token = tokenizer.decode([token_id])
                vec = last_hidden[0, i]  # [hidden_dim]
                
                # 找到这个 vector 在 W_q (查询投影) 中的响应
                # 或者：直接看 vec 本身的 SVD 分量
                
                print(f"\nToken: '{token}'")
                print(f"  Vector norm: {vec.norm().item():.2f}")
                
                # 简单分析：与哪些概念词最相关
                # （需要预定义概念向量，这里简化）
    
    # 更好的方法：直接分析 Attention 层的 W_q, W_k, W_v
    print("\n" + "="*60)
    print("Analyzing Attention Weight Matrices...")
    
    # 取中间层的 Q 投影
    layer_idx = 12  # 中间层
    attn = model.model.layers[layer_idx].self_attn
    
    W_q = attn.q_proj.weight  # [hidden_dim, hidden_dim]
    print(f"\nLayer {layer_idx} W_q shape: {W_q.shape}")
    
    # 对 W_q 做 SVD
    U, S, V = torch.svd(W_q.float())
    
    print(f"\nTop 10 singular values: {S[:10].tolist()}")
    
    # 关键：分析 V 的向量（输入空间）
    # 但 hidden_dim=896，无法直接解码为 token
    
    # 替代：用 lm_head 投影回 vocab 空间
    lm_head = model.lm_head.weight  # [vocab_size, hidden_dim]
    
    for i in range(10):
        v_i = V[:, i]  # 第 i 个右奇异向量 [hidden_dim]
        
        # 投影到 vocab 空间：lm_head @ v_i
        logits = lm_head @ v_i  # [vocab_size]
        
        top_values, top_indices = torch.topk(logits.abs(), k=15)
        
        tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
        
        print(f"\n--- Hidden Vector {i} (σ={S[i].item():.4f}) ---")
        print(f"  Tokens: {tokens[:10]}")
        
        # 人工判断是否有语义主题

if __name__ == "__main__":
    main()