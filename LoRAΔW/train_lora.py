import torch
import torch.nn as nn
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os

# ==================== 配置 ====================
class LoRAConfig:
    model_name = "models/Qwen2.5-0.5B"
    r = 16  # 低秩维度
    alpha = 32
    dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    epochs = 30
    batch_size = 16
    lr = 2e-4
    output_dir = "./AeloRu/experiment/LoRA/lora_checkpoints"

# ==================== 数据准备 ====================
def create_semantic_dataset():
    """创建多类别语义数据集"""
    with open("AeloRu\experiment\\tool\SENTENCES_V3.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = []
    for category, texts in data.items():
        for text in texts:
            samples.append({"text": text, "category": category})
    
    return Dataset.from_list(samples)

# ==================== LoRA 模型加载 ====================
def load_lora_model(config, checkpoint_path=None):
    """加载预训练模型 + LoRA"""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        # 加载已有 LoRA 权重
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
        print(f"✅ 已加载 LoRA 权重：{checkpoint_path}")
    else:
        # 初始化新 LoRA
        lora_config = LoraConfig(
            r=config.r,
            lora_alpha=config.alpha,
            target_modules=config.target_modules,
            lora_dropout=config.dropout,
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        print("✅ 已初始化新 LoRA 模型")
        model.print_trainable_parameters()
    
    return model, tokenizer

# ==================== 训练函数 ====================
def train_lora(model, tokenizer, dataset, config):
    """训练 LoRA 模型"""
    from transformers import TrainingArguments, Trainer
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=64
        )
        # 添加 labels，用于计算 loss
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.lr,
        save_steps=100,
        logging_steps=10,
        save_total_limit=2
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )
    
    trainer.train()
    model.save_pretrained(f"{config.output_dir}/final")
    tokenizer.save_pretrained(f"{config.output_dir}/final")
    
    print(f"✅ 训练完成，模型保存至：{config.output_dir}/final")
    return model

# ==================== 主函数 ====================
if __name__ == "__main__":
    config = LoRAConfig()
    dataset = create_semantic_dataset()
    model, tokenizer = load_lora_model(config)
    model = train_lora(model, tokenizer, dataset, config)