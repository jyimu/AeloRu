import sys
import os
sys.path.append('..')  # 添加父目录到路径

# 设置离线模式，避免网络连接
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import datasets
from aeloru_layer import inject_hidora, HiDoRALayer

# 配置
MODEL_NAME = "models/Qwen2.5-0.5B"  # 基础模型路径
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GSM8K 数据集类
class GSM8KDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']

        # 构建输入文本
        input_text = f"Question: {question}\nAnswer:"

        # 编码
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        labels = self.tokenizer(answer, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")['input_ids']

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

# 提取答案的函数
def extract_answer(text):
    # 从生成的文本中提取最后的数字
    numbers = re.findall(r'\d+\.?\d*', text)
    return float(numbers[-1]) if numbers else None

# 评估函数
def evaluate(model, dataloader, dataset, tokenizer):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=MAX_LENGTH,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

            for i in range(len(outputs)):
                generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
                predicted_answer = extract_answer(generated_text)

                # 获取真实答案
                idx = batch_idx * BATCH_SIZE + i
                if idx < len(dataset):
                    true_answer_text = dataset[idx]['answer']
                    true_answer = extract_answer(true_answer_text)

                    if predicted_answer is not None and true_answer is not None and abs(predicted_answer - true_answer) < 1e-6:
                        total_correct += 1
                total_samples += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy

# 训练函数
def train(model, train_dataloader, val_dataloader, val_dataset, optimizer, scheduler, tokenizer, epochs=3):
    model.train()
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_loss)

        # 验证
        val_accuracy = evaluate(model, val_dataloader, val_dataset, tokenizer)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")

    return train_losses, val_accuracies

# 主函数
def main():
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载 GSM8K 数据集
    dataset = datasets.load_dataset("AeloRu\data\gsm8k", "main")
    train_data = dataset['train']
    test_data = dataset['test']

    # 创建数据集
    train_dataset = GSM8KDataset(train_data, tokenizer, MAX_LENGTH)
    test_dataset = GSM8KDataset(test_data, tokenizer, MAX_LENGTH)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

    # 注入 Hi-DoRA
    model = inject_hidora(model, target_module_name="mlp", r=32, alpha=8.0)

    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=len(train_dataloader) * EPOCHS)

    # 训练
    train_losses, val_accuracies = train(model, train_dataloader, test_dataloader, test_data, optimizer, scheduler, tokenizer, EPOCHS)

    # 可视化
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    print("Training completed. Results saved to training_results.png")

if __name__ == "__main__":
    main()