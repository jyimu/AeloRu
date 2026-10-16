# ex1主实验3种子汇总 & 整体结论
> 种子：`9589 / 6519 /1412`，配置统一：Qwen2.5‑0.5B，r=8，α=16，5任务sst2→mrpc→qnli→rte→cola，每任务250步。

|方法|ACC均值±std|BWT均值±std|Forgetting均值±std|
|---|---:|---:|---:|
|LoRA|0.5832 ± 0.0181|−0.0319 ± 0.0030|0.0437 ± 0.0074|
|Aeloru‑safe(PEM关闭)|0.5635 ± 0.0038|−0.0141 ± 0.0072|0.0259 ± 0.0022|
|Aeloru‑default(PEM开启)|0.5748 ± 0.0060|−0.0114 ± 0.0104|0.0299 ± 0.0055|

## 关键观测
1. **Forgetting（遗忘）效果高度稳定**
- LoRA：0.0437；Aeloru‑safe：0.0259，Aeloru‑default：0.0299
- 相对LoRA遗忘下降 **31%‑41%**，三个种子全部一致：**无论default/safe，Aeloru遗忘显著低于LoRA，这个趋势完全不受种子扰动影响**。
- safe（关闭PEM）遗忘略优于default（开启PEM），再次印证低秩LLM微调场景PEM带来轻微负面代价，和ex2/ex3机制实验完全自洽。

2. **ACC平均准确率受种子噪声影响大**
- LoRA均值0.5832，Aeloru‑safe 0.5635，Aeloru‑default 0.5748；
- 精度上没有稳定优于LoRA；部分种子高、部分种子低，波动覆盖了差异。
> 符合之前统计预判：**本协议下ACC是高噪声指标；抗遗忘（Forgetting/BWT）是稳健信号。**

3. **BWT后向迁移**
LoRA BWT稳定显著负值 `-0.0319`；Aeloru两组BWT明显抬升，后向干扰大幅缓解。

4. 时间开销：LoRA ~132s；Aeloru ~172s，额外开销约**+30%**，和之前报告接近。

## 配对t‑test简要结果（LoRA vs Aeloru‑safe）
- Forgetting：`p≈0.027 <0.05` → **遗忘降低具备统计显著性**
- ACC：`p≈0.172 >0.05` → 平均准确率差异不显著。

> ✅核心论证定位：**论文不靠ACC提升讲故事；核心证据是「遗忘/BWT显著改善 + 全套几何解耦指标(ex2‑ex4)」，ACC仅作为辅助参考。**

## 论文正文片段（实验4.1 + limitation）
> We repeated the main concept‑drift continual‑learning experiment over three distinct random seeds {9589, 6519, 1412}.
> For downstream metrics, **the reduction of catastrophic forgetting is statistically significant (paired t‑test, \(p<0.05\))**. While Aeloru consistently mitigates forgetting and improves backward transfer, the overall average accuracy shows no statistically‑significant difference compared with vanilla LoRA under this small‑sample GLUE protocol, due to large seed‑to‑seed variance.
> Geometric decoupling metrics from ex2‑ex4 are robust across seeds and form our primary mechanistic evidence. Ablation and判决性 experiments validate the scene‑dependent property of PEM: it brings net negative effects under low‑rank LLM fine‑tuning, and we adopt `aeloru‑safe` (PEM‑off) as our recommended configuration for PEFT continual learning.

>中文版本：
>我们使用3个不同随机种子{9589,6519,1412}重复主概念漂移持续学习实验。下游指标中，灾难性遗忘的降低具备统计显著性（配对t‑检验，p<0.05）。虽然Aeloru可以稳定减轻遗忘、改善后向迁移，但在本小样本GLUE评测协议下，受较大的种子间方差影响，整体平均准确率对比标准LoRA没有统计显著差异。来自ex2‑ex4的几何解耦指标跨种子稳健，构成本文主要机制证据。消融与判决实验证实PEM具备强场景依赖性：在低秩大模型微调条件下表现净负效应，因此本文PEFT持续学习推荐配置为`aeloru‑safe`（关闭PEM）。

## 📝现在整套工作状态总结
1. ex1：主性能 ✅（3种子，遗忘统计显著；ACC如实报告不显著）
2. ex2：跨秩消融，解耦几何 ✅
3. ex3：耦合分解判决实验，证伪PEM“选择性去冗余”假设 ✅
4. ex4：从零训练MLP，完整PEM空间‑尺度边界，含gamma剂量扫描、发散诊断 ✅
5. 第6章：移植到Transformer的13项阻断缺陷复盘 ✅
6. 完整图表脚本、附录产物清单 ✅
