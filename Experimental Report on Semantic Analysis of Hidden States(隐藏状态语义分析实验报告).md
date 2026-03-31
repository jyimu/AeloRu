# 📊 Experimental Report on Semantic Analysis of Hidden States(隐藏状态语义分析实验报告)
[中文](https://github.com/jyimu/AeloRu/blob/main/experiment/log/logMD/Experimental%20Report%20on%20Semantic%20Analysis%20of%20Hidden%20States_zh.md)

> **Experiment ID**: `hidden_states_analysis_v3_20260319_132449`  
> **Experiment Time**: 2026-03-19 13:24:49 ~ 13:27:29  
> **Total Duration**: 160.11 seconds  
> **Running Device**: CUDA (PyTorch 2.6.0+cu126)

---

## 1️⃣ Experiment Overview

### 1.1 Experiment Objective

To investigate whether semantic structure information is encoded in the hidden states of the **Qwen2.5-1.5B** large language model, and to identify the optimal hidden state extraction strategy.

### 1.2 Experiment Configuration

| Configuration Item | Value |
|-------------------|-------|
| Model | `Qwen2.5-1.5B` |
| Model Path | `models/Qwen2.5-1.5B` |
| Running Device | `CUDA` |
| Samples per Category | 20 |
| Semantic Categories | `animal`, `code`, `emotion`, `number` |
| **Total Samples** | **80** |

### 1.3 Comparison Strategies

The experiment compared **4 hidden state extraction strategies**:

| Strategy ID | Strategy Name | Token Strategy | Layer Strategy |
|-------------|---------------|----------------|----------------|
| S1 | `Last_Token_Last_Layer` | Last token | Last layer |
| S2 | `Mean_Token_Last_Layer` | All tokens averaged | Last layer |
| S3 | `Last_Token_All_Layers_Mean` | Last token | All layers averaged |
| S4 | `Mean_Token_All_Layers_Mean` | All tokens averaged | All layers averaged |

---

## 2️⃣ Core Experimental Results

### 2.1 Strategy Comparison Summary Table

| Rank | Strategy | Separation Ratio | Silhouette Score | PCA Cumulative Variance (10D) | Intra-class Distance | Inter-class Distance |
|:----:|----------|:----------------:|:----------------:|:-----------------------------:|:--------------------:|:--------------------:|
| 🥇 | **Mean_Token_Last_Layer** | **2.303** | **0.503** | 76.18% | 30.06 | 69.22 |
| 🥈 | Last_Token_Last_Layer | 1.854 | 0.431 | 66.99% | 43.44 | 80.52 |
| 🥉 | Last_Token_All_Layers_Mean | 1.687 | 0.396 | 46.58% | 20.52 | 34.61 |
| 4️⃣ | Mean_Token_All_Layers_Mean | 1.467 | 0.305 | 99.72% | 348.88 | 511.70 |

---

### 2.2 Best Strategy Detailed Metrics

**🏆 Best Strategy: `Mean_Token_Last_Layer`**

![Best](experiment/log/hidden_states_analysis_v3_20260319_132449_Mean_Token_Last_Layer.png)

#### Clustering Quality Metrics

| Metric | Value | Evaluation |
|--------|-------|------------|
| Separation Ratio | 2.303 | ✅ Good (>2.0) |
| Silhouette Score | 0.503 | ✅ Above Average (>0.5) |
| PCA Top 2D Cumulative Variance | 51.11% | 🟡 Moderate |
| PCA Top 10D Cumulative Variance | 76.18% | ✅ Good |

#### Intra-class Distance

| Category | Intra-class Distance | Compactness Evaluation |
|----------|---------------------|------------------------|
| `emotion` | 24.15 | ✅ Most compact |
| `animal` | 25.03 | ✅ Compact |
| `number` | 28.89 | 🟡 Moderate |
| `code` | 42.15 | 🔴 Most dispersed |

**Average Intra-class Distance**: 30.06

#### Inter-class Distance

| Category Pair | Distance | Distinguishability |
|---------------|----------|-------------------|
| `code` ↔ `emotion` | 101.11 | 🔴 Most distinguishable |
| `code` ↔ `animal` | 89.47 | 🔴 Easily distinguishable |
| `code` ↔ `number` | 84.66 | 🔴 Easily distinguishable |
| `emotion` ↔ `number` | 53.57 | 🟡 Moderate |
| `animal` ↔ `emotion` | 45.51 | 🟡 Moderate |
| `animal` ↔ `number` | 40.99 | 🟡 Relatively close |

**Average Inter-class Distance**: 69.22

---

## 3️⃣ Key Findings

### 3.1 Finding 1: Last Layer Outperforms All Layers Averaged

| Comparison Item | Last Layer | All Layers Averaged | Gap |
|-----------------|------------|---------------------|-----|
| Separation Ratio (Mean Token) | 2.303 | 1.467 | **+57%** |
| Silhouette Score (Mean Token) | 0.503 | 0.305 | **+65%** |

**Conclusion**: Deep semantic information is concentrated in the last layer; averaging all layers dilutes the semantic signal.

---

### 3.2 Finding 2: Token Averaging Outperforms Last Token

| Comparison Item | Mean Token | Last Token | Gap |
|-----------------|------------|------------|-----|
| Separation Ratio (Last Layer) | 2.303 | 1.854 | **+24%** |
| Silhouette Score (Last Layer) | 0.503 | 0.431 | **+17%** |

**Conclusion**: Averaging all tokens better captures overall sentence meaning; Last Token may lose some information.

---

### 3.3 Finding 3: Code Category Specificity

| Metric | Code | Average of Other Categories | Difference |
|--------|------|----------------------------|------------|
| Intra-class Distance | 42.15 | 26.02 | **+62%** |
| Average Distance to Other Classes | 91.75 | 46.69 | **+96%** |

**Conclusion**: 
- Code has the largest semantic representation variability (large intra-class distance)
- Code differs most significantly from other semantic categories (large inter-class distance)
- The specificity of the code category may stem from its broad and abstract semantic scope, leading to greater dispersion and uniqueness in the hidden state encoding.

---

### 3.4 Finding 4: PCA Variance Trap

The `Mean_Token_All_Layers_Mean` strategy has a PCA cumulative variance as high as **99.72%**, but the clustering effect is the worst:

| Strategy | PCA 10D Variance | Separation Ratio | Silhouette Score |
|----------|-----------------|------------------|------------------|
| Mean_Token_All_Layers_Mean | 99.72% | 1.467 | 0.305 |
| Mean_Token_Last_Layer | 76.18% | 2.303 | 0.503 |

**Conclusion**: High variance explanation rate ≠ good semantic separation; overly dispersed information is actually detrimental to clustering.

---

## 4️⃣ Experiment Limitations

### 4.1 Deviation from Research Objectives

| Current Experiment | Original Research Objective |
|-------------------|----------------------------|
| Analyze hidden states | Analyze LoRA weight matrix W0 |
| Focus on output representations | Focus on the structure of weights themselves |
| Cluster semantic categories | Association between weights and semantics |

**⚠️ Important Note**: This experiment **did not directly study the W0 weights of LoRA**, but instead analyzed the model's hidden state representations. Although the results are valid, there is a deviation from the core research question of "the relationship between LoRA's W0 and semantics."

---

### 4.2 Other Limitations

| Limitation | Impact | Recommendation |
|------------|--------|----------------|
| Only 4 semantic categories | Limited generalizability of conclusions | Extend to 8-10 categories |
| Only 1.5B small model | Large models may perform differently | Validate on 7B/14B models |
| 20 samples per category | Moderate statistical stability | Increase to 50-100 samples |
| Only Chinese/English mixed | Language differences not controlled | Test separately by language |

---

## 5️⃣ Conclusions and Recommendations

### 5.1 Core Conclusions

1. ✅ **Hidden states do encode semantic structure**: Separation ratio 2.30, silhouette score 0.50, proving semantic categories are separable in hidden space.

2. ✅ **Optimal extraction strategy identified**: `Mean_Token_Last_Layer` (last layer + all tokens averaged)

3. ✅ **Layer selection is more important than token selection**: The gap between last layer vs. all layers averaged > the gap between token strategies

4. ⚠️ **Experiment did not directly answer the W0 semantic question**: Additional experiments are needed to analyze the weight matrix itself

---

### 5.2 Follow-up Experiment Recommendations

| Priority | Experiment Direction | Objective |
|:--------:|---------------------|-----------|
| 🔴 P0 | W0 Weight SVD Analysis | Directly study the semantic structure of the weight matrix |
| 🔴 P0 | Before/After LoRA Training Comparison | Analyze the semantic direction relationship between ΔW and W0 |
| 🟡 P1 | Extend Semantic Categories | Validate generalizability of conclusions |
| 🟡 P1 | Multi-Model Scale Comparison | 1.5B → 7B → 14B |
| 🟢 P2 | Downstream Task Validation | Use clustering results for classification tasks |

---

## 6️⃣ Experiment Outputs

| File Type | File Path |
|-----------|-----------|
| Log File | `AeLoRu/experiment/log/hidden_states_analysis_v3_20260319_132449.log` |
| JSON Results | `AeLoRu/experiment/log/hidden_states_analysis_v3_20260319_132449.json` |
| Visualization 1 | `../hidden_states_analysis_v3_20260319_132449_Last_Token_Last_Layer.png` |
| Visualization 2 | `../hidden_states_analysis_v3_20260319_132449_Mean_Token_Last_Layer.png` |
| Visualization 3 | `../hidden_states_analysis_v3_20260319_132449_Last_Token_All_Layers_Mean.png` |
| Visualization 4 | `../hidden_states_analysis_v3_20260319_132449_Mean_Token_All_Layers_Mean.png` |

---

## 7️⃣ Appendix: Raw Data Summary

**Best Strategy**: `Mean_Token_Last_Layer`

```json
{
  "session_id": "hidden_states_analysis_v3_20260319_132449",
  "total_steps": 320,
  "total_time": 160.10299825668335,
  "device": "cuda",
  "best_strategy": "Mean_Token_Last_Layer",
  "best_separation_ratio": 2.30297132670672,
  "best_silhouette_score": 0.5029744263236783
}

