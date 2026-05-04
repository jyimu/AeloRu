# AeloRu

**A**dvanced **E**fficient **L**ow-rank **O**ptimization **R**ule **U**nified

Beyond Low-Rank: Dynamic Plasticity through Amplitude-Direction Decoupling

---

## What is AeloRu?

AeloRu is a research framework for investigating **semantic structures in PEFT weight matrices** and developing **next-generation adaptive low-rank update methods**. It combines HiRA, DoRA, Hebbian learning, and asynchronous architectures to enable real-time model alignment with dynamic plasticity.

> **Core Research Question**: How do LoRA weight matrices (W0, ΔW) encode semantic information, and how can we enhance this encoding through adaptive modulation?

---

## Research Objectives([the latest progress](https://github.com/jyimu/AeloRu/blob/main/LoRA%20%CE%94W%20Semantic%20Similarity%20Verification%20Experiment%20Report(LoRA%20%CE%94W%20%E8%AF%AD%E4%B9%89%E7%9B%B8%E4%BC%BC%E6%80%A7%E9%AA%8C%E8%AF%81%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8A).md))

| Phase | Objective | Status |
|-------|-----------|--------|
| P0 | Hidden state semantic analysis | ✅ Done([Link](https://github.com/jyimu/AeloRu/blob/main/LoRA%20%CE%94W%20Semantic%20Similarity%20Verification%20Experiment%20Report(LoRA%20%CE%94W%20%E8%AF%AD%E4%B9%89%E7%9B%B8%E4%BC%BC%E6%80%A7%E9%AA%8C%E8%AF%81%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8A).md)) |
| P1 | HiRA-DoRA fusion implementation | 📋 Planned |
| P2 | Hebbian-RL hybrid learning | 📋 Planned |
| P3 | Asynchronous PEFT architecture | 📋 Planned |

---

## Core Technical Innovations

### 1️⃣ HiRA-DoRA Fusion

**Concept:** Apply amplitude-direction decoupling to the Hadamard modulation term instead of *W_0*.

```python
# Standard HiRA
delta = 1 + A @ B

# DoRA-style decoupling on delta
magnitude = torch.abs(delta) * m      # Learnable amplitude modulation
direction = delta / (torch.abs(delta) + 1e-8)  # Normalized direction

# Fusion
W = W0 * (magnitude * direction)
```

Innovation: First to combine HiRA's multiplicative structure with DoRA's magnitude-direction separation.
4GB VRAM Adaptation: Planned optimization via 4-bit quantization of W0 with HiRA-DoRA operating in 8-bit, reducing activation memory by ~60%.

### 2️⃣ Hebbian-RL Hybrid Learning

Concept: Detect similar inputs and amplify gradients based on historical success patterns.
``` python
# Hebbian memory bank
similarity = hebb_bank.find_similar(input_signature)

if similarity > 0.85:
    # Gradient amplification
    grad = grad * (1.0 + similarity)
    
    # Direction correction toward historical success
    historical_dir = hebb_bank.get_success_direction(input_signature)
    grad = 0.7 * grad + 0.3 * historical_dir
```

**Innovation**: Real-time fusion of Hebbian plasticity with RL gradients for adaptive learning.

maybe can 4GB VRAM Adaptation: Memory bank will use CPU offloading with async prefetch; only top-k similar entries cached in GPU memory (configurable k ≤ 16).

---

### 3️⃣ Asynchronous PEFT Architecture

**Concept**: Decouple inference (front-end, read-only) from training (back-end, read-write).

```
┌─────────────────────────────────────────────────────┐
│  Front Model (Read-Only)  │  Back Model (Read-Write)│
│  → Inference              │  → Gradient Updates     │
│  → Zero Latency           │  → Hebbian Enhancement  │
└─────────────────────────────────────────────────────┘
              ↓ Periodic Sync (Non-blocking)
```

**Innovation**: Adapter-level asynchrony, not full-model duplication.

---

### 4️⃣ Dynamic Memory Lifecycle

**Concept**: Reinforce successful patterns, forget failed ones.

| Operation | Condition | Action |
|-----------|-----------|--------|
| Store | reward > 0.5 | Save gradient direction |
| Reinforce | similarity > 0.85 | Amplify gradient |
| Decay | no activation (N steps) | Reduce memory weight |
| Prune | success_rate < threshold | Remove from bank |

**Innovation**: Adaptive memory with explicit lifecycle management.

---

## Research Methodology

### W0 Semantic Analysis Pipeline

```
1. Extract W0 from attention layers (q_proj, k_proj, v_proj, o_proj)
2. Apply SVD: W0 = U @ Σ @ Vh
3. Probe singular vectors with semantic tests
4. Compare W0 vs ΔW semantic directions
5. Analyze orthogonality between W0 and ΔW
```

### Evaluation Metrics

| Metric | Target | Baseline |
|--------|--------|----------|
| Accuracy | 56%+ | 55.0% (SOTA) |
| Convergence Speed | Faster than DoRA | DoRA (r=32) |
| Catastrophic Forgetting | <5% degradation | LoRA ~15% |
| Inference Throughput | +20% vs sync | Standard LoRA |

---

## Experimental Configuration

### Primary Setup

| Component | Configuration |
|-----------|---------------|
| Base Model | Qwen2.5-1.5B / Qwen3-1.5B |
| Adapter Rank | r=32 |
| Training Device | Single GPU (3060/4060 or higher) |
| Semantic Categories | 8-10 classes (expanded from 4) |
| Samples per Category | 50-100 |

### Baseline Comparisons

- Standard LoRA (r=32)
- DoRA (r=32)
- HiRA (r=32)
- Paper 2512.23165 Best Result (55.0%)

---

## Technical Roadmap

| Version | Feature | Timeline |
|---------|---------|----------|
| v0.1.0 | HiRA-DoRA fusion core | 2 weeks |
| v0.2.0 | Hebbian memory system | 4 weeks |
| v0.3.0 | Asynchronous architecture | 6 weeks |
| v0.4.0 | W0 semantic analysis tools | 8 weeks |
| v1.0.0 | Paper-ready release | 12 weeks |

---

## Expected Contributions

1. **First HiRA-DoRA fusion** with amplitude-direction decoupling on modulation terms
2. **Hebbian-RL hybrid learning** for real-time adaptive plasticity
3. **W0 semantic structure analysis** revealing weight-semantics relationships
4. **Adapter-level asynchrony** enabling zero-latency serving during continuous learning

---


## License

MIT License - see [LICENSE](LICENSE) for details.
