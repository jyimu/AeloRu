# AeloRu

**A**daptive **E**lastic **L**earning with **O**rthogonal **R**obust **U**nits

Beyond Low-Rank: Dynamic Plasticity through Amplitude-Direction Decoupling

Paper: [AeloRu: Adaptive Elastic Learning with Orthogonal Robust Units](paper/paper.md)

中文版：[README_zh.md](README_zh.md)

---

## What is AeloRu?

AeloRu is a **neuroscience-inspired low-rank adaptation (PEFT) framework** that integrates multiple synergistic cognitive learning mechanisms while staying memory-efficient for consumer-grade GPUs. Rather than treating fine-tuning purely as a statistical-optimization problem, AeloRu borrows principles from biological learning — adaptive synaptic plasticity, phasic cognitive cycles, offline sleep consolidation, and volatility-adaptive learning rates — and casts them into a unified low-rank adapter architecture.

> **Core Philosophy**: Pretrained weights are sacrosanct and inviolable; all learning outcomes are accumulated externally. Effective weight: `W_eff = W_0 + W_acc + Gate(ΔW)`.

> **Core Research Question**: How can integrating Hebbian Learning principles with LoRA-based parameter-efficient fine-tuning mitigate catastrophic forgetting during long-term sequential task learning, while enhancing adaptation speed in short-term scenarios?

### Core Mechanisms

| Mechanism | Technical Principle | Function |
| :--- | :--- | :--- |
| **Hi-DoRA** | Bidirectional amplitude–direction decoupling | Independent learnable row/column amplitude vectors decouple magnitude from direction at < 0.01% extra params. |
| **Dual 1-D HGF Plasticity Gate** | Closed-form precision propagation | Reduces Fisher gating from `O(d²)` to `O(d)` via two 1-D precision vectors (π_out, π_in); zero gradient cost. |
| **HongWen State Machine** | Four-phase cognitive cycle | EXPLORE → RED (conflict) → ANCHOR → SOLID, simulating fast/slow dual-timescale biological learning. |
| **HGF Closed-form Natural Gradient** | Volatility-coupled dynamic LR | Replaces costly squared-gradient Fisher with closed-form precision; auto-adapts to concept drift. |
| **DLAM Spectral Sleep** | Offline consolidation & pruning | Spectral filtering of Hebbian traces; prunes weak connections and boosts generalization. |

---

## Key Results

Evaluated on **Qwen2.5-0.5B** under an extreme concept-drift protocol (SST-2 → MRPC → QNLI → RTE → CoLA, never revisited) at an equal parameter budget:

- **Catastrophic forgetting** reduced from LoRA's 0.0538 to **0.0076 (−85.9%)**
- **Backward Transfer (BWT)** flipped from −0.0471 to **+0.0011**
- **Final accuracy** improved by **+0.83 pp**
- **Cost**: only **+22.1%** extra training time
- **Geometry**: inter-dimension correlation −46% ~ −65%, condition number −45% ~ −69%, rank-budget utilization 43% ~ 61% → 71% ~ 85%, task selectivity −78% ~ −86%

The gains stem from eliminating low-rank subspace redundancy rather than regularization alone. The paper further decomposes feature coupling into functional vs. redundant parts, runs a PEM dose–response experiment, and documents 13 blocking defects (and fixes) found while porting cognitive-science modules onto a pretrained Transformer. (Full details in [paper.md](paper/paper.md).)

---

## Project Roadmap

| Phase | Objective | Status |
|-------|-----------|--------|
| P0 | Hidden-state semantic analysis | ✅ Done ([Report](LoRA%20ΔW%20Semantic%20Similarity%20Verification%20Experiment%20Report(LoRA%20ΔW%20%E8%AF%AD%E4%B9%89%E7%9B%B8%E4%BC%BC%E6%80%A7%E9%AA%8C%E8%AF%81%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8A).md)) |
| P1 | HiRA-DoRA fusion implementation | ✅ Done ([V1](V1/README.md)) |
| P2 | Hebbian-RL hybrid learning | ✅ Done ([V1](V1/README.md)) |
| P3 | Training on real data & writing paper | ✅ Done ([paper](paper/paper.md)) |
| P4 | Asynchronous PEFT architecture | 🔄 Doing |

### Version Progression ([latest progress](paper/paper.md))

- **V1** — Core AeloRu layer, HongWen state machine, ReLoRA merge, Hebbian-Fisher → [V1.md](V1/V1.md) · [`V1/README.md`](V1/README.md)
- **V2** — Dual 1-D HGF plasticity gating, closed-form CE gradient, PEM/DLAM integration → [V2.md](V2/V2.md) · [`V2/HGF_CE.md`](V2/HGF_CE.md)
- **V3** — Full benchmark suite (ex1–ex4), PEM dose–response, subspace-geometry analysis, 13 defect fixes → [V3.md](V3/V3.md) · [`V3/md/`](V3/md/)

---

## Repository Structure

```
AeloRu/
├── V1/                 # Core implementation: aeloru_layer.py, HongWen state machine
├── V2/                 # Dual 1-D HGF gating, closed-form gradients (experiment_1.py)
├── V3/                 # Benchmark experiments ex1–ex4, subspace-geometry analysis
│   ├── md/             # Experiment notes (ex1, ex2, ex3, fix1…)
│   └── try_ex/         # Validation & probing scripts
├── experiment/         # Hidden-state semantic analysis (P0): analyze_hidden_states*.py
├── paper/              # Paper draft (paper.md), figures, experiment data (ex1–ex4)
├── tests/              # Unit tests (test_aeloru_logging.py)
├── logs/               # Training logs
├── output/ plots/ visualization/   # Generated artifacts
└── README.md
```

---

## Getting Started

```python
from V1.aeloru_layer import inject_aeloru, AeloruConfig

config = AeloruConfig(
    r=8,
    lora_alpha=4.0,
    use_hidora=True,
    use_relora=True,
    use_hongwen=True,
)
model = inject_aeloru(model, target_names=["q_proj", "v_proj"], cfg=config)
```

See [V1/README.md](V1/README.md) for the full configuration reference (`AeloruConfig`), the single-step training loop (`train_aeloru_step`), and adapter checkpoint save/load.

---

## Expected Contributions

1. **Hebbian-RL hybrid learning** for real-time adaptive plasticity
2. **Adapter-level asynchrony** enabling zero-latency serving during continuous learning
3. **Dual 1-D HGF plasticity gating** — `O(d²)` → `O(d)` Fisher gating for consumer-grade GPUs
4. **DLAM spectral sleep** for offline consolidation and anti-forgetting

---

## License

MIT License - see [LICENSE](LICENSE) for details.
