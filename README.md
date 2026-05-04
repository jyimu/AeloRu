# AeloRu

**A**daptive **E**lastic **L**earning with **O**rthogonal **R**obust **U**nits

Beyond Low-Rank: Dynamic Plasticity through Amplitude-Direction Decoupling

---

## What is AeloRu?

AeloRu is a research framework for investigating **semantic structures in PEFT weight matrices** and developing **next-generation adaptive low-rank update methods**. It combines HiRA, DoRA, Hebbian learning, and asynchronous architectures to enable real-time model alignment with dynamic plasticity.

> **Core Research Question**: How can integrating Hebbian Learning principles with LoRA-based parameter-efficient fine-tuning mitigate catastrophic forgetting during long-term sequential task learning, while enhancing adaptation speed in short-term scenarios?

---

## Research Objectives([the latest progress](./V1/README.md))

| Phase | Objective | Status |
|-------|-----------|--------|
| P0 | Hidden state semantic analysis | ✅ Done([Link](https://github.com/jyimu/AeloRu/blob/main/LoRA%20%CE%94W%20Semantic%20Similarity%20Verification%20Experiment%20Report(LoRA%20%CE%94W%20%E8%AF%AD%E4%B9%89%E7%9B%B8%E4%BC%BC%E6%80%A7%E9%AA%8C%E8%AF%81%E5%AE%9E%E9%AA%8C%E6%8A%A5%E5%91%8A).md)) |
| P1 | HiRA-DoRA fusion implementation | ✅ Done([Link](./V1/README.md)) |
| P2 | Hebbian-RL hybrid learning | ✅ Done([Link](./V1/README.md)) |
| P3 | Training on real data & writing paper | 🔄 Doing |
| P4 | Asynchronous PEFT architecture | 📋 Planned |

---

## Expected Contributions

1. **First HiRA-DoRA fusion** with amplitude-direction decoupling on modulation terms
2. **Hebbian-RL hybrid learning** for real-time adaptive plasticity
3. **Adapter-level asynchrony** enabling zero-latency serving during continuous learning

---


## License

MIT License - see [LICENSE](LICENSE) for details.
