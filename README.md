# AeloRu

**A**sync **e**dge **Lo**w-**R**ank **U**pdate

Real-time learning on consumer GPUs.

---

## What is AeloRu?

AeloRu is a lightweight, asynchronous low-rank update framework designed for edge devices with limited computational resources (e.g., RTX 3050 4GB). It enables real-time, continuous learning without the memory overhead of traditional fine-tuning methods.

&gt; *Inspired by Qelys.*

---

## Key Features

- 🚀 **Asynchronous Updates**: Decoupled inference and training for zero-latency serving
- 💾 **Memory Efficient**: LoRU (Low-Rank Update) reduces trainable parameters by 99%+
- ⚡ **Edge Optimized**: Designed for consumer GPUs with ≤4GB VRAM
- 🔌 **Plug & Play**: Drop-in memory module for any Transformer-based model

---

## Coming Soon

- [ ] Synchronous dual-buffer LoRU (v0.1.0) / 同步双缓冲 LoRU
- [ ] Asynchronous multi-process architecture (v0.2.0) / 异步多进程架构
- [ ] C++ extension for zero-copy synchronization (v0.3.0) / C++ 扩展零拷贝同步
- [ ] Production-ready release (v1.0.0) / 生产就绪版本
---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

&lt;p align="center"&gt;&lt;i&gt;AeloRu: Async Edge LoRU — where Qelys begins.&lt;/i&gt;&lt;/p&gt;
