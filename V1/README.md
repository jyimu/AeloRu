# Aeloru: LLM Real-Time Training Framework for Consumer-Grade GPUs

**Aeloru**(**A**daptive **E**lastic **L**earning with **O**rthogonal **R**obust **U**nits)
[中文README](./README_zh.md)
> **Core Philosophy**: Pretrained weights are sacrosanct and inviolable; all learning outcomes are accumulated externally.
> **Goal**: To achieve a brain-like closed-loop training cycle of "Exploration → Conflict → Anchoring → Consolidation" through cognitive conflict detection.
> **Project Repository**: [`www.github.com/jyimu/AeloRu`](www.github.com/jyimu/AeloRu)
> **Core Implementation**: [`www.github.com/jyimu/AeloRu/blob/main/V1/aeloru_layer.py`](www.github.com/jyimu/AeloRu/blob/main/V1/aeloru_layer.py)

---

## 1. Core Design Overview

Aeloru is a framework integrating multiple state-of-the-art fine-tuning techniques, designed to solve the challenge of real-time large language model (LLM) training on consumer-grade hardware. It decouples the learning process into three core components: **Sacrosanct Base**, **External Accumulation**, and **Working Memory of the Current Cycle**.

### 1.1 Core Integrated Technologies

| Module Name | Technical Principle | Function |
| :--- | :--- | :--- |
| **Hi-DoRA** | Low-rank adaptation with amplitude-direction decoupling | Applies row norm modulation to incremental weights, preserving the amplitude characteristics of pretrained weights. |
| **ReLoRA** | Periodic merge-and-reset | Merges low-rank increments into an external buffer to enable accumulation and consolidation of high-rank knowledge. |
| **Hebbian-Fisher** | Bidirectional synaptic plasticity | Fisher mask protects critical parameters (stabilization), while Hebbian learning rules record exploration traces (plasticity). |
| **Hong Wen** | Cognitive state machine | Four-phase cycle (Exploration / Hong Wen / Anchoring / Consolidation) driven by conflict score, simulating the brain-like learning process. |

### 1.2 Effective Weight Composition Formula

The inference weight $W_{eff}$ of Aeloru consists of three components, ensuring **zero forgetting** of pretrained knowledge:

$$W_{eff} = W_0 + W_{acc} + \text{Gate}(\Delta W)$$

Where:
*   $W_0$: Sacrosanct Base (Frozen Pretrained Weights)
*   $W_{acc}$: External Accumulation Buffer (Accumulated Knowledge)
*   $\Delta W$: Low-rank adapter increment of the current cycle (LoRA A & B)
*   $\text{Gate}(\cdot)$: Gating function integrating Hi-DoRA modulation, Fisher masking, and energy budget constraints

---

## 2. Core Algorithm Mechanisms

### 2.1 Hong Wen Cognitive State Machine
The framework simulates the human cognitive process with four states, driven by a conflict score (Fisher information change rate + exploration entropy):
> Closed-loop state flow: EXPLORE (Free Exploration) → RED (Cognitive Conflict / Hong Wen) → ANCHOR (Process Anchoring) → SOLID (Hebbian Consolidation) → Return to EXPLORE

1.  **EXPLORE (Free Exploration)**
    *   **Behavior**: Enables both Hebbian updates and backpropagation (BP).
    *   **Feature**: Low Fisher masking threshold, high parameter plasticity.
2.  **RED (Cognitive Conflict / Hong Wen)**
    *   **Trigger**: Conflict score exceeds the threshold (default: 0.65).
    *   **Behavior**: Pauses Hebbian updates, freezes Fisher information, and may trigger forced ReLoRA merge.
3.  **ANCHOR (Process Anchoring)**
    *   **Behavior**: Backpropagation-led learning, tightened Fisher protection.
    *   **Exit**: Transitions to consolidation phase when gradient norm converges (default: < 1e-4).
4.  **SOLID (Hebbian Consolidation)**
    *   **Behavior**: Only allows Hebbian reinforcement in low Fisher-sensitivity regions.
    *   **Duration**: Returns to exploration phase after a fixed number of steps (default: 200 steps).

### 2.2 Regularization and Constraints
*   **Orthogonal Penalty**
    *   Additional term to the loss function: $L_{ortho} = \lambda ||\Delta W^T \cdot W_0||_F^2$
    *   **Purpose**: Forces $\Delta W$ to fall into the left null space of $W_0$, avoiding damage to the pretrained manifold.
*   **Energy Budget**
    *   Hard constraint: $||\Delta W||_F \le \eta \cdot ||W_0||_F$
    *   **Purpose**: Prevents fine-tuning increments from overwhelming the pretrained weights and compromising model stability.

---

## 3. Core Formula System and Design Intent

The architecture design of Aeloru strictly follows mathematical definitions, with each formula corresponding to explicit cognitive computing logic. Below is the analysis of academic mapping and design intent for the core formulas.

### 3.1 Table of Four Core Formulas

| Core Formula | Academic Highlights | Design Intent |
| :--- | :--- | :--- |
| **1. Low-rank Increment**<br>$\Delta W = \frac{\alpha}{r} \cdot B A$ | **Ecosystem Compatibility**: Fully aligned with the standard scaling paradigm of LoRA, ensuring compatibility with the existing PEFT ecosystem and enabling seamless replacement of existing LoRA modules. | **Working Memory Isolation**: Serves as the "working memory of the current cycle", separating each round of increments from historically accumulated weights, laying the mathematical foundation for subsequent ReLoRA merging and Hong Wen anchoring. |
| **2. Hi-DoRA Modulation**<br>$\Delta W' = \text{diag}(m) \cdot \Delta W$ | **Amplitude Decoupling**: Replicates the core weight decomposition idea of DoRA via row-level amplitude modulation, while maintaining the lightweight incremental form with lower VRAM usage and more stable training than the original DoRA. | **Adaptive Elasticity**: Adds adaptive amplitude control to low-rank increments to avoid invalid updates, perfectly aligning with the core positioning of "adaptive elastic learning". |
| **3. Fisher Gating**<br>$\Delta W'' = \Delta W' \odot \frac{1}{1 + \gamma F}$ | **Soft Gating Protection**: A top-conference-recognized soft gating design, superior to hard truncation: it protects high-importance weights via Fisher information while preserving gradient backpropagation, avoiding gradient vanishing. | **Knowledge Moat**: Acts as the "core knowledge protection mechanism" to perfectly mitigate catastrophic forgetting, and forms bidirectional linkage with Hebbian trace consolidation to build the core moat of the architecture. |
| **4. Effective Weight**<br>$W_{eff} = W_0 + W_{acc} + \Delta W'''$ | **Dual-Track Structure**: The additive form is fully aligned with the non-intrusive design of LoRA, with the explicit introduction of accumulated weight $W_{acc}$ to formalize the mathematical definition of ReLoRA merging. | **Dual-Track Memory System**: Implements "long-term memory + working memory": the pretrained weight $W_0$ remains intact, long-term memory is consolidated in $W_{acc}$, and current learning is stored in $\Delta W'''$, with fully self-consistent logic. |

### 3.2 Auxiliary Regularization Formula

In addition to the core weight calculations above, Aeloru introduces orthogonal constraints to optimize learning efficiency:

*   **Orthogonal Penalty Loss**
    $$L_{ortho} = \lambda ||\Delta W^T W_0||_F^2$$
    *   **Physical Meaning**: Penalizes the directional overlap between the incremental weight $\Delta W$ and the pretrained weight $W_0$.
    *   **Design Value**: Acts as an "automatic avoidance learning mechanism", forcing the model to explore the null space of $W_0$ so that it only learns new knowledge without wasting computing power on content already mastered by the pretrained model, making it perfectly adapted to low-computing devices.

---

## 4. Detailed Configuration Parameters (`AeloruConfig`)

You can control the behavior of the framework by adjusting the `AeloruConfig`.

### 4.1 Basic Dimensions
*   `in_features`, `out_features`: Input and output dimensions of the layer
*   `r`: LoRA Rank, controlling the width of the adapter
*   `lora_alpha`: LoRA scaling factor
*   `LoRA_lr`: Dedicated learning rate for LoRA

### 4.2 Feature Toggles
*   `use_hidora`: Enable/disable amplitude modulation
*   `use_relora`: Enable/disable external accumulation and merging
*   `use_hebbian`: Enable/disable online Hebbian updates
*   `use_fisher`: Enable/disable Fisher cognitive masking
*   `use_hongwen`: Enable/disable cognitive state machine
*   `use_orthogonal_penalty`: Enable/disable orthogonal loss calculation
*   `use_energy_budget`: Enable/disable hard energy constraint

### 4.3 Key Hyperparameters
*   **ReLoRA**: `merge_every` (merge cycle steps), `merge_on_red` (forced merge on Hong Wen state)
*   **Hebbian**: `hebbian_lr` (Hebbian learning rate), `hebbian_decay` (forgetting decay factor)
*   **Fisher**: `fisher_gamma` (mask sharpness), `fisher_ema` (smoothing coefficient)
*   **Hong Wen**: `red_threshold` (Hong Wen state trigger threshold), `solid_steps` (consolidation phase steps)

---

## 5. Code Workflow Examples

### 5.1 Model Injection
Use the `inject_aeloru` function to recursively replace target layers (e.g., `q_proj`, `v_proj`) in the model.

```python
from aeloru_layer import inject_aeloru, AeloruConfig

# Define configuration
config = AeloruConfig(
    r=8, 
    lora_alpha=4.0,
    use_hidora=True,
    use_relora=True,
    use_hongwen=True
)

# Inject into the model
model = inject_aeloru(model, target_names=["q_proj", "v_proj"], cfg=config)
```

### 5.2 Training Steps
Use the encapsulated `train_aeloru_step` for single-step training, which automatically handles state machine transitions and ReLoRA merging.

```python
from aeloru_layer import train_aeloru_step

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for x, y_target in dataloader:
    # Automatically handle forward pass, state detection, backpropagation, and merging logic
    loss, metrics = train_aeloru_step(
        layer=model.your_aeloru_layer, 
        x=x, 
        y_target=y_target, 
        optimizer=optimizer
    )
    print(f"State: {metrics['state']}, Loss: {metrics['loss_total']}")
```

### 5.3 Saving and Loading
Aeloru supports independent adapter saving, storing only incremental parameters.

```python
# Save
layer.save_adapter("checkpoints/aeloru_step1000.pt")

# Load
layer.load_adapter("checkpoints/aeloru_step1000.pt")
```

---

## 6. Conclusion

**Aeloru** is far more than a fine-tuning algorithm: it is a **cognitive computing architecture**. By combining the high-rank accumulation capability of **ReLoRA**, the parameter efficiency of **Hi-DoRA**, the bio-inspired stability of **Hebbian-Fisher**, and the state machine control of **Hong Wen**, it provides a highly promising solution for the sustained training of large language models with limited computing power.