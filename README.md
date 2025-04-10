# Optimizing_Vision_Transformers
Optimizing Vision Transformers

# 🔍 Efficient Vision Transformers on CIFAR-10

This repository explores techniques to compress and accelerate Vision Transformers (ViT) using:

- 🔧 Structured **Pruning**
- 🧠 **Knowledge Distillation (KD)**
- 🤖 Compact ViT architectures (reduced depth, dim, and heads)

Our experiments demonstrate significant reductions in model size, FLOPs, and power usage — while preserving or even improving classification accuracy.

---

## 📁 Contents

- `train_cifar10.py` — Baseline ViT training script on CIFAR-10
- `vitprune.py` — Structured pruning implementation with `channel_selection`
- `distill_pruned.py` — Knowledge distillation from baseline to pruned ViT
- `metrics.py` — Evaluation utilities for measuring latency, FLOPs, model size, etc.
- `README.md` — You're here!

---

## 📊 Performance Summary

### 🧪 Metrics Comparison

| Model                                              | Accuracy | Latency (ms) | Speed (samples/sec) | Model Size (MB) | Memory (MB) | FLOPs (GFLOPs) | Parameters (M) | Power (W) |
|---------------------------------------------------|----------|--------------|----------------------|------------------|-------------|----------------|----------------|-----------|
| **Baseline ViT**                                  | 78.33%   | 2.17         | 461.39               | 39.20            | 1546.10     | 0.62           | 9.75           | 28.46     |
| **Pruned 20%**                                     | 71.50%   | 1.99         | 501.33               | 34.15            | 1546.28     | 0.54           | 8.50           | 26.83     |
| **Pruned 30%**                                     | 70.93%   | 1.95         | 511.57               | 31.49            | 1559.09     | 0.49           | 7.83           | 25.18     |
| **Pruned 40%**                                     | 69.25%   | 1.94         | 516.20               | 28.24            | 1625.50     | 0.44           | 7.02           | 22.68     |
| **Pruned 40% + KD Fine-tuned**                    | 80.67%   | 1.88         | 531.63               | 28.27            | 1627.34     | 0.44           | 7.02           | 18.94     |
| **KD: Reduced Dim + Depth + Heads (Tiny ViT)**    | 77.64%   | 1.32         | 755.03               | 6.75             | 1290.65     | 0.10           | 1.66           | 7.82      |

---

### 📉 % Change Compared to Baseline

| Model                                              | Accuracy Δ | Latency Δ | Speed Δ         | Model Size Δ | Memory Δ    | FLOPs Δ     | Params Δ     | Power Δ     |
|---------------------------------------------------|------------|-----------|------------------|---------------|--------------|--------------|---------------|-------------|
| **Pruned 20%**                                     | 🔽 -8.71%  | 🔽 -8.29% | 🔼 +8.66%        | 🔽 -12.88%    | 🔼 +0.01%    | 🔽 -12.90%   | 🔽 -12.82%    | 🔽 -5.73%    |
| **Pruned 30%**                                     | 🔽 -9.44%  | 🔽 -10.14%| 🔼 +10.88%       | 🔽 -19.67%    | 🔼 +0.84%    | 🔽 -20.97%   | 🔽 -19.69%    | 🔽 -11.52%   |
| **Pruned 40%**                                     | 🔽 -11.61% | 🔽 -10.60%| 🔼 +11.88%       | 🔽 -27.96%    | 🔼 +5.14%    | 🔽 -29.03%   | 🔽 -27.90%    | 🔽 -20.32%   |
| **Pruned 40% + KD Fine-tuned**                    | 🔼 +2.99%  | 🔽 -13.36%| 🔼 +15.23%       | 🔽 -27.89%    | 🔼 +5.25%    | 🔽 -29.03%   | 🔽 -27.90%    | 🔽 -33.45%   |
| **KD: Reduced Dim + Depth + Heads**    | 🔽 -0.88%  | 🔽 -39.17%| 🔼 +63.68%       | 🔽 -82.78%    | 🔽 -16.51%   | 🔽 -83.87%   | 🔽 -82.97%    | 🔽 -72.51%   |

---

## 🧠 Key Takeaways

- ✅ **Knowledge Distillation** can recover or improve performance after pruning.
- 💡 **Reduced dim ViT with KD** deliver excellent speed/efficiency for edge deployment.
- 🔍 Our pruning method achieves up to **28% model size reduction** with minimal accuracy loss.
- 🧪 Detailed evaluation includes accuracy, latency, throughput, FLOPs, memory, and power estimation.

---