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
- `vit_knowledge_distillation.py` — Knowledge distillation from baseline to pruned ViT
- `vit_evaluation.py` — Evaluation utilities for measuring latency, FLOPs, model size, etc.
- `README.md` — You're here!

---

## 📊 Performance Summary

### 🧪 Metrics Comparison

| Model         | Accuracy | CPU Mem Diff | GPU Mem Diff | Latency (ms) | Speed (samples/sec) | FLOPs       | Parameters     | Avg GPU Power (W) |
|---------------|----------|--------------|---------------|---------------|----------------------|-------------|----------------|--------------------|
| baseline_vit  | 0.8035   | 0.00 MB      | 1158.83 MB    | 10.85         | 9,218.05             | 0.62 GFLOPs | 9.75 Million   | 68.42              |
| Pruned_20     | 0.7192   | 0.00 MB      | 794.02 MB     | 8.35          | 11,975.93            | 0.54 GFLOPs | 8.50 Million   | 68.75              |
| Pruned_30     | 0.7133   | 0.00 MB      | 762.45 MB     | 5.91          | 16,916.51            | 0.49 GFLOPs | 7.83 Million   | 67.36              |
| Pruned_40     | 0.7038   | 0.00 MB      | 726.25 MB     | 5.98          | 16,714.57            | 0.44 GFLOPs | 7.02 Million   | 69.00              |
| Pruned_40_KD  | 0.8081   | 0.00 MB      | 726.25 MB     | 5.50          | 18,165.42            | 0.44 GFLOPs | 7.02 Million   | 67.01              |
| Student_KD    | 0.7645   | 0.00 MB      | 407.26 MB     | 4.60          | 21,731.78            | 0.10 GFLOPs | 1.66 Million   | 64.48              |
---

### 📉 % Change Compared to Baseline
| Model         | Accuracy (%) | CPU Mem Diff (%) | GPU Mem Diff (%) | Latency (%) | Speed (%) | FLOPs (%) | Parameters (%) | Avg GPU Power (%) |
|---------------|--------------|------------------|-------------------|-------------|------------|------------|-----------------|--------------------|
| Pruned_20     | -10.50       | 0.00             | -31.48            | -23.05      | +29.89     | -12.90     | -12.82          | +0.48              |
| Pruned_30     | -11.23       | 0.00             | -34.19            | -45.53      | +83.52     | -20.97     | -19.69          | -1.55              |
| Pruned_40     | -12.42       | 0.00             | -37.33            | -44.87      | +81.31     | -29.03     | -27.95          | +0.85              |
| Pruned_40_KD  | +0.57        | 0.00             | -37.33            | -49.30      | +97.02     | -29.03     | -27.95          | -2.06              |
| Student_KD    | -4.86        | 0.00             | -64.84            | -57.58      | +135.74    | -83.87     | -82.97          | -5.75              |


---

## 🧠 Key Takeaways

- ✅ **Knowledge Distillation** can recover or improve performance after pruning.
- 💡 **Reduced dim ViT with KD** deliver excellent speed/efficiency for edge deployment.
- 🧪 Detailed evaluation includes accuracy, latency, throughput, FLOPs, memory, and power estimation.

---
