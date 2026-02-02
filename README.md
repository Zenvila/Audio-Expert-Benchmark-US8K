# 🎧 UrbanSound8K Expert Benchmark: CNNs vs. Transformers
**A comprehensive benchmarking study of SOTA audio feature extractors (AST, PANNs, BEATs, YAMNet) on the UrbanSound8K dataset.**

---

## 🚀 Overview
This project evaluates the performance of four "Expert" audio models by using them as frozen-backbone feature extractors. By training a standardized downstream classifier on these embeddings, we measure the inherent quality and discriminative power of each architecture.

### 🏆 The Leaderboard (Fold 10)
| Model | Accuracy | Architecture | Feature Dim | Result |
| :--- | :--- | :--- | :--- | :--- |
| **AST** | **90.44%** | Transformer | 768 | **Champion** |
| **PANNs** | **86.62%** | CNN-14 | 2048 | **Strong CNN** |
| **YAMNet** | **80.17%** | MobileNet-V1 | 1024 | **Efficient** |
| **BEATs** | **20.91%** | Transformer | 64 | **Failed** |

---

## 🛠️ Methodology
1. **Feature Extraction:** Pre-trained weights for each model were loaded. Audio files from **UrbanSound8K** were passed through the models to extract high-dimensional embeddings.
2. **Standardized Probe:** A 3-layer MLP (Multi-Layer Perceptron) with **Batch Normalization** and **Dropout (0.3)** was used for all models to ensure a fair comparison.
3. **Evaluation:** Following official standards, **Fold 10** was used as the unseen test set, while Folds 1-9 were used for training.



## 🧪 Key Insights
* **Transformer Superiority:** The **Audio Spectrogram Transformer (AST)** significantly outperformed CNN-based models, proving its ability to capture long-range temporal dependencies in urban environments.
* **The BEATs Bottleneck:** The low accuracy (20.91%) for BEATs was identified as a "feature collapse" due to the extraction of compressed 64-D vectors rather than the full 768-D hidden states.
* **PANNs Robustness:** PANNs remains the most reliable CNN architecture, showing high resilience across stationary and impulsive noise classes.

---

## 📂 Project Structure
```text
├── embeddings/           # (Ignored) Pre-extracted .npy features
├── models/
│   └── checkpoints/      # (Ignored) SOTA Model weights (AST, PANNs, BEATs)
├── extract_ast.py        # Feature extraction script for AST
├── extract_panns.py      # Feature extraction script for PANNs
├── extract_beats.py      # Feature extraction script for BEATs
├── extract_yamnet.py     # Feature extraction script for YAMNet
├── benchmark_competition.py # Main training & evaluation engine
└── generate_confusion_matrix.py # Error analysis & visualization
