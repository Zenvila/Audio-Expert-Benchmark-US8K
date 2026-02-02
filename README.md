# 🎧 UrbanSound8K Expert Benchmark: CNNs vs. Transformers
**A comprehensive benchmarking study of SOTA audio feature extractors (AST, PANNs, BEATs, YAMNet) on the UrbanSound8K dataset.**

---

## 🚀 Overview
The core purpose of this project is to evaluate **Frozen Feature Extraction**—the practice of using massive, pre-trained "Expert" models as backbones and training lightweight downstream classifiers on top. This measures the inherent discriminative power of different architectures (CNNs vs. Transformers) for urban sound classification.

### 🏆 The Leaderboard (Fold 10)
| Model | Accuracy | Architecture | Feature Dim | Result |
| :--- | :--- | :--- | :--- | :--- |
| **AST** | **90.44%** | Transformer | 768 | **Champion** |
| **PANNs** | **86.62%** | CNN-14 | 2048 | **Strong CNN** |
| **YAMNet** | **80.17%** | MobileNet-V1 | 1024 | **Efficient** |
| **BEATs** | **20.91%** | Transformer | 64 | **Failed** |

---

## 📊 Dataset
This benchmark utilizes the **UrbanSound8K** dataset, which contains 8,732 labeled sound excerpts of urban sounds from 10 classes.
- **Official Dataset Link:** [UrbanSound8K (NYU)](https://urbansounddataset.weebly.com/urbansound8k.html)
- **Classes:** Air Conditioner, Car Horn, Children Playing, Dog Bark, Drilling, Engine Idling, Gun Shot, Jackhammer, Siren, and Street Music.

---

## 🛠️ Methodology
1. **Feature Extraction:** Pre-trained weights for each model were loaded. Audio files were passed through the models to extract high-dimensional embeddings.
2. **Standardized Probe:** To ensure a fair comparison, a 3-layer MLP (Multi-Layer Perceptron) with **Batch Normalization** and **Dropout (0.3)** was used as the classifier for all models.
3. **Evaluation:** Following official dataset standards, **Fold 10** was utilized as the hold-out test set, with Folds 1-9 used for training.



## 🧪 Key Insights
* **Transformer Dominance:** The **Audio Spectrogram Transformer (AST)** significantly outperformed CNN-based models, proving that self-attention mechanisms are superior for capturing temporal context in audio.
* **Feature Compression:** The low accuracy for **BEATs** (20.91%) was identified as a "feature collapse" due to the extraction of compressed 64-D vectors rather than the full 768-D hidden states.
* **CNN Efficiency:** **PANNs** remains a highly robust choice for scenarios where CNN-based architectures are preferred for their noise-handling capabilities.

---

## 📂 Project Structure
```text
├── embeddings/           # (Ignored) Pre-extracted .npy features
├── models/
│   └── checkpoints/      # (Ignored) SOTA Model weights (Cnn14.pth, etc.)
├── extract_ast.py        # Feature extraction script for AST
├── extract_panns.py      # Feature extraction script for PANNs
├── extract_beats.py      # Feature extraction script for BEATs
├── extract_yamnet.py     # Feature extraction script for YAMNet
├── benchmark_competition.py # Main training & evaluation engine
└── generate_confusion_matrix.py # Error analysis & visualization
