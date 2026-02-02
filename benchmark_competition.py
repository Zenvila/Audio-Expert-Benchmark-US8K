import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- 1. SETTINGS & PATHS ---
BASE_DIR = r"D:\UrbanSound-Feature-Benchmark"
CSV_PATH = os.path.join(BASE_DIR, "UrbanSound8K_Data", "metadata", "UrbanSound8K.csv")
EMBEDDING_ROOT = os.path.join(BASE_DIR, "embeddings")
MODELS = ['yamnet', 'panns', 'ast', 'beats']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Feature dimensions verified by your previous runs
DIMS = {'yamnet': 1024, 'panns': 2048, 'ast': 768, 'beats': 64}

# --- 2. THE DOWNSTREAM CLASSIFIER ---
class AudioClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(AudioClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# --- 3. ROBUST DATASET LOADER ---
class EmbeddingDataset(Dataset):
    def __init__(self, df, model_name):
        self.df = df
        self.model_name = model_name
        self.valid_indices = self._find_valid_files() # Pre-filter missing files

    def _find_valid_files(self):
        valid = []
        for i in range(len(self.df)):
            base_name = self.df.iloc[i]['slice_file_name']
            p1 = os.path.join(EMBEDDING_ROOT, self.model_name, f"{base_name}.npy")
            p2 = os.path.join(EMBEDDING_ROOT, self.model_name, f"{base_name}.wav.npy")
            if os.path.exists(p1) or os.path.exists(p2):
                valid.append(i)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        base_name = row['slice_file_name']
        
        # Check both naming conventions
        p1 = os.path.join(EMBEDDING_ROOT, self.model_name, f"{base_name}.npy")
        p2 = os.path.join(EMBEDDING_ROOT, self.model_name, f"{base_name}.wav.npy")
        path = p1 if os.path.exists(p1) else p2
        
        # Load and force 1D shape (Fixes the BEATs mat1/mat2 error)
        embedding = np.load(path).reshape(-1)
        
        # Final Shape Guard: Ensure embedding matches the expected Linear input
        expected_dim = DIMS[self.model_name]
        if embedding.shape[0] != expected_dim:
            if embedding.shape[0] > expected_dim:
                embedding = embedding[:expected_dim]
            else:
                # Pad with zeros if it's somehow shorter
                tmp = np.zeros(expected_dim)
                tmp[:embedding.shape[0]] = embedding
                embedding = tmp
                
        return torch.tensor(embedding).float(), torch.tensor(row['classID']).long()

# --- 4. COMPETITION LOGIC ---
def train_and_eval(model_name, input_dim):
    df = pd.read_csv(CSV_PATH)
    
    # Fold 10 test split
    train_df = df[df['fold'] != 10]
    test_df = df[df['fold'] == 10]

    train_loader = DataLoader(EmbeddingDataset(train_df, model_name), batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(EmbeddingDataset(test_df, model_name), batch_size=64, num_workers=0)

    model = AudioClassifier(input_dim).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n--- Competing: {model_name.upper()} ---")
    for epoch in range(20):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/20] complete.")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            preds = torch.argmax(model(x), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
    
    return accuracy_score(all_labels, all_preds)

# --- 5. RUN SHOWDOWN ---
if __name__ == "__main__":
    results = {}
    os.makedirs(os.path.join(BASE_DIR, "results", "plots"), exist_ok=True)

    for m in MODELS:
        try:
            acc = train_and_eval(m, DIMS[m])
            results[m] = acc
            print(f"  Final Accuracy for {m.upper()}: {acc:.4f}")
        except Exception as e:
            print(f"  Failed {m}: {e}")

    # --- 6. PLOT ---
    if results:
        plt.figure(figsize=(10, 6))
        colors = ['skyblue', 'salmon', 'lightgreen', 'plum']
        bars = plt.bar(results.keys(), results.values(), color=colors[:len(results)])
        plt.ylabel('Test Accuracy (Fold 10)')
        plt.title('UrbanSound8K: Expert Benchmark Results')
        plt.ylim(0, 1.0)
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2%}", ha='center', va='bottom')

        plt.savefig(os.path.join(BASE_DIR, "results", "plots", "final_benchmark_results.png"))
        print(f"\n[SUCCESS] Benchmark Complete! Check your 'results/plots' folder.")