import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- SETTINGS ---
BASE_DIR = r"D:\UrbanSound-Feature-Benchmark"
CSV_PATH = os.path.join(BASE_DIR, "UrbanSound8K_Data", "metadata", "UrbanSound8K.csv")
EMBEDDING_ROOT = os.path.join(BASE_DIR, "embeddings")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Official UrbanSound8K Class Names
CLASSES = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 
           'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

# Classifier must match your benchmark architecture
class AudioClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super(AudioClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
    def forward(self, x): return self.net(x)

class EmbeddingDataset(Dataset):
    def __init__(self, df, model_name):
        self.df = df
        self.model_name = model_name
        self.valid_indices = [i for i in range(len(df)) if os.path.exists(os.path.join(EMBEDDING_ROOT, model_name, f"{df.iloc[i]['slice_file_name']}.npy"))]

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx):
        row = self.df.iloc[self.valid_indices[idx]]
        path = os.path.join(EMBEDDING_ROOT, self.model_name, f"{row['slice_file_name']}.npy")
        return torch.tensor(np.load(path).reshape(-1)).float(), torch.tensor(row['classID']).long()

def run_analysis():
    df = pd.read_csv(CSV_PATH)
    # Use Fold 10 for the test set to match benchmark results
    train_df = df[df['fold'] != 10]
    test_df = df[df['fold'] == 10]

    train_loader = DataLoader(EmbeddingDataset(train_df, 'ast'), batch_size=64, shuffle=True)
    test_loader = DataLoader(EmbeddingDataset(test_df, 'ast'), batch_size=64)

    model = AudioClassifier(768).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training AST for Confusion Matrix analysis...")
    for epoch in range(20):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds = torch.argmax(model(x.to(DEVICE)), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    # Generate Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
    plt.title('AST Expert: UrbanSound8K Confusion Matrix (Fold 10)')
    
    output_path = os.path.join(BASE_DIR, "results", "plots", "ast_confusion_matrix.png")
    plt.savefig(output_path)
    print(f"[SUCCESS] Matrix saved to: {output_path}")

if __name__ == "__main__":
    run_analysis()