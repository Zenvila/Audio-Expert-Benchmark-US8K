import os
import torch
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# --- 1. IMPORT LOCAL ARCHITECTURE ---
# This uses the BEATs.py file in your D: drive
from BEATs import BEATs, BEATsConfig

# --- 2. SETUP ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- BEATs Expert (Offline Mode) on: {torch.cuda.get_device_name(0)} ---")

BASE_DIR = r"D:\UrbanSound-Feature-Benchmark"
CSV_PATH = os.path.join(BASE_DIR, "UrbanSound8K_Data", "metadata", "UrbanSound8K.csv")
AUDIO_ROOT = os.path.join(BASE_DIR, "UrbanSound8K_Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "embeddings", "beats")
CHECKPOINT = os.path.join(BASE_DIR, "models", "checkpoints", "BEATs_iter3_plus_AS2M.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 3. LOAD MODEL MANUALLY ---
print(f"Loading weights from: {CHECKPOINT}")
# weights_only=False fix for PyTorch 2.6
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
cfg = BEATsConfig(checkpoint['cfg'])
model = BEATs(cfg)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Starting extraction for {len(df)} files...")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="BEATs Processing"):
        file_path = os.path.join(AUDIO_ROOT, f"fold{row['fold']}", row['slice_file_name'])
        if not os.path.exists(file_path): continue

        try:
            # BEATs preprocesses internally, but needs 16kHz input
            wav, _ = librosa.load(file_path, sr=16000, mono=True)
            wav_tensor = torch.from_numpy(wav[None, :]).float().to(device)
            
            # Create the required padding mask (all zeros)
            padding_mask = torch.zeros(1, wav_tensor.shape[1]).bool().to(device)
            
            with torch.no_grad():
                # Extract features (returns a list, we want the first element)
                # representation shape: [Batch, Time, 768]
                representation = model.extract_features(wav_tensor, padding_mask=padding_mask)[0]
                
                # Mean pool across the time dimension to get a single 768-D vector
                embedding = representation.mean(dim=1).cpu().numpy()
                
                # Save as .npy
                np.save(os.path.join(OUTPUT_DIR, f"{row['slice_file_name']}.npy"), embedding[0])
        except Exception as e:
            continue

    print(f"\n[SUCCESS] BEATs Extraction Complete. Files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()