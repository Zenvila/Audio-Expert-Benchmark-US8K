import os
import torch
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from transformers import ASTFeatureExtractor, ASTModel

# --- 1. SYSTEM SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"AST Transformer Expert on: {torch.cuda.get_device_name(0)}")

# --- 2. CONFIGURATION ---
BASE_DIR = r"D:\UrbanSound-Feature-Benchmark"
CSV_PATH = os.path.join(BASE_DIR, "UrbanSound8K_Data", "metadata", "UrbanSound8K.csv")
AUDIO_ROOT = os.path.join(BASE_DIR, "UrbanSound8K_Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "embeddings", "ast")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 3. LOAD MODEL (Hugging Face - No wget needed!) ---
print("Loading AST Transformer from Hugging Face...")
# This will download about 300MB of weights automatically via Python
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(device)
model.eval()

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} records. Starting GPU Extraction...")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="AST Transformer"):
        file_name = row['slice_file_name']
        fold = f"fold{row['fold']}"
        full_path = os.path.join(AUDIO_ROOT, fold, file_name)
        
        if not os.path.exists(full_path):
            continue

        try:
            # AST requires 16kHz
            wav, _ = librosa.load(full_path, sr=16000, mono=True)
            
            # Transformers expect a specific input format (spectrogram patches)
            inputs = feature_extractor(wav, sampling_rate=16000, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # We take the mean of the hidden states to get a single 768-D vector
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
                np.save(os.path.join(OUTPUT_DIR, f"{file_name}.npy"), embedding[0])
                
        except Exception:
            continue

    print(f"\n[SUCCESS] AST Features saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()