import os
import time
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

# --- 1. SYSTEM SETUP ---
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --- 2. CONFIGURATION (STRICT WINDOWS PATHS) ---
BASE_DIR = r"D:\UrbanSound-Feature-Benchmark"
CSV_PATH = os.path.join(BASE_DIR, "UrbanSound8K_Data", "metadata", "UrbanSound8K.csv")
# Crucial: Your tree showed folds are inside UrbanSound8K_Data
AUDIO_ROOT = os.path.join(BASE_DIR, "UrbanSound8K_Data") 
OUTPUT_DIR = os.path.join(BASE_DIR, "embeddings", "yamnet")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 3. LOAD MODEL ---
print("Loading Frozen YAMNet...")
model = hub.load('https://tfhub.dev/google/yamnet/1')

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} records. Starting extraction...")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        file_name = row['slice_file_name']
        fold = f"fold{row['fold']}"
        
        # Use os.path.join to handle slashes correctly on Windows
        full_path = os.path.join(AUDIO_ROOT, fold, file_name)
        
        if not os.path.exists(full_path):
            # If a file is missing, we print it once to debug
            if index < 5: 
                print(f"\nMissing: {full_path}")
            continue

        try:
            # Step A: Load and Resample
            wav, _ = librosa.load(full_path, sr=16000, mono=True)
            # Step B: Extract
            _, embeddings, _ = model(wav)
            # Step C: Average and Save
            feat = np.mean(embeddings.numpy(), axis=0)
            np.save(os.path.join(OUTPUT_DIR, f"{file_name}.npy"), feat)
        except Exception as e:
            continue

    print(f"\nExtraction complete. Check your folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()