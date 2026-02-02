import os
import torch
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

# --- 1. DEFINE CNN14 ARCHITECTURE ---
def init_layer(layer):
    if layer.weight.ndimension() == 4:
        nn.init.xavier_uniform_(layer.weight)
    elif layer.weight.ndimension() == 2:
        nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        init_layer(self.conv1); init_layer(self.conv2)

    def forward(self, x):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, 2)

class Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):
        super(Cnn14, self).__init__()
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        init_layer(self.fc1); init_layer(self.fc_audioset)

    def forward(self, input):
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        embedding = F.relu_(self.fc1(x))
        return embedding

# --- 2. GPU EXECUTION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- PANNs Expert on: {torch.cuda.get_device_name(0)} ---")

BASE_DIR = r"D:\UrbanSound-Feature-Benchmark"
CSV_PATH = os.path.join(BASE_DIR, "UrbanSound8K_Data", "metadata", "UrbanSound8K.csv")
AUDIO_ROOT = os.path.join(BASE_DIR, "UrbanSound8K_Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "embeddings", "panns")
CHECKPOINT = os.path.join(BASE_DIR, "models", "checkpoints", "Cnn14.pth")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, 
              fmin=50, fmax=14000, classes_num=527).to(device)

# --- FIX: Set weights_only=False to allow loading of older PANNs checkpoint ---
checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model'])
model.eval()

def main():
    if not os.path.exists(CSV_PATH):
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} files. Starting GPU Extraction...")

    for index, row in tqdm(df.iterrows(), total=len(df), desc="PANNs CNN14"):
        file_name = row['slice_file_name']
        fold = f"fold{row['fold']}"
        full_path = os.path.join(AUDIO_ROOT, fold, file_name)
        
        if not os.path.exists(full_path):
            continue

        try:
            wav, _ = librosa.load(full_path, sr=32000, mono=True)
            wav = torch.tensor(wav[None, :]).float().to(device)
            
            with torch.no_grad():
                embedding = model(wav).cpu().numpy()
                save_path = os.path.join(OUTPUT_DIR, f"{file_name}.npy")
                np.save(save_path, embedding[0])
                
        except Exception:
            continue

    print(f"\n[SUCCESS] PANNs features saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()