"""
=============================================================================
SPEECH EMOTION RECOGNITION (SER) MODEL TRAINING
=============================================================================
Project: Multimodal Emotion Recognition (DSN2098 - Group-52)
Architecture: Hybrid Deep CRNN (2D-CNN + BiLSTM + Attention Head)
Target Emotions (7 classes):
  ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
Hardware: NVIDIA GeForce RTX 5050 Laptop GPU (CUDA)
=============================================================================
"""

import os
import sys
import time
import math
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from concurrent.futures import ThreadPoolExecutor

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTIONS)
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}

SAMPLE_RATE = 22050
DURATION = 3.0  # seconds
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)  # 66,150 audio samples
N_MELS = 128
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 512
MAX_TIME_FRAMES = 130  # ~3.0s / (512/22050)

BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_AUDIO_DIR = os.path.join(BASE_DIR, "part2_datasets", "cleaned", "audio")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(MODELS_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# FEATURE EXTRACTION & AUDIO PREPROCESSING
# =============================================================================

def extract_features_from_audio(filepath, augment=False):
    """
    Loads an audio file and extracts a combined (168, 130) feature representation:
      - 128 Log-Mel Spectrogram bands
      - 40 MFCC coefficients
    """
    try:
        full_path = os.path.join(BASE_DIR, filepath) if not os.path.isabs(filepath) else filepath
        y, sr = sf.read(full_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # Pad or trim to fixed duration (3 seconds)
        if len(y) < TARGET_LENGTH:
            pad_width = TARGET_LENGTH - len(y)
            y = np.pad(y, (0, pad_width), mode='constant')
        else:
            y = y[:TARGET_LENGTH]

        # Data augmentation for training
        if augment:
            # 1. Random noise injection
            if np.random.rand() < 0.4:
                noise_amp = 0.005 * np.random.rand() * np.max(np.abs(y))
                y = y + noise_amp * np.random.normal(size=len(y))
            # 2. Random pitch shift
            if np.random.rand() < 0.3:
                n_steps = np.random.randint(-2, 3)
                y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=n_steps)

        # 1. Log-Mel Spectrogram (128 x Time)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # 2. MFCCs (40 x Time)
        mfcc = librosa.feature.mfcc(
            y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
        )

        # Stack features: (168, Time)
        combined = np.vstack([log_mel, mfcc])  # (168, time_steps)

        # Ensure fixed width (MAX_TIME_FRAMES = 130)
        if combined.shape[1] < MAX_TIME_FRAMES:
            pad_len = MAX_TIME_FRAMES - combined.shape[1]
            combined = np.pad(combined, ((0, 0), (0, pad_len)), mode='constant')
        else:
            combined = combined[:, :MAX_TIME_FRAMES]

        # Standardize (Z-score normalize)
        mean = np.mean(combined)
        std = np.std(combined) + 1e-6
        norm_features = (combined - mean) / std

        return norm_features.astype(np.float32)

    except Exception as e:
        print(f"Error extracting {filepath}: {e}")
        return np.zeros((N_MELS + N_MFCC, MAX_TIME_FRAMES), dtype=np.float32)


# =============================================================================
# DATASET & CACHING
# =============================================================================

class SpeechEmotionDataset(Dataset):
    def __init__(self, df, is_train=False, cache_name=None):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.cache_file = os.path.join(CLEANED_AUDIO_DIR, f"{cache_name}.npz") if cache_name else None
        
        self.features = []
        self.labels = self.df['label_idx'].values.astype(np.int64)

        self._load_or_extract()

    def _load_or_extract(self):
        if self.cache_file and os.path.exists(self.cache_file):
            print(f" [+] Loading cached features from: {os.path.basename(self.cache_file)}...")
            data = np.load(self.cache_file)
            self.features = data['features']
            self.labels = data['labels']
            print(f"     -> Loaded {len(self.features)} cached audio feature maps.")
            return

        print(f" [+] Extracting acoustic features for {len(self.df)} samples (multithreaded)...")
        t0 = time.time()

        def process_row(idx):
            row = self.df.iloc[idx]
            feats = extract_features_from_audio(row['filepath'], augment=False)
            return idx, feats

        extracted = [None] * len(self.df)
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(process_row, range(len(self.df)))
            for idx, feats in results:
                extracted[idx] = feats

        self.features = np.array(extracted, dtype=np.float32)
        elapsed = time.time() - t0
        print(f"     -> Extraction finished in {elapsed:.1f}s ({len(self.features)} samples).")

        if self.cache_file:
            np.savez_compressed(self.cache_file, features=self.features, labels=self.labels)
            print(f"     -> Saved cache: {os.path.basename(self.cache_file)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feat = self.features[idx]
        # On the fly SpecAugment for training
        if self.is_train and np.random.rand() < 0.5:
            feat = feat.copy()
            # Frequency masking
            f_mask = np.random.randint(0, 15)
            f_start = np.random.randint(0, feat.shape[0] - f_mask)
            feat[f_start:f_start+f_mask, :] = 0
            # Time masking
            t_mask = np.random.randint(0, 15)
            t_start = np.random.randint(0, feat.shape[1] - t_mask)
            feat[:, t_start:t_start+t_mask] = 0

        tensor_feat = torch.from_numpy(feat).unsqueeze(0)  # (1, 168, 130)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor_feat, label


# =============================================================================
# MODEL ARCHITECTURE: DEEP SPEECH EMOTION CRNN
# =============================================================================

class SpeechEmotionCRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SpeechEmotionCRNN, self).__init__()

        # Feature Extractor (2D CNN)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d((2, 1)),  # pool frequency only, keep time resolution
            nn.Dropout2d(0.3)
        )

        # Temporal Sequence Modeling (BiLSTM)
        # After 4 pools: Frequency = 168 // (2*2*2*2) = 10 bands, Channels = 256
        self.rnn_input_dim = 256 * 10
        self.lstm = nn.LSTM(
            input_size=self.rnn_input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Self-Attention Pooling Layer
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (Batch, 1, 168, 130)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)  # (Batch, 256, 10, Time)

        b, c, f, t = c4.size()
        # Reshape for RNN: (Batch, Time, c * f)
        rnn_in = c4.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # BiLSTM
        lstm_out, _ = self.lstm(rnn_in)  # (Batch, Time, 256)

        # Attention weights across time
        att_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (Batch, Time, 1)
        context = torch.sum(lstm_out * att_weights, dim=1)  # (Batch, 256)

        out = self.classifier(context)  # (Batch, num_classes)
        return out


# =============================================================================
# TRAINING & EVALUATION PIPELINE
# =============================================================================

def train_speech_model():
    print("\n" + "=" * 70)
    print(" SPEECH EMOTION RECOGNITION (SER) TRAINING PIPELINE")
    print("=" * 70)

    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dev_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        dev_name = "CPU"
    print(f" [★] Compute Device: {device} ({dev_name})")
    print(f" [★] PyTorch Version: {torch.__version__}")
    print("=" * 70 + "\n")

    # 2. Load Split CSVs
    train_csv = os.path.join(CLEANED_AUDIO_DIR, "audio_train.csv")
    val_csv = os.path.join(CLEANED_AUDIO_DIR, "audio_val.csv")
    test_csv = os.path.join(CLEANED_AUDIO_DIR, "audio_test.csv")

    if not os.path.exists(train_csv):
        print(f" [!] Error: Cleaned audio files not found. Run clean_part2_datasets.py first.")
        return

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    print(f" Dataset Splits: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # 3. Create Datasets & Loaders
    train_dataset = SpeechEmotionDataset(train_df, is_train=True, cache_name="train_features")
    val_dataset = SpeechEmotionDataset(val_df, is_train=False, cache_name="val_features")
    test_dataset = SpeechEmotionDataset(test_df, is_train=False, cache_name="test_features")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # 4. Model, Loss, Optimizer
    model = SpeechEmotionCRNN(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n [★] Initialized SpeechEmotionCRNN ({total_params:,} trainable parameters)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    best_val_acc = 0.0
    history = []
    best_model_path = os.path.join(MODELS_DIR, "speech_emotion_model.pth")

    print("\n" + "=" * 70)
    print(" STARTING TRAINING LOOP")
    print("=" * 70 + "\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        t_start = time.time()

        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * feats.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = (train_correct / train_total) * 100.0

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                outputs = model(feats)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * feats.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = (val_correct / val_total) * 100.0
        scheduler.step()

        elapsed = time.time() - t_start
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        is_best = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'emotions': EMOTIONS
            }, best_model_path)
            is_best = " ⭐ [BEST]"

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} [{elapsed:.1f}s] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%{is_best}",
            flush=True
        )

    # Save training history
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(MODELS_DIR, "speech_training_history.csv")
    history_df.to_csv(history_csv, index=False)
    print(f"\n [+] Saved training history: {history_csv}")

    # =========================================================================
    # FINAL EVALUATION ON UNSEEN TEST SPLIT
    # =========================================================================
    print("\n" + "=" * 70)
    print(" EVALUATING ON UNSEEN TEST DATASET (424 Audio Files)")
    print("=" * 70)

    # Load best checkpoint
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for feats, labels in test_loader:
            feats = feats.to(device)
            outputs = model(feats)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_acc = (all_preds == all_targets).mean() * 100.0

    print(f"\n [★] Fresh Unseen Test Accuracy: {test_acc:.2f}%\n")

    # Classification Report
    report = classification_report(all_targets, all_preds, target_names=[e.capitalize() for e in EMOTIONS], digits=4)
    print(report)

    report_path = os.path.join(RESULTS_DIR, "speech_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("SPEECH EMOTION RECOGNITION (SER) - CLASSIFICATION REPORT\n")
        f.write("=" * 65 + "\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Unseen Test Accuracy    : {test_acc:.2f}%\n\n")
        f.write(report + "\n")
    print(f" [+] Saved classification report: {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[e.capitalize() for e in EMOTIONS],
        yticklabels=[e.capitalize() for e in EMOTIONS]
    )
    plt.title(f"Speech Emotion Confusion Matrix (Test Acc: {test_acc:.2f}%)")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.tight_layout()
    cm_img_path = os.path.join(RESULTS_DIR, "speech_confusion_matrix.png")
    plt.savefig(cm_img_path, dpi=300)
    plt.close()
    print(f" [+] Saved confusion matrix: {cm_img_path}")

    # Accuracy / Loss Curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_df['epoch'], history_df['train_acc'], label='Train Accuracy', color='royalblue')
    plt.plot(history_df['epoch'], history_df['val_acc'], label='Val Accuracy', color='darkorange')
    plt.title('Speech Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', color='royalblue')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', color='crimson')
    plt.title('Speech Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, "speech_training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()
    print(f" [+] Saved training curves: {curves_path}")

    print("\n" + "=" * 70)
    print(" [✓] SPEECH EMOTION MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print(f" Best Checkpoint: {best_model_path}")
    print(f" Final Test Acc : {test_acc:.2f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    train_speech_model()
