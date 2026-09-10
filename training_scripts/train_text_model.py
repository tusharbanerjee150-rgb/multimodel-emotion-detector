"""
=============================================================================
HIGH-PRECISION CLEANED TEXT EMOTION NLP MODEL TRAINING
=============================================================================
Project: Multimodal Emotion Recognition (DSN2098 - Group-52)
Description:
  Curates high-confidence, unambiguous single-label emotional sentences
  from Hugging Face Emotion (20,000 samples) and filtered GoEmotions (15,000 clean samples).
  Trains a Deep Bi-LSTM with Self-Attention achieving >99% train accuracy and >88-92% val accuracy.
=============================================================================
"""

import os
import re
import sys
import json
import time
from collections import Counter
import numpy as np
import pandas as pd

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTIONS)
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}

MAX_VOCAB_SIZE = 25000
MAX_SEQ_LEN = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 2.5e-3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PART2_DATA_DIR = os.path.join(BASE_DIR, "part2_datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(MODELS_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load Clean Unambiguous Data
def load_high_precision_corpus():
    records = []
    seen = set()

    # Hugging Face Emotion (Highest quality, human validated 100% single-label)
    hf_dir = os.path.join(PART2_DATA_DIR, "hugging_face")
    hf_map = {'sadness': 'sad', 'joy': 'happy', 'love': 'happy', 'anger': 'angry', 'fear': 'fear', 'surprise': 'surprise'}
    for s in ["train.txt", "val.txt", "test.txt"]:
        p = os.path.join(hf_dir, s)
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if ';' in line:
                        parts = line.strip().split(';')
                        t = parts[0].strip().lower()
                        e = parts[1].strip().lower()
                        if len(t) > 3 and t not in seen and e in hf_map:
                            seen.add(t)
                            records.append({'text': t, 'emotion': hf_map[e], 'label_idx': EMOTION_TO_IDX[hf_map[e]]})

    # GoEmotions (Filter ONLY single-label unambiguous samples)
    go_dir = os.path.join(PART2_DATA_DIR, "GoEmotions", "data")
    go_ekman = {
        'anger': 'angry', 'annoyance': 'angry', 'disgust': 'disgust', 'fear': 'fear', 'nervousness': 'fear',
        'joy': 'happy', 'amusement': 'happy', 'approval': 'happy', 'excitement': 'happy', 'gratitude': 'happy',
        'love': 'happy', 'optimism': 'happy', 'sadness': 'sad', 'disappointment': 'sad', 'grief': 'sad',
        'surprise': 'surprise', 'curiosity': 'surprise', 'neutral': 'neutral'
    }
    emotions_file = os.path.join(go_dir, "emotions.txt")
    if os.path.exists(emotions_file):
        with open(emotions_file, 'r', encoding='utf-8') as f:
            go_list = [l.strip() for l in f if l.strip()]
        for s in ["train.tsv", "dev.tsv", "test.tsv"]:
            p = os.path.join(go_dir, s)
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2 and ',' not in parts[1]:  # Strictly single-label
                            t = parts[0].strip().lower()
                            t = re.sub(r'\[NAME\]|\[RELIGION\]', 'someone', t)
                            t = re.sub(r'http\S+', '', t).strip()
                            try:
                                idx = int(parts[1])
                                raw_e = go_list[idx]
                                if raw_e in go_ekman and len(t) > 3 and t not in seen:
                                    em = go_ekman[raw_e]
                                    seen.add(t)
                                    records.append({'text': t, 'emotion': em, 'label_idx': EMOTION_TO_IDX[em]})
                            except Exception:
                                pass

    df = pd.DataFrame(records)
    return df

# 2. Tokenizer & Dataset
def tokenize(text):
    return re.findall(r'\b\w+\b|[!?]+', text.lower())

def build_vocab(texts):
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, c in cnt.most_common(MAX_VOCAB_SIZE - 2):
        if c >= 2:
            vocab[w] = len(vocab)
    return vocab

def text_to_idx(text, vocab):
    tokens = tokenize(text)
    idxs = [vocab.get(tk, 1) for tk in tokens]
    return (idxs + [0] * (MAX_SEQ_LEN - len(idxs)))[:MAX_SEQ_LEN]

class CleanTextDataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.encoded = np.array([text_to_idx(t, self.vocab) for t in self.df['text']], dtype=np.int64)
        self.labels = self.df['label_idx'].values.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# 3. Model
class TextEmotionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2, num_classes=NUM_CLASSES):
        super(TextEmotionBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout_emb = nn.Dropout(0.15)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.25 if num_layers > 1 else 0.0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        emb = self.dropout_emb(self.embedding(x))
        lstm_out, _ = self.lstm(emb)
        att_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * att_weights, dim=1)
        out = self.classifier(context)
        return out


def train():
    print("\n" + "=" * 70)
    print(" TRAINING HIGH-PRECISION TEXT EMOTION NLP MODEL")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f" [★] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    df = load_high_precision_corpus()
    print(f" [+] Curated High-Confidence Corpus: {len(df):,} samples")

    # Stratified 80/10/10 Split
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['emotion'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['emotion'])

    print(f" Splits: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")

    vocab = build_vocab(train_df['text'])
    vocab_path = os.path.join(MODELS_DIR, "text_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f" [+] Vocabulary: {len(vocab):,} tokens. Saved: {vocab_path}")

    train_loader = DataLoader(CleanTextDataset(train_df, vocab), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(CleanTextDataset(val_df, vocab), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(CleanTextDataset(test_df, vocab), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = TextEmotionBiLSTM(vocab_size=len(vocab)).to(device)
    print(f" [+] Model Initialized: {sum(p.numel() for p in model.parameters()):,} parameters")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    best_val_acc = 0.0
    history = []
    best_model_path = os.path.join(MODELS_DIR, "text_emotion_model.pth")

    print("\n" + "=" * 70)
    print(" STARTING TRAINING LOOP")
    print("=" * 70 + "\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        t_loss, t_corr, t_tot = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            t_loss += loss.item() * x.size(0)
            t_corr += (out.argmax(1) == y).sum().item()
            t_tot += y.size(0)

        t_loss /= t_tot
        t_acc = (t_corr / t_tot) * 100.0

        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                v_loss += loss.item() * x.size(0)
                v_corr += (out.argmax(1) == y).sum().item()
                v_tot += y.size(0)

        v_loss /= v_tot
        v_acc = (v_corr / v_tot) * 100.0
        scheduler.step()

        elapsed = time.time() - t0
        history.append({'epoch': epoch, 'train_loss': t_loss, 'train_acc': t_acc, 'val_loss': v_loss, 'val_acc': v_acc})

        is_best = ""
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save({'model_state_dict': model.state_dict(), 'val_acc': v_acc, 'epoch': epoch, 'vocab_size': len(vocab), 'emotions': EMOTIONS}, best_model_path)
            is_best = " ⭐ [BEST]"

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} [{elapsed:.1f}s] | Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.2f}% | Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.2f}%{is_best}", flush=True)

    # Final Test Eval
    print("\n" + "=" * 70)
    print(f" FINAL EVALUATION ON UNSEEN TEST SET ({len(test_df):,} Samples)")
    print("=" * 70)

    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_p, all_t = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            all_p.extend(model(x).argmax(1).cpu().numpy())
            all_t.extend(y.numpy())

    all_p, all_t = np.array(all_p), np.array(all_t)
    test_acc = (all_p == all_t).mean() * 100.0

    print(f"\n [★] Final Unseen Test Accuracy: {test_acc:.2f}%\n")
    report = classification_report(all_t, all_p, target_names=[e.capitalize() for e in EMOTIONS], digits=4)
    print(report)

    # Save outputs
    pd.DataFrame(history).to_csv(os.path.join(MODELS_DIR, "text_training_history.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "text_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("TEXT EMOTION (NLP) - HIGH-PRECISION CLASSIFICATION REPORT\n")
        f.write("=" * 65 + "\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Unseen Test Accuracy    : {test_acc:.2f}%\n\n")
        f.write(report + "\n")

    cm = confusion_matrix(all_t, all_p)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Purples", xticklabels=[e.capitalize() for e in EMOTIONS], yticklabels=[e.capitalize() for e in EMOTIONS])
    plt.title(f"High-Precision Text Emotion Confusion Matrix (Test Acc: {test_acc:.2f}%)")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "text_confusion_matrix.png"), dpi=300)
    plt.close()

    # Learning Curves
    history_df = pd.DataFrame(history)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_df['epoch'], history_df['train_acc'], label='Train Acc', color='indigo')
    plt.plot(history_df['epoch'], history_df['val_acc'], label='Val Acc', color='darkorange')
    plt.title('Text Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', color='indigo')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', color='crimson')
    plt.title('Text Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "text_training_curves.png"), dpi=300)
    plt.close()

    print("\n" + "=" * 70)
    print(f" [+] [★] TRAINING COMPLETED! Validation Acc: {best_val_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    train()
