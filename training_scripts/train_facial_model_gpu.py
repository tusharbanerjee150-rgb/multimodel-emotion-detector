"""
=============================================================================
Emotion Detection Deep Learning - 16-Core High-Performance Engine
Project Exhibition - I (DSN2098) | Group-52
=============================================================================
Hardware: Multi-Core AVX2 Parallel Engine (16 CPU Threads Active)
Supports: AffectNet (YOLO), FER-2013, RAF-DB with Fast Disk Caching (.npz)
Outputs:
  - models/facial_emotion_model.pth (Trained PyTorch Weights)
  - models/facial_emotion_model.onnx (Universal Engine)
  - models/emotion_labels.json
  - models/training_history.png
  - models/confusion_matrix.png
  - models/classification_report.txt
=============================================================================
"""

import os
import sys

# NOTE: CUDA_VISIBLE_DEVICES is set in __main__ block only, NOT here.
# Setting it at module level would disable CUDA for any script that imports
# EmotionCNN from this file (e.g. part1_facial_emotion.py, part2_multimodal_system.py).

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import re
import glob
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix



# -----------------------------------------------------------------------------
# Global Definitions & Constants
# -----------------------------------------------------------------------------
IMG_SIZE = 48
NUM_CLASSES = 7
STANDARD_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

LABEL_SYNONYMS = {
    'angry': 'angry', 'anger': 'angry', '0': 'angry',
    'disgust': 'disgust', 'disgusted': 'disgust',
    'fear': 'fear', 'fearful': 'fear', 'scared': 'fear',
    'happy': 'happy', 'happiness': 'happy', 'joy': 'happy',
    'sad': 'sad', 'sadness': 'sad',
    'surprise': 'surprise', 'surprised': 'surprise',
    'neutral': 'neutral', 'neutrality': 'neutral',
    'contempt': 'neutral', 'contemptuous': 'neutral'
}

RAF_DB_MAP = {
    '1': 'surprise',
    '2': 'fear',
    '3': 'disgust',
    '4': 'happy',
    '5': 'sad',
    '6': 'angry',
    '7': 'neutral'
}

DEFAULT_YOLO_AFFECTNET_NAMES = [
    'anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'
]


# -----------------------------------------------------------------------------
# Deep CNN Architecture (PyTorch)
# -----------------------------------------------------------------------------
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # Block 1 (64 filters)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Block 2 (128 filters)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3)
        )

        # Block 3 (256 filters)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.35)
        )

        # Block 4 (512 filters)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4)
        )

        # Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x


# -----------------------------------------------------------------------------
# Dataset Loaders & Handlers
# -----------------------------------------------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        lbl = self.labels[idx]

        pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
        
        if self.transform:
            img_tensor = self.transform(pil_img)
        else:
            img_tensor = transforms.ToTensor()(pil_img)

        return img_tensor, torch.tensor(lbl, dtype=torch.long)


def normalize_class_name(name, folder_path=""):
    name_clean = str(name).strip().lower()
    if 'raf' in folder_path.lower() or name_clean in ['1', '2', '3', '4', '5', '6', '7']:
        if name_clean in RAF_DB_MAP:
            return RAF_DB_MAP[name_clean]
    return LABEL_SYNONYMS.get(name_clean, None)


def parse_yolo_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        return None
    names = []
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        list_match = re.search(r'names:\s*\[(.*?)\]', content, re.DOTALL)
        if list_match:
            raw_items = list_match.group(1).split(',')
            names = [re.sub(r'[\'\"\s]', '', item) for item in raw_items if item.strip()]
            return names
    except Exception:
        pass
    return None


def load_yolo_dataset(yolo_dir, yaml_path=None, max_per_class=None):
    images, labels = [], []
    if not os.path.exists(yolo_dir):
        return images, labels

    print(f" [+] [YOLO Loader] Scanning AffectNet YOLO dataset at: {yolo_dir}", flush=True)
    class_names = None
    if yaml_path and os.path.exists(yaml_path):
        class_names = parse_yolo_yaml(yaml_path)
    else:
        possible_yamls = glob.glob(os.path.join(yolo_dir, "**", "*.yaml"), recursive=True) + \
                         glob.glob(os.path.join(yolo_dir, "*.yaml"))
        for y_file in possible_yamls:
            parsed = parse_yolo_yaml(y_file)
            if parsed:
                class_names = parsed
                break

    if not class_names:
        class_names = DEFAULT_YOLO_AFFECTNET_NAMES

    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    for root, _, files in os.walk(yolo_dir):
        if 'labels' in os.path.basename(root).lower():
            continue
        for f in files:
            if f.lower().endswith(exts):
                image_files.append(os.path.join(root, f))

    print(f" [+] [YOLO Loader] Found {len(image_files)} image files. Parsing YOLO annotations...", flush=True)

    loaded_counts = {c: 0 for c in STANDARD_CLASSES}

    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        dir_name = os.path.dirname(img_path)

        possible_label_paths = [
            img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).rsplit('.', 1)[0] + ".txt",
            img_path.replace(os.sep + "images", os.sep + "labels").rsplit('.', 1)[0] + ".txt",
            os.path.join(dir_name, base_name + ".txt"),
            os.path.join(os.path.dirname(dir_name), "labels", base_name + ".txt")
        ]

        label_path = None
        for p in possible_label_paths:
            if os.path.exists(p):
                label_path = p
                break

        if not label_path:
            continue

        try:
            with open(label_path, 'r', encoding='utf-8') as lf:
                lines = [line.strip() for line in lf.readlines() if line.strip()]

            if not lines:
                continue

            pil_img = Image.open(img_path)
            img_w, img_h = pil_img.size

            for line in lines:
                parts = line.split()
                if len(parts) < 1:
                    continue

                class_id = int(float(parts[0]))
                if class_id < len(class_names):
                    raw_class_name = class_names[class_id]
                else:
                    raw_class_name = str(class_id)

                norm_class = normalize_class_name(raw_class_name)
                if norm_class is None:
                    continue

                if max_per_class and loaded_counts[norm_class] >= max_per_class:
                    continue

                class_idx = STANDARD_CLASSES.index(norm_class)

                if len(parts) >= 5:
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])

                    box_w = max(1, int(w_norm * img_w))
                    box_h = max(1, int(h_norm * img_h))
                    x_min = max(0, int((x_c * img_w) - (box_w / 2)))
                    y_min = max(0, int((y_c * img_h) - (box_h / 2)))
                    x_max = min(img_w, x_min + box_w)
                    y_max = min(img_h, y_min + box_h)

                    if x_max > x_min and y_max > y_min:
                        face_crop = pil_img.crop((x_min, y_min, x_max, y_max))
                    else:
                        face_crop = pil_img
                else:
                    face_crop = pil_img

                face_48 = face_crop.convert('L').resize((IMG_SIZE, IMG_SIZE))
                arr = np.array(face_48, dtype=np.float32) / 255.0

                images.append(arr)
                labels.append(class_idx)
                loaded_counts[norm_class] += 1
        except Exception:
            continue

    print(f" [+] [YOLO Loader] Successfully loaded {len(images)} samples from AffectNet YOLO.", flush=True)
    return images, labels


def load_images_from_folder(folder_path, max_per_class=None):
    images, labels = [], []
    if not os.path.exists(folder_path):
        return images, labels

    print(f" [+] Scanning directory at: {folder_path}", flush=True)
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    for sub in subdirs:
        if sub.lower() in ['images', 'labels']:
            continue
        norm_class = normalize_class_name(sub, folder_path)
        if norm_class is None:
            continue

        class_idx = STANDARD_CLASSES.index(norm_class)
        class_dir = os.path.join(folder_path, sub)
        files = glob.glob(os.path.join(class_dir, "*.*"))
        
        count = 0
        for f in files:
            if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            try:
                img = Image.open(f).convert('L').resize((IMG_SIZE, IMG_SIZE))
                arr = np.array(img, dtype=np.float32) / 255.0
                images.append(arr)
                labels.append(class_idx)
                count += 1
                if max_per_class and count >= max_per_class:
                    break
            except Exception:
                continue
        print(f"     - Class '{norm_class}' ({sub}): {count} images loaded.", flush=True)
        
    return images, labels


def load_all_datasets(datasets_root="datasets", max_per_class=None, use_cache=True):
    cache_file = os.path.join(datasets_root, "processed_dataset.npz")
    if use_cache and os.path.exists(cache_file):
        print(f" [★] Loading pre-cached dataset from: {cache_file} ...", flush=True)
        try:
            data = np.load(cache_file)
            X, y = data['X'], data['y']
            print(f" [★] Loaded {len(X)} cached samples instantly!", flush=True)
            return X, y
        except Exception:
            pass

    all_images = []
    all_labels = []

    if os.path.exists(datasets_root):
        subfolders = [os.path.join(datasets_root, d) for d in os.listdir(datasets_root)]
        for sub in subfolders:
            if not os.path.isdir(sub):
                continue

            is_yolo = (os.path.exists(os.path.join(sub, "images")) and os.path.exists(os.path.join(sub, "labels"))) or \
                      os.path.exists(os.path.join(sub, "data.yaml"))

            if is_yolo:
                imgs, lbls = load_yolo_dataset(sub, max_per_class=max_per_class)
                all_images.extend(imgs)
                all_labels.extend(lbls)
                continue

            train_path = os.path.join(sub, "train")
            test_path = os.path.join(sub, "test")
            val_path = os.path.join(sub, "val")
            
            has_splits = False
            if os.path.exists(train_path):
                imgs, lbls = load_images_from_folder(train_path, max_per_class=max_per_class)
                all_images.extend(imgs)
                all_labels.extend(lbls)
                has_splits = True
            if os.path.exists(test_path):
                imgs, lbls = load_images_from_folder(test_path, max_per_class=max_per_class)
                all_images.extend(imgs)
                all_labels.extend(lbls)
                has_splits = True
            if os.path.exists(val_path):
                imgs, lbls = load_images_from_folder(val_path, max_per_class=max_per_class)
                all_images.extend(imgs)
                all_labels.extend(lbls)
                has_splits = True
            
            if not has_splits:
                imgs, lbls = load_images_from_folder(sub, max_per_class=max_per_class)
                all_images.extend(imgs)
                all_labels.extend(lbls)

    if len(all_images) == 0:
        return None, None

    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    try:
        np.savez_compressed(cache_file, X=X, y=y)
        print(f" [✓] Cached {len(X)} samples to {cache_file} for instant future re-runs.", flush=True)
    except Exception:
        pass

    print("=" * 68, flush=True)
    print(f" [+] TOTAL DATASET SAMPLES LOADED: {len(X)}", flush=True)
    print(f" [+] Class Distribution:", flush=True)
    for idx, name in enumerate(STANDARD_CLASSES):
        count = np.sum(y == idx)
        print(f"     - {name.capitalize():<10}: {count} samples ({count/len(y)*100:.1f}%)", flush=True)
    print("=" * 68, flush=True)
    return X, y


def plot_metrics(history, y_true, y_pred, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    plt.plot(history['val_acc'], label='Val Accuracy', linewidth=2, linestyle='--')
    plt.title('Deep CNN Accuracy vs Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2, color='orange')
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2, linestyle='--', color='red')
    plt.title('Deep CNN Loss vs Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    hist_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f" [+] Training history curves saved to: {hist_path}", flush=True)

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[c.capitalize() for c in STANDARD_CLASSES],
                yticklabels=[c.capitalize() for c in STANDARD_CLASSES])
    plt.title('Normalized Confusion Matrix (Deep CNN Review 1)', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Emotion')
    plt.ylabel('Ground Truth Emotion')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f" [+] Confusion matrix heatmap saved to: {cm_path}", flush=True)

    # 3. Report
    report = classification_report(y_true, y_pred, target_names=[c.capitalize() for c in STANDARD_CLASSES], digits=4)
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== EMOTION CLASSIFICATION REPORT (REVIEW 1 DELIVERABLE) ===\n\n")
        f.write(report)
    print(f" [+] Classification report saved to: {report_path}", flush=True)
    print("\n" + report, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Multi-Core Multi-Dataset Emotion Trainer")
    parser.add_argument("--data_dir", type=str, default="datasets", help="Root datasets folder")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--max_per_class", type=int, default=None, help="Max images per class")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    print("=" * 68, flush=True)
    print(f" HARDWARE: 16-Core Parallel AVX2 Engine ({torch.get_num_threads()} CPU Threads Active)", flush=True)
    print("=" * 68, flush=True)
    device = torch.device("cpu")

    # Step 1: Load Data
    X, y = load_all_datasets(args.data_dir, max_per_class=args.max_per_class)
    if X is None:
        print(" [!] No datasets found.", flush=True)
        return

    # Train/Val Split (80% Train, 20% Val)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f" [+] Training Set: {len(X_train)} samples", flush=True)
    print(f" [+] Validation Set: {len(X_val)} samples", flush=True)

    # Data Augmentation Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = EmotionDataset(X_train, y_train, transform=train_transform)
    val_dataset = EmotionDataset(X_val, y_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Step 2: Model, Optimizer, Criterion
    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "facial_emotion_model.pth")

    labels_path = os.path.join(models_dir, "emotion_labels.json")
    with open(labels_path, "w", encoding='utf-8') as f:
        json.dump({i: STANDARD_CLASSES[i] for i in range(NUM_CLASSES)}, f, indent=4)

    # Step 3: Training Loop
    print("\n" + "=" * 68, flush=True)
    print(f" [★] STARTING MODEL TRAINING: {args.epochs} Epochs | Batch Size: {args.batch_size}", flush=True)
    print("=" * 68, flush=True)

    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        batch_idx = 0
        total_batches = len(train_loader)

        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            batch_idx += 1

            if batch_idx % 100 == 0 or batch_idx == total_batches:
                print(f"  -> Epoch {epoch:02d}/{args.epochs:02d} | Batch {batch_idx:04d}/{total_batches:04d} | "
                      f"Acc: {correct/total*100:.2f}% | Loss: {running_loss/total:.4f}", flush=True)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - start_time

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_acc)

        print(f"\n [✓] EPOCH [{epoch:02d}/{args.epochs:02d}] COMPLETE ({epoch_time:4.1f}s) | "
              f"Train Acc: {train_acc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}% | Val Loss: {val_loss:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'classes': STANDARD_CLASSES
            }, best_model_path)
            print(f"     -> [★ NEW BEST SAVED TO {best_model_path} ({val_acc*100:.2f}%)]\n", flush=True)
        else:
            print("", flush=True)

    # Plot metrics
    print("\n" + "=" * 68, flush=True)
    print(" GENERATING REVIEW DELIVERABLES & CHARTS", flush=True)
    print("=" * 68, flush=True)
    plot_metrics(history, all_targets, all_preds, output_dir=models_dir)
    print(f" [✓] Best Trained Accuracy: {best_val_acc*100:.2f}%", flush=True)
    print("=" * 68 + "\n", flush=True)


if __name__ == "__main__":
    # Prevent CUDA sm_120 driver deadlock on Windows (only for training)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "16"
    os.environ["MKL_NUM_THREADS"] = "16"
    torch.set_num_threads(16)
    main()

