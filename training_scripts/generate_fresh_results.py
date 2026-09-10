import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, models, transforms

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

FER_TEST = os.path.join(
    BASE_DIR,
    "datasets",
    "fer2013",
    "test"
)

RAF_TEST = os.path.join(
    BASE_DIR,
    "datasets",
    "raf_db",
    "test"
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "facial_emotion_model_cuda.pth"
)

HISTORY_PATH = os.path.join(
    MODELS_DIR,
    "results",
    "training_history.csv"
) if os.path.exists(os.path.join(MODELS_DIR, "results", "training_history.csv")) else os.path.join(
    MODELS_DIR,
    "training_history.csv"
)

BATCH_SIZE = 64
NUM_WORKERS = 0
NUM_CLASSES = 7


# ============================================================
# OUTPUT CLASS NAMES
# ============================================================

# These correspond to the output indices used by the model.
# RAF-DB numeric folders are kept exactly as ImageFolder loaded
# them, matching the original training pipeline.

CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("FRESH PART 1 MODEL EVALUATION")
print("=" * 70)

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print("=" * 70)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_paths = {
    "Model": MODEL_PATH,
    "Training history": HISTORY_PATH,
    "FER2013 test": FER_TEST,
    "RAF-DB test": RAF_TEST
}

for name, path in required_paths.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\n{name} not found:\n{path}"
        )


# ============================================================
# 1. LOAD TRAINING HISTORY
# ============================================================

print("\n[1/5] Reading training history...")

history = pd.read_csv(
    HISTORY_PATH
)

history.columns = [
    column.strip()
    for column in history.columns
]

fresh_history_path = os.path.join(
    MODELS_DIR,
    "training_history_latest.csv"
)

history.to_csv(
    fresh_history_path,
    index=False
)

print(
    f"Saved: {fresh_history_path}"
)


# ============================================================
# FIND BEST EPOCH
# ============================================================

best_index = history[
    "validation_accuracy"
].idxmax()

best_row = history.loc[
    best_index
]

best_epoch = int(
    best_row["epoch"]
)

best_train_accuracy = float(
    best_row["train_accuracy"]
)

best_validation_accuracy = float(
    best_row["validation_accuracy"]
)

best_train_loss = float(
    best_row["train_loss"]
)

best_validation_loss = float(
    best_row["validation_loss"]
)

print(
    f"\nBest Epoch: {best_epoch}"
)

print(
    f"Best Training Accuracy: "
    f"{best_train_accuracy:.2f}%"
)

print(
    f"Best Validation Accuracy: "
    f"{best_validation_accuracy:.2f}%"
)


# ============================================================
# 2. CREATE TRAINING HISTORY GRAPHS
# ============================================================

print(
    "\n[2/5] Creating training history plots..."
)


# ----------------------------
# Accuracy plot
# ----------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history["epoch"],
    history["train_accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    history["epoch"],
    history["validation_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.scatter(
    best_epoch,
    best_validation_accuracy,
    s=100,
    label=(
        f"Best Epoch {best_epoch} "
        f"({best_validation_accuracy:.2f}%)"
    )
)

plt.title(
    "Facial Emotion Model Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

accuracy_path = os.path.join(
    MODELS_DIR,
    "training_accuracy_latest.png"
)

plt.savefig(
    accuracy_path,
    dpi=300
)

plt.close()


# ----------------------------
# Loss plot
# ----------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history["epoch"],
    history["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    history["epoch"],
    history["validation_loss"],
    marker="o",
    label="Validation Loss"
)

plt.title(
    "Facial Emotion Model Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

loss_path = os.path.join(
    MODELS_DIR,
    "training_loss_latest.png"
)

plt.savefig(
    loss_path,
    dpi=300
)

plt.close()


# ============================================================
# 3. LOAD TEST DATASETS
# ============================================================

print(
    "\n[3/5] Loading evaluation datasets..."
)


test_transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ------------------------------------------------------------
# FER2013
# ------------------------------------------------------------

fer_test_dataset = datasets.ImageFolder(
    root=FER_TEST,
    transform=test_transform
)


# ------------------------------------------------------------
# RAF-DB
# ------------------------------------------------------------

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

RAF_DB_INDEX_MAP = {
    '1': EMOTIONS.index('surprise'),  # 6
    '2': EMOTIONS.index('fear'),      # 2
    '3': EMOTIONS.index('disgust'),   # 1
    '4': EMOTIONS.index('happy'),     # 3
    '5': EMOTIONS.index('sad'),       # 5
    '6': EMOTIONS.index('angry'),     # 0
    '7': EMOTIONS.index('neutral')    # 4
}

raf_test_dataset = datasets.ImageFolder(
    root=RAF_TEST,
    transform=test_transform
)

# Remap RAF-DB samples to standard emotion indices
remapped_samples = []
for file_path, class_idx in raf_test_dataset.samples:
    folder_name = raf_test_dataset.classes[class_idx]
    correct_idx = RAF_DB_INDEX_MAP.get(folder_name, class_idx)
    remapped_samples.append((file_path, correct_idx))

raf_test_dataset.samples = remapped_samples
raf_test_dataset.targets = [s[1] for s in remapped_samples]
raf_test_dataset.classes = EMOTIONS


print(
    f"\nFER2013 test images: "
    f"{len(fer_test_dataset)}"
)

print(
    f"RAF-DB test images: "
    f"{len(raf_test_dataset)} (Remapped to standard 7 classes)"
)


print(
    "\nFER2013 class mapping:"
)

print(
    fer_test_dataset.class_to_idx
)

print(
    "\nRAF-DB class mapping:"
)

print(
    raf_test_dataset.class_to_idx
)


# ============================================================
# COMBINE DATASETS
# ============================================================

# This intentionally reproduces the same ConcatDataset approach
# used during the original CUDA training.

combined_test_dataset = ConcatDataset([
    fer_test_dataset,
    raf_test_dataset
])


test_loader = DataLoader(
    combined_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print(
    f"\nTotal evaluation images: "
    f"{len(combined_test_dataset)}"
)


# ============================================================
# 4. CREATE SAME RESNET18 ARCHITECTURE
# ============================================================

print(
    "\n[4/5] Loading trained ResNet18 model..."
)


model = models.resnet18(
    weights=None
)

num_features = (
    model.fc.in_features
)

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(
        num_features,
        NUM_CLASSES
    )
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint
)

model = model.to(
    DEVICE
)

model.eval()

print(
    "Best model loaded successfully."
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

all_predictions = []
all_targets = []

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(
            images
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_targets.extend(
            labels.cpu().numpy()
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


all_predictions = np.array(
    all_predictions
)

all_targets = np.array(
    all_targets
)

actual_accuracy = (
    correct / total
) * 100


print(
    f"\nFresh evaluation accuracy: "
    f"{actual_accuracy:.2f}%"
)


# ============================================================
# 5. CLASSIFICATION REPORT
# ============================================================

print(
    "\n[5/5] Creating classification report..."
)


report = classification_report(
    all_targets,
    all_predictions,
    labels=list(
        range(NUM_CLASSES)
    ),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


report_path = os.path.join(
    MODELS_DIR,
    "classification_report_latest.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "=" * 65 + "\n"
    )

    file.write(
        "FACIAL EMOTION MODEL — CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 65 + "\n\n"
    )

    file.write(
        f"Best Epoch: {best_epoch}\n"
    )

    file.write(
        f"Training Accuracy at Best Epoch: "
        f"{best_train_accuracy:.2f}%\n"
    )

    file.write(
        f"Validation Accuracy from Training History: "
        f"{best_validation_accuracy:.2f}%\n"
    )

    file.write(
        f"Fresh Evaluation Accuracy: "
        f"{actual_accuracy:.2f}%\n"
    )

    file.write(
        f"Training Loss at Best Epoch: "
        f"{best_train_loss:.4f}\n"
    )

    file.write(
        f"Validation Loss at Best Epoch: "
        f"{best_validation_loss:.4f}\n\n"
    )

    file.write(
        "PER-CLASS PERFORMANCE\n"
    )

    file.write(
        "-" * 65 + "\n"
    )

    file.write(
        report
    )


print("\n" + report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_targets,
    all_predictions,
    labels=list(
        range(NUM_CLASSES)
    )
)


# ------------------------------------------------------------
# Normalized confusion matrix
# ------------------------------------------------------------

cm_normalized = (
    cm.astype(np.float64)
    /
    np.maximum(
        cm.sum(
            axis=1,
            keepdims=True
        ),
        1
    )
)


plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    vmin=0,
    vmax=1
)

plt.title(
    "Normalized Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.tight_layout()

confusion_path = os.path.join(
    MODELS_DIR,
    "confusion_matrix_latest.png"
)

plt.savefig(
    confusion_path,
    dpi=300
)

plt.close()


# ============================================================
# SAVE RAW CONFUSION MATRIX
# ============================================================

cm_csv_path = os.path.join(
    MODELS_DIR,
    "confusion_matrix_latest.csv"
)

pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
).to_csv(
    cm_csv_path
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FRESH RESULTS GENERATED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nBest Epoch: {best_epoch}"
)

print(
    f"Training Accuracy: "
    f"{best_train_accuracy:.2f}%"
)

print(
    f"Validation Accuracy: "
    f"{best_validation_accuracy:.2f}%"
)

print(
    f"Fresh Evaluation Accuracy: "
    f"{actual_accuracy:.2f}%"
)

print("\nGenerated files:")

print(
    "models/training_history_latest.csv"
)

print(
    "models/training_accuracy_latest.png"
)

print(
    "models/training_loss_latest.png"
)

print(
    "models/classification_report_latest.txt"
)

print(
    "models/confusion_matrix_latest.png"
)

print(
    "models/confusion_matrix_latest.csv"
)

print("=" * 70)