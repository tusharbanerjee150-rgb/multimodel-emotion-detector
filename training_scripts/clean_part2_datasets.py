"""
=============================================================================
PART 2 DATASET CLEANING & STANDARDIZATION PIPELINE
=============================================================================
Project: Multimodal Emotion Recognition (DSN2098 - Group-52)
Description:
  Cleans, verifies, and standardizes all Part 2 datasets:
    1. Audio (RAVDESS + TESS) -> Unified Speech Emotion Manifest & Splits
    2. Text (Hugging Face + GoEmotions) -> Unified NLP Emotion Manifest & Splits
  
  All samples are verified and mapped to the standard 7 project emotions:
    ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
=============================================================================
"""

import os
import sys
import re
import csv
import json
import glob
import wave
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Standard project emotions & label indices
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PART2_DATA_DIR = os.path.join(BASE_DIR, "part2_datasets")
CLEANED_DIR = os.path.join(PART2_DATA_DIR, "cleaned")

os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(os.path.join(CLEANED_DIR, "audio"), exist_ok=True)
os.makedirs(os.path.join(CLEANED_DIR, "text"), exist_ok=True)


# =============================================================================
# PART 1: AUDIO DATASET CLEANING (RAVDESS + TESS)
# =============================================================================

RAVDESS_EMOTION_MAP = {
    '01': 'neutral',
    '02': 'neutral',  # calm mapped to neutral
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise'
}

TESS_FOLDER_MAP = {
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
    'pleasant_surprise': 'surprise',
    'pleasant_surprised': 'surprise'
}

def verify_audio_file(filepath):
    """Verifies that an audio file can be read and is not empty."""
    try:
        if os.path.getsize(filepath) < 1024:
            return False, 0, 0
        with wave.open(filepath, 'rb') as wf:
            n_frames = wf.getnframes()
            sr = wf.getframerate()
            duration = n_frames / float(sr) if sr > 0 else 0
            if duration < 0.2:
                return False, 0, 0
            return True, duration, sr
    except Exception:
        try:
            import soundfile as sf
            info = sf.info(filepath)
            if info.duration < 0.2:
                return False, 0, 0
            return True, info.duration, info.samplerate
        except Exception:
            return False, 0, 0

def clean_audio_datasets():
    print("\n" + "=" * 65)
    print(" [1/2] CLEANING SPEECH EMOTION (AUDIO) DATASETS")
    print("=" * 65)

    audio_records = []
    seen_files = set()

    # 1. Process RAVDESS
    ravdess_dir = os.path.join(PART2_DATA_DIR, "ravdess")
    ravdess_count = 0
    if os.path.exists(ravdess_dir):
        print(" [+] Scanning RAVDESS audio files...")
        wav_files = glob.glob(os.path.join(ravdess_dir, "**", "*.wav"), recursive=True)
        for wav_path in wav_files:
            fname = os.path.basename(wav_path)
            # Prevent duplicate processing between Actor_XX and audio_speech_actors_01-24
            if fname in seen_files:
                continue

            parts = fname.split(".")[0].split("-")
            if len(parts) == 7:
                emotion_code = parts[2]
                actor_id = parts[6]
                if emotion_code in RAVDESS_EMOTION_MAP:
                    emotion = RAVDESS_EMOTION_MAP[emotion_code]
                    is_valid, duration, sr = verify_audio_file(wav_path)
                    if is_valid:
                        seen_files.add(fname)
                        audio_records.append({
                            'filepath': os.path.relpath(wav_path, BASE_DIR).replace("\\", "/"),
                            'filename': fname,
                            'dataset': 'RAVDESS',
                            'emotion': emotion,
                            'label_idx': EMOTION_TO_IDX[emotion],
                            'duration_sec': round(duration, 3),
                            'sample_rate': sr,
                            'actor': f"Actor_{actor_id}"
                        })
                        ravdess_count += 1

    print(f"     -> Verified & Cleaned RAVDESS samples: {ravdess_count}")

    # 2. Process TESS
    tess_dir = os.path.join(PART2_DATA_DIR, "tess")
    tess_count = 0
    if os.path.exists(tess_dir):
        print(" [+] Scanning TESS audio files...")
        wav_files = glob.glob(os.path.join(tess_dir, "**", "*.wav"), recursive=True)
        for wav_path in wav_files:
            fname = os.path.basename(wav_path)
            if fname in seen_files:
                continue

            # Determine emotion from folder name or filename
            parent_folder = os.path.basename(os.path.dirname(wav_path)).lower()
            matched_emotion = None
            for key, em in TESS_FOLDER_MAP.items():
                if key in parent_folder or key in fname.lower():
                    matched_emotion = em
                    break

            if matched_emotion:
                is_valid, duration, sr = verify_audio_file(wav_path)
                if is_valid:
                    seen_files.add(fname)
                    speaker = "OAF" if "OAF" in parent_folder.upper() or "OAF" in fname else "YAF"
                    audio_records.append({
                        'filepath': os.path.relpath(wav_path, BASE_DIR).replace("\\", "/"),
                        'filename': fname,
                        'dataset': 'TESS',
                        'emotion': matched_emotion,
                        'label_idx': EMOTION_TO_IDX[matched_emotion],
                        'duration_sec': round(duration, 3),
                        'sample_rate': sr,
                        'actor': speaker
                    })
                    tess_count += 1

    print(f"     -> Verified & Cleaned TESS samples: {tess_count}")

    total_audio = len(audio_records)
    print(f"\n [✓] Total clean audio samples compiled: {total_audio}")
    df_audio = pd.DataFrame(audio_records)

    # Class distribution
    print("\n --- Audio Emotion Distribution ---")
    dist = df_audio['emotion'].value_counts()
    for em, count in dist.items():
        pct = (count / total_audio) * 100
        print(f"  • {em.capitalize():<10}: {count:>5} samples ({pct:>5.1f}%)")

    # Save Master Audio Manifest
    manifest_path = os.path.join(CLEANED_DIR, "audio", "audio_manifest.csv")
    df_audio.to_csv(manifest_path, index=False)
    print(f"\n [+] Saved master audio manifest: {manifest_path}")

    # Stratified Train/Val/Test Split (80% / 10% / 10%)
    train_df, temp_df = train_test_split(df_audio, test_size=0.20, random_state=42, stratify=df_audio['emotion'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['emotion'])

    train_path = os.path.join(CLEANED_DIR, "audio", "audio_train.csv")
    val_path = os.path.join(CLEANED_DIR, "audio", "audio_val.csv")
    test_path = os.path.join(CLEANED_DIR, "audio", "audio_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f" [+] Saved Train Split ({len(train_df)} samples): {train_path}")
    print(f" [+] Saved Val Split   ({len(val_df)} samples): {val_path}")
    print(f" [+] Saved Test Split  ({len(test_df)} samples): {test_path}")

    return df_audio


# =============================================================================
# PART 2: TEXT DATASET CLEANING (HUGGING FACE + GOEMOTIONS)
# =============================================================================

HF_TEXT_MAP = {
    'sadness': 'sad',
    'joy': 'happy',
    'love': 'happy',
    'anger': 'angry',
    'fear': 'fear',
    'surprise': 'surprise'
}

GOEMOTIONS_EKMAN_MAP = {
    'anger': 'angry',
    'annoyance': 'angry',
    'disapproval': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'nervousness': 'fear',
    'joy': 'happy',
    'amusement': 'happy',
    'approval': 'happy',
    'excitement': 'happy',
    'gratitude': 'happy',
    'love': 'happy',
    'optimism': 'happy',
    'relief': 'happy',
    'pride': 'happy',
    'admiration': 'happy',
    'desire': 'happy',
    'caring': 'happy',
    'sadness': 'sad',
    'disappointment': 'sad',
    'embarrassment': 'sad',
    'grief': 'sad',
    'remorse': 'sad',
    'surprise': 'surprise',
    'realization': 'surprise',
    'confusion': 'surprise',
    'curiosity': 'surprise',
    'neutral': 'neutral'
}

def clean_text_sample(text):
    """Normalizes text by removing unwanted artifacts, lowercasing, and normalizing whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[NAME\]', 'someone', text)
    text = re.sub(r'\[RELIGION\]', 'faith', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def clean_text_datasets():
    print("\n" + "=" * 65)
    print(" [2/2] CLEANING TEXT SENTIMENT & NLP DATASETS")
    print("=" * 65)

    text_records = []
    seen_texts = set()

    # 1. Process Hugging Face emotion dataset (train.txt, val.txt, test.txt)
    hf_dir = os.path.join(PART2_DATA_DIR, "hugging_face")
    hf_count = 0
    if os.path.exists(hf_dir):
        print(" [+] Processing Hugging Face Emotion text files...")
        for split_file in ["train.txt", "val.txt", "test.txt"]:
            fpath = os.path.join(hf_dir, split_file)
            if os.path.exists(fpath):
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if ';' in line:
                            parts = line.split(';')
                            raw_text = parts[0]
                            raw_emotion = parts[1].strip()
                            clean_t = clean_text_sample(raw_text)
                            if len(clean_t) > 3 and clean_t not in seen_texts:
                                if raw_emotion in HF_TEXT_MAP:
                                    emotion = HF_TEXT_MAP[raw_emotion]
                                    seen_texts.add(clean_t)
                                    text_records.append({
                                        'text': clean_t,
                                        'emotion': emotion,
                                        'label_idx': EMOTION_TO_IDX[emotion],
                                        'dataset': 'HuggingFace'
                                    })
                                    hf_count += 1

    print(f"     -> Verified & Cleaned Hugging Face samples: {hf_count}")

    # 2. Process GoEmotions
    go_dir = os.path.join(PART2_DATA_DIR, "GoEmotions", "data")
    go_count = 0
    if os.path.exists(go_dir):
        print(" [+] Processing GoEmotions dataset...")
        # Load emotion names
        emotions_file = os.path.join(go_dir, "emotions.txt")
        go_emotions_list = []
        if os.path.exists(emotions_file):
            with open(emotions_file, 'r', encoding='utf-8') as f:
                go_emotions_list = [line.strip() for line in f if line.strip()]

        for split_file in ["train.tsv", "dev.tsv", "test.tsv"]:
            fpath = os.path.join(go_dir, split_file)
            if os.path.exists(fpath):
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            raw_text = parts[0]
                            label_indices_str = parts[1].split(',')
                            clean_t = clean_text_sample(raw_text)
                            if len(clean_t) > 3 and clean_t not in seen_texts:
                                mapped_emotions = []
                                for idx_s in label_indices_str:
                                    try:
                                        idx = int(idx_s.strip())
                                        if idx < len(go_emotions_list):
                                            raw_em = go_emotions_list[idx]
                                            if raw_em in GOEMOTIONS_EKMAN_MAP:
                                                mapped_em = GOEMOTIONS_EKMAN_MAP[raw_em]
                                                mapped_emotions.append(mapped_em)
                                    except Exception:
                                        pass
                                
                                if mapped_emotions:
                                    chosen_emotion = Counter(mapped_emotions).most_common(1)[0][0]
                                    seen_texts.add(clean_t)
                                    text_records.append({
                                        'text': clean_t,
                                        'emotion': chosen_emotion,
                                        'label_idx': EMOTION_TO_IDX[chosen_emotion],
                                        'dataset': 'GoEmotions'
                                    })
                                    go_count += 1

    print(f"     -> Verified & Cleaned GoEmotions samples: {go_count}")

    total_text = len(text_records)
    print(f"\n [✓] Total clean text samples compiled: {total_text}")
    df_text = pd.DataFrame(text_records)

    # Class distribution
    print("\n --- Text Emotion Distribution ---")
    dist = df_text['emotion'].value_counts()
    for em, count in dist.items():
        pct = (count / total_text) * 100
        print(f"  • {em.capitalize():<10}: {count:>6} samples ({pct:>5.1f}%)")

    # Save Master Text Manifest
    manifest_path = os.path.join(CLEANED_DIR, "text", "text_manifest.csv")
    df_text.to_csv(manifest_path, index=False)
    print(f"\n [+] Saved master text manifest: {manifest_path}")

    # Stratified Train/Val/Test Split (80% / 10% / 10%)
    train_df, temp_df = train_test_split(df_text, test_size=0.20, random_state=42, stratify=df_text['emotion'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['emotion'])

    train_path = os.path.join(CLEANED_DIR, "text", "text_train.csv")
    val_path = os.path.join(CLEANED_DIR, "text", "text_val.csv")
    test_path = os.path.join(CLEANED_DIR, "text", "text_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f" [+] Saved Train Split ({len(train_df)} samples): {train_path}")
    print(f" [+] Saved Val Split   ({len(val_df)} samples): {val_path}")
    print(f" [+] Saved Test Split  ({len(test_df)} samples): {test_path}")

    return df_text


# =============================================================================
# MAIN PIPELINE
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print(" STARTING PART 2 DATASET CLEANING & STANDARDIZATION")
    print(" Target Emotions: " + str(EMOTIONS))
    print("=" * 65)

    df_audio = clean_audio_datasets()
    df_text = clean_text_datasets()

    print("\n" + "=" * 65)
    print(" [✓] ALL PART 2 DATASETS SUCCESSFULLY CLEANED & STANDARDIZED!")
    print(f" Total Clean Audio Samples: {len(df_audio)}")
    print(f" Total Clean Text Samples : {len(df_text)}")
    print(f" Destination Directory    : {CLEANED_DIR}")
    print("=" * 65 + "\n")
