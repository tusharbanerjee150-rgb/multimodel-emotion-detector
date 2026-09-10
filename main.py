"""
=============================================================================
SERENE EARTH AI — FASTAPI MULTIMODAL EMOTION RECOGNITION SERVER
=============================================================================
Project: Multimodal Emotion Recognition (DSN2098 - Group-52)
Framework: FastAPI + Uvicorn + PyTorch CUDA
Hardware: NVIDIA GeForce RTX 5050 Laptop GPU
=============================================================================
"""

import os
import io
import re
import json
import time
import uuid
import base64
from typing import Optional, List, Dict, Any

if hasattr(os, 'add_dll_directory'):
    pass

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =============================================================================
# CONSTANTS & PROJECT PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
STORAGE_FILE = os.path.join(DATA_DIR, "app_storage.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_EMOJIS = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😔',
    'surprise': '😲'
}

DOMAIN_RECOMMENDATIONS = {
    'happy': {
        'state': 'Positive / High Engagement',
        'education': 'Learner is actively engaged and receptive. Introduce advanced concepts.',
        'customer': 'High customer delight. Request feedback or positive review.',
        'driver': 'Driver alert and relaxed. Safe driving conditions.',
        'mental_health': 'Emotional state optimal. Reinforce healthy habit continuity.'
    },
    'sad': {
        'state': 'Low Arousal / Negative Valence',
        'education': 'Learner shows signs of discouragement. Offer a short break or simpler practice problem.',
        'customer': 'Customer dissatisfaction detected. Escalate to senior support representative.',
        'driver': 'Potential driver inattention. Activate gentle auditory alert.',
        'mental_health': 'Low mood detected. Suggest mindfulness breathing or mood journaling.'
    },
    'angry': {
        'state': 'High Arousal / Negative Valence',
        'education': 'Learner is frustrated. Offer interactive hints or switch difficulty mode.',
        'customer': 'Frustration alert! Offer immediate apology and compensation coupon.',
        'driver': 'Aggressive driving risk (Road Rage). Trigger calming music and caution warning.',
        'mental_health': 'Elevated distress. Suggest de-escalation breathing exercise.'
    },
    'fear': {
        'state': 'High Stress / Anxious',
        'education': 'Exam anxiety or confusion detected. Provide reassuring encouragement.',
        'customer': 'Customer is anxious or overwhelmed. Provide clear step-by-step guidance.',
        'driver': 'High anxiety / unexpected hazard panic. Alert driver safety assist system.',
        'mental_health': 'Acute anxiety detected. Trigger grounding 5-4-3-2-1 exercise.'
    },
    'surprise': {
        'state': 'High Arousal / Novelty Stimulus',
        'education': 'Novelty detected. Capitalize on curiosity to explain core concepts.',
        'customer': 'Unexpected reaction. Verify if customer expectations were met.',
        'driver': 'Sudden road event reaction. Ensure vehicle stability and lane control.',
        'mental_health': 'Novelty experience. Log trigger in emotion timeline.'
    },
    'disgust': {
        'state': 'Negative Valence / Rejection',
        'education': 'Disinterest or disengagement. Change presentation style or topic format.',
        'customer': 'Strong product/service disapproval. Initiate quality review process.',
        'driver': 'Discomfort detected. Check vehicle climate and ergonomics.',
        'mental_health': 'Negative aversion. Encourage reflection on underlying stressor.'
    },
    'neutral': {
        'state': 'Baseline / Balanced',
        'education': 'Learner is focused and attentive. Maintain steady instructional pace.',
        'customer': 'Standard interaction. Proceed with standard workflow.',
        'driver': 'Driver focused and in baseline state. All systems nominal.',
        'mental_health': 'Stable baseline. Continue routine emotion tracking.'
    }
}


# =============================================================================
# IN-WEBSITE PERSISTENT STORAGE
# =============================================================================

def load_storage() -> Dict[str, List[Any]]:
    if not os.path.exists(STORAGE_FILE):
        return {"history": [], "snapshots": []}
    try:
        with open(STORAGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"history": [], "snapshots": []}

def save_storage(data: Dict[str, List[Any]]):
    try:
        with open(STORAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f" [!] Storage save error: {e}")


# =============================================================================
# MODALITY 1: HIGH-FPS FACIAL EMOTION ENGINE (ResNet-18 CUDA)
# =============================================================================

class FacialEmotionEngine:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector_cache = {}
        self.yunet_path = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")

        # High-Speed RGB ImageNet Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = None
        self._init_model(model_path)

    def _init_model(self, model_path=None):
        m_path = model_path
        if not m_path or not os.path.exists(m_path):
            m_path = os.path.join(MODELS_DIR, "facial_emotion_model_final.pth")
        if not os.path.exists(m_path):
            m_path = os.path.join(MODELS_DIR, "facial_emotion_model_cuda.pth")
        try:
            if os.path.exists(m_path):
                self.model = models.resnet18(weights=None)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(num_features, len(EMOTIONS))
                )
                
                ckpt = torch.load(m_path, map_location=self.device, weights_only=False)
                sd = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
                self.model.load_state_dict(sd)
                self.model.to(self.device)
                self.model.eval()

                if torch.cuda.is_available():
                    torch.backends.cudnn.benchmark = True
                    # Warmup pass
                    dummy = torch.zeros(1, 3, 224, 224, device=self.device)
                    with torch.inference_mode():
                        self.model(dummy)

                dev_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
                print(f" [+] [Face Engine] Loaded ResNet-18 Model on: {dev_name}")
            else:
                print(f" [!] [Face Engine] Model file not found at {m_path}")
        except Exception as e:
            print(f" [!] [Face Engine] Load error: {e}")

    def _get_detector(self, w, h):
        key = (w, h)
        if key not in self.detector_cache:
            try:
                if os.path.exists(self.yunet_path) and hasattr(cv2, 'FaceDetectorYN'):
                    self.detector_cache[key] = cv2.FaceDetectorYN.create(
                        model=self.yunet_path,
                        config='',
                        input_size=(w, h),
                        score_threshold=0.20,
                        nms_threshold=0.3,
                        top_k=5000
                    )
                else:
                    self.detector_cache[key] = None
            except Exception as e:
                print(f" [!] Detector create note: {e}")
                self.detector_cache[key] = None
        return self.detector_cache.get(key)

    def analyze_image_cv(self, img_bgr):
        t0 = time.time()
        if img_bgr is None or img_bgr.size == 0:
            return {'status': 'error', 'message': 'Empty or corrupted image frame.'}

        h, w = img_bgr.shape[:2]
        detections = []
        overall_probs = {e: 0.0 for e in EMOTIONS}
        detected_faces = []

        detector = self._get_detector(w, h)
        if detector is not None:
            try:
                _, faces = detector.detect(img_bgr)
                if faces is not None:
                    for f in faces:
                        fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                        conf_score = float(f[14])
                        if conf_score >= 0.20 and fw > 20 and fh > 20:
                            detected_faces.append((fx, fy, fw, fh))
            except Exception as e:
                print(f" [!] Face detect error: {e}")

        # Fallback: if no face detected in dim lighting, isolate centered portrait region
        if len(detected_faces) == 0:
            pw, ph = int(w * 0.50), int(h * 0.60)
            px, py = int((w - pw) / 2), int(h * 0.10)
            detected_faces.append((px, py, pw, ph))

        for (x, y, fw, fh) in detected_faces:
            # 20% Margin Padding for full facial expression capture
            pad_x = int(fw * 0.20)
            pad_y = int(fh * 0.20)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + fw + pad_x)
            y2 = min(h, y + fh + pad_y)
            
            face_crop_bgr = img_bgr[y1:y2, x1:x2]
            if face_crop_bgr.size == 0:
                continue

            face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)
            
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                outputs = self.model(tensor)
                probs_arr = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            probs_dict = {EMOTIONS[i]: float(probs_arr[i]) for i in range(len(EMOTIONS))}
            dom_emotion = max(probs_dict, key=probs_dict.get)
            conf = float(probs_dict[dom_emotion])

            detections.append({
                'box': [int(x), int(y), int(fw), int(fh)],
                'dominant_emotion': dom_emotion,
                'confidence': conf,
                'emoji': EMOTION_EMOJIS[dom_emotion],
                'probabilities': probs_dict
            })

        if len(detections) > 0:
            primary = max(detections, key=lambda d: d['confidence'])
            overall_probs = primary['probabilities']
            dom_emotion = primary['dominant_emotion']
            conf = primary['confidence']
        else:
            dom_emotion = 'neutral'
            conf = 0.5

        elapsed_ms = (time.time() - t0) * 1000.0

        return {
            'status': 'success',
            'face_count': len(detected_faces),
            'dominant_emotion': dom_emotion,
            'confidence': conf,
            'emoji': EMOTION_EMOJIS[dom_emotion],
            'probabilities': overall_probs,
            'detections': detections,
            'latency_ms': round(elapsed_ms, 1)
        }



# =============================================================================
# MODALITY 2: SPEECH EMOTION RECOGNITION (Deep CRNN CUDA)
# =============================================================================

class SpeechEmotionCRNN(nn.Module):
    def __init__(self, num_classes=len(EMOTIONS)):
        super(SpeechEmotionCRNN, self).__init__()
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
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.3)
        )
        self.rnn_input_dim = 256 * 10
        self.lstm = nn.LSTM(
            input_size=self.rnn_input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        b, c, f, t = c4.size()
        rnn_in = c4.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        lstm_out, _ = self.lstm(rnn_in)
        att_w = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * att_w, dim=1)
        out = self.classifier(context)
        return out


class VoiceEmotionEngine:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._init_model(model_path)

    def _init_model(self, model_path=None):
        m_path = model_path or os.path.join(MODELS_DIR, "speech_emotion_model.pth")
        try:
            if os.path.exists(m_path):
                self.model = SpeechEmotionCRNN().to(self.device)
                ckpt = torch.load(m_path, map_location=self.device, weights_only=False)
                sd = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
                self.model.load_state_dict(sd)
                self.model.eval()

                # Warmup pass
                dummy = torch.zeros(1, 1, 168, 130, device=self.device)
                with torch.inference_mode():
                    self.model(dummy)

                dev_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
                print(f" [+] [Voice Engine] Loaded Deep CRNN Model on: {dev_name}")
            else:
                print(f" [!] [Voice Engine] Model file not found at {m_path}")
        except Exception as e:
            print(f" [!] [Voice Engine] Load error: {e}")

    def analyze_audio(self, audio_bytes: bytes):
        if not audio_bytes or len(audio_bytes) < 44:
            return {'status': 'error', 'message': 'Audio data is empty or too short.'}

        import librosa
        import soundfile as sf

        data = None
        sr = 22050

        # 1. Try soundfile
        try:
            data, in_sr = sf.read(io.BytesIO(audio_bytes))
            sr = in_sr
        except Exception:
            data = None

        # 2. Try scipy.io.wavfile
        if data is None:
            try:
                from scipy.io import wavfile
                in_sr, raw_wav = wavfile.read(io.BytesIO(audio_bytes))
                sr = in_sr
                if raw_wav.dtype == np.int16:
                    data = raw_wav.astype(np.float32) / 32768.0
                elif raw_wav.dtype == np.int32:
                    data = raw_wav.astype(np.float32) / 2147483648.0
                elif raw_wav.dtype == np.uint8:
                    data = (raw_wav.astype(np.float32) - 128.0) / 128.0
                else:
                    data = raw_wav.astype(np.float32)
            except Exception:
                data = None

        # 3. Try standard wave module
        if data is None:
            try:
                import wave
                with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    sr = wf.getframerate()
                    raw_frames = wf.readframes(wf.getnframes())
                    if sampwidth == 2:
                        data = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sampwidth == 1:
                        data = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                    if n_channels > 1 and data is not None:
                        data = data[::n_channels]
            except Exception:
                data = None

        if data is None or len(data) == 0:
            return {'status': 'error', 'message': 'Could not decode audio data.'}

        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        if sr != 22050:
            data = librosa.resample(data, orig_sr=sr, target_sr=22050)
            sr = 22050

        # Amplitude normalization & Silence Trimming
        max_val = np.max(np.abs(data))
        if max_val > 1e-4:
            data = data / max_val * 0.95

        try:
            trimmed, _ = librosa.effects.trim(data, top_db=22)
            if len(trimmed) >= 22050 * 0.5:  # At least 0.5s of speech
                data = trimmed
        except Exception:
            pass

        # Acoustic Metrics
        rms = float(np.mean(librosa.feature.rms(y=data)[0]))
        spec = float(np.mean(librosa.feature.spectral_centroid(y=data, sr=sr)[0]))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(data)[0]))

        try:
            onset_env = librosa.onset.onset_strength(y=data, sr=sr)
            tempo = float(librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)[0])
        except Exception:
            tempo = 120.0

        # Deep CRNN Audio Representation (3.0s window)
        target_len = 22050 * 3  # 3.0s
        if len(data) < target_len:
            pad_left = (target_len - len(data)) // 2
            pad_right = target_len - len(data) - pad_left
            padded_data = np.pad(data, (pad_left, pad_right), mode='constant')
        else:
            padded_data = data[:target_len]

        mel_spec = librosa.feature.melspectrogram(y=padded_data, sr=22050, n_fft=1024, hop_length=512, n_mels=128)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        mfcc = librosa.feature.mfcc(y=padded_data, sr=22050, n_mfcc=40, n_fft=1024, hop_length=512)
        combined = np.vstack([log_mel, mfcc])

        if combined.shape[1] < 130:
            combined = np.pad(combined, ((0, 0), (0, 130 - combined.shape[1])), mode='constant')
        else:
            combined = combined[:, :130]

        norm_feat = (combined - np.mean(combined)) / (np.std(combined) + 1e-6)
        tensor_feat = torch.from_numpy(norm_feat.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            out = self.model(tensor_feat)
            probs_arr = torch.softmax(out, dim=1).cpu().numpy()[0]

        probabilities = {EMOTIONS[i]: float(probs_arr[i]) for i in range(len(EMOTIONS))}
        dom_emotion = max(probabilities, key=probabilities.get)

        return {
            'status': 'success',
            'dominant_emotion': dom_emotion,
            'confidence': float(probabilities[dom_emotion]),
            'emoji': EMOTION_EMOJIS[dom_emotion],
            'probabilities': probabilities,
            'acoustic_metrics': {
                'energy_rms': f"{rms:.4f}",
                'estimated_tempo_bpm': f"{tempo:.1f}",
                'spectral_centroid_hz': f"{spec:.1f}",
                'zero_crossing_rate': f"{zcr:.4f}"
            }
        }


# =============================================================================
# MODALITY 3: TEXT EMOTION NLP ENGINE (Deep BiLSTM + Attention CUDA)
# =============================================================================

class TextEmotionBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2, num_classes=len(EMOTIONS)):
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
        att_w = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * att_w, dim=1)
        out = self.classifier(context)
        return out


class TextEmotionEngine:
    def __init__(self, model_path=None, vocab_path=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vocab = None
        self.lexicon = {
            'happy': ['happy', 'happiness', 'joy', 'joyful', 'joyous', 'great', 'awesome', 'excellent', 'love', 'fantastic', 'good', 'wonderful', 'glad', 'delighted', 'pleased', 'smiling', 'super', 'best', 'blessed', 'excited', 'excitement', 'win', 'winning', 'celebrate', 'celebration', 'amazing', 'proud', 'promotion', 'promoted', 'grateful', 'gratitude', 'thrilled', 'content', 'relief', 'laughter', 'laughing'],
            'sad': ['sad', 'sadness', 'unhappy', 'depressed', 'depression', 'cry', 'crying', 'grief', 'sorrow', 'lonely', 'loneliness', 'miserable', 'heartbroken', 'hopeless', 'gloomy', 'down', 'pain', 'hurt', 'tear', 'failure', 'disappointed', 'disappointment', 'loss', 'bad', 'mourning'],
            'angry': ['angry', 'anger', 'mad', 'furious', 'fury', 'rage', 'hate', 'annoyed', 'irritated', 'irritation', 'outraged', 'pissed', 'hostile', 'screaming', 'frustrated', 'frustration', 'agitated', 'violence', 'damn', 'hell', 'dispute', 'fight', 'enemy'],
            'fear': ['fear', 'fearful', 'scared', 'afraid', 'terrified', 'panic', 'horror', 'anxious', 'anxiety', 'nervous', 'threat', 'danger', 'dread', 'worry', 'worried', 'creep', 'phobia', 'shock', 'warning', 'risk', 'trembling'],
            'surprise': ['surprise', 'surprised', 'surprising', 'wow', 'unexpected', 'astonished', 'astonishing', 'unbelievable', 'omg', 'incredible', 'shocking', 'sudden', 'whoa', 'stumbled', 'miracle'],
            'disgust': ['disgust', 'disgusted', 'disgusting', 'gross', 'nasty', 'awful', 'revolting', 'yuck', 'sickening', 'horrible', 'repulsive', 'offensive', 'vomit', 'trash'],
            'neutral': ['okay', 'fine', 'normal', 'standard', 'average', 'information', 'report', 'note', 'status', 'fact', 'regular', 'device', 'system']
        }
        self.idiom_overrides = [
            (r'\b(?:crying|cried|tears|wept)\s+(?:out\s+of|with|from)\s+(?:happiness|joy|joyousness|laughter|delight|pride|relief)\b', 'happy', 0.94),
            (r'\b(?:tears\s+of\s+(?:joy|happiness|laughter|relief)|happy\s+tears)\b', 'happy', 0.95),
            (r'\bso\s+happy\s+(?:that\s+)?(?:i|we|they)\s+(?:could|started\s+to|was|were)\s+(?:cry|crying|weep|weeping)\b', 'happy', 0.94),
            (r'\b(?:dying|dead)\s+(?:of|from)\s+(?:laughter|laughing|fun)\b', 'happy', 0.90),
            (r'\b(?:screaming|shouting)\s+(?:with|in|out\s+of)\s+(?:joy|happiness|excitement|delight)\b', 'happy', 0.92)
        ]
        self.negation_patterns = [
            r'\b(?:not|didn\'?t|did\s+not|never|couldn\'?t|could\s+not|failed\s+to|without|no|hardly|won\'?t|cannot)\s+(?:\w+\s+){0,3}(?:promotion|promoted|happy|happiness|joy|win|winning|pass|passed|success|succeed|well|better|good)\b'
        ]
        self._init_model(model_path, vocab_path)

    def _init_model(self, model_path=None, vocab_path=None):
        m_path = model_path or os.path.join(MODELS_DIR, "text_emotion_model.pth")
        v_path = vocab_path or os.path.join(MODELS_DIR, "text_vocab.json")

        try:
            if os.path.exists(m_path) and os.path.exists(v_path):
                with open(v_path, "r", encoding="utf-8") as f:
                    self.vocab = json.load(f)

                self.model = TextEmotionBiLSTM(vocab_size=len(self.vocab)).to(self.device)
                ckpt = torch.load(m_path, map_location=self.device, weights_only=False)
                sd = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
                self.model.load_state_dict(sd)
                self.model.eval()

                # Warmup pass
                dummy = torch.zeros(1, 64, dtype=torch.long, device=self.device)
                with torch.inference_mode():
                    self.model(dummy)

                dev_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
                print(f" [+] [Text Engine] Loaded Deep Bi-LSTM Model on: {dev_name}")
            else:
                print(" [!] [Text Engine] Trained model not found, using rule-based lexicon.")
        except Exception as e:
            print(f" [!] [Text Engine] Load error: {e}")

    def _tokenize_text(self, text: str, max_len=64):
        words = re.findall(r'\b\w+\b|[!?]+', text.lower())
        indices = [self.vocab.get(w, self.vocab.get("<UNK>", 1)) for w in words]
        if len(indices) < max_len:
            indices = indices + [self.vocab.get("<PAD>", 0)] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return indices

    def analyze_text(self, text: str):
        if not text or not text.strip():
            return {'status': 'error', 'message': 'Empty text input'}

        clean_text = text.lower()
        words = re.findall(r'\b\w+\b', clean_text)
        detected_keywords = []

        # 1. Check strict tears-of-joy idiom overrides (e.g. 'crying out of happiness' -> Happy)
        matched_idiom = None
        for pattern, override_emotion, override_conf in self.idiom_overrides:
            if re.search(pattern, clean_text):
                matched_idiom = (override_emotion, override_conf)
                break

        # 2. Check negation (e.g. 'didnt get a promotion', 'not happy')
        has_negated_positive = any(re.search(p, clean_text) for p in self.negation_patterns)

        for word in words:
            for emotion, word_list in self.lexicon.items():
                if word in word_list:
                    detected_keywords.append({'word': word, 'emotion': emotion})

        pos_words = set(self.lexicon['happy'])
        neg_words = set(self.lexicon['sad'] + self.lexicon['angry'] + self.lexicon['fear'] + self.lexicon['disgust'])
        pos_count = sum(1 for w in words if w in pos_words)
        neg_count = sum(1 for w in words if w in neg_words)

        if has_negated_positive:
            # Shift positive count to negative because the positive event was negated
            pos_count = max(0, pos_count - 1)
            neg_count += 1

        if matched_idiom:
            polarity = 0.85 if matched_idiom[0] == 'happy' else -0.85
        else:
            total_matched = pos_count + neg_count
            polarity = (pos_count - neg_count) / total_matched if total_matched > 0 else 0.0

        if matched_idiom:
            override_em, override_conf = matched_idiom
            probabilities = {e: 0.01 for e in EMOTIONS}
            probabilities[override_em] = override_conf
            rem = (1.0 - override_conf) / (len(EMOTIONS) - 1)
            for e in EMOTIONS:
                if e != override_em:
                    probabilities[e] = rem

            return {
                'status': 'success',
                'dominant_emotion': override_em,
                'confidence': override_conf,
                'emoji': EMOTION_EMOJIS[override_em],
                'sentiment_polarity': round(polarity, 2),
                'sentiment_label': 'Positive' if polarity > 0.15 else ('Negative' if polarity < -0.15 else 'Neutral'),
                'probabilities': probabilities,
                'highlighted_keywords': detected_keywords
            }

        if self.model is not None and self.vocab is not None:
            indices = self._tokenize_text(text)
            tensor_text = torch.tensor([indices], dtype=torch.long, device=self.device)
            with torch.inference_mode():
                out = self.model(tensor_text)
                probs_arr = torch.softmax(out, dim=1).cpu().numpy()[0]

            probabilities = {EMOTIONS[i]: float(probs_arr[i]) for i in range(len(EMOTIONS))}

            # If text has adverse negation + crying/sad tokens, ensure model doesn't falsely vote happy
            if has_negated_positive and any(w in clean_text for w in ['cry', 'crying', 'cried', 'sad', 'heartbroken', 'sorrow']):
                probabilities['sad'] = max(0.85, probabilities.get('sad', 0.85))
                probabilities['happy'] = min(0.02, probabilities.get('happy', 0.02))
                total_p = sum(probabilities.values())
                probabilities = {k: v / total_p for k, v in probabilities.items()}

            dom_emotion = max(probabilities, key=probabilities.get)

            return {
                'status': 'success',
                'dominant_emotion': dom_emotion,
                'confidence': float(probabilities[dom_emotion]),
                'emoji': EMOTION_EMOJIS[dom_emotion],
                'sentiment_polarity': round(polarity, 2),
                'sentiment_label': 'Positive' if polarity > 0.15 else ('Negative' if polarity < -0.15 else 'Neutral'),
                'probabilities': probabilities,
                'highlighted_keywords': detected_keywords
            }

        # Lexicon Fallback
        scores = {e: 0.1 for e in EMOTIONS}
        for kw in detected_keywords:
            scores[kw['emotion']] += 1.0
        if (pos_count + neg_count) == 0:
            scores['neutral'] += 0.5

        exp_scores = {e: np.exp(scores[e]) for e in EMOTIONS}
        total_exp = sum(exp_scores.values())
        probabilities = {e: float(exp_scores[e] / total_exp) for e in EMOTIONS}
        dom_emotion = max(probabilities, key=probabilities.get)

        return {
            'status': 'success',
            'dominant_emotion': dom_emotion,
            'confidence': float(probabilities[dom_emotion]),
            'emoji': EMOTION_EMOJIS[dom_emotion],
            'sentiment_polarity': round(polarity, 2),
            'sentiment_label': 'Positive' if polarity > 0.15 else ('Negative' if polarity < -0.15 else 'Neutral'),
            'probabilities': probabilities,
            'highlighted_keywords': detected_keywords
        }


# =============================================================================
# MODALITY 4: MULTIMODAL DECISION FUSION ENGINE
# =============================================================================

class MultimodalFusionEngine:
    def __init__(self, face_weight=0.45, voice_weight=0.35, text_weight=0.20):
        self.weights = {'face': face_weight, 'voice': voice_weight, 'text': text_weight}

    def fuse(self, face_res=None, voice_res=None, text_res=None):
        active_modalities = {}
        if face_res and face_res.get('status') == 'success':
            active_modalities['face'] = (self.weights['face'], face_res['probabilities'])
        if voice_res and voice_res.get('status') == 'success':
            active_modalities['voice'] = (self.weights['voice'], voice_res['probabilities'])
        if text_res and text_res.get('status') == 'success':
            active_modalities['text'] = (self.weights['text'], text_res['probabilities'])

        if not active_modalities:
            return {'status': 'error', 'message': 'No valid modality input received for fusion.'}

        total_w = sum(w for w, _ in active_modalities.values())
        fused_probabilities = {e: 0.0 for e in EMOTIONS}

        for mod_name, (w, probs) in active_modalities.items():
            norm_w = w / total_w
            for e in EMOTIONS:
                fused_probabilities[e] += norm_w * probs.get(e, 0.0)

        fused_dominant = max(fused_probabilities, key=fused_probabilities.get)
        confidence = float(fused_probabilities[fused_dominant])

        dominant_votes = [
            (face_res.get('dominant_emotion') if 'face' in active_modalities else None),
            (voice_res.get('dominant_emotion') if 'voice' in active_modalities else None),
            (text_res.get('dominant_emotion') if 'text' in active_modalities else None)
        ]
        valid_votes = [v for v in dominant_votes if v is not None]
        agreement_ratio = valid_votes.count(fused_dominant) / len(valid_votes) if valid_votes else 1.0
        recommendations = DOMAIN_RECOMMENDATIONS.get(fused_dominant, DOMAIN_RECOMMENDATIONS['neutral'])

        return {
            'status': 'success',
            'dominant_emotion': fused_dominant,
            'confidence': confidence,
            'emoji': EMOTION_EMOJIS[fused_dominant],
            'active_modalities': list(active_modalities.keys()),
            'modality_consensus': f"{agreement_ratio*100:.0f}%",
            'fused_probabilities': fused_probabilities,
            'recommendations': recommendations
        }


# =============================================================================
# FASTAPI APP & ROUTER CONFIGURATION
# =============================================================================

app = FastAPI(
    title="Serene Earth AI - Multimodal Emotion Recognition",
    description="Exhibition – I (DSN2098) Group-52 Deep Learning Emotion AI System",
    version="2.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
results_dir = os.path.join(BASE_DIR, "models", "results")
if os.path.exists(results_dir):
    app.mount("/results", StaticFiles(directory=results_dir), name="results")

face_engine = FacialEmotionEngine()
voice_engine = VoiceEmotionEngine()
text_engine = TextEmotionEngine()
fusion_engine = MultimodalFusionEngine()


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class TextPayload(BaseModel):
    text: str

class FacePayload(BaseModel):
    image: str

class MultimodalPayload(BaseModel):
    image: Optional[str] = None
    audio: Optional[str] = None
    text: Optional[str] = None

class HistoryItem(BaseModel):
    id: Optional[str] = None
    timestamp: Optional[str] = None
    modality: Optional[str] = "Multimodal"
    dominant_emotion: Optional[str] = "neutral"
    confidence: Optional[float] = 0.90
    emoji: Optional[str] = "✨"
    probabilities: Optional[Dict[str, float]] = None
    details: Optional[Dict[str, Any]] = None

class SnapshotItem(BaseModel):
    id: Optional[str] = None
    timestamp: Optional[str] = None
    image: str
    dominant_emotion: Optional[str] = "neutral"
    confidence: Optional[float] = 0.90
    emoji: Optional[str] = "📸"
    notes: Optional[str] = ""


# =============================================================================
# API ROUTES
# =============================================================================

@app.get("/test_babel", response_class=HTMLResponse)
async def serve_test_babel():
    test_path = os.path.join(BASE_DIR, "templates", "test_babel.html")
    if os.path.exists(test_path):
        with open(test_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("Not found")

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = os.path.join(BASE_DIR, "templates", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h3>Serene Earth AI Dashboard template not found.</h3>")


@app.get("/api/health")
async def health_check():
    return {
        'status': 'healthy',
        'project': 'Serene Earth AI — Multimodal Emotion Recognition',
        'group': 'DSN2098 - Group-52',
        'device': str(face_engine.device),
        'supported_emotions': EMOTIONS,
        'engines': {
            'face_resnet18': face_engine.model is not None,
            'voice_deep_crnn': voice_engine.model is not None,
            'text_bilstm': text_engine.model is not None,
            'multimodal_fusion': True
        }
    }


@app.post("/api/predict/face")
async def predict_face(request: Request, payload: Optional[FacePayload] = None, image: Optional[UploadFile] = File(None)):
    try:
        img_bgr = None
        if image is not None:
            content = await image.read()
            np_arr = np.frombuffer(content, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif payload is not None and payload.image:
            b64_str = payload.image
            if ',' in b64_str:
                b64_str = b64_str.split(',')[1]
            img_bytes = base64.b64decode(b64_str)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            json_data = await request.json()
            if 'image' in json_data:
                b64_str = json_data['image']
                if ',' in b64_str:
                    b64_str = b64_str.split(',')[1]
                img_bytes = base64.b64decode(b64_str)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return JSONResponse(status_code=400, content={'status': 'error', 'message': 'No valid image received.'})

        result = face_engine.analyze_image_cv(img_bgr)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})


@app.post("/api/predict/voice")
async def predict_voice(request: Request, audio: Optional[UploadFile] = File(None)):
    try:
        audio_bytes = None
        if audio is not None:
            audio_bytes = await audio.read()
        else:
            try:
                json_data = await request.json()
                if 'audio' in json_data and json_data['audio']:
                    b64_str = json_data['audio']
                    if ',' in b64_str:
                        b64_str = b64_str.split(',')[1]
                    audio_bytes = base64.b64decode(b64_str)
            except Exception:
                form = await request.form()
                if 'audio' in form:
                    upload = form['audio']
                    if hasattr(upload, 'read'):
                        audio_bytes = await upload.read()

        if audio_bytes is None:
            return JSONResponse(status_code=400, content={'status': 'error', 'message': 'No audio received.'})

        result = voice_engine.analyze_audio(audio_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})


@app.post("/api/predict/text")
async def predict_text(payload: TextPayload):
    try:
        result = text_engine.analyze_text(payload.text)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})


@app.post("/api/predict/multimodal")
async def predict_multimodal(request: Request):
    try:
        face_res, voice_res, text_res = None, None, None

        # Check Multipart or JSON
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type:
            form = await request.form()
            if 'image' in form and hasattr(form['image'], 'read'):
                img_bytes = await form['image'].read()
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    face_res = face_engine.analyze_image_cv(img_bgr)

            if 'audio' in form and hasattr(form['audio'], 'read'):
                audio_bytes = await form['audio'].read()
                voice_res = voice_engine.analyze_audio(audio_bytes)

            text_val = form.get('text', '')
            if text_val and text_val.strip():
                text_res = text_engine.analyze_text(text_val.strip())
        else:
            json_data = await request.json()
            if json_data.get('image'):
                b64_str = json_data['image']
                if ',' in b64_str:
                    b64_str = b64_str.split(',')[1]
                img_bytes = base64.b64decode(b64_str)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    face_res = face_engine.analyze_image_cv(img_bgr)

            if json_data.get('audio'):
                b64_str = json_data['audio']
                if ',' in b64_str:
                    b64_str = b64_str.split(',')[1]
                audio_bytes = base64.b64decode(b64_str)
                voice_res = voice_engine.analyze_audio(audio_bytes)

            text_val = json_data.get('text', '')
            if text_val and text_val.strip():
                text_res = text_engine.analyze_text(text_val.strip())

        fusion_res = fusion_engine.fuse(face_res, voice_res, text_res)

        return JSONResponse(content={
            'multimodal_result': fusion_res,
            'individual_modalities': {
                'face': face_res,
                'voice': voice_res,
                'text': text_res
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})


# =============================================================================
# IN-WEBSITE PERSISTENCE & HISTORY ENDPOINTS
# =============================================================================

@app.get("/api/history")
async def get_history():
    storage = load_storage()
    return storage.get("history", [])


@app.post("/api/history")
async def add_history(item: HistoryItem):
    storage = load_storage()
    record = item.model_dump()
    if not record.get("id"):
        record["id"] = str(uuid.uuid4())[:8]
    if not record.get("timestamp"):
        record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    storage.setdefault("history", []).insert(0, record)
    # Cap history at 100 entries
    storage["history"] = storage["history"][:100]
    save_storage(storage)
    return {"status": "success", "item": record}


@app.delete("/api/history/{item_id}")
async def delete_history_item(item_id: str):
    storage = load_storage()
    orig_len = len(storage.get("history", []))
    storage["history"] = [h for h in storage.get("history", []) if h.get("id") != item_id]
    save_storage(storage)
    return {"status": "success", "deleted": orig_len - len(storage["history"])}


@app.delete("/api/history")
async def clear_all_history():
    storage = load_storage()
    storage["history"] = []
    save_storage(storage)
    return {"status": "success", "message": "All history cleared"}


@app.get("/api/snapshots")
async def get_snapshots():
    storage = load_storage()
    return storage.get("snapshots", [])


@app.post("/api/snapshots")
async def save_snapshot(item: SnapshotItem):
    storage = load_storage()
    record = item.model_dump()
    if not record.get("id"):
        record["id"] = str(uuid.uuid4())[:8]
    if not record.get("timestamp"):
        record["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    storage.setdefault("snapshots", []).insert(0, record)
    storage["snapshots"] = storage["snapshots"][:50]
    save_storage(storage)
    return {"status": "success", "snapshot": record}


@app.delete("/api/snapshots/{snapshot_id}")
async def delete_snapshot(snapshot_id: str):
    storage = load_storage()
    orig_len = len(storage.get("snapshots", []))
    storage["snapshots"] = [s for s in storage.get("snapshots", []) if s.get("id") != snapshot_id]
    save_storage(storage)
    return {"status": "success", "deleted": orig_len - len(storage["snapshots"])}


# =============================================================================
# SERVER ENTRYPOINT
# =============================================================================

def start_server(host="127.0.0.1", port=5000):
    import uvicorn
    print("\n" + "=" * 70)
    print(" SERENE EARTH AI - FASTAPI MULTIMODAL EMOTION RECOGNITION HUB")
    print(" Project Exhibition - I (DSN2098) | Group-52")
    print("=" * 70)
    print(f" [*] Web Dashboard URL : http://{host}:{port}")
    print(f" [*] Interactive OpenAPI Docs: http://{host}:{port}/docs")
    print(f" [*] REST API Base URL : http://{host}:{port}/api")
    print("=" * 70 + "\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()

