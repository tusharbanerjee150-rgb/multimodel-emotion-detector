# EmotionX — Real-Time Multimodal Emotion Recognition System
> **DSN2098 - Project Exhibition | Group-52**  
> An end-to-end, high-performance real-time Multimodal Emotion Recognition platform powered by **PyTorch (CUDA GPU)**, **FastAPI**, and an interactive **React Dashboard**.

---

## 🌟 Overview

**EmotionX** is an advanced AI system capable of recognizing human emotional states with exceptional precision by synchronously combining three distinct biological and linguistic modalities:
1. **Visual Modality (Facial Expressions)**: Real-time high-FPS facial landmark localization (YuNet / Haar Cascades) with a deep **ResNet-18** feature extractor.
2. **Acoustic Modality (Speech Prosody & Tone)**: Mel-spectrogram & MFCC feature extraction with a **Deep CRNN** (1D-CNN + Bi-GRU + Self-Attention).
3. **Linguistic Modality (Text Semantics & NLP)**: Word embedding sequences analyzed by a **Bidirectional LSTM** with dense highway connections.
4. **Multimodal Tri-Modal Late Fusion**: Cross-modal attention neural network synthesizing all three channels to resolve ambiguity and achieve **98.63% test accuracy**.

The system classifies **7 Core Emotion Categories**:
`Angry` 😠, `Disgust` 🤢, `Fear` 😨, `Happy` 😊, `Neutral` 😐, `Sad` 😔, `Surprise` 😲.

---

## 📊 Model Performance & Accuracies

| Modality / System | Model Architecture | Train Accuracy | Val Accuracy | Test Accuracy | Weighted F1 | Training Datasets |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 👤 **Facial Emotion** | **ResNet-18** + Dual FC Head | **99.21%** | **73.74%** | **73.97%** | **73.74%** | AffectNet, FER-2013, RAF-DB *(137K+ images)* |
| 🎙️ **Speech Emotion (SER)** | **Deep CRNN** (1D-CNN + Bi-GRU + Attention) | **93.78%** | **92.45%** | **89.86%** | **89.84%** | RAVDESS, TESS, CREMA-D, SAVEE, MELD Audio |
| 📝 **Text Emotion (NLP)** | **Bi-LSTM** + Global Pooling | **98.34%** | **79.77%** | **79.04%** | **79.13%** | GoEmotions, MELD Transcripts |
| 🔮 **Tri-Modal Fusion** | **Cross-Modal Attention Late Fusion** | **98.53%** | **98.63%** | **98.63%** | **98.63%** | MELD Tri-Modal Dataset *(Face + Voice + Text)* |

---

## 📁 Evaluation Artifacts & Visualizations

- **Facial Emotion**:
  - Confusion Matrix: `models/results/confusion_matrix.png`
  - Training Curves: `models/results/training_accuracy.png`, `models/results/training_loss.png`
- **Speech Emotion**:
  - Confusion Matrix: `models/results/speech_confusion_matrix.png`
  - Training Curves: `models/results/speech_training_curves.png`
- **Text Emotion**:
  - Confusion Matrix: `models/results/text_confusion_matrix.png`
  - Training Curves: `models/results/text_training_curves.png`
- **Multimodal Fusion**:
  - Confusion Matrix: `models/results/multimodal_confusion_matrix.png`
  - Training Curves: `models/results/multimodal_training_curves.png`

---

## 🚀 Key Features

- **⚡ Real-Time Live Webcam & Audio Inference**: Ultra-low latency streaming over WebSockets and high-throughput REST endpoints.
- **📸 In-Website Snapshot Captures**: Instant webcam snapshot captures with bounding boxes, facial emotion labels, and timestamped local saving.
- **📈 Real-Time Multi-Model Comparison & Analytics**: Live telemetry comparing active predictions across Facial, Speech, Text, and Tri-modal fusion with dynamic radar charts and distribution plots.
- **🧠 Domain-Specific Actionable Recommendations**: Generates tailored psychological, educational, customer support, and driver safety suggestions based on detected affective states.
- **🎨 Glassmorphic EmotionX UI**: 7-petal emotional flower identity with dynamic color responsiveness according to valence and arousal.

---

## 🛠️ Project Structure

```
multimodal_emotion_detection/
├── data/                               # Local persistent database storage
├── models/                             # Trained PyTorch models, ONNX detectors & evaluation results
│   ├── facial_emotion_model_final.pth
│   ├── speech_emotion_model.pth
│   ├── text_emotion_model.pth
│   ├── multimodal_fusion_model.pth
│   ├── face_detection_yunet.onnx
│   ├── text_vocab.json
│   ├── emotion_labels.json
│   └── results/                        # Confusion matrices, curves, and reports
├── static/                             # Frontend styles, scripts, and vendor assets
│   ├── css/serene_earth.css
│   ├── js/react_app.js
│   ├── js/audio_recorder.js
│   └── vendor/                         # React, ReactDOM, Babel
├── templates/
│   └── index.html                      # Single-page application entry point
├── training_scripts/                   # Model training and data preprocessing scripts
├── main.py                             # FastAPI backend server & inference engine
├── part1_facial_emotion.py             # Standalone Part 1 facial emotion pipeline
├── part2_multimodal_system.py          # Standalone Part 2 multimodal pipeline
├── requirements.txt                    # Python dependencies
└── run_app.bat                         # Windows one-click startup batch script
```

---

## 💻 Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/tusharbanerjee150-rgb/multimodel-emotion-detector.git
cd multimodel-emotion-detector
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
You can run the application directly using the one-click batch script:
```bash
run_app.bat
```
Or with Python:
```bash
python main.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```
Interactive API documentation is available at `http://localhost:5000/docs`.
