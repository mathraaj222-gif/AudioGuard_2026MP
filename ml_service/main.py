import os
import tempfile
import requests
import numpy as np
import torch
import librosa
import noisereduce as nr
import gc
import traceback

from fastapi import FastAPI
from pydantic import BaseModel
from moviepy.editor import VideoFileClip

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

# ------------------ CONFIG ------------------
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ------------------ CUSTOM SER MODEL ------------------
import torch.nn as nn

class SERModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.lstm = nn.LSTM(
            input_size=32 * 32,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(256, 8)

    def forward(self, x):
        x = self.conv(x)                     # [B, 32, H, W]
        b, c, h, w = x.shape
        x = x.view(b, w, c * h)             # [B, T, Features]
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)

# ------------------ ENGINE ------------------
class InferenceEngine:
    def __init__(self):
        self.whisper = None
        self.translator = None
        self.tca = None
        self.ser_model = None

    def clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- LOAD MODELS ----------
    def load_whisper(self):
        if self.whisper is None:
            path = snapshot_download("Systran/faster-whisper-small")
            self.whisper = WhisperModel(path, device="cpu", compute_type="int8")
        return self.whisper

    def load_translator(self):
        if self.translator is None:
            self.translator = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-mul-en",
                device=-1
            )
        return self.translator

    def load_tca(self):
        if self.tca is None:
            model_id = "MathRaaj/T1_bert_nli_2"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            self.tca = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return self.tca

    def load_ser(self):
        if self.ser_model is None:
            try:
                model_dir = snapshot_download("MathRaaj/ser-fast-cnn-bilstm")
                weights_path = os.path.join(model_dir, "pytorch_model.bin")

                model = SERModel()
                state_dict = torch.load(weights_path, map_location="cpu")

                # Fix DataParallel issue
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

                model.load_state_dict(state_dict)
                model.eval()

                self.ser_model = model
                print("✅ SER MODEL LOADED")
            except Exception as e:
                print(f"❌ SER LOAD ERROR: {e}")
                self.ser_model = None

        return self.ser_model

    # ---------- FEATURE EXTRACTION ----------
    def extract_features(self, audio, sr=16000, n_mels=128, n_frames=400):
        try:
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            delta = librosa.feature.delta(mel_db)
            delta2 = librosa.feature.delta(mel_db, order=2)

            features = np.stack([mel_db, delta, delta2], axis=0)

            if features.shape[2] < n_frames:
                pad = n_frames - features.shape[2]
                features = np.pad(features, ((0,0),(0,0),(0,pad)))
            else:
                features = features[:, :, :n_frames]

            mean = features.mean()
            std = features.std()
            if std > 0:
                features = (features - mean) / std

            return torch.tensor(features).unsqueeze(0).float()
        except Exception as e:
            print(f"Feature error: {e}")
            return None

    # ---------- SER ----------
    def run_ser(self, audio):
        model = self.load_ser()
        if model is None:
            return "error", 0.0

        features = self.extract_features(audio)
        if features is None:
            return "error", 0.0

        with torch.no_grad():
            logits = model(features)
            probs = torch.softmax(logits, dim=-1)

        score, idx = torch.max(probs, dim=-1)

        labels = ["neutral","calm","happy","sad","angry","fearful","disgust","surprise"]
        return labels[idx.item()], score.item()

    # ---------- WHISPER ----------
    def run_whisper(self, audio):
        model = self.load_whisper()
        segments, info = model.transcribe(audio)
        text = "".join([s.text for s in segments]).strip()
        return text, info.language

    # ---------- MAIN PIPELINE ----------
    def analyze(self, url):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = os.path.join(tmp, "input")

            # Download
            r = requests.get(url)
            with open(file_path, "wb") as f:
                f.write(r.content)

            # Extract audio
            if file_path.endswith((".mp4",".mov",".avi")):
                video = VideoFileClip(file_path)
                audio_path = os.path.join(tmp, "audio.wav")
                video.audio.write_audiofile(audio_path, logger=None)
                video.close()
            else:
                audio_path = file_path

            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)

            # Noise reduction
            y = nr.reduce_noise(y=y, sr=sr)

            # -------- SEQUENTIAL (SAFE) --------
            text, lang = self.run_whisper(y)
            emotion, ser_score = self.run_ser(y)

            # Translation
            if lang != "en" and text:
                translator = self.load_translator()
                text_en = translator(text)[0]['translation_text']
            else:
                text_en = text

            # TCA
            tca = self.load_tca()
            tca_res = tca(text_en)[0]

            is_hate = "1" in tca_res["label"]

            return {
                "transcription": text,
                "translation": text_en,
                "language": lang,
                "emotion": emotion,
                "ser_score": ser_score,
                "is_hatespeech": is_hate,
                "confidence": tca_res["score"]
            }

# ------------------ FASTAPI ------------------
app = FastAPI()
engine = InferenceEngine()

@app.on_event("startup")
def startup():
    engine.load_ser()

class Request(BaseModel):
    url: str

@app.post("/process")
def process(req: Request):
    try:
        return engine.analyze(req.url)
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}