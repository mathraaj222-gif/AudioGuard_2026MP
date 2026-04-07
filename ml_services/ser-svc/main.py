import os
import tempfile
import requests
import torch
import numpy as np
import librosa
import noisereduce as nr
import gc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import snapshot_download
from model import SERModel

class SEREngine:
    def __init__(self):
        self.model = None
        self.labels = ["neutral","calm","happy","sad","angry","fearful","disgust","surprise"]

    def load_model(self):
        if self.model is None:
            model_dir = snapshot_download("MathRaaj/ser-fast-cnn-bilstm")
            weights_path = os.path.join(model_dir, "pytorch_model.bin")
            
            model = SERModel()
            state_dict = torch.load(weights_path, map_location="cpu")
            # Remove module. prefix if present from DataParallel
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
        return self.model

    def extract_features(self, audio, sr=16000, n_mels=128, n_frames=400):
        try:
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            delta = librosa.feature.delta(mel_db)
            delta2 = librosa.feature.delta(mel_db, order=2)
            
            features = np.stack([mel_db, delta, delta2], axis=0)
            
            if features.shape[2] < n_frames:
                pad = n_frames - features.shape[2]
                features = np.pad(features, ((0,0), (0,0), (0, pad)))
            else:
                features = features[:, :, :n_frames]
            
            mean = features.mean()
            std = features.std()
            if std > 0: features = (features - mean) / std
            
            return torch.tensor(features).unsqueeze(0).float()
        except Exception:
            return None

    def process(self, audio_url: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "audio.wav")
            resp = requests.get(audio_url)
            with open(path, "wb") as f:
                f.write(resp.content)
            
            y, sr = librosa.load(path, sr=16000)
            y = nr.reduce_noise(y=y, sr=sr)
            
            features = self.extract_features(y)
            if features is None:
                raise ValueError("Feature extraction failed")
            
            model = self.load_model()
            with torch.no_grad():
                logits = model(features)
                probs = torch.softmax(logits, dim=-1)
            
            score, idx = torch.max(probs, dim=-1)
            gc.collect()
            
            return {
                "detected_emotion": self.labels[idx.item()],
                "ser_confidence": float(score)
            }

app = FastAPI(title="AudioGuard SER-Svc")
engine = SEREngine()

class EmotionRequest(BaseModel):
    audio_url: str

@app.post("/emotion")
def get_emotion(req: EmotionRequest):
    try:
        return engine.process(req.audio_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
