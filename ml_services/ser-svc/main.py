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
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

class SEREngine:
    def __init__(self):
        print("SER-Svc: Loading MathRaaj/ser-optimized during startup...")
        model_id = "MathRaaj/ser-optimized"
        
        # Load Hugging Face components
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id)
        self.model.eval()
        
        # Labels are automatically loaded from model config
        self.labels = self.model.config.id2label
        print(f"SER-Svc: Models Loaded Successfully with labels: {list(self.labels.values())}")

    def process(self, audio_url: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "audio.wav")
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AudioGuard/1.0"}
            resp = requests.get(audio_url, headers=headers, timeout=30)
            if resp.status_code != 200:
                err_msg = f"Failed to download audio. Status: {resp.status_code}, URL: {audio_url}"
                print(f"SER-Svc Error: {err_msg}")
                raise Exception(err_msg)
                
            with open(path, "wb") as f:
                f.write(resp.content)
            
            # Load and preprocess
            y, sr = librosa.load(path, sr=16000)
            y = nr.reduce_noise(y=y, sr=sr)
            
            # Prepare inputs for WavLM
            inputs = self.extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                # We need output_hidden_states to extract the 768-dim embedding from the base model
                outputs = self.model(**inputs, output_hidden_states=True)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                # Embedding: Take the mean of the hidden states from the last transformer layer
                # This gives us a robust 768-dim feature vector for context
                last_hidden_state = outputs.hidden_states[-1]
                embedding = torch.mean(last_hidden_state, dim=1).squeeze().tolist()
            
            score, idx = torch.max(probs, dim=-1)
            gc.collect()
            
            return {
                "detected_emotion": self.labels[idx.item()],
                "ser_confidence": float(score),
                "embedding": embedding
            }

app = FastAPI(title="AudioGuard SER-Svc")
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    engine = SEREngine()

class EmotionRequest(BaseModel):
    audio_url: str

@app.post("/emotion")
def get_emotion(req: EmotionRequest):
    try:
        if engine is None:
            raise Exception("SER Engine not initialized")
        return engine.process(req.audio_url)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
