import os
import tempfile
import requests
import torch
import gc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel
from transformers import pipeline
from huggingface_hub import snapshot_download

# --- CONFIG ---
torch.set_num_threads(1)

class InferenceEngine:
    def __init__(self):
        print("Whisper-Svc: Loading Models during startup...")
        path = snapshot_download("Systran/faster-whisper-small")
        self.whisper = WhisperModel(path, device="cpu", compute_type="int8")
        self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en", device=-1)
        print("Whisper-Svc: Models Loaded Successfully!")

    def process(self, audio_url: str):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = os.path.join(tmp, "audio.wav")
            
            # 1. Download from Cloudinary URL
            resp = requests.get(audio_url)
            with open(audio_path, "wb") as f:
                f.write(resp.content)

            # 2. Transcribe
            segments, info = self.whisper.transcribe(audio_path)
            text = "".join([s.text for s in segments]).strip()

            # 3. Translate if not English
            if info.language != "en" and text:
                english_text = self.translator(text)[0]['translation_text']
            else:
                english_text = text

            gc.collect()
            return {
                "transcription": text,
                "translation_en": english_text,
                "original_language": info.language
            }

app = FastAPI(title="AudioGuard Whisper-Svc")
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    engine = InferenceEngine()

class AudioRequest(BaseModel):
    audio_url: str

@app.post("/transcribe")
def transcribe(req: AudioRequest):
    try:
        if engine is None:
            raise Exception("Inference Engine not initialized")
        return engine.process(req.audio_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
