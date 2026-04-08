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

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from huggingface_hub import snapshot_download

# --- CONFIG ---
torch.set_num_threads(1)

class InferenceEngine:
    def __init__(self):
        print("Whisper-Svc: Loading Models during startup...")
        # 1. Load Whisper
        path = snapshot_download("Systran/faster-whisper-small")
        self.whisper = WhisperModel(path, device="cpu", compute_type="int8")
        
        # 2. Load Translator (Directly using Model+Tokenizer to bypass pipeline task errors)
        translator_id = "Helsinki-NLP/opus-mt-mul-en"
        self.trans_tok = AutoTokenizer.from_pretrained(translator_id)
        self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(translator_id)
        print("Whisper-Svc: Models Loaded Successfully!")

    def process(self, audio_url: str):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = os.path.join(tmp, "audio.wav")
            
            # 1. Download from Cloudinary URL
            resp = requests.get(audio_url)
            if resp.status_code != 200:
                raise Exception(f"Failed to download audio from Cloudinary: {resp.status_code} - {resp.text[:200]}")
            
            with open(audio_path, "wb") as f:
                f.write(resp.content)

            # 2. Transcribe
            segments, info = self.whisper.transcribe(audio_path)
            text = "".join([s.text for s in segments]).strip()

            # 3. Translate if not English
            if info.language != "en" and text:
                try:
                    inputs = self.trans_tok(text, return_tensors="pt", max_length=512, truncation=True)
                    translated_tokens = self.trans_model.generate(**inputs)
                    english_text = self.trans_tok.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                except Exception as e:
                    print(f"Translation Error: {e}")
                    english_text = text
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
