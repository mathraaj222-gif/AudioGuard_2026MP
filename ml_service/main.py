import os
import tempfile
import requests
import numpy as np
import torch
import librosa
import noisereduce as nr
import gc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from moviepy import VideoFileClip
from transformers import pipeline, AutoModelForAudioClassification, AutoModelForSequenceClassification, AutoTokenizer

# --- Fusion Logic ---
def fallback_fusion(tca_probs, ser_probs, tca_threshold=0.75):
    tca_max = np.max(tca_probs)
    if tca_max >= tca_threshold:
        return np.argmax(tca_probs), tca_probs
    else:
        return np.argmax(ser_probs), ser_probs

# --- Model Pipeline ---
class InferencePipeline:
    def __init__(self):
        print("Initializing ML Service (Lazy Loading Mode)...")
        self.hf_token = os.getenv("HF_TOKEN")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def clear_memory(self):
        """Force garbage collection to prevent 4GB RAM overflow"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_url(self, video_url: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = os.path.join(tmp_dir, "input.mp4")
            audio_path = os.path.join(tmp_dir, "input.wav")
            
            # 1. Download
            print(f"Downloading {video_url}...")
            resp = requests.get(video_url, stream=True)
            with open(video_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            # 2. Extract & Preprocess
            print("Processing audio...")
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, logger=None)
            video.close()
            
            data, rate = librosa.load(audio_path, sr=16000)
            clean_audio = nr.reduce_noise(y=data, sr=rate, stationary=True)
            
            # 3. Transcription (Voxtral Mini V2)
            print("Step 1: Transcribing (Voxtral Mini V2)...")
            trans_pipe = pipeline(
                "automatic-speech-recognition", 
                model="Voxtral/Voxtral-Mini-Transcribe-v2", 
                device=self.device,
                torch_dtype=self.dtype
            )
            asr_result = trans_pipe(clean_audio, return_timestamps=False)
            original_text = asr_result["text"]
            
            # Note: Voxtral/Whisper provides language chunks, but we'll use a simple length check 
            # for the orchestrator until we implement deep language ID.
            detected_lang = "auto" 
            del trans_pipe
            self.clear_memory()

            # 4. Translation (NLLB-200)
            print("Step 2: Translating (NLLB-200)...")
            # We use NLLB to ensure the safety check (TCA) gets high-quality English
            translator = pipeline(
                "translation", 
                model="facebook/nllb-200-distilled-600M",
                device=self.device,
                torch_dtype=self.dtype,
                tgt_lang="eng_Latn"
            )
            # NLLB handles the source language automatically
            translation_result = translator(original_text, max_length=512)
            english_text = translation_result[0]['translation_text']
            del translator
            self.clear_memory()

            # 5. Emotion Recognition (Wav2Vec-BERT)
            print("Step 3: Analyzing Emotion...")
            ser_pipe = pipeline(
                "audio-classification", 
                model="MathRaaj/s3-wav2vec-bert", 
                token=self.hf_token,
                device=self.device,
                torch_dtype=self.dtype
            )
            ser_preds = ser_pipe({"array": clean_audio, "sampling_rate": 16000})
            top_ser = ser_preds[0]
            detected_emotion = top_ser['label']
            ser_probs = np.zeros(2) # Fallback binary for fusion
            if any(x in detected_emotion.lower() for x in ['angry', 'anger', 'disgust']):
                ser_probs[1] = top_ser['score']
            else:
                ser_probs[0] = top_ser['score']
            del ser_pipe
            self.clear_memory()

            # 6. Hate Speech Detection (BERT-NLI)
            print("Step 4: Analyzing Safety (TCA)...")
            tca_pipe = pipeline(
                "text-classification", 
                model="MathRaaj/t1-bert-nli-baseline", 
                token=self.hf_token,
                device=self.device,
                torch_dtype=self.dtype
            )
            tca_preds = tca_pipe(english_text)[0]
            tca_probs = np.zeros(2)
            if 'label_1' in tca_preds['label'].lower() or 'hate' in tca_preds['label'].lower():
                tca_probs[1] = tca_preds['score']
            else:
                tca_probs[0] = tca_preds['score']
            del tca_pipe
            self.clear_memory()

            # 7. Final Fusion
            final_class, _ = fallback_fusion(tca_probs, ser_probs)
            
            return {
                "transcription": original_text,
                "translation_en": english_text,
                "is_hatespeech": bool(final_class == 1),
                "confidence": f"{float(max(tca_probs[1], ser_probs[1]) if final_class == 1 else max(tca_probs[0], ser_probs[0])) * 100:.2f}%",
                "tca_confidence": f"{float(tca_probs[1] if final_class == 1 else tca_probs[0]) * 100:.2f}%",
                "ser_confidence": f"{float(ser_probs[1] if final_class == 1 else ser_probs[0]) * 100:.2f}%",
                "detected_emotion": detected_emotion,
                "original_language": "Detected via NLLB"
            }

import traceback

# --- FastAPI ---
app = FastAPI()
pipeline_instance = None

class VideoRequest(BaseModel):
    video_url: str
    hf_token: str = None

@app.on_event("startup")
async def startup():
    global pipeline_instance
    pipeline_instance = InferencePipeline()

@app.get("/")
def health(): return {"status": "ok"}

@app.post("/process")
async def process(request: VideoRequest):
    try:
        if request.hf_token:
            pipeline_instance.hf_token = request.hf_token
        return pipeline_instance.process_url(request.video_url)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
