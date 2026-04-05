import os
import tempfile
import requests
import numpy as np
import torch
import whisper
import librosa
import noisereduce as nr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from moviepy import VideoFileClip
from transformers import pipeline

# --- Fusion Logic (Inlined to keep service standalone) ---
def fallback_fusion(tca_probs, ser_probs, tca_threshold=0.75):
    tca_max = np.max(tca_probs)
    if tca_max >= tca_threshold:
        return np.argmax(tca_probs), tca_probs
    else:
        return np.argmax(ser_probs), ser_probs

# --- Model Pipeline ---
class InferencePipeline:
    def __init__(self):
        print("Initializing ML Service Models...")
        
        # 1. Whisper (Transcription)
        print("Loading Whisper 'base' model...")
        self.whisper_model = whisper.load_model("base")
        
        # 2. SER Model (Emotion) - Loading from Hugging Face Hub
        print("Loading SER Model (wav2vec2-large)...")
        ser_model_id = "MathRaaj/S4_wav2vec2_large"
        try:
            self.ser_model = pipeline("audio-classification", model=ser_model_id)
            print(f"SER Model Loaded: {ser_model_id}")
        except Exception as e:
            print(f"Warning: Failed to load SER model from {ser_model_id} ({e}). Using mock data.")
            self.ser_model = None
            
        # 3. TCA Model (Context) - Loading from Hugging Face Hub
        print("Loading TCA Model (deberta-large-nli)...")
        tca_model_id = "MathRaaj/T3_deberta_large_nli"
        try:
            self.tca_model = pipeline("text-classification", model=tca_model_id, top_k=None)
            print(f"TCA Model Loaded: {tca_model_id}")
        except Exception as e:
            print(f"Warning: Failed to load TCA model from {tca_model_id} ({e}). Using mock data.")
            self.tca_model = None
        
        print("All models ready.")

    def process_url(self, video_url: str):
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = os.path.join(tmp_dir, "input_video.mp4")
            audio_path = os.path.join(tmp_dir, "extracted_audio.wav")
            
            # 1. Download Video
            print(f"Downloading video from {video_url}...")
            resp = requests.get(video_url, stream=True)
            if resp.status_code != 200:
                raise Exception(f"Failed to download video: {resp.status_code}")
            with open(video_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            # 2. Extract Audio
            print("Extracting audio...")
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, logger=None)
            video.close()

            # 3. Clean Audio
            print("Cleaning audio...")
            data, rate = librosa.load(audio_path, sr=16000)
            clean_audio_data = nr.reduce_noise(y=data, sr=rate)

            # 4. Transcribe
            print("Transcribing...")
            trans_result = self.whisper_model.transcribe(clean_audio_data, task="translate")
            transcription = trans_result["text"]

            # 5. Predict SER
            print("Predicting SER...")
            if self.ser_model:
                preds = self.ser_model({"array": clean_audio_data, "sampling_rate": 16000})
                ser_probs = np.zeros(7)
                # Map specific emotion labels to index 1 (hate/anger) or 0 (neutral)
                # This matches the logic from your original inference_pipeline.py
                pred_label = preds[0]['label'].lower()
                pred_score = preds[0]['score']
                if any(a in pred_label for a in ['angry', 'anger', 'hate']):
                    ser_probs[1] = pred_score
                else:
                    ser_probs[0] = pred_score
            else:
                ser_probs = np.array([0.9, 0.1, 0, 0, 0, 0, 0])

            # 6. Predict TCA
            print("Predicting TCA...")
            if self.tca_model:
                preds = self.tca_model(transcription)[0]
                tca_probs = np.zeros(7)
                for p in preds:
                    label = str(p['label']).lower()
                    score = float(p['score'])
                    if 'hate' in label or 'offensive' in label or 'label_1' in label:
                        tca_probs[1] += score
                    else:
                        tca_probs[0] += score
            else:
                tca_probs = np.array([0.9, 0.1, 0, 0, 0, 0, 0])

            # 7. Fusion
            final_class, merged_probs = fallback_fusion(tca_probs, ser_probs)
            
            return {
                "transcription": transcription,
                "is_hatespeech": bool(final_class == 1),
                "confidence": f"{float(np.max(merged_probs)) * 100:.2f}%",
                "tca_confidence": f"{float(np.max(tca_probs)) * 100:.2f}%",
                "ser_confidence": f"{float(np.max(ser_probs)) * 100:.2f}%",
                "detected_emotion": "Aggressive/Angry" if np.argmax(ser_probs) == 1 else "Neutral"
            }

# --- FastAPI App ---
app = FastAPI(title="AudioGuard ML Service")
pipeline = None

class VideoRequest(BaseModel):
    video_url: str

@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = InferencePipeline()

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "audioguard-ml"}

@app.post("/process")
async def process_video(request: VideoRequest):
    try:
        result = pipeline.process_url(request.video_url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
