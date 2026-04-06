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
        
        # 0. Get HF Token from Env (For private models)
        self.hf_token = os.getenv("HF_TOKEN")
        
        # 1. Whisper (Transcription)
        print("Loading Whisper 'small' model (Verbatim Update)...")
        self.whisper_model = whisper.load_model("small")
        
        # 2. SER Model (Emotion) - Loading from Hugging Face Hub
        print("Loading SER Model (wav2vec-bert)...")
        ser_model_id = "MathRaaj/s3-wav2vec-bert"
        try:
            self.ser_model = pipeline(
                "audio-classification", 
                model=ser_model_id, 
                token=self.hf_token,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print(f"SER Model Loaded: {ser_model_id}")
        except Exception as e:
            print(f"Warning: Failed to load SER model from {ser_model_id} ({e}). Using mock data.")
            self.ser_model = None
            
        # 3. TCA Model (Context) - Loading from Hugging Face Hub
        print("Loading TCA Model (bert-nli)...")
        tca_model_id = "MathRaaj/t1-bert-nli-baseline"
        try:
            self.tca_model = pipeline(
                "text-classification", 
                model=tca_model_id, 
                top_k=None, 
                token=self.hf_token,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
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

            # 4. Transcribe (Optimized Single-Pass)
            print("Transcribing and Translating (Single Pass)...")
            # Running with task="translate" gives us EN text + metadata about the original language
            trans_result = self.whisper_model.transcribe(clean_audio_data, task="translate")
            
            tca_input_text = trans_result["text"] # This is the English version for the AI check
            detected_lang = trans_result.get("language", "en")
            
            # Since task="translate" only returns English, we'll use the EN text as the main transcript
            # for maximum speed to avoid timeouts.
            original_transcription = tca_input_text 

            # 5. Predict SER (7-Emotion Expansion)
            print("Predicting SER...")
            if self.ser_model:
                preds = self.ser_model({"array": clean_audio_data, "sampling_rate": 16000})
                ser_probs = np.zeros(7)
                
                # Get the detailed top predicted emotion from your S3 model
                top_pred = preds[0]
                detected_emotion_label = top_pred['label'] # Full emotion name (e.g., Happy, Sad)
                pred_score = top_pred['score']
                
                # Still map for the Fusion logic (Binary safety check)
                if any(a in detected_emotion_label.lower() for a in ['angry', 'anger', 'disgust', 'hate']):
                    ser_probs[1] = pred_score # Aggressive/High-risk
                else:
                    ser_probs[0] = pred_score # Normal/Low-risk
            else:
                detected_emotion_label = "Neutral"
                ser_probs = np.array([0.9, 0.1, 0, 0, 0, 0, 0])

            # 6. Predict TCA
            print("Predicting TCA...")
            if self.tca_model:
                preds = self.tca_model(tca_input_text)[0]
                tca_probs = np.zeros(7)
                for p in preds:
                    label = str(p['label']).lower()
                    score = float(p['score'])
                    if any(l in label for l in ['hate', 'offensive', 'label_1', 'negative']):
                        tca_probs[1] += score
                    else:
                        tca_probs[0] += score
            else:
                tca_probs = np.array([0.9, 0.1, 0, 0, 0, 0, 0])

            # 7. Fusion
            final_class, merged_probs = fallback_fusion(tca_probs, ser_probs)
            
            return {
                "transcription": original_transcription,
                "translation_en": tca_input_text if detected_lang != "en" else "Already in English",
                "is_hatespeech": bool(final_class == 1),
                "confidence": f"{float(np.max(merged_probs)) * 100:.2f}%",
                "tca_confidence": f"{float(np.max(tca_probs)) * 100:.2f}%",
                "ser_confidence": f"{float(np.max(ser_probs)) * 100:.2f}%",
                "detected_emotion": detected_emotion_label,
                "original_language": detected_lang
            }

import traceback

# --- FastAPI App ---
app = FastAPI(title="AudioGuard ML Service")
pipeline = None

class VideoRequest(BaseModel):
    video_url: str
    hf_token: str = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        pipeline = InferencePipeline()
    except Exception as e:
        print(f"CRITICAL: Failed to initialize pipeline: {e}")
        traceback.print_exc()

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "audioguard-ml"}

@app.post("/process")
async def process_video(request: VideoRequest):
    try:
        if pipeline is None:
            raise Exception("ML Pipeline not initialized properly on startup.")
        
        # Override the pipeline's token if one is provided in the request
        if request.hf_token:
            pipeline.hf_token = request.hf_token
            
        result = pipeline.process_url(request.video_url)
        return result
    except Exception as e:
        print(f"ML Service Error: {e}")
        traceback.print_exc() # This prints to Google Cloud Logs
        # Return the actual error message to the Orchestrator
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
