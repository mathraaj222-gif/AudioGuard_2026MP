import os
import tempfile
import requests
import numpy as np
import torch
import librosa
import noisereduce as nr
import gc
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from moviepy import VideoFileClip
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForAudioClassification, AutoModelForSequenceClassification
from faster_whisper import WhisperModel
from huggingface_hub import login, snapshot_download

# --- PRODUCTION CONFIG: 4GB RAM CPU ---
# Limit CPU contention to prevent OOM
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# --- Fusion Logic (Same as before) ---
def fallback_fusion(tca_probs, ser_probs, tca_threshold=0.75):
    tca_max = np.max(tca_probs)
    if tca_max >= tca_threshold:
        return np.argmax(tca_probs), tca_probs
    else:
        return np.argmax(ser_probs), ser_probs

# --- PRODUCTION SINGLETON ENGINE ---
class InferenceEngine:
    """Production wrapper for lazy loading and memory management on 4GB Cloud Run."""
    def __init__(self):
        self.device = "cpu"
        self.hf_token = os.getenv("HF_TOKEN")
        
        # Models (Lazy Initialization)
        self.whisper = None
        self.translator = None
        self.ser_model = None
        self.tca_model = None
        
        # Authentication
        if self.hf_token:
            try:
                login(token=self.hf_token)
            except Exception as e:
                print(f"Auth Warning: {e}")

    def clear_memory(self):
        """Force cleanup between heavy AI stages."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Lazy Loaders ---

    def get_whisper(self):
        if self.whisper is None:
            print("Loading Faster-Whisper Small (CPU INT8)...")
            # Snapshot ensures we use the cached version from Docker build
            path = snapshot_download(repo_id="Systran/faster-whisper-small")
            self.whisper = WhisperModel(path, device="cpu", compute_type="int8")
        return self.whisper

    def get_translator(self):
        if self.translator is None:
            # SWAP: NLLB-200 (2.4GB) -> OPUS-MT (300MB)
            print("Loading Helsinki-NLP/opus-mt-mul-en (Lightweight)...")
            model_id = "Helsinki-NLP/opus-mt-mul-en"
            self.translator = pipeline("translation", model=model_id, device=-1)
        return self.translator

    def get_ser(self):
        if self.ser_model is None:
            print("Loading MathRaaj/ser-fast-cnn-bilstm (PyTorch)...")
            model_id = "MathRaaj/ser-fast-cnn-bilstm"
            self.ser_model = pipeline("audio-classification", model=model_id, device=-1, trust_remote_code=True)
        return self.ser_model

    def get_tca(self):
        if self.tca_model is None:
            print("Loading MathRaaj/t1-bert-nli-baseline (Safety)...")
            model_id = "MathRaaj/t1-bert-nli-baseline"
            self.tca_model = pipeline("text-classification", model=model_id, device=-1, trust_remote_code=True)
        return self.tca_model

    # --- Processing Stages ---

    def analyze_source(self, url: str):
        """Master inference loop with internal fallbacks."""
        is_audio = any(url.lower().endswith(ext) for ext in ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'])
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = os.path.join(tmp_dir, "input")
            audio_path = os.path.join(tmp_dir, "clean.wav")

            # 1. DOWNLOAD
            print("Action: Downloading source stream...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(source_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            # 2. AUDIO EXTRACTION
            if is_audio:
                audio_entry = source_path
            else:
                try:
                    video = VideoFileClip(source_path)
                    video.audio.write_audiofile(audio_path, logger=None)
                    video.close()
                    audio_entry = audio_path
                except Exception:
                    audio_entry = source_path # Fallback

            # PRE-PROCESSING
            y, sr = librosa.load(audio_entry, sr=16000)
            clean_y = nr.reduce_noise(y=y, sr=sr, stationary=True)
            self.clear_memory()

            # 3. TRANSCRIPTION
            try:
                whisper = self.get_whisper()
                segments, info = whisper.transcribe(clean_y, beam_size=5)
                transcription = "".join([s.text for s in segments]).strip()
                detected_lang = info.language
                print(f"Whisper Info: Language={detected_lang}")
            except Exception as e:
                print(f"Whisper Error: {e}")
                transcription = ""
                detected_lang = "en"

            self.clear_memory()

            # 4. TRANSLATION (Lightweight Opus-MT)
            if detected_lang == "en" or not transcription:
                english_text = transcription
            else:
                try:
                    translator = self.get_translator()
                    # Opus-MT translates automatically from multilingual to EN
                    res = translator(transcription, max_length=512)
                    english_text = res[0]['translation_text']
                except Exception as e:
                    print(f"Translation Error: {e}")
                    english_text = transcription
            
            self.clear_memory()

            # 5. SER (EMOTION)
            try:
                ser = self.get_ser()
                ser_res = ser({"array": clean_y, "sampling_rate": 16000})
                top_ser = ser_res[0]
                emotion = top_ser['label']
                ser_probs = np.zeros(2)
                if any(x in emotion.lower() for x in ['angry', 'anger', 'disgust', 'fear']):
                    ser_probs[1] = top_ser['score']
                else:
                    ser_probs[0] = top_ser['score']
            except Exception as e:
                print(f"SER Error: {e}")
                emotion = "Analysis Error"
                ser_probs = np.array([1.0, 0.0])

            self.clear_memory()

            # 6. TCA (SAFETY)
            try:
                tca = self.get_tca()
                tca_res = tca(english_text)[0]
                tca_probs = np.zeros(2)
                if 'label_1' in tca_res['label'].lower() or 'hate' in tca_res['label'].lower():
                    tca_label = "Hostile Context Detected"
                    tca_probs[1] = tca_res['score']
                else:
                    tca_label = "Safe Social Context"
                    tca_probs[0] = tca_res['score']
            except Exception as e:
                print(f"TCA Error: {e}")
                tca_label = "Analysis Not Available"
                tca_probs = np.array([1.0, 0.0])

            self.clear_memory()

            # 7. FINAL FUSION
            final_class, _ = fallback_fusion(tca_probs, ser_probs)
            
            return {
                "transcription": transcription,
                "translation_en": english_text,
                "is_hatespeech": bool(final_class == 1),
                "confidence": f"{float(max(tca_probs[1], ser_probs[1]) if final_class == 1 else max(tca_probs[0], ser_probs[0])) * 100:.2f}%",
                "tca_confidence": f"{float(tca_probs[1] if final_class == 1 else tca_probs[0]) * 100:.2f}%",
                "ser_confidence": f"{float(ser_probs[1] if final_class == 1 else ser_probs[0]) * 100:.2f}%",
                "tca_label": tca_label,
                "detected_emotion": emotion,
                "original_language": detected_lang
            }

# --- WEB SERVICE ---
app = FastAPI()
engine = InferenceEngine()

class RequestModel(BaseModel):
    video_url: str
    hf_token: str = None

@app.get("/")
def health(): return {"status": "ok"}

@app.post("/process")
async def process(request: RequestModel):
    try:
        # Override token if provided
        if request.hf_token:
            engine.hf_token = request.hf_token
            login(token=request.hf_token)
            
        return engine.analyze_source(request.video_url)
    except Exception as e:
        print(f"System Failure: {e}")
        traceback.print_exc()
        return {
            "status": "failed",
            "error_detail": f"Production Engine Failure: {str(e)}"
        }
