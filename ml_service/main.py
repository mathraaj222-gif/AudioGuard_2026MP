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
from moviepy.editor import VideoFileClip
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForAudioClassification, AutoModelForSequenceClassification
from faster_whisper import WhisperModel
from huggingface_hub import login, snapshot_download, hf_hub_download

# --- PRODUCTION CONFIG: 4GB RAM CPU ---
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# --- Fusion Logic ---
def fallback_fusion(tca_probs, ser_probs, tca_threshold=0.75):
    tca_max = np.max(tca_probs)
    if tca_max >= tca_threshold:
        return np.argmax(tca_probs), tca_probs
    else:
        return np.argmax(ser_probs), ser_probs

# --- PRODUCTION SINGLETON ENGINE ---
class InferenceEngine:
    """Manual Model Loading Engine for Production (4GB RAM)."""
    def __init__(self):
        self.device = "cpu"
        self.hf_token = os.getenv("HF_TOKEN")
        
        # Singleton models
        self.whisper = None
        self.translator = None
        self.ser_model = None
        self.tca_model = None
        self.tca_tokenizer = None
        
        # Prediction Label Maps
        self.ser_id2label = {0:"neutral",1:"calm",2:"happy",3:"sad",4:"angry",5:"fearful",6:"disgust"}
        
        if self.hf_token:
            try:
                login(token=self.hf_token)
            except Exception as e:
                print(f"Auth Warning: {e}")

    def clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Lazy Loaders (Robust Expert Mode) ---

    def get_whisper(self):
        if self.whisper is None:
            print("Action: Loading Faster-Whisper Small...")
            path = snapshot_download(repo_id="Systran/faster-whisper-small")
            self.whisper = WhisperModel(path, device="cpu", compute_type="int8")
        return self.whisper

    def get_translator(self):
        if self.translator is None:
            print("Action: Loading Helsinki-NLP/opus-mt-mul-en...")
            model_id = "Helsinki-NLP/opus-mt-mul-en"
            self.translator = pipeline("translation", model=model_id, device=-1)
        return self.translator

    def get_ser(self):
        """Expert Loading: MathRaaj/ser-fast-cnn-bilstm."""
        if self.ser_model is None:
            print("Action: Manual Loading Custom SER Architecture...")
            model_id = "MathRaaj/ser-fast-cnn-bilstm"
            try:
                # Use standard AutoModel with trust_remote_code=True
                # This automatically downloads modeling_*.py from HF
                self.ser_model = AutoModelForAudioClassification.from_pretrained(
                    model_id, 
                    trust_remote_code=True
                )
                self.ser_model.eval()
                print("SER Activation: Success!")
            except Exception as e:
                print(f"SER Activation Failure: {e}")
                traceback.print_exc()
                raise e
        return self.ser_model

    def get_tca(self):
        """Expert Loading: MathRaaj/T1_bert_nli_2."""
        if self.tca_model is None:
            print("Action: Manual Loading Custom TCA Architecture (T1_bert_nli_2)...")
            model_id = "MathRaaj/T1_bert_nli_2"
            try:
                self.tca_tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.tca_model = AutoModelForSequenceClassification.from_pretrained(
                    model_id, 
                    trust_remote_code=True
                )
                self.tca_model.eval()
                print("TCA Activation: Success!")
            except Exception as e:
                print(f"TCA Activation Failure: {e}")
                traceback.print_exc()
                raise e
        return self.tca_model, self.tca_tokenizer

    # --- Processing Stages ---

    def analyze_source(self, url: str):
        is_audio = any(url.lower().endswith(ext) for ext in ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'])
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = os.path.join(tmp_dir, "input")
            audio_path = os.path.join(tmp_dir, "clean.wav")

            # 1. DOWNLOAD
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(source_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            # 2. AUDIO EXTRACTION (Stable MoviePy v1 Syntax)
            if is_audio:
                audio_entry = source_path
            else:
                try:
                    video = VideoFileClip(source_path)
                    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
                    video.close()
                    audio_entry = audio_path
                except Exception:
                    audio_entry = source_path # Fallback

            y, sr = librosa.load(audio_entry, sr=16000)
            clean_y = nr.reduce_noise(y=y, sr=sr, stationary=True)
            self.clear_memory()

            # 3. TRANSCRIPTION
            whisper = self.get_whisper()
            segments, info = whisper.transcribe(clean_y, beam_size=5)
            transcription = "".join([s.text for s in segments]).strip()
            detected_lang = info.language
            self.clear_memory()

            # 4. TRANSLATION
            if detected_lang == "en" or not transcription:
                english_text = transcription
            else:
                try:
                    translator = self.get_translator()
                    res = translator(transcription, max_length=512)
                    english_text = res[0]['translation_text']
                except Exception:
                    english_text = transcription
            self.clear_memory()

            # 5. SER (EMOTION) - Explicit Calculation
            try:
                ser_model = self.get_ser()
                # Dummy processor for custom mode (manual array loading)
                with torch.no_grad():
                    # MathRaaj custom models expect input_values usually
                    inputs = torch.tensor(clean_y).unsqueeze(0)
                    outputs = ser_model(inputs)
                    logits = outputs.logits
                    pred_id = torch.argmax(logits, dim=-1).item()
                    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
                    
                    emotion = self.ser_id2label.get(pred_id, "Unknown")
                    ser_probs = np.zeros(2)
                    # Mapping: angry, disgust, fearful are hostile (index 1)
                    if pred_id in [4, 5, 6]:
                        ser_probs[1] = probs[pred_id]
                    else:
                        ser_probs[0] = probs[pred_id]
            except Exception as e:
                print(f"Expert Emotion AI Failure: {e}")
                emotion = "Analysis Error"
                ser_probs = np.array([1.0, 0.0])
            self.clear_memory()

            # 6. TCA (SAFETY) - Explicit Calculation
            try:
                tca_model, tokenizer = self.get_tca()
                inputs = tokenizer(english_text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = tca_model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
                    pred_id = np.argmax(probs)
                    
                    if pred_id == 1 or "label_1" in str(tca_model.config.id2label.get(pred_id, "")):
                        tca_label = "Hostile Context Detected"
                        tca_probs = np.array([probs[0], probs[1]])
                    else:
                        tca_label = "Safe Social Context"
                        tca_probs = np.array([probs[0], probs[1]])
            except Exception as e:
                print(f"Expert Safety AI Failure: {e}")
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
        if request.hf_token:
            engine.hf_token = request.hf_token
            login(token=request.hf_token)
        return engine.analyze_source(request.video_url)
    except Exception as e:
        print(f"ML Service Failure: {str(e)}")
        traceback.print_exc()
        return {"status": "failed", "error_detail": str(e)}
