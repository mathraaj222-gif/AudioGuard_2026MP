import os
import tempfile
import requests
import numpy as np
import torch
import librosa
import noisereduce as nr
import gc
import traceback
import concurrent.futures
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from moviepy.editor import VideoFileClip
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForAudioClassification, AutoModelForSequenceClassification, AutoConfig
from faster_whisper import WhisperModel
from huggingface_hub import login, snapshot_download

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

class InferenceEngine:
    def __init__(self):
        self.device = "cpu"
        self.hf_token = os.getenv("HF_TOKEN")
        self.whisper = None
        self.translator = None
        self.ser_pipe = None
        self.tca_pipe = None
        
        if self.hf_token:
            try: login(token=self.hf_token)
            except Exception: pass

    def clear_memory(self):
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def get_whisper(self):
        if self.whisper is None:
            path = snapshot_download(repo_id="Systran/faster-whisper-small")
            self.whisper = WhisperModel(path, device="cpu", compute_type="int8")
        return self.whisper

    def get_translator(self):
        if self.translator is None:
            self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en", device=-1)
        return self.translator

    def get_ser(self):
        if self.ser_pipe is None:
            model_id = "MathRaaj/ser-fast-cnn-bilstm"
            try:
                self.ser_pipe = pipeline("audio-classification", model=model_id, device=-1, trust_remote_code=True)
            except Exception:
                model = AutoModelForAudioClassification.from_pretrained(model_id, trust_remote_code=True)
                self.ser_pipe = pipeline("audio-classification", model=model, device=-1)
        return self.ser_pipe

    def get_tca(self):
        if self.tca_pipe is None:
            model_id = "MathRaaj/T1_bert_nli_2"
            try:
                self.tca_pipe = pipeline("text-classification", model=model_id, device=-1, trust_remote_code=True)
            except Exception:
                tok = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
                self.tca_pipe = pipeline("text-classification", model=model, tokenizer=tok, device=-1)
        return self.tca_pipe

    def _run_whisper(self, audio):
        model = self.get_whisper()
        segments, info = model.transcribe(audio, beam_size=5)
        text = "".join([s.text for s in segments]).strip()
        return text, info.language

    def _run_ser(self, audio):
        try:
            pipe = self.get_ser()
            # FIX: Ensure audio is float32 for custom BiLSTM layers
            input_audio = audio.astype(np.float32)
            res = pipe({"array": input_audio, "sampling_rate": 16000})[0]
            return res['label'].lower(), res['score']
        except Exception as e:
            print(f"SER Thread Error: {e}")
            return "Analysis Error", 0.0

    def analyze_source(self, url: str):
        is_audio = any(url.lower().endswith(ext) for ext in ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'])
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            source_path = os.path.join(tmp_dir, "input")
            audio_path = os.path.join(tmp_dir, "clean.wav")

            # 1. DOWNLOAD
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(source_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)

            # 2. AUDIO EXTRACTION
            if is_audio: audio_entry = source_path
            else:
                try:
                    video = VideoFileClip(source_path)
                    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
                    video.close()
                    audio_entry = audio_path
                except Exception: audio_entry = source_path

            y, sr = librosa.load(audio_entry, sr=16000)
            clean_y = nr.reduce_noise(y=y, sr=sr, stationary=True)
            self.clear_memory()

            # 3. PARALLEL EXECUTION (Whisper + SER)
            print("Action: Starting Parallel Track (Whisper & SER)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_whisper = executor.submit(self._run_whisper, clean_y)
                future_ser = executor.submit(self._run_ser, clean_y)
                
                transcription, detected_lang = future_whisper.result()
                emotion, ser_score = future_ser.result()

            self.clear_memory()

            # 4. TRANSLATION & TCA (Sequential because they depend on text)
            if detected_lang == "en" or not transcription:
                english_text = transcription
            else:
                try:
                    translator = self.get_translator()
                    res = translator(transcription, max_length=512)
                    english_text = res[0]['translation_text']
                except Exception: english_text = transcription
            
            # TCA Processing
            try:
                tca = self.get_tca()
                tca_res = tca(english_text)[0]
                t_score = tca_res['score']
                t_label = tca_res['label'].lower()
                
                tca_probs = np.zeros(2)
                if 'label_1' in t_label or 'hate' in t_label:
                    tca_label = "Hostile Context Detected"
                    tca_probs[1] = t_score; tca_probs[0] = 1 - t_score
                else:
                    tca_label = "Safe Social Context"
                    tca_probs[0] = t_score; tca_probs[1] = 1 - t_score
            except Exception:
                tca_label = "Analysis Not Available"
                tca_probs = np.array([1.0, 0.0])

            # SER Probability Mapping
            ser_probs = np.zeros(2)
            if any(x in emotion for x in ['angry', 'anger', 'disgust', 'fear']):
                ser_probs[1] = ser_score; ser_probs[0] = 1 - ser_score
            else:
                ser_probs[0] = ser_score; ser_probs[1] = 1 - ser_score

            # 5. FINAL FUSION
            final_class, _ = fallback_fusion(tca_probs, ser_probs)
            
            # FIX: Restore all keys for Dashboard (tca_confidence, ser_confidence, lang, translation)
            return {
                "transcription": transcription,
                "translation_en": english_text,
                "original_language": detected_lang,
                "is_hatespeech": bool(final_class == 1),
                "confidence": f"{float(max(tca_probs[final_class], ser_probs[final_class])) * 100:.2f}%",
                "tca_label": tca_label,
                "tca_confidence": f"{float(tca_probs[1] if final_class == 1 else tca_probs[0]) * 100:.2f}%",
                "ser_confidence": f"{float(ser_probs[1] if final_class == 1 else ser_probs[0]) * 100:.2f}%",
                "detected_emotion": emotion
            }

app = FastAPI()
engine = InferenceEngine()

class RequestModel(BaseModel):
    video_url: str
    hf_token: str = None

@app.post("/process")
async def process(request: RequestModel):
    try: return engine.analyze_source(request.video_url)
    except Exception as e:
        traceback.print_exc()
        return {"status": "failed", "error_detail": str(e)}
