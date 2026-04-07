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
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForAudioClassification, AutoModelForSequenceClassification, AutoConfig
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

class InferenceEngine:
    def __init__(self):
        self.device = "cpu"
        self.hf_token = os.getenv("HF_TOKEN")
        self.whisper = None
        self.translator = None
        self.ser_pipe = None
        self.tca_pipe = None
        self.ser_id2label = {0:"neutral",1:"calm",2:"happy",3:"sad",4:"angry",5:"fearful",6:"disgust"}
        
        if self.hf_token:
            try:
                login(token=self.hf_token)
            except Exception: pass

    def clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_whisper(self):
        if self.whisper is None:
            path = snapshot_download(repo_id="Systran/faster-whisper-small")
            self.whisper = WhisperModel(path, device="cpu", compute_type="int8")
        return self.whisper

    def get_translator(self):
        if self.translator is None:
            model_id = "Helsinki-NLP/opus-mt-mul-en"
            self.translator = pipeline("translation", model=model_id, device=-1)
        return self.translator

    def get_ser(self):
        """Robust SER Loading: MathRaaj/ser-fast-cnn-bilstm."""
        if self.ser_pipe is None:
            model_id = "MathRaaj/ser-fast-cnn-bilstm"
            print(f"Action: Registering and Loading SER {model_id}...")
            try:
                # Attempt high-level pipeline first (often handles mappings better)
                self.ser_pipe = pipeline(
                    "audio-classification", 
                    model=model_id, 
                    device=-1, 
                    trust_remote_code=True
                )
                print("SER Status: Online via Pipeline.")
            except (KeyError, Exception) as e:
                print(f"Pipeline fail ({type(e).__name__}), trying explicit AutoModel...")
                try:
                    # Fallback to explicit AutoModel if registry fails
                    model = AutoModelForAudioClassification.from_pretrained(model_id, trust_remote_code=True)
                    self.ser_pipe = pipeline("audio-classification", model=model, device=-1)
                    print("SER Status: Online via Explicit AutoModel.")
                except Exception as final_e:
                    print(f"Critical SER Failure: {final_e}")
                    raise final_e
        return self.ser_pipe

    def get_tca(self):
        """Robust TCA Loading: MathRaaj/T1_bert_nli_2."""
        if self.tca_pipe is None:
            model_id = "MathRaaj/T1_bert_nli_2"
            print(f"Action: Registering and Loading TCA {model_id}...")
            try:
                self.tca_pipe = pipeline(
                    "text-classification", 
                    model=model_id, 
                    device=-1, 
                    trust_remote_code=True
                )
                print("TCA Status: Online via Pipeline.")
            except (KeyError, Exception) as e:
                print(f"TCA Pipeline fail ({type(e).__name__}), trying explicit AutoModel...")
                try:
                    tok = AutoTokenizer.from_pretrained(model_id, use_auth_token=self.hf_token)
                    model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
                    self.tca_pipe = pipeline("text-classification", model=model, tokenizer=tok, device=-1)
                    print("TCA Status: Online via Explicit AutoModel.")
                except Exception as final_e:
                    print(f"Critical TCA Failure: {final_e}")
                    raise final_e
        return self.tca_pipe

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

            # 2. AUDIO EXTRACTION (v1 stability)
            if is_audio:
                audio_entry = source_path
            else:
                try:
                    video = VideoFileClip(source_path)
                    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
                    video.close()
                    audio_entry = audio_path
                except Exception:
                    audio_entry = source_path

            y, sr = librosa.load(audio_entry, sr=16000)
            clean_y = nr.reduce_noise(y=y, sr=sr, stationary=True)
            self.clear_memory()

            # 3. TRANSCRIPTION
            whisper = self.get_whisper()
            segments, _ = whisper.transcribe(clean_y, beam_size=5)
            transcription = "".join([s.text for s in segments]).strip()
            self.clear_memory()

            # 4. SER
            try:
                ser = self.get_ser()
                res = ser({"array": clean_y, "sampling_rate": 16000})[0]
                label = res['label'].lower()
                score = res['score']
                ser_probs = np.zeros(2)
                if any(x in label for x in ['angry', 'anger', 'disgust', 'fear']):
                    ser_probs[1] = score
                else: ser_probs[0] = score
                emotion = label
            except Exception:
                emotion = "Analysis Error"
                ser_probs = np.array([1.0, 0.0])
            self.clear_memory()

            # 5. TCA (using T1_bert_nli_2)
            try:
                tca = self.get_tca()
                res = tca(transcription)[0]
                score = res['score']
                tca_probs = np.zeros(2)
                if 'label_1' in res['label'].lower() or 'hate' in res['label'].lower():
                    tca_label = "Hostile Context Detected"
                    tca_probs[1] = score
                    tca_probs[0] = 1 - score
                else:
                    tca_label = "Safe Social Context"
                    tca_probs[0] = score
                    tca_probs[1] = 1 - score
            except Exception:
                tca_label = "Not Available"
                tca_probs = np.array([1.0, 0.0])
            self.clear_memory()

            # 6. FINAL FUSION
            final_class, _ = fallback_fusion(tca_probs, ser_probs)
            
            return {
                "transcription": transcription,
                "is_hatespeech": bool(final_class == 1),
                "confidence": f"{float(max(tca_probs[final_class], ser_probs[final_class])) * 100:.2f}%",
                "tca_label": tca_label,
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
        return {"status": "failed", "error": str(e)}
