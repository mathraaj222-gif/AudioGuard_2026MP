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
from transformers import pipeline
from faster_whisper import WhisperModel
from huggingface_hub import login, snapshot_download
import traceback

# --- Fusion Logic ---
def fallback_fusion(tca_probs, ser_probs, tca_threshold=0.75):
    tca_max = np.max(tca_probs)
    if tca_max >= tca_threshold:
        return np.argmax(tca_probs), tca_probs
    else:
        return np.argmax(ser_probs), ser_probs

# --- Language Mapping (Whisper to NLLB-200) ---
LANGUAGE_MAPPING = {
    "en": "eng_Latn", "ms": "zsm_Latn", "id": "ind_Latn", "es": "spa_Latn",
    "fr": "fra_Latn", "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn",
    "ar": "ara_Arab", "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Kore",
    "hi": "hin_Deva", "ta": "tam_Taml", "te": "tel_Telu", "ru": "rus_Cyrl"
}

# --- Model Pipeline ---
class InferencePipeline:
    def __init__(self):
        print("Initializing ML Service (Universal + Fast-CNN + AUTH Mode)...")
        self.hf_token = os.getenv("HF_TOKEN")
        
        # LOG IN TO HF HUB AT STARTUP
        if self.hf_token:
            try:
                login(token=self.hf_token)
                os.environ["HF_HUB_TOKEN"] = self.hf_token
                os.environ["HF_TOKEN"] = self.hf_token
                print("Hugging Face Hub authenticated successfully!")
            except Exception as e:
                print(f"Warning: HF Login Failed: {e}")
                
        self.device = "cpu" 
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def clear_memory(self):
        """Force garbage collection to prevent 4GB RAM overflow"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_url(self, video_url: str):
        # 0. DETECT IF INPUT IS AUDIO OR VIDEO
        is_audio = any(video_url.lower().endswith(ext) for ext in ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'])
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "input_source")
            audio_path = os.path.join(tmp_dir, "processed_audio.wav")
            
            # 1. Download
            print(f"Downloading source from {video_url}...")
            resp = requests.get(video_url, stream=True)
            with open(file_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            # 2. Extract & Preprocess (Conditional Format Switch)
            if is_audio:
                print("Step 0: Detected pure audio. Skipping video extraction.")
                audio_source = file_path
            else:
                print("Step 0: Detected video. Extracting audio track...")
                try:
                    video = VideoFileClip(file_path)
                    video.audio.write_audiofile(audio_path, logger=None)
                    video.close()
                    audio_source = audio_path
                except Exception as e:
                    print(f"Warning: Extraction issue: {e}. Falling back to direct load.")
                    audio_source = file_path
            
            print("Step 0.1: Resampling audio for AI analysis...")
            data, rate = librosa.load(audio_source, sr=16000)
            clean_audio = nr.reduce_noise(y=data, sr=rate, stationary=True)
            
            # 3. Transcription (Manual Snapshot Download for Stability)
            print("Step 1: Fetching & Transcribing (Small INT8)...")
            try:
                model_path = snapshot_download(
                    repo_id="Systran/faster-whisper-small", 
                    token=self.hf_token,
                    local_files_only=False
                )
                
                whisper_model = WhisperModel(model_path, device="cpu", compute_type="int8")
                segments, info = whisper_model.transcribe(clean_audio, beam_size=5)
                
                detected_lang = info.language
                original_text = "".join([s.text for s in segments]).strip()
                print(f"Detected language: {detected_lang}")
                
                del whisper_model
                self.clear_memory()
            except Exception as e:
                print(f"Whisper Fetch/Optimization Error: {e}")
                raise e

            # 4. Translation (NLLB-200) - Smarter Polyglot logic
            if detected_lang == "en" or not original_text.strip():
                print("Step 2: Skipping Translation (Already English or Empty)")
                english_text = original_text
            else:
                print(f"Step 2: Translating from {detected_lang} (NLLB-200)...")
                try:
                    src_lang_code = LANGUAGE_MAPPING.get(detected_lang, "eng_Latn")
                    translator = pipeline(
                        "translation", 
                        model="facebook/nllb-200-distilled-600M",
                        device=-1, # CPU
                        torch_dtype=self.dtype,
                        src_lang=src_lang_code,
                        tgt_lang="eng_Latn"
                    )
                    translation_result = translator(original_text, max_length=512)
                    english_text = translation_result[0]['translation_text']
                    del translator
                    self.clear_memory()
                except Exception as e:
                    print(f"NLLB Translation Error: {e}")
                    english_text = original_text # Fallback

            # 5. Emotion Recognition (Fast-CNN-BiLSTM)
            print("Step 3: Analyzing Emotion (MathRaaj/ser-fast-cnn-bilstm)...")
            try:
                ser_pipe = pipeline(
                    "audio-classification", 
                    model="MathRaaj/ser-fast-cnn-bilstm", 
                    token=self.hf_token,
                    device=-1, # CPU
                    torch_dtype=self.dtype,
                    trust_remote_code=True
                )
                ser_preds = ser_pipe({"array": clean_audio, "sampling_rate": 16000})
                top_ser = ser_preds[0]
                detected_emotion = top_ser['label']
                
                # REAL SCORE FUSION (NO DEFAULTS)
                ser_probs = np.zeros(2) 
                if any(x in detected_emotion.lower() for x in ['angry', 'anger', 'disgust', 'fear']):
                    ser_probs[1] = top_ser['score']
                else:
                    ser_probs[0] = top_ser['score']
                del ser_pipe
                self.clear_memory()
            except Exception as e:
                print(f"SER Error: {e}")
                detected_emotion = "Analysis Error"
                ser_probs = np.array([1.0, 0.0])

            # 6. Hate Speech Detection (BERT-NLI)
            print("Step 4: Analyzing Safety (TCA)...")
            try:
                tca_pipe = pipeline(
                    "text-classification", 
                    model="MathRaaj/t1-bert-nli-baseline", 
                    token=self.hf_token,
                    device=-1, # CPU
                    torch_dtype=self.dtype,
                    trust_remote_code=True
                )
                tca_preds = tca_pipe(english_text)[0]
                tca_probs = np.zeros(2)
                if 'label_1' in tca_preds['label'].lower() or 'hate' in tca_preds['label'].lower():
                    tca_probs[1] = tca_preds['score']
                else:
                    tca_probs[0] = tca_preds['score']
                del tca_pipe
                self.clear_memory()
            except Exception as e:
                print(f"TCA Error: {e}")
                tca_probs = np.array([1.0, 0.0])

            # 7. Final Fusion (Dynamic Result Rendering)
            final_class, _ = fallback_fusion(tca_probs, ser_probs)
            
            return {
                "transcription": original_text,
                "translation_en": english_text,
                "is_hatespeech": bool(final_class == 1),
                "confidence": f"{float(max(tca_probs[1], ser_probs[1]) if final_class == 1 else max(tca_probs[0], ser_probs[0])) * 100:.2f}%",
                "tca_confidence": f"{float(tca_probs[1] if final_class == 1 else tca_probs[0]) * 100:.2f}%",
                "ser_confidence": f"{float(ser_probs[1] if final_class == 1 else ser_probs[0]) * 100:.2f}%",
                "detected_emotion": detected_emotion,
                "original_language": detected_lang
            }

# --- FastAPI ---
app = FastAPI()
pipeline_instance = None

@app.on_event("startup")
async def startup():
    global pipeline_instance
    pipeline_instance = InferencePipeline()

class VideoRequest(BaseModel):
    video_url: str
    hf_token: str = None

@app.get("/")
def health(): return {"status": "ok"}

@app.post("/process")
async def process(request: VideoRequest):
    try:
        if request.hf_token:
            pipeline_instance.hf_token = request.hf_token
        return pipeline_instance.process_url(request.video_url)
    except Exception as e:
        error_name = type(e).__name__
        error_msg = str(e)
        print(f"ML Service Failure ({error_name}): {error_msg}")
        traceback.print_exc()
        return {
            "status": "failed",
            "error_detail": f"AI Brain Error ({error_name}): {error_msg}"
        }
