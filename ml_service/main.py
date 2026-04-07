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
                # Load specialized model manually to avoid pipeline auto-feature issues
                config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                model = AutoModelForAudioClassification.from_pretrained(model_id, trust_remote_code=True)
                model.eval()
                self.ser_pipe = (model, config)
                print(f"SER Model Loaded: {model_id}")
            except Exception as e:
                print(f"SER Load Critical Failure: {e}")
                self.ser_pipe = None
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

    def _extract_ser_features(self, audio, sr=16000, n_mels=128, n_frames=400):
        """
        Expert Feature Extraction: Generates 3-channel Neural Image [1, 3, 128, 400]
        Channels: Mel-Spectrogram DB, Delta, Delta2
        """
        try:
            # 1. Base Mel Spectrogram
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=512)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # 2. Delta & Delta2 (Captures temporal dynamics)
            delta = librosa.feature.delta(mel_db)
            delta2 = librosa.feature.delta(mel_db, order=2)

            # 3. Stack into 3 channels
            features = np.stack([mel_db, delta, delta2], axis=0) # [3, 128, frames]

            # 4. Padding / Truncating to exact target (400 frames)
            if features.shape[2] < n_frames:
                pad_width = n_frames - features.shape[2]
                features = np.pad(features, ((0,0), (0,0), (0, pad_width)), mode='constant')
            else:
                features = features[:, :, :n_frames]

            # 5. Normalize (Z-score for Neural Input)
            mean = features.mean()
            std = features.std()
            if std > 0: features = (features - mean) / std

            return torch.from_numpy(features).unsqueeze(0).float() # [1, 3, 128, 400]
        except Exception:
            return None

    def _run_ser(self, audio):
        try:
            ser_data = self.get_ser()
            if not ser_data: return "Analysis Error", 0.0
            
            model, config = ser_data
            
            # Extract 3-channel features
            input_tensor = self._extract_ser_features(audio)
            if input_tensor is None: return "Analysis Error", 0.0

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                score, idx = torch.max(probs, dim=-1)
                
                label_map = config.id2label if hasattr(config, 'id2label') else {
                    0:"neutral", 1:"calm", 2:"happy", 3:"sad", 
                    4:"angry", 5:"fearful", 6:"disgust", 7:"surprise"
                }
                label = label_map.get(idx.item(), "unknown").lower()
                return label, score.item()
        except Exception as e:
            print(f"SER Neural Execution Error: {e}")
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
