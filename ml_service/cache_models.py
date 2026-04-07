import os
import torch
from huggingface_hub import snapshot_download, login
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForAudioClassification, AutoModelForSequenceClassification
from faster_whisper import WhisperModel

def cache_all_models():
    print("--- PRE-CACHING MODELS FOR PRODUCTION (4GB RAM LIMIT) ---")
    hf_token = os.getenv("HF_TOKEN")
    
    if hf_token:
        login(token=hf_token)
        print("Authenticated with HF Hub.")

    # 1. TRANSCRIBER (Faster-Whisper)
    print("Pre-downloading Faster-Whisper Small...")
    snapshot_download(repo_id="Systran/faster-whisper-small")

    # 2. TRANSLATOR (Opus-MT - Lightweight replacement for NLLB)
    print("Pre-downloading Helsinki-NLP/opus-mt-mul-en (300MB)...")
    model_name = "Helsinki-NLP/opus-mt-mul-en"
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 3. SER (Speech Emotion Recognition)
    print("Pre-downloading MathRaaj/ser-fast-cnn-bilstm...")
    ser_id = "MathRaaj/ser-fast-cnn-bilstm"
    AutoModelForAudioClassification.from_pretrained(ser_id, trust_remote_code=True)

    # 4. TCA (Text Context Analysis)
    print("Pre-downloading MathRaaj/t1-bert-nli-baseline...")
    tca_id = "MathRaaj/t1-bert-nli-baseline"
    AutoTokenizer.from_pretrained(tca_id)
    AutoModelForSequenceClassification.from_pretrained(tca_id, trust_remote_code=True)

    print("--- ALL MODELS CACHED SUCCESSFULLY ---")

if __name__ == "__main__":
    cache_all_models()
