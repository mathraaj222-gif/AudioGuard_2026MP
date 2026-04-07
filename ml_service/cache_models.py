import os
import torch
from huggingface_hub import snapshot_download, login, hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForAudioClassification, AutoModelForSequenceClassification
from faster_whisper import WhisperModel

def cache_all_models():
    print("--- PRE-CACHING MODELS FOR PRODUCTION (4GB RAM) ---")
    hf_token = os.getenv("HF_TOKEN")
    
    if hf_token:
        try:
            login(token=hf_token)
            print("Authenticated with HF Hub.")
        except Exception as e:
            print(f"Login skip: {e}")

    # 1. TRANSCRIBER
    print("Caching Faster-Whisper Small...")
    snapshot_download(repo_id="Systran/faster-whisper-small")

    # 2. TRANSLATOR
    print("Caching Helsinki-NLP/opus-mt-mul-en...")
    model_name = "Helsinki-NLP/opus-mt-mul-en"
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 3. SER (Custom Architecture Bypass)
    # Use snapshot_download to avoid KeyError during build
    print("Caching MathRaaj/ser-fast-cnn-bilstm (Silent Cache)...")
    ser_id = "MathRaaj/ser-fast-cnn-bilstm"
    snapshot_download(repo_id=ser_id) # Caches weights, config, and modeling_*.py
    hf_hub_download(repo_id=ser_id, filename="pytorch_model.bin")

    # 4. TCA (Updated Version Bypass)
    print("Caching MathRaaj/T1_bert_nli_2 (Silent Cache)...")
    tca_id = "MathRaaj/T1_bert_nli_2"
    snapshot_download(repo_id=tca_id)
    hf_hub_download(repo_id=tca_id, filename="pytorch_model.bin")

    print("--- ALL MODELS BAKED SUCCESSFULLY ---")

if __name__ == "__main__":
    cache_all_models()
