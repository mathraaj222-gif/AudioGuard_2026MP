import os
import torch
from huggingface_hub import snapshot_download, login
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

    # 3. SER (Speech Emotion Recognition)
    # Using snapshot_download captures everything (safetensors or pytorch bin) automatically
    print("Caching MathRaaj/ser-fast-cnn-bilstm...")
    ser_id = "MathRaaj/ser-fast-cnn-bilstm"
    snapshot_download(repo_id=ser_id) # Captures all weight formats

    # 4. TCA (Text Context Analysis - T1_bert_nli_2)
    # Safetensors support is handled automatically by snapshot_download
    print("Caching MathRaaj/T1_bert_nli_2...")
    tca_id = "MathRaaj/T1_bert_nli_2"
    snapshot_download(repo_id=tca_id)

    print("--- ALL MODELS BAKED SUCCESSFULLY (SAFETENSORS SUPPORTED) ---")

if __name__ == "__main__":
    cache_all_models()
