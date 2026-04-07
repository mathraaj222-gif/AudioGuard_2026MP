FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache models for Cloud Run fast startup
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-small')"
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    model_id='Helsinki-NLP/opus-mt-mul-en'; \
    AutoTokenizer.from_pretrained(model_id); \
    AutoModelForSeq2SeqLM.from_pretrained(model_id)"

COPY . .

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
