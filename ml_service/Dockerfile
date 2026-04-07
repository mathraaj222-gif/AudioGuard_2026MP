# Use a specific stable version
FROM python:3.11-slim-bookworm

# 1. Install ONLY necessary system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libpq-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Optimization: Install requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Pre-cache models (Passing HF_TOKEN if needed)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
COPY cache_models.py .
# Run caching - we use a shell to ensure it cleans up
RUN python cache_models.py && rm cache_models.py

# 4. Copy app code
COPY . .

# 5. Production Env Vars
ENV TORCH_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1"]