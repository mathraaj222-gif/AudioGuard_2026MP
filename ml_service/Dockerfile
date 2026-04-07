# Use official Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the caching utility and pre-download models
# Note: Use --build-arg HF_TOKEN=... if pre-downloading private models
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
COPY cache_models.py .
RUN python cache_models.py

# Copy the rest of the application code
COPY . .

# Expose port (Cloud Run sets $PORT)
EXPOSE 8080

# Production settings: One thread, minimal workers for 4GB RAM
ENV TORCH_NUM_THREADS=1

# Command to run the application using uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
