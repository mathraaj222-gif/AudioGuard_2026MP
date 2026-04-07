FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache TCA model
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='MathRaaj/T1_bert_nli_2')"

COPY . .

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
