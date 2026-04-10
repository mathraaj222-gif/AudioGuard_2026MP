# 🛡️ AudioGuard: Intelligent Multi-Modal Hate Speech Detection

AudioGuard is a state-of-the-art AI pipeline designed to detect hostile and toxic content in audio streams. By leveraging a **Meta-Classification** architecture, it combines acoustic embeddings (emotions/events) with linguistic context (text) to provide a robust, AI-driven safety decision.

---

## 🏗️ System Architecture & Structure

AudioGuard is built as a **Microservices-based Orchestration Pipeline**, ensuring scalability and modular model management.

### 1. Core Components
- **Backend Orchestrator**: The central "Postman" (FastAPI) that coordinates parallel extraction from AI services and hands the data to the Meta-Classifier.
- **Whisper-Service**: Specialized endpoint for Speech-to-Text and automated translation.
- **SER-Service**: Extracts acoustic features and 768-dim embeddings using **MathRaaj/ser-optimized**.
- **TCA-Service**: Performs linguistic analysis and extracts 768-dim textual embeddings.
- **Meta-Service**: The "Brain" of the system. An ensemble AI (LightGBM + Cross-Modal Attention) that makes the final safety verdict.
- **Frontend Dashboard**: interface for visual analysis and historical record tracking.

---

## 🧠 Model Ecosystem & Fusion logic

### 1. Base AI Models
The system uses specialized architectures to generate rich context:

| Model ID | Service | Task | Output |
| :--- | :--- | :--- | :--- |
| `MathRaaj/ser-optimized` | SER | Acoustic | 8-class labels + 768-dim Embedding |
| `MathRaaj/T1_bert_nli_3` | TCA | Linguistic| Hostility labels + 768-dim Embedding |
| `whisper-small` | STT | Speech | Multilingual Transcription & Translation |

### 2. The Meta-Classifier (The Final Judge)
AudioGuard has moved away from simple "weighted average" math. The **Meta-Service** now receives a concatenated **1536-dimensional vector** (768 Audio + 768 Text) and uses an ensemble approach:

- **LightGBM Component**: Trained to detect tabular patterns in multi-modal probabilities.
- **Cross-Modal Attention Model**: A neural network that learns which modality (text or audio) to trust more based on the specific context.
- **Decision Logic**: The system uses a configurable `HATE_THRESHOLD` (default: `0.35`) to flag content as "Hate/Offensive".

---

## 🔄 The Pipeline Workflow

The process follows a high-performance asynchronous flow:

### 1. Input & Parallel Extraction
The system receives an Audio/Video URL. It simultaneously triggers **Whisper** (for text) and **SER** (for audio embeddings).

### 2. Linguistic Analysis
Whisper's English translation is sent to **TCA**, which generates a textual understanding and a corresponding embedding.

### 3. Meta-Fusion
The Backend grabs the **768-dim thoughts** from both SER and TCA, glues them into a **1536-dim vector**, and sends them to the **Meta-Service**.

### 4. Final Verdict
The Meta-Service provides the final verdict ($is\_hateful: true/false$). results are persisted in the database and broadcasted to the dashboard. **The Meta-Classifier's output is the sole authority for the final safety flag.**

---

## 🧪 Data Transformation Flow

1.  **Remote File (URL)**: Input medium.
2.  **Raw Waveform (`.wav`)**: Standardized to 16kHz Mono.
3.  **Embeddings (Numerical Thoughts)**: 
    - **SER**: 768 floating-point numbers representing acoustic context.
    - **TCA**: 768 floating-point numbers representing linguistic context.
4.  **1536-dim Vector**: The combined "world-view" of the audio clip.
5.  **Final Score**: A single probability score (0 to 1) from the Meta-Ensemble.
6.  **JSON Output**: Consumed by the Frontend for rendering.

---

## ⚡ Deployment & Running
The project is containerized using **Docker Compose**.

```bash
docker-compose up --build
```

- **Backend**: `http://localhost:8000`
- **Whisper Service**: `http://localhost:8081`
- **SER Service**: `http://localhost:8082`
- **TCA Service**: `http://localhost:8083`

---
*Created for AudioGuard Major Project 2026. Designed for safety, speed, and accuracy.*
