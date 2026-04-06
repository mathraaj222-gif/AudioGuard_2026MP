import os
import shutil
import traceback
import tempfile
import requests
from fastapi import FastAPI, Depends, Request, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn

from database import engine, get_db, SessionLocal, Base
import models
from models import VideoRecord

# Environment variables for service communication
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8080").rstrip("/")
HF_TOKEN = os.getenv("HF_TOKEN")

from sqlalchemy import text

# Initialize DB tables with Auto-Migration for the Async Upgrade
try:
    # Check if the 'status' column exists (to detect outdated DB on Railway)
    with engine.connect() as conn:
        conn.execute(text("SELECT status FROM video_records LIMIT 1"))
    print("Database schema is up-to-date.")
except Exception:
    print("Outdated/Empty Database detected. Initializing schema for Async Upgrade...")
    try:
        # If it fails, we drop and recreate to ensure the new columns (status, video_url) are added
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        print("Database schema successfully recreated.")
    except Exception as e:
        print(f"Database Reset Warning (Expected on first run): {e}")

app = FastAPI(title="AudioGuard Orchestrator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://audio-guard-2026-mp.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Background Task ---
def perform_analysis_task(video_url: str, record_id: int):
    """Heavy lifting AI task that runs in the background"""
    db = SessionLocal()
    try:
        print(f"Bkg Task: Starting analysis for {video_url}...")
        payload = {"video_url": video_url, "hf_token": HF_TOKEN}
        resp = requests.post(f"{ML_SERVICE_URL}/process", json=payload, timeout=240)
        
        record = db.query(VideoRecord).filter(VideoRecord.id == record_id).first()
        if not record: return

        if resp.status_code != 200:
            print(f"Bkg Task: ML Service Error: {resp.text}")
            record.status = "FAILED"
        else:
            data = resp.json()
            record.transcription = data.get("transcription")
            record.translation_en = data.get("translation_en")
            record.original_language = data.get("original_language")
            record.detected_emotion = data.get("detected_emotion")
            record.is_hatespeech = data.get("is_hatespeech")
            record.confidence = data.get("confidence")
            record.tca_confidence = data.get("tca_confidence")
            record.ser_confidence = data.get("ser_confidence")
            record.status = "COMPLETED"
        
        db.commit()
    except Exception as e:
        print(f"Bkg Task: Failed: {e}")
        record = db.query(VideoRecord).filter(VideoRecord.id == record_id).first()
        if record:
            record.status = "FAILED"
            db.commit()
    finally:
        db.close()

# --- Routes ---
@app.get("/")
def health_check():
    return {"status": "online", "service": "audioguard-orchestrator"}

@app.post("/api/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    video_url: str = Form(...), 
    db: Session = Depends(get_db)
):
    """Immediately returns 202 Accepted and starts AI in background"""
    print(f"Orchestrator: Received Job for {video_url}")
    
    # 1. Create Initial Pending Record
    new_record = VideoRecord(
        filename=video_url.split("/")[-1],
        video_url=video_url,
        status="PENDING"
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)

    # 2. Add to Background Tasks
    background_tasks.add_task(perform_analysis_task, video_url, new_record.id)

    return JSONResponse(status_code=202, content={
        "status": "PENDING",
        "video_url": video_url,
        "message": "Analysis started in background. Polling required."
    })

@app.get("/api/status")
def get_status(video_url: str, db: Session = Depends(get_db)):
    """Frontend checks this endpoint every few seconds"""
    record = db.query(VideoRecord).filter(VideoRecord.video_url == video_url).order_by(VideoRecord.timestamp.desc()).first()
    
    if not record:
        return JSONResponse(status_code=404, content={"error": "Not found"})
    
    response = {
        "status": record.status,
        "video_url": record.video_url,
        "results": None
    }

    if record.status == "COMPLETED":
        response["results"] = {
            "transcription": record.transcription,
            "translation_en": record.translation_en,
            "original_language": record.original_language,
            "detected_emotion": record.detected_emotion,
            "is_hatespeech": record.is_hatespeech,
            "confidence": record.confidence,
            "tca_confidence": record.tca_confidence,
            "ser_confidence": record.ser_confidence
        }
    
    return response

@app.get("/api/videos")
def get_videos(db: Session = Depends(get_db)):
    return db.query(VideoRecord).order_by(VideoRecord.timestamp.desc()).all()

@app.delete("/api/videos/{video_id}")
def delete_video(video_id: int, db: Session = Depends(get_db)):
    record = db.query(VideoRecord).filter(VideoRecord.id == video_id).first()
    if record:
        db.delete(record)
        db.commit()
        return {"status": "success"}
    return JSONResponse(status_code=404, content={"error": "Not found"})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
