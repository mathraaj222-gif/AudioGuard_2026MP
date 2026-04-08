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

from sqlalchemy import text
import asyncio
import httpx

# Microservices URLs (Set these in your Cloud Run Environment Variables)
WHISPER_URL = os.getenv("WHISPER_URL", "http://localhost:8081").rstrip("/")
SER_URL = os.getenv("SER_URL", "http://localhost:8082").rstrip("/")
TCA_URL = os.getenv("TCA_URL", "http://localhost:8083").rstrip("/")

# Initialize DB tables with Auto-Migration for the Dashboard Restoration
try:
    # Check if the latest feature columns exist (to detect outdated DB on Railway)
    with engine.connect() as conn:
        conn.execute(text("SELECT tca_confidence FROM video_records LIMIT 1"))
    print("Database schema is up-to-date.")
except Exception:
    print("Outdated Database detected. Forcing schema update for Dashboard Restoration...")
    try:
        # If it fails, we drop and recreate to ensure all columns (tca_confidence, etc) are added.
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        print("Database schema successfully synchronized.")
    except Exception as e:
        print(f"Database Sync Warning: {e}")

app = FastAPI(title="AudioGuard Orchestrator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://audio-guard-2026-mp.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def to_audio_url(video_url: str) -> str:
    """
    Cloudinary Magic: Transforms a Video URL into an optimized 16kHz Mono WAV URL
    Example: .../upload/v123/video.mp4 -> .../upload/f_wav,ac_1,ar_16000/v123/video.wav
    """
    # Simply replace the extension and inject the transformation parameters
    if "/upload/" in video_url:
        # Standardize extension to .wav and inject high-speed audio params
        transformed = video_url.replace("/upload/", "/upload/f_wav,ach_1,ar_16000/")
        # Ensure it ends with .wav for some older librosa versions
        base, _ = os.path.splitext(transformed)
        return base + ".wav"
    return video_url

# Fusion logic from monolith
def fallback_fusion(tca_label, tca_conf, ser_emotion, ser_conf):
    """Combines TCA and SER results into a final safety decision"""
    is_hostile_tca = "hostile" in tca_label.lower()
    is_hostile_ser = any(x in ser_emotion.lower() for x in ['angry', 'anger', 'disgust', 'fear'])
    
    # Priority 1: High confidence TCA detection
    if is_hostile_tca and tca_conf > 0.7:
        return True, f"{tca_conf * 100:.1f}%"
    
    # Priority 2: Agreement between models
    if is_hostile_tca and is_hostile_ser:
        return True, f"{max(tca_conf, ser_conf) * 100:.1f}%"
        
    # Default: Safe if not explicitly hostile
    return False, f"{max(tca_conf, ser_conf) * 100:.1f}%"

# --- Async Background Task ---
async def perform_analysis_task(video_url: str, record_id: int):
    """Master Orchestrator: Coordinates parallel tasks across AI services"""
    db = SessionLocal()
    try:
        print(f"Orchestration: Starting Distributed Analysis for Job {record_id}...")
        audio_url = to_audio_url(video_url)
        
        async with httpx.AsyncClient(timeout=300) as client:
            # 1. Start Whisper and SER in Parallel
            print(f" -> Dispatching Parallel Tracks (Whisper & SER)...")
            whisper_future = client.post(f"{WHISPER_URL}/transcribe", json={"audio_url": audio_url})
            ser_future = client.post(f"{SER_URL}/emotion", json={"audio_url": audio_url})
            
            whisper_resp, ser_resp = await asyncio.gather(whisper_future, ser_future)
            
            if whisper_resp.status_code != 200 or ser_resp.status_code != 200:
                w_err = whisper_resp.text if whisper_resp.status_code != 200 else "OK"
                s_err = ser_resp.text if ser_resp.status_code != 200 else "OK"
                raise Exception(f"Phase 1 Failure: Whisper({whisper_resp.status_code}: {w_err}) SER({ser_resp.status_code}: {s_err})")
            
            w_data = whisper_resp.json()
            s_data = ser_resp.json()
            
            # 2. Run TCA Sequential (Uses translation from Whisper)
            print(f" -> Dispatching TCA Analysis...")
            tca_resp = await client.post(f"{TCA_URL}/analyze", json={"text": w_data["translation_en"]})
            
            if tca_resp.status_code != 200:
                t_err = tca_resp.text
                raise Exception(f"Phase 2 Failure: TCA({tca_resp.status_code}: {t_err})")
                
            t_data = tca_resp.json()
            
            # 3. Perform Fusion
            is_hatespeech, final_conf = fallback_fusion(
                t_data["tca_label"], t_data["tca_confidence"],
                s_data["detected_emotion"], s_data["ser_confidence"]
            )
            
            # 4. Update Record
            record = db.query(VideoRecord).filter(VideoRecord.id == record_id).first()
            if record:
                record.transcription = w_data["transcription"]
                record.translation_en = w_data["translation_en"]
                record.original_language = w_data["original_language"]
                record.detected_emotion = s_data["detected_emotion"]
                record.ser_confidence = f"{s_data['ser_confidence'] * 100:.1f}%"
                record.tca_label = t_data["tca_label"]
                record.tca_confidence = f"{t_data['tca_confidence'] * 100:.1f}%"
                record.is_hatespeech = is_hatespeech
                record.confidence = final_conf
                record.status = "COMPLETED"
                db.commit()
                print(f" ✅ Job {record_id} COMPLETED successfully.")

    except Exception as e:
        print(f" ❌ Job {record_id} FAILED: {e}")
        traceback.print_exc()
        record = db.query(VideoRecord).filter(VideoRecord.id == record_id).first()
        if record:
            record.status = "FAILED"
            record.transcription = f"Analysis Error: {str(e)}"
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
    """Immediately returns 202 Accepted and starts distributed AI in background"""
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

    # 2. Add to Background Tasks (Async function works natively)
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

    # If it is COMPLETED or FAILED, we should still send the record details 
    # (specifically the error message if it failed)
    if record.status in ["COMPLETED", "FAILED"]:
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
    # Cloud Run provides the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
