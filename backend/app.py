import os
import shutil
import tempfile
import requests
from fastapi import FastAPI, Depends, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn

from database import engine, get_db, Base
from models import VideoRecord

# Environment variables for service communication
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8080")

# Initialize DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AudioGuard Orchestrator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "audioguard-backend"}

@app.post("/api/analyze")
async def analyze_video(
    video_url: str = Form(...), 
    db: Session = Depends(get_db)
):
    """
    Receives a Cloudinary URL, delegates to ML Service, and stores results.
    """
    print(f"Received analyzed request for URL: {video_url}")
    
    try:
        # 1. Delegate to ML Service (Google Cloud Run)
        print(f"Forwarding to ML service: {ML_SERVICE_URL}/process")
        resp = requests.post(f"{ML_SERVICE_URL}/process", json={"video_url": video_url})
        
        if resp.status_code != 200:
            return JSONResponse(
                status_code=resp.status_code, 
                content={"error": f"ML Service failure: {resp.text}"}
            )
            
        result = resp.json()
        
        # 2. Save to Database
        db_record = VideoRecord(
            filename=video_url.split("/")[-1],
            transcription=result["transcription"],
            detected_emotion=result["detected_emotion"],
            is_hatespeech=result["is_hatespeech"],
            confidence=result["confidence"],
            tca_confidence=result["tca_confidence"],
            ser_confidence=result["ser_confidence"]
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        
        # Return merged result
        return JSONResponse(content={
            "db_id": db_record.id,
            **result
        })
            
    except Exception as e:
        print(f"Orchestrator Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/videos")
def get_videos(db: Session = Depends(get_db)):
    """Retrieve history of analyzed videos"""
    records = db.query(VideoRecord).order_by(VideoRecord.timestamp.desc()).all()
    return records

@app.delete("/api/videos/{video_id}")
def delete_video(video_id: int, db: Session = Depends(get_db)):
    """Delete a video record"""
    record = db.query(VideoRecord).filter(VideoRecord.id == video_id).first()
    if record:
        db.delete(record)
        db.commit()
        return {"status": "success", "message": "Record deleted"}
    return JSONResponse(content={"error": "Not found"}, status_code=404)

if __name__ == "__main__":
    print("Run this app via: uvicorn app:app --reload")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
