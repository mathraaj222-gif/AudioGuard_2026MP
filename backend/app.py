import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn

from inference_pipeline import InferencePipeline
from database import engine, get_db, Base
from models import VideoRecord

# Initialize DB tables
Base.metadata.create_all(bind=engine)

from inference_pipeline import InferencePipeline

app = FastAPI(title="Hate Speech Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_engine = None

@app.on_event("startup")
def startup_event():
    global pipeline_engine
    print("Starting up API, loading models...")
    pipeline_engine = InferencePipeline()

@app.post("/api/analyze")
async def analyze_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    print(f"Received file: {file.filename}")
    
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.gettempdir()
    safe_filename = file.filename.replace(" ", "_")
    temp_file_path = os.path.join(temp_dir, safe_filename)
    
    contents = await file.read()
    with open(temp_file_path, "wb") as buffer:
        buffer.write(contents)
        
    print(f"Saved {len(contents)} bytes to {temp_file_path}")
        
    try:
        # Run inference pipeline
        result = pipeline_engine.process_file(temp_file_path)
        
        # Save to Database
        db_record = VideoRecord(
            filename=safe_filename,
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
        
        result["db_id"] = db_record.id
        
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    return JSONResponse(content=result)

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
