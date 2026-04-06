from sqlalchemy import Column, Integer, String, Boolean, DateTime
from datetime import datetime
from database import Base

class VideoRecord(Base):
    __tablename__ = "video_records"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    video_url = Column(String, index=True)
    status = Column(String, default="PENDING") # PENDING, COMPLETED, FAILED
    transcription = Column(String)
    translation_en = Column(String)
    original_language = Column(String)
    detected_emotion = Column(String)
    is_hatespeech = Column(Boolean)
    confidence = Column(String)
    tca_confidence = Column(String)
    ser_confidence = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
