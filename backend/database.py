from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Uses Railway's Postgres URL if provided, otherwise defaults to local SQLite for testing.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local_videos.db")

# Fix for SQLAlchemy 2.0: Railway provides 'postgres://', but it must be 'postgresql://'
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(
    DATABASE_URL, 
    connect_args=connect_args,
    pool_pre_ping=True  # Automatically reconnect if the connection dropped
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
