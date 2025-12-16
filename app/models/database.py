from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.utils.config import settings

Base = declarative_base()

class ChatHistory(Base):
    """Model for storing chat history."""
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create database engine (optional, only if using database)
if settings.database_url:
    engine = create_engine(settings.database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)