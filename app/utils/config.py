from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application configuration from environment variables."""
    
    openai_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    database_url: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()