from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Physical AI Textbook API",
    version="1.0.0",
    description="RAG chatbot backend for Physical AI & Humanoid Robotics textbook"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sab origins allow karo temporarily
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)

@app.get("/")
async def root():
    return {
        "message": "Physical AI Textbook API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}