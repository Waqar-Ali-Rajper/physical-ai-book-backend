from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.rag_service import RAGService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Don't initialize here - do it lazily
rag_service = None

def get_rag_service():
    """Lazy initialization of RAG service."""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    context: Optional[str] = None
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    sources: list[str]
    confidence: float

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process user question using RAG."""
    try:
        if not request.message.strip():
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )
        
        service = get_rag_service()
        result = await service.generate_answer(
            question=request.message,
            context=request.context
        )
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your request"
        )

@router.post("/index-document")
async def index_document(text: str, source: str, doc_id: int):
    """Index a document in the vector store."""
    try:
        service = get_rag_service()
        await service.add_document(text, source, doc_id)
        return {"status": "success", "message": f"Indexed document: {source}"}
    except Exception as e:
        logger.error(f"Indexing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error indexing document: {str(e)}"
        )