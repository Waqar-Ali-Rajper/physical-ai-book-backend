from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.utils.config import settings
from sentence_transformers import SentenceTransformer
import logging
import httpx

logger = logging.getLogger(__name__)

class RAGService:
    """Service for RAG operations using Hugging Face and Qdrant."""
    
    def __init__(self):
        # Use FREE Hugging Face embeddings instead of OpenAI
        print("Loading embedding model... (first time takes a minute)")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded!")
        
        # Still need OpenAI for chat (but not for embeddings)
        http_client = httpx.Client(timeout=30.0)
        self.openai_client = OpenAI(
            api_key=settings.openai_api_key,
            http_client=http_client
        )
        
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self.collection_name = "physical_ai_textbook"
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # Hugging Face model dimension
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
    
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding using FREE Hugging Face model."""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise
    
    async def add_document(self, text: str, source: str, doc_id: int):
        """Add a document to the vector store."""
        try:
            embedding = self.embed_text(text)
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload={
                            "text": text,
                            "source": source
                        }
                    )
                ]
            )
            logger.info(f"Added document: {source}")
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    async def search(self, query: str, limit: int = 3) -> list[dict]:
        """Search for relevant content in the vector store."""
        try:
            query_vector = self.embed_text(query)
            
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            return [
                {
                    "text": hit.payload["text"],
                    "source": hit.payload["source"],
                    "score": hit.score
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    async def generate_answer(self, question: str, context: str = None) -> dict:
        """Generate answer using OpenAI with RAG context."""
        try:
            if context:
                relevant_docs = [{"text": context, "source": "Selected Text", "score": 1.0}]
            else:
                relevant_docs = await self.search(question, limit=3)
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find relevant information in the textbook to answer your question.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            context_text = "\n\n".join([
                f"Source: {doc['source']}\n{doc['text']}"
                for doc in relevant_docs
            ])
            
            system_prompt = """You are a helpful assistant for a Physical AI & Humanoid Robotics textbook. 
Answer questions based on the provided context from the textbook. 
If the answer is not in the context, say so clearly.
Be concise and technical but easy to understand."""
            
            user_prompt = f"""Context from textbook:
{context_text}

Question: {question}

Please provide a clear and accurate answer based on the context above."""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            sources = [doc["source"] for doc in relevant_docs]
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": relevant_docs[0]["score"] if relevant_docs else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise