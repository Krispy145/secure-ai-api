from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

from app.services.rag_service import get_rag_service

logger = logging.getLogger(__name__)
router = APIRouter()

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query/question")
    max_tokens: Optional[int] = Field(default=500, ge=1, le=2000, description="Maximum tokens in response")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    n_results: Optional[int] = Field(default=3, ge=1, le=10, description="Number of relevant documents to retrieve")

class RAGResponse(BaseModel):
    query: str
    response: str
    sources: Optional[List[str]] = None
    confidence: Optional[float] = None
    retrieved_docs: Optional[int] = None

@router.post("/query", response_model=RAGResponse)
async def rag_query(payload: QueryRequest):
    """
    Query the RAG (Retrieval-Augmented Generation) system.
    
    This endpoint uses vector database retrieval and LLM generation to provide
    contextual answers based on the knowledge base. It retrieves relevant documents
    and generates responses using the retrieved context.
    
    The system supports:
    - Semantic search using vector embeddings
    - Context-aware response generation
    - Source attribution for retrieved documents
    
    Note: For full functionality, install chromadb and sentence-transformers:
    pip install chromadb sentence-transformers
    
    For LLM generation, configure OPENAI_API_KEY environment variable.
    """
    # Basic validation
    if not payload.query or len(payload.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Get RAG service instance
        rag_service = get_rag_service()
        
        # Query the RAG system
        result = rag_service.query(
            query_text=payload.query,
            n_results=payload.n_results or 3,
            max_tokens=payload.max_tokens or 500,
            temperature=payload.temperature or 0.7
        )
        
        return RAGResponse(
            query=result.get("query", payload.query),
            response=result.get("response", ""),
            sources=result.get("sources"),
            confidence=result.get("confidence"),
            retrieved_docs=result.get("retrieved_docs", 0)
        )
        
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
