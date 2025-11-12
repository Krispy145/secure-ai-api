from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

router = APIRouter()

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query/question")
    max_tokens: Optional[int] = Field(default=500, ge=1, le=2000, description="Maximum tokens in response")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")

class RAGResponse(BaseModel):
    query: str
    response: str
    sources: Optional[List[str]] = None
    confidence: Optional[float] = None

@router.post("/query", response_model=RAGResponse)
async def rag_query(payload: QueryRequest):
    """
    Query the RAG (Retrieval-Augmented Generation) system.
    
    This is a stub endpoint that returns mock responses.
    Will be replaced with actual vector database retrieval and LLM generation in the next milestone.
    """
    # Basic validation
    if not payload.query or len(payload.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Stub response with some basic logic
    query_lower = payload.query.lower()
    
    # Simple keyword-based stub responses
    if "phishing" in query_lower or "security" in query_lower:
        response = (
            "Phishing is a type of cyber attack where attackers attempt to trick users "
            "into revealing sensitive information by impersonating legitimate entities. "
            "Common indicators include suspicious URLs, unexpected requests for credentials, "
            "and urgent language designed to create panic."
        )
        sources = ["security_guide.pdf", "phishing_detection_handbook.pdf"]
        confidence = 0.85
    elif "api" in query_lower or "endpoint" in query_lower:
        response = (
            "This API provides endpoints for phishing detection and RAG-based question answering. "
            "The phishing endpoint accepts URLs and returns classification results. "
            "The RAG endpoint processes natural language queries and returns contextual responses."
        )
        sources = ["api_documentation.md"]
        confidence = 0.90
    else:
        response = (
            f"This is a placeholder response to your query: '{payload.query}'. "
            "The actual RAG system will retrieve relevant information from a vector database "
            "and generate contextual responses using a language model."
        )
        sources = None
        confidence = 0.75
    
    return RAGResponse(
        query=payload.query,
        response=response,
        sources=sources,
        confidence=confidence
    )
