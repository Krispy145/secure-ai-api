from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.post("/query")
async def rag_query(payload: QueryRequest):
    # Stub response for now
    return {
        "query": payload.query,
        "response": "This is a placeholder response from the RAG system."
    }
