from fastapi import APIRouter
from . import phishing, rag

router = APIRouter()
router.include_router(phishing.router, prefix="/predict", tags=["phishing"])
router.include_router(rag.router, prefix="/rag", tags=["rag"])
