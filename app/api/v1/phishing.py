from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class PhishingInput(BaseModel):
    url: str

@router.post("/phishing")
async def predict_phishing(payload: PhishingInput):
    # Stub response for now
    return {
        "input_url": payload.url,
        "prediction": "legitimate",
        "confidence": 0.87
    }
