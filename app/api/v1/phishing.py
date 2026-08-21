import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.deps import get_current_user
from app.core.limiter import limiter
from app.services.phishing_classifier import get_classifier_service

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(get_current_user)])


class PhishingInput(BaseModel):
    url: str = Field(..., description="URL to classify as phishing or legitimate")


class PhishingPrediction(BaseModel):
    input_url: str
    prediction: str
    confidence: float
    score: Optional[float] = None


class PhishingSample(BaseModel):
    id: str
    url: str
    label: str
    score: float


# Mock samples data for stub endpoint
MOCK_SAMPLES = [
    {"id": "1", "url": "https://www.google.com", "label": "legitimate", "score": 0.12},
    {
        "id": "2",
        "url": "https://example.com/login",
        "label": "legitimate",
        "score": 0.23,
    },
    {
        "id": "3",
        "url": "https://bit.ly/suspicious-link",
        "label": "phishing",
        "score": 0.88,
    },
    {
        "id": "4",
        "url": "http://192.168.1.1/login.php",
        "label": "phishing",
        "score": 0.92,
    },
    {"id": "5", "url": "https://secure-bank.com", "label": "legitimate", "score": 0.15},
    {
        "id": "6",
        "url": "https://secure-bank.com.fake-site.net",
        "label": "phishing",
        "score": 0.85,
    },
]


@router.post("/phishing", response_model=PhishingPrediction)
@limiter.limit(settings.API_RATE_LIMIT)
async def predict_phishing(request: Request, response: Response, payload: PhishingInput):
    """
    Classify a URL as phishing or legitimate using the trained ML model.

    This endpoint uses the trained phishing classifier model to make predictions.
    Falls back to stub logic if the model is not available.
    """
    # Basic URL validation
    if not payload.url or len(payload.url.strip()) == 0:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    try:
        # Try to use the real model
        classifier_service = get_classifier_service()
        if classifier_service.is_loaded:
            result = classifier_service.predict(payload.url)
            return PhishingPrediction(
                input_url=payload.url,
                prediction=result["prediction"],
                confidence=result["confidence"],
                score=result["score"],
            )
        else:
            logger.warning("Model not loaded, using fallback logic")
            raise RuntimeError("Model not available")

    except (RuntimeError, ValueError, Exception) as e:
        # Fallback to stub logic if model is not available
        logger.warning(f"Using fallback prediction logic: {e}")
        url_lower = payload.url.lower()
        is_suspicious = any(
            [
                "bit.ly" in url_lower or "tinyurl" in url_lower,
                "192.168" in url_lower or "10.0" in url_lower,
                ".fake" in url_lower or "-fake" in url_lower,
                len(payload.url) > 100,
            ]
        )

        prediction = "phishing" if is_suspicious else "legitimate"
        confidence = 0.9 if is_suspicious else 0.8
        score = confidence if is_suspicious else 1.0 - confidence

        return PhishingPrediction(
            input_url=payload.url,
            prediction=prediction,
            confidence=confidence,
            score=score,
        )


@router.get("/samples", response_model=List[PhishingSample])
@limiter.limit(settings.API_RATE_LIMIT)
async def get_phishing_samples(
    request: Request,
    response: Response,
    limit: Optional[int] = Query(
        default=10, ge=1, le=100, description="Maximum number of samples to return"
    ),
    label: Optional[str] = Query(
        default=None, description="Filter by label: 'phishing' or 'legitimate'"
    ),
):
    """
    Get a list of phishing samples for testing and demonstration.

    This is a stub endpoint that returns mock sample data.
    Will be replaced with actual database queries in the next milestone.
    """
    samples = MOCK_SAMPLES.copy()

    # Apply label filter if provided
    if label:
        label_lower = label.lower()
        if label_lower not in ["phishing", "legitimate"]:
            raise HTTPException(status_code=400, detail="Label must be 'phishing' or 'legitimate'")
        samples = [s for s in samples if s["label"].lower() == label_lower]

    # Apply limit
    samples = samples[:limit]

    return [PhishingSample(**sample) for sample in samples]
