from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import router as api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="FastAPI service for RAG + inference with OAuth2/JWT, rate limiting, Docker, CI/CD.",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "version": settings.VERSION}

@app.get("/ping")
def ping():
    """Simple ping endpoint for connectivity testing."""
    return {"message": "pong", "status": "ok"}

# Include API v1 router
app.include_router(api_router, prefix="/v1")

# Add direct phishing routes for convenience (without /v1 prefix)
from app.api.v1 import phishing as phishing_module
app.include_router(phishing_module.router, prefix="/phishing", tags=["phishing"])
