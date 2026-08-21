import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.v1 import auth as auth_module
from app.api.v1 import phishing as phishing_module
from app.api.v1 import rag as rag_module
from app.api.v1 import router as api_router
from app.core.config import settings
from app.core.limiter import limiter
from app.services.phishing_classifier import initialize_classifier_service
from app.services.rag_service import initialize_rag_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing application services...")
    if settings.secret_is_insecure:
        logger.warning(
            "SECRET_KEY is using the insecure default. "
            "Set SECRET_KEY in the environment before deploying."
        )

    model_loaded = initialize_classifier_service()
    if model_loaded:
        logger.info("Phishing classifier model loaded successfully")
    else:
        logger.warning("Failed to load phishing classifier model - stub mode will be used")

    rag_initialized = initialize_rag_service()
    if rag_initialized:
        logger.info("RAG service initialized successfully")
    else:
        logger.warning("RAG service initialization failed - fallback mode will be used")

    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "FastAPI service for RAG + inference with OAuth2/JWT, rate limiting, Docker, CI/CD."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

cors_origins = settings.cors_origin_list
allow_credentials = cors_origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
@limiter.exempt
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
@limiter.exempt
def health():
    """Health check endpoint."""
    return {"status": "ok", "version": settings.VERSION}


@app.get("/ping")
@limiter.exempt
def ping():
    """Simple ping endpoint for connectivity testing."""
    return {"message": "pong", "status": "ok"}


app.include_router(api_router, prefix="/v1")
app.include_router(auth_module.router, prefix="/auth", tags=["auth"])
app.include_router(phishing_module.router, prefix="/phishing", tags=["phishing"])
app.include_router(rag_module.router, prefix="/rag", tags=["rag"])
