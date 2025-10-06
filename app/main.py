from fastapi import FastAPI
from app.api.v1 import router as api_router

app = FastAPI(title="Secure AI API", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(api_router, prefix="/v1")
