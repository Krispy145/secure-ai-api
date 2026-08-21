import os
from typing import List

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Secure AI API"
    VERSION: str = "0.2.0"
    API_V1_STR: str = "/v1"

    SECRET_KEY: str = os.getenv("SECRET_KEY", "supersecret")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    DEMO_USERNAME: str = os.getenv("DEMO_USERNAME", "demo")
    DEMO_PASSWORD: str = os.getenv("DEMO_PASSWORD", "changeme")

    RATE_LIMIT_ENABLED: bool = True
    AUTH_RATE_LIMIT: str = "5/minute"
    API_RATE_LIMIT: str = "30/minute"
    RAG_RATE_LIMIT: str = "20/minute"
    DEFAULT_RATE_LIMIT: str = "60/minute"

    CORS_ORIGINS: str = (
        "http://localhost:5173,http://localhost:3000,http://localhost:8080,"
        "http://127.0.0.1:5173,http://127.0.0.1:3000"
    )

    class Config:
        env_file = ".env"
        case_sensitive = True

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def secret_is_insecure(self) -> bool:
        return self.SECRET_KEY in {"supersecret", "", "changeme"}


settings = Settings()
