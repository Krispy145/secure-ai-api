from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

import bcrypt
import jwt

from app.core.config import settings


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))


def _encode(payload: Dict[str, Any]) -> str:
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_access_token(subject: str) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return _encode(
        {
            "sub": subject,
            "type": "access",
            "jti": uuid4().hex,
            "iat": now,
            "exp": expire,
        }
    )


def create_refresh_token(subject: str) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    return _encode(
        {
            "sub": subject,
            "type": "refresh",
            "jti": uuid4().hex,
            "iat": now,
            "exp": expire,
        }
    )


def decode_token(token: str, expected_type: Optional[str] = None) -> Dict[str, Any]:
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    if expected_type and payload.get("type") != expected_type:
        raise jwt.InvalidTokenError("Unexpected token type")
    if not payload.get("sub"):
        raise jwt.InvalidTokenError("Token is missing subject")
    return payload
