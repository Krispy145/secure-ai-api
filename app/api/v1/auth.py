from datetime import datetime, timezone
from typing import Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.deps import get_current_user
from app.core.limiter import limiter
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from app.core.users import (
    User,
    allow_refresh_token,
    get_user,
    refresh_token_is_allowed,
    revoke_refresh_token,
)

router = APIRouter()


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class LoginRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: str = Field(..., min_length=1)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)


class UserPublic(BaseModel):
    username: str


def _issue_tokens(username: str) -> TokenResponse:
    access_token = create_access_token(username)
    refresh_token = create_refresh_token(username)
    refresh_payload = decode_token(refresh_token, expected_type="refresh")
    allow_refresh_token(
        refresh_payload["jti"],
        datetime.fromtimestamp(refresh_payload["exp"], tz=timezone.utc),
    )
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


def _authenticate(username: str, password: str) -> User:
    user = get_user(username)
    if user is None or not verify_password(password, user.hashed_password) or user.disabled:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@router.post("/token", response_model=TokenResponse)
@limiter.limit(settings.AUTH_RATE_LIMIT)
async def login_for_access_token(
    request: Request,
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    """OAuth2 password flow. Used by Swagger 'Authorize'."""
    _authenticate(form_data.username, form_data.password)
    return _issue_tokens(form_data.username)


@router.post("/login", response_model=TokenResponse)
@limiter.limit(settings.AUTH_RATE_LIMIT)
async def login_json(request: Request, response: Response, payload: LoginRequest):
    """JSON login for web/mobile clients. `username` or `email` is accepted."""
    identity = (payload.username or payload.email or "").strip()
    if not identity:
        raise HTTPException(status_code=400, detail="username or email is required")
    _authenticate(identity, payload.password)
    return _issue_tokens(identity)


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit(settings.AUTH_RATE_LIMIT)
async def refresh_tokens(request: Request, response: Response, payload: RefreshRequest):
    """Rotate refresh token. Previous refresh token is revoked."""
    try:
        token_data = decode_token(payload.refresh_token, expected_type="refresh")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    jti = token_data.get("jti")
    username = token_data.get("sub")
    if not jti or not username or not refresh_token_is_allowed(jti):
        raise HTTPException(status_code=401, detail="Refresh token is not active")

    user = get_user(username)
    if user is None or user.disabled:
        raise HTTPException(status_code=401, detail="User is not active")

    revoke_refresh_token(jti)
    return _issue_tokens(username)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(payload: RefreshRequest):
    """Revoke a refresh token. Access tokens stay valid until they expire."""
    try:
        token_data = decode_token(payload.refresh_token, expected_type="refresh")
    except jwt.PyJWTError:
        return None
    jti = token_data.get("jti")
    if jti:
        revoke_refresh_token(jti)
    return None


@router.get("/me", response_model=UserPublic)
async def read_me(current_user: User = Depends(get_current_user)):
    return UserPublic(username=current_user.username)
