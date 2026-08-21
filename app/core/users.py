from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Optional

from app.core.config import settings
from app.core.security import hash_password


@dataclass
class User:
    username: str
    hashed_password: str
    disabled: bool = False


_users: Optional[Dict[str, User]] = None
_refresh_allowlist: Dict[str, datetime] = {}
_lock = Lock()


def _seed_users() -> Dict[str, User]:
    hashed = hash_password(settings.DEMO_PASSWORD)
    return {
        settings.DEMO_USERNAME: User(
            username=settings.DEMO_USERNAME,
            hashed_password=hashed,
        )
    }


def get_user(username: str) -> Optional[User]:
    global _users
    with _lock:
        if _users is None:
            _users = _seed_users()
        return _users.get(username)


def allow_refresh_token(jti: str, expires_at: datetime) -> None:
    with _lock:
        _refresh_allowlist[jti] = expires_at


def refresh_token_is_allowed(jti: str) -> bool:
    with _lock:
        expires_at = _refresh_allowlist.get(jti)
        if expires_at is None:
            return False
        if expires_at < datetime.now(timezone.utc):
            _refresh_allowlist.pop(jti, None)
            return False
        return True


def revoke_refresh_token(jti: str) -> None:
    with _lock:
        _refresh_allowlist.pop(jti, None)


def clear_refresh_tokens() -> None:
    with _lock:
        _refresh_allowlist.clear()


def reset_auth_state() -> None:
    """Test helper: drop seeded users and refresh tokens."""
    global _users
    with _lock:
        _users = None
        _refresh_allowlist.clear()
