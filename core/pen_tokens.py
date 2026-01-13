"""
Pen token helpers for Stoody BLE stroke ingestion.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import time
import jwt

from config_async import JWT_SECRET_KEY, JWT_ALGORITHM

PEN_TOKEN_TTL_SECONDS = 60 * 60 * 8  # 8 hours


def create_pen_token(
    student_id: str,
    tenant_id: Optional[str],
    db_name: Optional[str],
    institution_id: Optional[str],
    ttl_seconds: int = PEN_TOKEN_TTL_SECONDS,
) -> Tuple[str, datetime]:
    """Create a short-lived pen token (JWT) with tenant context."""
    now = datetime.utcnow()
    expire = now + timedelta(seconds=ttl_seconds)

    payload = {
        "sub": student_id,
        "type": "pen",
        "tenant_id": tenant_id,
        "db_name": db_name,
        "institution_id": institution_id,
        "iat": now,
        "exp": expire,
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token, expire


def decode_pen_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a pen token."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "pen":
            return None
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None
