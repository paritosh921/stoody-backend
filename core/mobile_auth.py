"""Mobile auth session policy helpers."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Optional

from config_async import settings


MOBILE_APP_SOURCE = "stoody-mobile"


def is_mobile_auth_request(request: Any) -> bool:
    headers = getattr(request, "headers", {}) or {}
    source = headers.get("x-app-source") if hasattr(headers, "get") else None
    return str(source or "").strip().lower() == MOBILE_APP_SOURCE


def mobile_session_delta_for_request(request: Any) -> Optional[timedelta]:
    if not is_mobile_auth_request(request):
        return None
    return timedelta(minutes=settings.MOBILE_SESSION_EXPIRE_MINUTES)
