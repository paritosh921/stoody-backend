import os
import time
import logging
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import jwt

logger = logging.getLogger(__name__)


class JitsiProviderService:
    def __init__(self):
        self._domain = os.environ.get("ONLINE_CLASS_JITSI_DOMAIN", "").strip()
        self._base_url = os.environ.get("ONLINE_CLASS_JITSI_BASE_URL", "").strip()
        self._jwt_enabled = os.environ.get("ONLINE_CLASS_JITSI_JWT_ENABLED", "").strip().lower() in ("true", "1", "yes")
        self._jwt_app_id = os.environ.get("ONLINE_CLASS_JITSI_JWT_APP_ID", "stoody").strip()
        self._jwt_secret = os.environ.get("ONLINE_CLASS_JITSI_JWT_SECRET", "").strip()
        self._jwt_audience = os.environ.get("ONLINE_CLASS_JITSI_JWT_AUDIENCE", "jitsi").strip()
        self._jwt_ttl = self._read_positive_int("ONLINE_CLASS_JITSI_JWT_TTL_SECONDS", 7200)

    @staticmethod
    def _read_positive_int(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Invalid %s value; using default %s", name, default)
            return default
        return value if value > 0 else default

    @property
    def configured(self) -> bool:
        return bool(self._domain or self._base_url) and not self.missing_required_jwt_secret

    @property
    def domain(self) -> str:
        if not self._domain and self._base_url:
            return urlparse(self._base_url).hostname or ""
        return self._domain

    @property
    def base_url(self) -> str:
        if self._base_url:
            return self._base_url.rstrip("/")
        if self._domain:
            scheme = "https"
            return f"{scheme}://{self._domain}"
        return ""

    @property
    def jwt_available(self) -> bool:
        return self._jwt_enabled and bool(self._jwt_secret)

    @property
    def missing_required_jwt_secret(self) -> bool:
        return self._jwt_enabled and not self._jwt_secret

    def generate_room_name(self, meeting_id: str) -> str:
        safe_id = "".join(c if c.isalnum() else "-" for c in meeting_id)
        return f"stoody-{safe_id}"

    def generate_canvas_room_name(self, meeting_id: str, kind: str, student_id: str = "") -> str:
        safe_kind = "".join(c if c.isalnum() else "-" for c in kind).strip("-").lower()
        if not safe_kind:
            safe_kind = "canvas"
        base = f"{self.generate_room_name(meeting_id)}-canvas-{safe_kind}"
        if student_id:
            safe_student = "".join(c if c.isalnum() else "-" for c in student_id).strip("-")
            return f"{base}-{safe_student}".lower()
        return base.lower()

    def get_room_url(self, room_name: str) -> str:
        if not self.configured:
            return ""
        base = self.base_url
        return f"{base}/{room_name}"

    def generate_jwt(
        self,
        room_name: str,
        user_id: str,
        user_name: str,
        user_email: str = "",
        moderator: bool = False,
    ) -> Optional[str]:
        if not self.jwt_available:
            if self._jwt_enabled and not self._jwt_secret:
                logger.warning(
                    "Jitsi JWT is enabled but ONLINE_CLASS_JITSI_JWT_SECRET is not set. "
                    "Refusing to emit an insecure token."
                )
            return None
        if not user_id:
            logger.warning("Refusing to emit Jitsi JWT for room %s without a user id", room_name)
            return None

        now = int(time.time())
        context = {
            "user": {
                "id": user_id,
                "name": user_name,
                "email": user_email,
                "moderator": moderator,
            }
        }

        payload = {
            "aud": self._jwt_audience,
            "iss": self._jwt_app_id,
            "sub": self.domain,
            "room": room_name,
            "nbf": now,
            "exp": now + self._jwt_ttl,
            "context": context,
        }

        return jwt.encode(payload, self._jwt_secret, algorithm="HS256")

    def get_provider_details_for_room(
        self,
        room_name: str,
        user_id: str = "",
        user_name: str = "",
        user_email: str = "",
        moderator: bool = False,
    ) -> Dict[str, Any]:
        url = self.get_room_url(room_name)

        token = None
        token_required = self._jwt_enabled

        if self._jwt_enabled:
            if self._jwt_secret:
                token = self.generate_jwt(
                    room_name=room_name,
                    user_id=user_id,
                    user_name=user_name,
                    user_email=user_email,
                    moderator=moderator,
                )
            else:
                logger.warning(
                    "Jitsi JWT enabled but secret absent for room %s; "
                    "marking provider unconfigured instead of returning a public join path.",
                    room_name,
                )

        response_configured = self.configured and (not self._jwt_enabled or bool(token))

        return {
            "provider": "jitsi",
            "domain": self.domain,
            "room_name": room_name,
            "url": url,
            "token_required": token_required,
            "token": token,
            "configured": response_configured,
        }

    def get_provider_details(
        self,
        meeting_id: str,
        user_id: str = "",
        user_name: str = "",
        user_email: str = "",
        moderator: bool = False,
    ) -> Dict[str, Any]:
        room_name = self.generate_room_name(meeting_id)
        return self.get_provider_details_for_room(
            room_name=room_name,
            user_id=user_id,
            user_name=user_name,
            user_email=user_email,
            moderator=moderator,
        )


jitsi_provider_service = JitsiProviderService()
