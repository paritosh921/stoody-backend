import os
import logging
from typing import Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class JitsiProviderService:
    def __init__(self):
        self._domain = os.environ.get("ONLINE_CLASS_JITSI_DOMAIN", "").strip()
        self._base_url = os.environ.get("ONLINE_CLASS_JITSI_BASE_URL", "").strip()
        self._jwt_enabled = os.environ.get("ONLINE_CLASS_JITSI_JWT_ENABLED", "").strip().lower() in ("true", "1", "yes")

    @property
    def configured(self) -> bool:
        return bool(self._domain or self._base_url)

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

    def generate_room_name(self, meeting_id: str) -> str:
        safe_id = "".join(c if c.isalnum() else "-" for c in meeting_id)
        return f"stoody-{safe_id}"

    def get_room_url(self, room_name: str) -> str:
        if not self.configured:
            return ""
        base = self.base_url
        return f"{base}/{room_name}"

    def get_provider_details(self, meeting_id: str) -> Dict[str, Any]:
        room_name = self.generate_room_name(meeting_id)
        url = self.get_room_url(room_name)

        token_required = self._jwt_enabled

        return {
            "provider": "jitsi",
            "domain": self.domain,
            "room_name": room_name,
            "url": url,
            "token_required": token_required,
            "token": None,
            "configured": self.configured,
        }


jitsi_provider_service = JitsiProviderService()
