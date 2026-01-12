from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Dict, Optional, Any


class PenRouter:
    """Resolves BLE MAC addresses into canonical pen IDs."""

    def __init__(
        self,
        registry_path: Path,
        *,
        cache_ttl: float = 60.0,
    ) -> None:
        self._registry_path = registry_path
        self._cache_ttl = cache_ttl

        self._lock = asyncio.Lock()
        self._cache: Dict[str, str] = {}
        self._records_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry = 0.0

    async def resolve_pen_id(self, pen_mac: str) -> str:
        """Return the canonical pen ID for a MAC address."""

        normalized_mac = pen_mac.upper()
        await self._ensure_cache()

        try:
            return self._cache[normalized_mac]
        except KeyError as exc:
            raise LookupError(f"Unknown pen MAC {normalized_mac}") from exc

    async def get_pen_info(self, pen_id: str) -> Optional[Dict[str, Any]]:
        """Return full pen record by pen_id (includes student_name, etc)."""
        await self._ensure_cache()
        return self._records_cache.get(pen_id)

    async def get_all_pens(self) -> Dict[str, Dict[str, Any]]:
        """Return all pen records keyed by pen_id."""
        await self._ensure_cache()
        return self._records_cache.copy()

    async def reload(self) -> None:
        """Force reload the registry from disk (used after sync)."""
        async with self._lock:
            self._cache, self._records_cache = self._load_registry()
            import time
            self._cache_expiry = time.time() + self._cache_ttl

    async def _ensure_cache(self) -> None:
        import time

        now = time.time()
        if now < self._cache_expiry and self._cache:
            return

        async with self._lock:
            if now < self._cache_expiry and self._cache:
                return
            self._cache, self._records_cache = self._load_registry()
            self._cache_expiry = now + self._cache_ttl

    def _load_registry(self) -> tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
        if not self._registry_path.exists():
            raise FileNotFoundError(
                f"Pen registry missing at {self._registry_path}. "
                "Provision the file before starting the backend."
            )

        with self._registry_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)

        mapping: Dict[str, str] = {}
        records_by_id: Dict[str, Dict[str, Any]] = {}
        for record in records:
            mac = record["pen_mac"].upper()
            pen_id = record["pen_id"]
            mapping[mac] = pen_id
            records_by_id[pen_id] = record

        return mapping, records_by_id

