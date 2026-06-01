"""Token-aware gateway for outbound OCR and LLM calls."""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from math import ceil
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError


class AIUsageLimitExceeded(Exception):
    """Raised when a configured AI usage limit blocks a provider call."""

    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        super().__init__(payload.get("error", "ai_token_limit_exceeded"))


def estimate_text_tokens(text: str) -> int:
    text = text or ""
    if not text:
        return 0
    return max(1, ceil(len(text) / 4))


def estimate_ocr_tokens(*, pdf_bytes: int = 0, image_bytes: int = 0, page_count: int = 1) -> int:
    byte_count = max(0, int(pdf_bytes or 0)) + max(0, int(image_bytes or 0))
    return max(1, ceil(byte_count / 50) + (max(1, int(page_count or 1)) * 750))


class AIGatewayService:
    """Reserve usage, call the provider, and record an audit event."""

    def __init__(self, db: Any = None, *, is_b2c: bool = False):
        self.db = db
        self.is_b2c = is_b2c

    async def call(
        self,
        *,
        user_id: str,
        tenant_id: Optional[str],
        document_id: Optional[str],
        region_id: Optional[str],
        region_scope: Optional[str],
        stage: str,
        provider: str,
        model: str,
        input_kind: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int = 0,
        max_output_tokens: Optional[int] = None,
        input_units: Optional[Dict[str, Any]] = None,
        call_fn: Callable[[], Awaitable[Any]],
    ) -> Any:
        if not self._enabled():
            return await call_fn()

        estimated_input_tokens = max(0, int(estimated_input_tokens or 0))
        estimated_output_tokens = max(0, int(estimated_output_tokens or 0))
        estimated_total_tokens = estimated_input_tokens + estimated_output_tokens
        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        daily_period_key = now.strftime("%Y-%m-%d")
        monthly_period_key = now.strftime("%Y-%m")
        tenant_id = tenant_id or "unknown"
        user_id = user_id or "unknown"

        base_event = {
            "event_id": event_id,
            "user_id": user_id,
            "tenant_id": tenant_id,
            "document_id": document_id,
            "region_id": region_id,
            "region_scope": region_scope,
            "stage": stage,
            "provider": provider,
            "model": model,
            "input_kind": input_kind,
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_total_tokens": estimated_total_tokens,
            "max_output_tokens": max_output_tokens,
            "actual_input_tokens": None,
            "actual_output_tokens": None,
            "usage_source": "estimated",
            "input_units": input_units or {},
            "status": "reserved",
            "error": None,
            "latency_ms": None,
            "created_at": now,
            "updated_at": now,
        }

        blocked_payload = await self._reserve_or_block(
            user_id=user_id,
            tenant_id=tenant_id,
            daily_period_key=daily_period_key,
            monthly_period_key=monthly_period_key,
            tokens=estimated_total_tokens,
        )
        if blocked_payload is not None:
            blocked_event = dict(base_event)
            blocked_event.update({"status": "blocked", "error": blocked_payload["error"]})
            await self._insert_event(blocked_event)
            if self._block_on_limit():
                raise AIUsageLimitExceeded(blocked_payload)
            soft_event = dict(base_event)
            soft_event.update(
                {
                    "event_id": str(uuid.uuid4()),
                    "status": "reserved",
                    "soft_limit_override": True,
                    "limit_error": blocked_payload,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            return await self._execute_and_record_event(soft_event, call_fn)

        return await self._execute_and_record_event(base_event, call_fn)

    async def _execute_and_record_event(
        self,
        event: Dict[str, Any],
        call_fn: Callable[[], Awaitable[Any]],
    ) -> Any:
        event_id = event["event_id"]
        await self._insert_event(event)
        started = time.monotonic()
        try:
            result = await call_fn()
            actual_input, actual_output = self._extract_usage(result)
            await self._update_event(
                event_id,
                {
                    "status": "success",
                    "actual_input_tokens": actual_input,
                    "actual_output_tokens": actual_output,
                    "usage_source": "provider_reported" if actual_input is not None or actual_output is not None else "estimated",
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    "updated_at": datetime.now(timezone.utc),
                },
            )
            return result
        except Exception as exc:
            await self._update_event(
                event_id,
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    "updated_at": datetime.now(timezone.utc),
                },
            )
            raise

    async def _reserve_or_block(
        self,
        *,
        user_id: str,
        tenant_id: str,
        daily_period_key: str,
        monthly_period_key: str,
        tokens: int,
    ) -> Optional[Dict[str, Any]]:
        user_override = await self._limit_override("user", user_id)
        max_region = user_override.get("max_tokens_per_region") if user_override else _env_int("AI_MAX_TOKENS_PER_REGION")
        if max_region is not None and tokens > max_region:
            return self._limit_payload("region", user_id, max_region, 0, tokens)

        reservations: list[Dict[str, str]] = []
        user_limit = user_override.get("daily_token_limit") if user_override else _env_int("AI_DAILY_TOKEN_LIMIT_PER_USER")
        allowed, used = await self._reserve_usage(
            scope="user",
            subject_id=user_id,
            period="daily",
            period_key=daily_period_key,
            tokens=tokens,
            limit=user_limit,
        )
        if not allowed:
            return self._limit_payload("user_daily", user_id, user_limit, used, tokens)
        reservations.append({"scope": "user", "subject_id": user_id, "period": "daily", "period_key": daily_period_key})

        user_monthly_limit = user_override.get("monthly_token_limit") if user_override else _env_int("AI_MONTHLY_TOKEN_LIMIT_PER_USER")
        allowed, used = await self._reserve_usage(
            scope="user",
            subject_id=user_id,
            period="monthly",
            period_key=monthly_period_key,
            tokens=tokens,
            limit=user_monthly_limit,
        )
        if not allowed:
            await self._release_reservations(reservations, tokens)
            return self._limit_payload("user_monthly", user_id, user_monthly_limit, used, tokens)
        reservations.append({"scope": "user", "subject_id": user_id, "period": "monthly", "period_key": monthly_period_key})

        tenant_override = await self._limit_override("tenant", tenant_id)
        tenant_limit = tenant_override.get("daily_token_limit") if tenant_override else _env_int("AI_DAILY_TOKEN_LIMIT_PER_TENANT")
        allowed, used = await self._reserve_usage(
            scope="tenant",
            subject_id=tenant_id,
            period="daily",
            period_key=daily_period_key,
            tokens=tokens,
            limit=tenant_limit,
        )
        if not allowed:
            await self._release_reservations(reservations, tokens)
            return self._limit_payload("tenant_daily", user_id, tenant_limit, used, tokens)
        reservations.append({"scope": "tenant", "subject_id": tenant_id, "period": "daily", "period_key": daily_period_key})

        tenant_monthly_limit = tenant_override.get("monthly_token_limit") if tenant_override else _env_int("AI_MONTHLY_TOKEN_LIMIT_PER_TENANT")
        allowed, used = await self._reserve_usage(
            scope="tenant",
            subject_id=tenant_id,
            period="monthly",
            period_key=monthly_period_key,
            tokens=tokens,
            limit=tenant_monthly_limit,
        )
        if not allowed:
            await self._release_reservations(reservations, tokens)
            return self._limit_payload("tenant_monthly", user_id, tenant_monthly_limit, used, tokens)

        return None

    def _limit_payload(
        self,
        limit_type: str,
        user_id: str,
        limit: Optional[int],
        used: int,
        required: int,
    ) -> Dict[str, Any]:
        limit_value = int(limit or 0)
        return {
            "error": "ai_token_limit_exceeded",
            "limit_type": limit_type,
            "user_id": user_id,
            "limit": limit_value,
            "used": used,
            "estimated_required": required,
            "remaining": max(0, limit_value - used),
        }

    async def _insert_event(self, event: Dict[str, Any]) -> Optional[str]:
        if hasattr(self.db, "insert_event"):
            return await self.db.insert_event(event)
        collection = await self._collection("ai_usage_events")
        if collection is None:
            return None
        result = await collection.insert_one(event)
        return str(result.inserted_id)

    async def _update_event(self, event_id: str, updates: Dict[str, Any]) -> bool:
        if hasattr(self.db, "update_event"):
            return await self.db.update_event(event_id, updates)
        collection = await self._collection("ai_usage_events")
        if collection is None:
            return False
        result = await collection.update_one({"event_id": event_id}, {"$set": updates})
        return bool(result.modified_count)

    async def _reserve_usage(
        self,
        *,
        scope: str,
        subject_id: str,
        period: str,
        period_key: str,
        tokens: int,
        limit: Optional[int],
    ) -> Tuple[bool, int]:
        if hasattr(self.db, "reserve_usage"):
            return await self.db.reserve_usage(
                scope=scope,
                subject_id=subject_id,
                period=period,
                period_key=period_key,
                tokens=tokens,
                limit=limit,
            )
        collection = await self._collection("ai_usage_counters")
        if collection is None:
            return True, tokens

        counter_id = f"{scope}:{subject_id}:{period}:{period_key}"
        if limit is None:
            updated = await collection.find_one_and_update(
                {"counter_id": counter_id},
                {
                    "$setOnInsert": {
                        "counter_id": counter_id,
                        "scope": scope,
                        "subject_id": subject_id,
                        "period": period,
                        "period_key": period_key,
                        "created_at": datetime.now(timezone.utc),
                    },
                    "$inc": {"reserved_tokens": tokens},
                    "$set": {"updated_at": datetime.now(timezone.utc)},
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            return True, int((updated or {}).get("reserved_tokens", tokens))

        await self._ensure_counter_doc(collection, counter_id, scope, subject_id, period, period_key)
        updated = await collection.find_one_and_update(
            {
                "counter_id": counter_id,
                "$or": [
                    {"reserved_tokens": {"$exists": False}},
                    {"reserved_tokens": {"$lte": max(0, limit - tokens)}},
                ],
            },
            {
                "$inc": {"reserved_tokens": tokens},
                "$set": {"updated_at": datetime.now(timezone.utc)},
            },
            return_document=ReturnDocument.AFTER,
        )
        if updated:
            return True, int(updated.get("reserved_tokens", tokens))

        existing = await collection.find_one({"counter_id": counter_id})
        return False, int((existing or {}).get("reserved_tokens", 0))

    async def _ensure_counter_doc(
        self,
        collection: Any,
        counter_id: str,
        scope: str,
        subject_id: str,
        period: str,
        period_key: str,
    ) -> None:
        try:
            await collection.insert_one(
                {
                    "counter_id": counter_id,
                    "scope": scope,
                    "subject_id": subject_id,
                    "period": period,
                    "period_key": period_key,
                    "reserved_tokens": 0,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        except DuplicateKeyError:
            return

    async def _release_reservations(self, reservations: list[Dict[str, str]], tokens: int) -> None:
        for reservation in reversed(reservations):
            await self._release_usage(tokens=tokens, **reservation)

    async def _release_usage(
        self,
        *,
        scope: str,
        subject_id: str,
        period: str,
        period_key: str,
        tokens: int,
    ) -> bool:
        if hasattr(self.db, "release_usage"):
            return await self.db.release_usage(
                scope=scope,
                subject_id=subject_id,
                period=period,
                period_key=period_key,
                tokens=tokens,
            )
        collection = await self._collection("ai_usage_counters")
        if collection is None:
            return False
        counter_id = f"{scope}:{subject_id}:{period}:{period_key}"
        await collection.update_one(
            {"counter_id": counter_id},
            {
                "$inc": {"reserved_tokens": -tokens},
                "$set": {"updated_at": datetime.now(timezone.utc)},
            },
        )
        await collection.update_one(
            {"counter_id": counter_id, "reserved_tokens": {"$lt": 0}},
            {"$set": {"reserved_tokens": 0}},
        )
        return True

    async def _limit_override(self, scope: str, subject_id: str) -> Dict[str, Any]:
        collection = await self._collection("ai_usage_limits")
        if collection is None:
            return {}
        doc = await collection.find_one({"scope": scope, "subject_id": subject_id})
        if not doc or doc.get("enabled") is False:
            return {}
        return doc

    async def _collection(self, name: str) -> Any:
        if self.db is None:
            return None
        if self.is_b2c and hasattr(self.db, "get_b2c_collection"):
            return await self.db.get_b2c_collection(name)
        if hasattr(self.db, "get_context_db"):
            context_db = await self.db.get_context_db()
            return context_db[name] if context_db is not None else None
        return None

    def _extract_usage(self, result: Any) -> Tuple[Optional[int], Optional[int]]:
        usage = None
        if isinstance(result, dict):
            usage = result.get("usage")
        else:
            usage = getattr(result, "usage", None)
        if not usage:
            return None, None
        if isinstance(usage, dict):
            return usage.get("prompt_tokens"), usage.get("completion_tokens")
        return getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)

    def _enabled(self) -> bool:
        return os.getenv("AI_GATEWAY_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}

    def _block_on_limit(self) -> bool:
        return os.getenv("AI_BLOCK_ON_LIMIT", "true").strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None
