"""
LLM Gate — MongoDB repository layer.

Manages three collections inside the per-tenant DB (``skb_<tenant>``):

* ``llm_gate_config``       — singleton config (§7.1)
* ``llm_token_usage_log``   — append-only call log with 7-day TTL (§7.2)
* ``llm_token_usage_rollup``— daily/weekly/monthly rollups (§7.3)

Spec authority  : new-docs/architecture/LLM_GATE_SPEC.md §7
Ownership       : LLM gate (STATE_OWNERSHIP_MAP.md)
Failure modes   : GATE-03 (token logging missing / inconsistent)
Test IDs        : U-GATE-03 (budget checks), U-GATE-04 (log shape), I-GATE-03 (usage API)
Hard constraint : C1 — MongoDB only.  No PostgreSQL.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING
from pymongo.errors import OperationFailure

from .models import (
    GateConfig,
    PeriodType,
    TokenUsageLogEntry,
    TokenUsageRollup,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collection names (LLM_GATE_SPEC.md §7)
# ---------------------------------------------------------------------------
COL_CONFIG = "llm_gate_config"
COL_LOG = "llm_token_usage_log"
COL_ROLLUP = "llm_token_usage_rollup"

# TTL for raw log rows: 7 days (§7.2 / §8)
_LOG_TTL_SECONDS = 7 * 24 * 60 * 60  # 604 800


class GateRepository:
    """
    Async MongoDB repository for LLM gate state.

    Ownership declaration (STATE_OWNERSHIP_MAP.md §5):
        Writes : llm_gate_config, llm_token_usage_log, llm_token_usage_rollup
        Reads  : (same collections — consumed by usage APIs and budget checks)
        Never writes to : conducted-exam artifacts, DCR/PCR evaluation state, practice persistence
        Transactional boundaries : provider response + append-only usage log
    """

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._config_col = db[COL_CONFIG]
        self._log_col = db[COL_LOG]
        self._rollup_col = db[COL_ROLLUP]

    # ------------------------------------------------------------------
    # Index bootstrap (idempotent — safe to call on every startup)
    # ------------------------------------------------------------------

    async def ensure_indexes(self) -> None:
        """Create indexes required by LLM_GATE_SPEC.md §7.2 and §7.3."""
        try:
            # llm_token_usage_log indexes
            # TTL index on called_at — 7 day retention
            try:
                await self._log_col.create_index(
                    [("called_at", ASCENDING)],
                    name="ttl_called_at",
                    expireAfterSeconds=_LOG_TTL_SECONDS,
                )
            except OperationFailure:
                # Index may already exist with different TTL — leave it
                logger.debug("TTL index on %s.called_at already exists", COL_LOG)

            await self._log_col.create_index(
                [("caller", ASCENDING)],
                name="idx_caller",
            )
            await self._log_col.create_index(
                [("model", ASCENDING)],
                name="idx_model",
            )

            # llm_token_usage_rollup unique compound index
            await self._rollup_col.create_index(
                [("period_type", ASCENDING), ("period_start", ASCENDING)],
                name="uniq_period",
                unique=True,
            )

            logger.info("LLM gate indexes ensured on %s", self._db.name)
        except Exception:
            logger.exception("Failed to ensure LLM gate indexes on %s", self._db.name)

    # ------------------------------------------------------------------
    # Config (§7.1)
    # ------------------------------------------------------------------

    async def get_config(self) -> GateConfig:
        """Return the current gate config or sensible defaults if unset."""
        doc = await self._config_col.find_one({"_id": "gate_config"})
        if doc is None:
            return GateConfig()
        # Strip Mongo _id before hydrating Pydantic model
        doc.pop("_id", None)
        return GateConfig(**doc)

    async def upsert_config(self, config: GateConfig) -> None:
        """Upsert the singleton config document."""
        payload = config.model_dump(mode="json")
        payload["updated_at"] = datetime.utcnow()
        await self._config_col.update_one(
            {"_id": "gate_config"},
            {"$set": payload},
            upsert=True,
        )

    # ------------------------------------------------------------------
    # Append-only token log (§7.2) — GATE-03 mitigation
    # ------------------------------------------------------------------

    async def append_log(self, entry: TokenUsageLogEntry) -> None:
        """Insert a single append-only log row.  Never updates or deletes."""
        doc = entry.model_dump(mode="json")
        await self._log_col.insert_one(doc)

    # ------------------------------------------------------------------
    # Rollup helpers (§7.3 / §8)
    # ------------------------------------------------------------------

    async def upsert_rollup(self, rollup: TokenUsageRollup) -> None:
        """Upsert a rollup document keyed by (period_type, period_start)."""
        flt = {
            "period_type": rollup.period_type.value,
            "period_start": rollup.period_start,
        }
        payload = rollup.model_dump(mode="json")
        await self._rollup_col.update_one(flt, {"$set": payload}, upsert=True)

    async def get_rollup(
        self,
        period_type: PeriodType,
        period_start: datetime,
    ) -> Optional[TokenUsageRollup]:
        """Fetch a single rollup for a given period."""
        doc = await self._rollup_col.find_one({
            "period_type": period_type.value,
            "period_start": period_start,
        })
        if doc is None:
            return None
        doc.pop("_id", None)
        return TokenUsageRollup(**doc)

    async def list_rollups(
        self,
        period_type: Optional[PeriodType] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[TokenUsageRollup]:
        """List rollups, optionally filtered."""
        flt: Dict[str, Any] = {}
        if period_type is not None:
            flt["period_type"] = period_type.value
        if since is not None:
            flt["period_start"] = {"$gte": since}
        cursor = self._rollup_col.find(flt).sort("period_start", ASCENDING).limit(limit)
        results: List[TokenUsageRollup] = []
        async for doc in cursor:
            doc.pop("_id", None)
            results.append(TokenUsageRollup(**doc))
        return results

    # ------------------------------------------------------------------
    # Token aggregation queries (used by budget enforcement — §4 step 4)
    # ------------------------------------------------------------------

    async def sum_tokens_since(self, since: datetime) -> int:
        """
        Sum ``total_tokens`` from the raw log for rows with
        ``called_at >= since``.  Falls back to 0 on error.

        For budget enforcement the caller should combine this with any
        existing rollup whose period overlaps.
        """
        pipeline = [
            {"$match": {"called_at": {"$gte": since}}},
            {"$group": {"_id": None, "total": {"$sum": "$total_tokens"}}},
        ]
        results = await self._log_col.aggregate(pipeline).to_list(length=1)
        if results:
            return int(results[0].get("total", 0))
        return 0

    async def sum_tokens_in_range(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """
        Aggregate log rows in ``[start, end)`` and return a summary dict
        suitable for rollup creation.
        """
        pipeline = [
            {"$match": {"called_at": {"$gte": start, "$lt": end}}},
            {
                "$group": {
                    "_id": None,
                    "total_tokens": {"$sum": "$total_tokens"},
                    "total_input": {"$sum": "$input_tokens"},
                    "total_output": {"$sum": "$output_tokens"},
                    "total_cost_usd": {"$sum": "$estimated_cost_usd"},
                    "call_count": {"$sum": 1},
                }
            },
        ]
        results = await self._log_col.aggregate(pipeline).to_list(length=1)
        if results:
            r = results[0]
            r.pop("_id", None)
            return r
        return {
            "total_tokens": 0,
            "total_input": 0,
            "total_output": 0,
            "total_cost_usd": 0.0,
            "call_count": 0,
        }

    async def breakdown_in_range(
        self,
        start: datetime,
        end: datetime,
        group_field: str,
    ) -> List[Dict[str, Any]]:
        """
        Aggregate log rows grouped by ``group_field`` (``model`` or ``caller``)
        within ``[start, end)``.
        """
        pipeline = [
            {"$match": {"called_at": {"$gte": start, "$lt": end}}},
            {
                "$group": {
                    "_id": f"${group_field}",
                    "total_tokens": {"$sum": "$total_tokens"},
                    "total_input": {"$sum": "$input_tokens"},
                    "total_output": {"$sum": "$output_tokens"},
                    "total_cost_usd": {"$sum": "$estimated_cost_usd"},
                    "call_count": {"$sum": 1},
                }
            },
        ]
        results = await self._log_col.aggregate(pipeline).to_list(length=100)
        return [
            {
                "key": r["_id"],
                "total_tokens": r["total_tokens"],
                "total_input": r["total_input"],
                "total_output": r["total_output"],
                "total_cost_usd": r["total_cost_usd"],
                "call_count": r["call_count"],
            }
            for r in results
            if r["_id"] is not None
        ]
