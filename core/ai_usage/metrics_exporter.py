"""Safe Prometheus export for separate general and ExamPen AI usage."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

MAX_TENANT_DATABASES = 250
DEFAULT_TOP_N = 10

PERIOD_WINDOWS = {
    "today": "today",
    "7d": "7d",
    "30d": "30d",
}


def public_identity_ref(value: Any, *, prefix: str = "user") -> str:
    raw = str(value or "unknown").strip() or "unknown"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


async def build_ai_usage_metric_rows(db_manager: Any, *, now: datetime | None = None, top_n: int = DEFAULT_TOP_N) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    now = _utc(now)
    top_n = max(1, min(int(top_n or DEFAULT_TOP_N), DEFAULT_TOP_N))
    databases = await _usage_databases(db_manager)

    for period, start in _period_starts(now).items():
        general_breakdown: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        general_users: dict[str, float] = {}
        general_tenants: dict[str, float] = {}
        exampen_breakdown: dict[tuple[str, str], dict[str, Any]] = {}

        for tenant_ref, database in databases:
            _merge_general_breakdown(
                general_breakdown,
                await _aggregate_general_breakdown(database, start),
            )
            _merge_identity_totals(
                general_users,
                await _aggregate_general_identity(database, start, identity_field="user_id"),
                identity_field="user_id",
            )
            tenant_total = await _aggregate_general_total(database, start)
            if tenant_total > 0:
                general_tenants[tenant_ref] = general_tenants.get(tenant_ref, 0.0) + tenant_total

            _merge_exampen_breakdown(
                exampen_breakdown,
                await _aggregate_exampen_breakdown(database, start),
            )

        rows.extend(
            build_general_ai_usage_metric_rows_from_aggregates(
                period=period,
                breakdown=general_breakdown.values(),
                top_users=[
                    {"user_id": user_id, "total_tokens": total}
                    for user_id, total in _top_items(general_users, top_n)
                ],
                top_tenants=[
                    {"tenant_id": tenant_ref, "total_tokens": total}
                    for tenant_ref, total in _top_items(general_tenants, top_n)
                ],
            )
        )
        rows.extend(
            build_exampen_ai_usage_metric_rows_from_aggregates(
                period=period,
                breakdown=exampen_breakdown.values(),
            )
        )

    return rows


def build_general_ai_usage_metric_rows_from_aggregates(
    *,
    period: str,
    breakdown: Iterable[dict[str, Any]],
    top_users: Iterable[dict[str, Any]],
    top_tenants: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in breakdown:
        labels = {
            "period": _safe_label(period),
            "provider": _safe_label(item.get("provider")),
            "model": _safe_label(item.get("model")),
            "stage": _safe_label(item.get("stage")),
            "status": _safe_label(item.get("status")),
        }
        for token_type, field in (
            ("input", "input_tokens"),
            ("output", "output_tokens"),
            ("total", "total_tokens"),
        ):
            rows.append(
                {
                    "metric": "general_tokens",
                    "labels": {**labels, "token_type": token_type},
                    "value": _float(item.get(field)),
                }
            )
        rows.append(
            {
                "metric": "general_calls",
                "labels": labels,
                "value": _float(item.get("calls")),
            }
        )

    for rank, item in enumerate(top_users, start=1):
        rows.append(
            {
                "metric": "general_top_user_tokens",
                "labels": {
                    "period": _safe_label(period),
                    "rank": str(rank),
                    "user_ref": public_identity_ref(item.get("user_id"), prefix="user"),
                },
                "value": _float(item.get("total_tokens")),
            }
        )

    for rank, item in enumerate(top_tenants, start=1):
        rows.append(
            {
                "metric": "general_top_tenant_tokens",
                "labels": {
                    "period": _safe_label(period),
                    "rank": str(rank),
                    "tenant_ref": public_identity_ref(item.get("tenant_id"), prefix="tenant"),
                },
                "value": _float(item.get("total_tokens")),
            }
        )

    return rows


def build_exampen_ai_usage_metric_rows_from_aggregates(
    *,
    period: str,
    breakdown: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in breakdown:
        labels = {
            "period": _safe_label(period),
            "caller": _safe_label(item.get("caller")),
            "model": _safe_label(item.get("model")),
        }
        for token_type, field in (
            ("input", "input_tokens"),
            ("output", "output_tokens"),
            ("cache_read", "cache_read_tokens"),
            ("cache_creation", "cache_creation_tokens"),
            ("total", "total_tokens"),
        ):
            rows.append(
                {
                    "metric": "exampen_tokens",
                    "labels": {**labels, "token_type": token_type},
                    "value": _float(item.get(field)),
                }
            )
        rows.append(
            {
                "metric": "exampen_calls",
                "labels": labels,
                "value": _float(item.get("calls")),
            }
        )
    return rows


async def _usage_databases(db_manager: Any) -> list[tuple[str, Any]]:
    databases: list[tuple[str, Any]] = []

    b2c_db = await _maybe_await(getattr(db_manager, "get_b2c_db", lambda: None)())
    if b2c_db is not None:
        databases.append(("b2c", b2c_db))

    master_db = await _maybe_await(getattr(db_manager, "get_master_db", lambda: None)())
    if master_db is None:
        return databases

    tenants = await _tenant_docs(master_db)
    seen = {"b2c"}
    for tenant in tenants:
        db_name = str(tenant.get("db_name") or "").strip()
        if not db_name or db_name in seen:
            continue
        tenant_db = await _maybe_await(getattr(db_manager, "get_tenant_db")(db_name))
        if tenant_db is None:
            continue
        tenant_ref = str(tenant.get("tenant_id") or tenant.get("_id") or db_name)
        databases.append((tenant_ref, tenant_db))
        seen.add(db_name)
    return databases


async def _tenant_docs(master_db: Any) -> list[dict[str, Any]]:
    try:
        cursor = master_db["tenants"].find(
            {"db_name": {"$exists": True, "$nin": ["", None]}},
            {"_id": 1, "tenant_id": 1, "db_name": 1},
        ).limit(MAX_TENANT_DATABASES)
        return await _cursor_to_list(cursor, MAX_TENANT_DATABASES)
    except Exception:
        return []


async def _aggregate_general_breakdown(database: Any, start: datetime) -> list[dict[str, Any]]:
    pipeline = [
        {"$match": {"created_at": {"$gte": start}}},
        {
            "$project": {
                "provider": {"$ifNull": ["$provider", "unknown"]},
                "model": {"$ifNull": ["$model", "unknown"]},
                "stage": {"$ifNull": ["$stage", "unknown"]},
                "status": {"$ifNull": ["$status", "unknown"]},
                "input_tokens": {"$ifNull": ["$actual_input_tokens", 0]},
                "output_tokens": {"$ifNull": ["$actual_output_tokens", 0]},
                "total_tokens": _general_total_tokens_expression(),
            }
        },
        {
            "$group": {
                "_id": {
                    "provider": "$provider",
                    "model": "$model",
                    "stage": "$stage",
                    "status": "$status",
                },
                "input_tokens": {"$sum": "$input_tokens"},
                "output_tokens": {"$sum": "$output_tokens"},
                "total_tokens": {"$sum": "$total_tokens"},
                "calls": {"$sum": 1},
            }
        },
    ]
    rows = await _aggregate(database, "ai_usage_events", pipeline)
    return [
        {
            "provider": (row.get("_id") or {}).get("provider"),
            "model": (row.get("_id") or {}).get("model"),
            "stage": (row.get("_id") or {}).get("stage"),
            "status": (row.get("_id") or {}).get("status"),
            "input_tokens": row.get("input_tokens"),
            "output_tokens": row.get("output_tokens"),
            "total_tokens": row.get("total_tokens"),
            "calls": row.get("calls"),
        }
        for row in rows
    ]


async def _aggregate_general_identity(database: Any, start: datetime, *, identity_field: str) -> list[dict[str, Any]]:
    pipeline = [
        {"$match": {"created_at": {"$gte": start}, identity_field: {"$exists": True, "$nin": ["", None]}}},
        {"$project": {identity_field: f"${identity_field}", "total_tokens": _general_total_tokens_expression()}},
        {"$group": {"_id": f"${identity_field}", "total_tokens": {"$sum": "$total_tokens"}}},
        {"$sort": {"total_tokens": -1}},
        {"$limit": DEFAULT_TOP_N},
    ]
    rows = await _aggregate(database, "ai_usage_events", pipeline)
    return [{identity_field: row.get("_id"), "total_tokens": row.get("total_tokens")} for row in rows]


async def _aggregate_general_total(database: Any, start: datetime) -> float:
    pipeline = [
        {"$match": {"created_at": {"$gte": start}}},
        {"$project": {"total_tokens": _general_total_tokens_expression()}},
        {"$group": {"_id": None, "total_tokens": {"$sum": "$total_tokens"}}},
    ]
    rows = await _aggregate(database, "ai_usage_events", pipeline)
    return _float(rows[0].get("total_tokens")) if rows else 0.0


async def _aggregate_exampen_breakdown(database: Any, start: datetime) -> list[dict[str, Any]]:
    pipeline = [
        {"$match": {"called_at": {"$gte": start}}},
        {
            "$group": {
                "_id": {
                    "caller": {"$ifNull": ["$caller", "unknown"]},
                    "model": {"$ifNull": ["$model", "unknown"]},
                },
                "input_tokens": {"$sum": "$input_tokens"},
                "output_tokens": {"$sum": "$output_tokens"},
                "cache_read_tokens": {"$sum": "$cache_read_tokens"},
                "cache_creation_tokens": {"$sum": "$cache_creation_tokens"},
                "total_tokens": {"$sum": "$total_tokens"},
                "calls": {"$sum": 1},
            }
        },
    ]
    rows = await _aggregate(database, "llm_token_usage_log", pipeline)
    return [
        {
            "caller": (row.get("_id") or {}).get("caller"),
            "model": (row.get("_id") or {}).get("model"),
            "input_tokens": row.get("input_tokens"),
            "output_tokens": row.get("output_tokens"),
            "cache_read_tokens": row.get("cache_read_tokens"),
            "cache_creation_tokens": row.get("cache_creation_tokens"),
            "total_tokens": row.get("total_tokens"),
            "calls": row.get("calls"),
        }
        for row in rows
    ]


async def _aggregate(database: Any, collection_name: str, pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        cursor = database[collection_name].aggregate(pipeline)
        return await _cursor_to_list(cursor, 1000)
    except Exception:
        return []


def _general_total_tokens_expression() -> dict[str, Any]:
    return {
        "$let": {
            "vars": {
                "actual": {
                    "$add": [
                        {"$ifNull": ["$actual_input_tokens", 0]},
                        {"$ifNull": ["$actual_output_tokens", 0]},
                    ]
                }
            },
            "in": {
                "$cond": [
                    {"$gt": ["$$actual", 0]},
                    "$$actual",
                    {"$ifNull": ["$estimated_total_tokens", 0]},
                ]
            },
        }
    }


def _merge_general_breakdown(target: dict[tuple[str, str, str, str], dict[str, Any]], rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        key = (
            _safe_label(row.get("provider")),
            _safe_label(row.get("model")),
            _safe_label(row.get("stage")),
            _safe_label(row.get("status")),
        )
        current = target.setdefault(
            key,
            {
                "provider": key[0],
                "model": key[1],
                "stage": key[2],
                "status": key[3],
                "input_tokens": 0.0,
                "output_tokens": 0.0,
                "total_tokens": 0.0,
                "calls": 0.0,
            },
        )
        for field in ("input_tokens", "output_tokens", "total_tokens", "calls"):
            current[field] += _float(row.get(field))


def _merge_exampen_breakdown(target: dict[tuple[str, str], dict[str, Any]], rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        key = (_safe_label(row.get("caller")), _safe_label(row.get("model")))
        current = target.setdefault(
            key,
            {
                "caller": key[0],
                "model": key[1],
                "input_tokens": 0.0,
                "output_tokens": 0.0,
                "cache_read_tokens": 0.0,
                "cache_creation_tokens": 0.0,
                "total_tokens": 0.0,
                "calls": 0.0,
            },
        )
        for field in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_creation_tokens", "total_tokens", "calls"):
            current[field] += _float(row.get(field))


def _merge_identity_totals(target: dict[str, float], rows: Iterable[dict[str, Any]], *, identity_field: str) -> None:
    for row in rows:
        identity = str(row.get(identity_field) or "").strip()
        if identity:
            target[identity] = target.get(identity, 0.0) + _float(row.get("total_tokens"))


def _top_items(values: dict[str, float], top_n: int) -> list[tuple[str, float]]:
    return sorted(values.items(), key=lambda item: item[1], reverse=True)[:top_n]


def _period_starts(now: datetime) -> dict[str, datetime]:
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return {
        "today": today,
        "7d": now - timedelta(days=7),
        "30d": now - timedelta(days=30),
    }


async def _cursor_to_list(cursor: Any, length: int) -> list[dict[str, Any]]:
    if hasattr(cursor, "to_list"):
        return await _maybe_await(cursor.to_list(length=length))
    if isinstance(cursor, list):
        return cursor[:length]
    return list(cursor)[:length]


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _safe_label(value: Any, fallback: str = "unknown") -> str:
    text = str(value if value is not None else fallback).strip()
    return text or fallback


def _float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0
