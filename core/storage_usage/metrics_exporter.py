"""Safe Prometheus rows for backend private storage and MongoDB usage."""

from __future__ import annotations

import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config_async import settings
from core.ai_usage.metrics_exporter import MAX_TENANT_DATABASES, public_identity_ref


PRIVATE_UPLOAD_STORAGE = "private_uploads"
PLATFORM_REF = "platform"
B2C_REF = "b2c"

_CACHE_LOCK = asyncio.Lock()
_CACHE_ROOT: str | None = None
_CACHE_ROWS: list[dict[str, Any]] | None = None
_CACHE_ERRORS = 0
_CACHE_SUCCESS = True
_CACHE_GENERATED_AT = 0.0
_CACHE_EXPIRES_AT = 0.0


async def build_storage_usage_metric_rows(
    db_manager: Any,
    *,
    local_root: str | Path | None = None,
    now: datetime | None = None,
    cache_ttl_seconds: int | None = None,
) -> list[dict[str, Any]]:
    root = Path(local_root or settings.UPLOAD_PRIVATE_LOCAL_DIR)
    current_time = _utc(now)
    current_ts = current_time.timestamp()
    ttl = settings.STORAGE_USAGE_METRICS_CACHE_TTL_SECONDS if cache_ttl_seconds is None else int(cache_ttl_seconds)

    if ttl <= 0:
        rows, errors, success = await _collect_metric_rows(db_manager, root)
        return rows + _collector_rows(success=success, errors=errors, generated_at=current_ts, now_ts=current_ts)

    root_key = str(root.resolve(strict=False))
    if _cache_valid(root_key, current_ts):
        return list(_CACHE_ROWS or []) + _collector_rows(
            success=_CACHE_SUCCESS,
            errors=_CACHE_ERRORS,
            generated_at=_CACHE_GENERATED_AT,
            now_ts=current_ts,
        )

    async with _CACHE_LOCK:
        if _cache_valid(root_key, current_ts):
            return list(_CACHE_ROWS or []) + _collector_rows(
                success=_CACHE_SUCCESS,
                errors=_CACHE_ERRORS,
                generated_at=_CACHE_GENERATED_AT,
                now_ts=current_ts,
            )

        rows, errors, success = await _collect_metric_rows(db_manager, root)
        _store_cache(
            root_key=root_key,
            rows=rows,
            errors=errors,
            success=success,
            generated_at=current_ts,
            expires_at=current_ts + ttl,
        )
        return rows + _collector_rows(success=success, errors=errors, generated_at=current_ts, now_ts=current_ts)


def reset_storage_usage_metric_cache() -> None:
    _store_cache(root_key=None, rows=None, errors=0, success=True, generated_at=0.0, expires_at=0.0)


async def _collect_metric_rows(db_manager: Any, root: Path) -> tuple[list[dict[str, Any]], int, bool]:
    rows: list[dict[str, Any]] = []
    errors = 0
    success = True

    try:
        rows.extend(_private_upload_storage_rows(root))
    except Exception:
        success = False
        errors += 1

    mongo_rows, mongo_errors = await _mongodb_storage_rows(db_manager)
    rows.extend(mongo_rows)
    errors += mongo_errors
    return rows, errors, success and errors == 0


def _private_upload_storage_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    objects_total = 0

    for prefix in _private_prefixes():
        base = root / prefix
        if not base.exists():
            continue
        for tenant_path in base.iterdir():
            if tenant_path.is_dir():
                tenant_segment = tenant_path.name
                used = _directory_size(tenant_path)
            elif tenant_path.is_file():
                tenant_segment = "unknown"
                used = _safe_file_size(tenant_path)
            else:
                continue
            if used <= 0:
                continue
            objects_total += used
            rows.append(
                {
                    "metric": "tenant_storage",
                    "labels": {
                        "tenant_ref": public_identity_ref(tenant_segment, prefix="tenant"),
                        "storage": PRIVATE_UPLOAD_STORAGE,
                        "prefix": _safe_label(prefix),
                        "kind": "objects_used",
                    },
                    "value": float(used),
                }
            )

    rows.extend(
        [
            {
                "metric": "backend_storage",
                "labels": {"storage": PRIVATE_UPLOAD_STORAGE, "kind": "objects_used"},
                "value": float(objects_total),
            },
            *_filesystem_rows(root),
        ]
    )
    return rows


def _filesystem_rows(root: Path) -> list[dict[str, Any]]:
    usage_path = _nearest_existing_path(root)
    usage = shutil.disk_usage(usage_path)
    return [
        {
            "metric": "backend_storage",
            "labels": {"storage": PRIVATE_UPLOAD_STORAGE, "kind": "filesystem_capacity"},
            "value": float(usage.total),
        },
        {
            "metric": "backend_storage",
            "labels": {"storage": PRIVATE_UPLOAD_STORAGE, "kind": "filesystem_used"},
            "value": float(usage.used),
        },
        {
            "metric": "backend_storage",
            "labels": {"storage": PRIVATE_UPLOAD_STORAGE, "kind": "filesystem_free"},
            "value": float(usage.free),
        },
    ]


async def _mongodb_storage_rows(db_manager: Any) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    errors = 0
    if db_manager is None:
        return rows, errors

    master_db = await _maybe_await(getattr(db_manager, "get_master_db", lambda: None)())
    b2c_db = await _maybe_await(getattr(db_manager, "get_b2c_db", lambda: None)())

    if master_db is not None:
        stats, failed = await _db_stats(master_db)
        errors += failed
        rows.extend(_mongodb_rows_for_stats(database_role="master", tenant_ref=PLATFORM_REF, stats=stats))

    if b2c_db is not None:
        stats, failed = await _db_stats(b2c_db)
        errors += failed
        rows.extend(_mongodb_rows_for_stats(database_role="b2c", tenant_ref=B2C_REF, stats=stats))

    if master_db is None:
        return rows, errors

    for tenant in await _tenant_docs(master_db):
        db_name = str(tenant.get("db_name") or "").strip()
        if not db_name:
            continue
        get_tenant_db = getattr(db_manager, "get_tenant_db", None)
        if get_tenant_db is None:
            continue
        tenant_db = await _maybe_await(get_tenant_db(db_name))
        if tenant_db is None:
            continue
        stats, failed = await _db_stats(tenant_db)
        errors += failed
        tenant_ref = public_identity_ref(tenant.get("tenant_id") or tenant.get("_id") or db_name, prefix="tenant")
        rows.extend(_mongodb_rows_for_stats(database_role="tenant", tenant_ref=tenant_ref, stats=stats))

    return rows, errors


def _mongodb_rows_for_stats(*, database_role: str, tenant_ref: str, stats: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not stats:
        return []
    data_size = _float(stats.get("dataSize"))
    storage_size = _float(stats.get("storageSize"))
    index_size = _float(stats.get("indexSize"))
    total_size = _float(stats.get("totalSize")) or (storage_size + index_size)
    rows = []
    for kind, value in (
        ("data", data_size),
        ("storage", storage_size),
        ("index", index_size),
        ("total", total_size),
    ):
        rows.append(
            {
                "metric": "mongodb_storage",
                "labels": {
                    "database_role": _safe_label(database_role),
                    "tenant_ref": _safe_label(tenant_ref),
                    "kind": kind,
                },
                "value": value,
            }
        )
    return rows


async def _db_stats(database: Any) -> tuple[dict[str, Any] | None, int]:
    try:
        return await _maybe_await(database.command("dbStats", scale=1)), 0
    except Exception:
        return None, 1


async def _tenant_docs(master_db: Any) -> list[dict[str, Any]]:
    try:
        cursor = master_db["tenants"].find(
            {"db_name": {"$exists": True, "$nin": ["", None]}},
            {"_id": 1, "tenant_id": 1, "db_name": 1},
        ).limit(MAX_TENANT_DATABASES)
        return await _cursor_to_list(cursor, MAX_TENANT_DATABASES)
    except Exception:
        return []


def _collector_rows(*, success: bool, errors: int, generated_at: float, now_ts: float) -> list[dict[str, Any]]:
    return [
        {
            "metric": "storage_collector",
            "labels": {"field": "success"},
            "value": 1.0 if success else 0.0,
        },
        {
            "metric": "storage_collector",
            "labels": {"field": "snapshot_age_seconds"},
            "value": max(float(now_ts - generated_at), 0.0),
        },
        {
            "metric": "storage_collector",
            "labels": {"field": "collection_errors"},
            "value": float(max(int(errors), 0)),
        },
    ]


def _private_prefixes() -> tuple[str, ...]:
    return (
        settings.UPLOAD_QUARANTINE_PREFIX,
        settings.UPLOAD_REJECTED_PREFIX,
        settings.UPLOAD_RELEASED_PREFIX,
        settings.UPLOAD_DERIVED_PREFIX,
    )


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        total += _safe_file_size(item)
    return total


def _safe_file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _nearest_existing_path(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            return Path(".")
        current = parent
    return current


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


def _cache_valid(root_key: str, now_ts: float) -> bool:
    return _CACHE_ROWS is not None and _CACHE_ROOT == root_key and now_ts < _CACHE_EXPIRES_AT


def _store_cache(
    *,
    root_key: str | None,
    rows: list[dict[str, Any]] | None,
    errors: int,
    success: bool,
    generated_at: float,
    expires_at: float,
) -> None:
    global _CACHE_ROOT, _CACHE_ROWS, _CACHE_ERRORS, _CACHE_SUCCESS, _CACHE_GENERATED_AT, _CACHE_EXPIRES_AT
    _CACHE_ROOT = root_key
    _CACHE_ROWS = list(rows) if rows is not None else None
    _CACHE_ERRORS = int(errors)
    _CACHE_SUCCESS = bool(success)
    _CACHE_GENERATED_AT = float(generated_at)
    _CACHE_EXPIRES_AT = float(expires_at)


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
