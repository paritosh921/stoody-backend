"""Upload verdict persistence helpers."""

from __future__ import annotations

from datetime import datetime
from inspect import iscoroutinefunction
from typing import Any, Literal


UploadVerdictStatus = Literal["pending", "clean", "rejected", "scan_failed"]
_INDEXED_DB_IDS: set[int] = set()


def build_upload_verdict(
    *,
    upload_id: str,
    policy_id: str,
    status: UploadVerdictStatus,
    sha256: str,
    size_bytes: int,
    original_filename: str,
    declared_content_type: str,
    detected_magic_type: str,
    scanner_name: str | None,
    scanner_version: str | None,
    scan_started_at: datetime,
    scan_finished_at: datetime,
    tenant_db: str | None,
    user_id: str | None,
    purpose_metadata: dict[str, Any],
    authorization_subject: str,
    quarantine_storage_path: str,
    released_storage_path: str | None = None,
    rejection_reason: str | None = None,
) -> dict[str, Any]:
    now = datetime.utcnow()
    return {
        "upload_id": upload_id,
        "policy_id": policy_id,
        "status": status,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "original_filename": original_filename,
        "declared_content_type": declared_content_type,
        "detected_magic_type": detected_magic_type,
        "scanner_name": scanner_name,
        "scanner_version": scanner_version,
        "scan_started_at": scan_started_at,
        "scan_finished_at": scan_finished_at,
        "tenant_db": tenant_db,
        "user_id": user_id,
        "purpose_metadata": purpose_metadata,
        "authorization_subject": authorization_subject,
        "quarantine_storage_path": quarantine_storage_path,
        "released_storage_path": released_storage_path,
        "rejection_reason": rejection_reason,
        "created_at": now,
        "updated_at": now,
    }


async def persist_upload_verdict(db: Any, verdict: dict[str, Any]) -> None:
    # ``AsyncIOMotorDatabase`` dynamically returns a collection for arbitrary
    # attributes.  ``hasattr(db, "mongo_insert_one")`` is therefore true for a
    # Motor database even though it is not the DatabaseManager helper method.
    # Motor collections are themselves callable (they raise a helpful error
    # only when called), so ``callable(...)`` is not enough here.  The
    # DatabaseManager adapter is an async method; only use it when the
    # resolved attribute is actually an async function.  Otherwise persist
    # through the tenant collection directly.
    insert_with_manager = getattr(db, "mongo_insert_one", None)
    if iscoroutinefunction(insert_with_manager):
        await insert_with_manager("upload_security_verdicts", verdict)
        return
    await ensure_upload_verdict_indexes(db)
    collection = db["upload_security_verdicts"]
    await collection.insert_one(verdict)


async def ensure_upload_verdict_indexes(db: Any) -> None:
    db_id = id(db)
    if db_id in _INDEXED_DB_IDS:
        return
    collection = db["upload_security_verdicts"]
    for field in ("upload_id", "sha256", "tenant_db", "status", "created_at"):
        await collection.create_index(field)
    _INDEXED_DB_IDS.add(db_id)
