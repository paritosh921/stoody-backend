"""High-level secure upload orchestration."""

from __future__ import annotations

import hashlib
import inspect
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable

from fastapi import HTTPException, status

from core.observability import record_upload_security_decision

from .detection import DetectedFileType, detect_file_type
from .policies import UploadPolicy, get_upload_policy
from .scanner import ClamAVScanner, ScanResult
from .storage import PrivateUploadStorage, safe_filename
from .validation import read_upload_file_limited, run_post_scan_parser_guards
from .verdicts import build_upload_verdict, persist_upload_verdict


@dataclass(frozen=True)
class CleanUpload:
    upload_id: str
    original_filename: str
    content_type: str
    size_bytes: int
    sha256: str
    detected_magic_type: str
    released_storage_path: str
    purpose_metadata: dict[str, Any]
    bytes: bytes | None = None


async def secure_upload(
    *,
    file: Any,
    policy_id: str,
    actor: dict[str, Any] | None,
    db: Any,
    purpose_metadata: dict[str, Any],
    authorization_subject: str,
    scanner: Any | None = None,
    storage: PrivateUploadStorage | None = None,
    include_bytes: bool = True,
) -> CleanUpload:
    policy = get_upload_policy(policy_id)
    _validate_metadata(purpose_metadata, authorization_subject)
    data = await read_upload_file_limited(file, policy)
    return await _secure_upload_bytes(
        data=data,
        filename=getattr(file, "filename", None),
        content_type=getattr(file, "content_type", None),
        policy=policy,
        actor=actor,
        db=db,
        purpose_metadata=purpose_metadata,
        authorization_subject=authorization_subject,
        scanner=scanner,
        storage=storage,
        include_bytes=include_bytes,
    )


async def secure_upload_many(
    *,
    files: Iterable[Any],
    policy_id: str,
    actor: dict[str, Any] | None,
    db: Any,
    purpose_metadata_factory: Callable[[Any, int], dict[str, Any]],
    authorization_subject_factory: Callable[[Any, int], str],
    scanner: Any | None = None,
    storage: PrivateUploadStorage | None = None,
    include_bytes: bool = True,
) -> list[CleanUpload]:
    policy = get_upload_policy(policy_id)
    file_list = list(files)
    if policy.max_files is not None and len(file_list) > policy.max_files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Too many files for {policy.policy_id}")

    prepared: list[tuple[Any, bytes]] = []
    total = 0
    for file in file_list:
        data = await read_upload_file_limited(file, policy)
        total += len(data)
        if policy.max_total_size_bytes is not None and total > policy.max_total_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Total upload size exceeds {policy.max_total_size_bytes} bytes",
            )
        prepared.append((file, data))

    results: list[CleanUpload] = []
    for index, (file, data) in enumerate(prepared):
        purpose_metadata = purpose_metadata_factory(file, index)
        authorization_subject = authorization_subject_factory(file, index)
        _validate_metadata(purpose_metadata, authorization_subject)
        results.append(
            await _secure_upload_bytes(
                data=data,
                filename=getattr(file, "filename", None),
                content_type=getattr(file, "content_type", None),
                policy=policy,
                actor=actor,
                db=db,
                purpose_metadata=purpose_metadata,
                authorization_subject=authorization_subject,
                scanner=scanner,
                storage=storage,
                include_bytes=include_bytes,
            )
        )
    return results


async def _secure_upload_bytes(
    *,
    data: bytes,
    filename: str | None,
    content_type: str | None,
    policy: UploadPolicy,
    actor: dict[str, Any] | None,
    db: Any,
    purpose_metadata: dict[str, Any],
    authorization_subject: str,
    scanner: Any | None,
    storage: PrivateUploadStorage | None,
    include_bytes: bool,
) -> CleanUpload:
    detected = detect_file_type(data, filename, content_type, policy)
    upload_id = str(uuid.uuid4())
    tenant_db = _tenant_db(actor)
    user_id = _user_id(actor)
    sha256 = hashlib.sha256(data).hexdigest()
    storage = storage or PrivateUploadStorage()
    scanner = scanner or ClamAVScanner()

    quarantine_path = await storage.write_quarantine(
        data=data,
        tenant=tenant_db,
        upload_id=upload_id,
        original_filename=filename,
    )

    scan_started_at = datetime.utcnow()
    scan_result: ScanResult = await scanner.scan_path(quarantine_path, filename=filename or "", policy_id=policy.policy_id)
    scan_finished_at = datetime.utcnow()

    if scan_result.status != "clean":
        await storage.mark_rejected(quarantine_path=quarantine_path, tenant=tenant_db, upload_id=upload_id)
        status_value = "rejected" if scan_result.status == "rejected" else "scan_failed"
        _record_decision(policy.policy_id, status_value)
        await _persist_verdict(
            db=db,
            upload_id=upload_id,
            policy=policy,
            status=status_value,
            sha256=sha256,
            size_bytes=len(data),
            filename=filename,
            content_type=content_type,
            detected=detected,
            scan_result=scan_result,
            scan_started_at=scan_started_at,
            scan_finished_at=scan_finished_at,
            tenant_db=tenant_db,
            user_id=user_id,
            purpose_metadata=purpose_metadata,
            authorization_subject=authorization_subject,
            quarantine_path=quarantine_path,
            released_path=None,
            rejection_reason=scan_result.signature or scan_result.error,
        )
        if scan_result.status == "rejected":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Malware detected in upload")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Malware scanner unavailable")

    try:
        guard_result = run_post_scan_parser_guards(data, detected, policy)
        if inspect.isawaitable(guard_result):
            await guard_result
    except HTTPException as exc:
        await storage.mark_rejected(quarantine_path=quarantine_path, tenant=tenant_db, upload_id=upload_id)
        _record_decision(policy.policy_id, "parser_rejected")
        await _persist_verdict(
            db=db,
            upload_id=upload_id,
            policy=policy,
            status="rejected",
            sha256=sha256,
            size_bytes=len(data),
            filename=filename,
            content_type=content_type,
            detected=detected,
            scan_result=scan_result,
            scan_started_at=scan_started_at,
            scan_finished_at=scan_finished_at,
            tenant_db=tenant_db,
            user_id=user_id,
            purpose_metadata=purpose_metadata,
            authorization_subject=authorization_subject,
            quarantine_path=quarantine_path,
            released_path=None,
            rejection_reason=str(exc.detail),
        )
        raise

    released_path = await storage.release_clean(
        quarantine_path=quarantine_path,
        tenant=tenant_db,
        policy_id=policy.policy_id,
        upload_id=upload_id,
        safe_filename=safe_filename(filename),
        content_type=detected.declared_mime_type,
        metadata={"upload_id": upload_id, "policy_id": policy.policy_id, "sha256": sha256, "verdict": "clean"},
    )
    await _persist_verdict(
        db=db,
        upload_id=upload_id,
        policy=policy,
        status="clean",
        sha256=sha256,
        size_bytes=len(data),
        filename=filename,
        content_type=content_type,
        detected=detected,
        scan_result=scan_result,
        scan_started_at=scan_started_at,
        scan_finished_at=scan_finished_at,
        tenant_db=tenant_db,
        user_id=user_id,
        purpose_metadata=purpose_metadata,
        authorization_subject=authorization_subject,
        quarantine_path=quarantine_path,
        released_path=released_path,
        rejection_reason=None,
    )
    _record_decision(policy.policy_id, "accepted")
    return CleanUpload(
        upload_id=upload_id,
        original_filename=filename or "",
        content_type=detected.declared_mime_type,
        size_bytes=len(data),
        sha256=sha256,
        detected_magic_type=detected.magic_type,
        released_storage_path=released_path,
        purpose_metadata=purpose_metadata,
        bytes=data if include_bytes else None,
    )


def _validate_metadata(purpose_metadata: dict[str, Any], authorization_subject: str) -> None:
    if not isinstance(purpose_metadata, dict) or not purpose_metadata.get("purpose"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload purpose metadata is required")
    if not authorization_subject:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload authorization subject is required")


def _tenant_db(actor: dict[str, Any] | None) -> str | None:
    if not actor:
        return None
    return actor.get("db_name") or actor.get("tenant_db") or actor.get("tenant_id")


def _user_id(actor: dict[str, Any] | None) -> str | None:
    if not actor:
        return None
    return actor.get("user_id") or actor.get("_id") or actor.get("id")


async def _persist_verdict(
    *,
    db: Any,
    upload_id: str,
    policy: UploadPolicy,
    status: str,
    sha256: str,
    size_bytes: int,
    filename: str | None,
    content_type: str | None,
    detected: DetectedFileType,
    scan_result: ScanResult,
    scan_started_at: datetime,
    scan_finished_at: datetime,
    tenant_db: str | None,
    user_id: str | None,
    purpose_metadata: dict[str, Any],
    authorization_subject: str,
    quarantine_path: str,
    released_path: str | None,
    rejection_reason: str | None,
) -> None:
    verdict = build_upload_verdict(
        upload_id=upload_id,
        policy_id=policy.policy_id,
        status=status,  # type: ignore[arg-type]
        sha256=sha256,
        size_bytes=size_bytes,
        original_filename=filename or "",
        declared_content_type=content_type or "",
        detected_magic_type=detected.magic_type,
        scanner_name=scan_result.scanner_name,
        scanner_version=scan_result.scanner_version,
        scan_started_at=scan_started_at,
        scan_finished_at=scan_finished_at,
        tenant_db=tenant_db,
        user_id=user_id,
        purpose_metadata=purpose_metadata,
        authorization_subject=authorization_subject,
        quarantine_storage_path=quarantine_path,
        released_storage_path=released_path,
        rejection_reason=rejection_reason,
    )
    await persist_upload_verdict(db, verdict)


def _record_decision(policy_id: str, outcome: str) -> None:
    try:
        record_upload_security_decision(policy_id, outcome)
    except Exception:
        pass
