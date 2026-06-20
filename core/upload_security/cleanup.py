"""Retention cleanup for private upload storage."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config_async import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UploadCleanupResult:
    dry_run: bool
    scanned_files: int
    candidate_files: int
    deleted_files: int
    reclaimed_bytes: int
    quarantine_prefix: str
    rejected_prefix: str

    def to_dict(self) -> dict[str, int | bool | str]:
        return asdict(self)


def cleanup_private_upload_storage(
    local_root: str | Path | None = None,
    *,
    now: datetime | None = None,
    rejected_retention_days: int | None = None,
    quarantine_retention_hours: int | None = None,
    dry_run: bool = True,
) -> UploadCleanupResult:
    root = Path(local_root or settings.UPLOAD_PRIVATE_LOCAL_DIR)
    current_time = _as_utc(now or datetime.now(timezone.utc))
    rejected_days = (
        settings.UPLOAD_REJECTED_RETENTION_DAYS
        if rejected_retention_days is None
        else rejected_retention_days
    )
    quarantine_hours = (
        settings.UPLOAD_QUARANTINE_RETENTION_HOURS
        if quarantine_retention_hours is None
        else quarantine_retention_hours
    )
    expired_roots = (
        (root / settings.UPLOAD_QUARANTINE_PREFIX, current_time - timedelta(hours=quarantine_hours)),
        (root / settings.UPLOAD_REJECTED_PREFIX, current_time - timedelta(days=rejected_days)),
    )

    scanned = 0
    candidates: list[Path] = []
    for base_path, cutoff in expired_roots:
        if not base_path.exists():
            continue
        for path in base_path.rglob("*"):
            if not path.is_file():
                continue
            scanned += 1
            if _mtime_utc(path) <= cutoff:
                candidates.append(path)

    reclaimed = 0
    deleted = 0
    for path in candidates:
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if dry_run:
            continue
        try:
            path.unlink()
            reclaimed += size
            deleted += 1
            _remove_empty_parents(path.parent, stop_at=root)
        except OSError as exc:
            logger.warning("upload_cleanup_delete_failed", extra={"path": str(path), "error": str(exc)})

    result = UploadCleanupResult(
        dry_run=dry_run,
        scanned_files=scanned,
        candidate_files=len(candidates),
        deleted_files=deleted,
        reclaimed_bytes=reclaimed,
        quarantine_prefix=settings.UPLOAD_QUARANTINE_PREFIX,
        rejected_prefix=settings.UPLOAD_REJECTED_PREFIX,
    )
    logger.info("upload_cleanup_completed", extra=result.to_dict())
    return result


def collect_upload_storage_usage(local_root: str | Path | None = None) -> dict[str, int]:
    root = Path(local_root or settings.UPLOAD_PRIVATE_LOCAL_DIR)
    usage: dict[str, int] = {}
    for prefix in (
        settings.UPLOAD_QUARANTINE_PREFIX,
        settings.UPLOAD_REJECTED_PREFIX,
        settings.UPLOAD_RELEASED_PREFIX,
        settings.UPLOAD_DERIVED_PREFIX,
    ):
        usage[prefix] = _directory_size(root / prefix)
    return usage


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            total += item.stat().st_size
        except OSError:
            continue
    return total


def _mtime_utc(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _remove_empty_parents(path: Path, *, stop_at: Path) -> None:
    stop = stop_at.resolve(strict=False)
    current = path
    while True:
        try:
            current_resolved = current.resolve(strict=False)
        except OSError:
            return
        if current_resolved == stop or stop not in current_resolved.parents:
            return
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent
