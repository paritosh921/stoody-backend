"""Dry-run audit helpers for legacy public/local uploads."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .detection import detect_magic_type
from .storage import safe_filename, safe_storage_segment


@dataclass(frozen=True)
class LegacyUploadRecord:
    original_path: str
    relative_path: str
    size_bytes: int
    sha256: str
    detected_magic_type: str
    likely_policy_id: str
    status: str
    issue: str | None = None


_MAGIC_POLICY_HINTS = {
    "pdf": "pdf_document",
    "png": "generic_image_upload",
    "jpeg": "generic_image_upload",
    "gif": "generic_image_upload",
    "bmp": "generic_image_upload",
    "webp": "generic_image_upload",
    "zip": "teaching_material",
    "ole": "teaching_material",
    "csv": "bulk_students",
}


def audit_local_uploads(root: str | Path) -> list[LegacyUploadRecord]:
    root_path = Path(root)
    if not root_path.exists():
        return []

    records: list[LegacyUploadRecord] = []
    for path in sorted(item for item in root_path.rglob("*") if item.is_file()):
        try:
            data = path.read_bytes()
            magic_type = detect_magic_type(data)
            sha256 = hashlib.sha256(data).hexdigest()
            issue = None
        except OSError as exc:
            data = b""
            magic_type = "unknown"
            sha256 = ""
            issue = str(exc)
        likely_policy = _MAGIC_POLICY_HINTS.get(magic_type, "unknown")
        records.append(
            LegacyUploadRecord(
                original_path=str(path),
                relative_path=str(path.relative_to(root_path)),
                size_bytes=len(data) if data else path.stat().st_size if path.exists() else 0,
                sha256=sha256,
                detected_magic_type=magic_type,
                likely_policy_id=likely_policy,
                status="legacy_unverified",
                issue=issue,
            )
        )
    return records


def write_legacy_audit_report(records: Iterable[LegacyUploadRecord], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return

    fieldnames = [
        "original_path",
        "relative_path",
        "size_bytes",
        "sha256",
        "detected_magic_type",
        "likely_policy_id",
        "status",
        "issue",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def migrate_verified_legacy_uploads(
    records: Iterable[LegacyUploadRecord],
    *,
    clean_sha256: set[str],
    destination_root: str | Path,
    tenant: str = "legacy",
    dry_run: bool = True,
) -> list[dict[str, str]]:
    destination_base = Path(destination_root)
    planned: list[dict[str, str]] = []
    for record in records:
        if record.sha256 not in clean_sha256:
            continue
        policy = safe_storage_segment(record.likely_policy_id, fallback="legacy")
        target = (
            destination_base
            / "clean"
            / safe_storage_segment(tenant)
            / policy
            / record.sha256
            / safe_filename(Path(record.relative_path).name)
        )
        planned.append(
            {
                "source": record.original_path,
                "destination": str(target),
                "sha256": record.sha256,
                "policy_id": record.likely_policy_id,
            }
        )
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record.original_path, target)
    return planned
