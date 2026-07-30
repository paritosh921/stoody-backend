"""Migrate one immutable PCR paper and solution to private S3 safely.

The migration is intentionally narrow and defaults to read-only verification.
It verifies each local asset against the SHA-256 frozen in the document,
uploads under a tenant-scoped private key, and changes MongoDB paths with a
compare-and-set update.  Question content, answer mappings, paper versions,
and frozen hashes are never modified.

Examples:
    py -3 scripts/migrate_pcr_canonical_assets.py \
        --db-name skb_sgtb-0001 --document-id UT1

    py -3 scripts/migrate_pcr_canonical_assets.py \
        --db-name skb_sgtb-0001 --document-id UT1 --apply
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_async import settings  # noqa: E402
from services.canonical_asset_storage import (  # noqa: E402
    CanonicalAssetStorageError,
    read_canonical_asset,
    store_canonical_asset,
)
from utils.s3_storage import is_s3_enabled  # noqa: E402


SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]+$")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify or migrate one PCR document's immutable assets",
    )
    parser.add_argument("--db-name", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Upload verified bytes and atomically update document storage paths",
    )
    return parser.parse_args()


def _validate_identifier(value: str, label: str) -> str:
    cleaned = str(value or "").strip()
    if not SAFE_IDENTIFIER.fullmatch(cleaned):
        raise SystemExit(f"{label} contains unsupported characters")
    return cleaned


async def _verified_asset(
    document: dict[str, Any],
    *,
    path_field: str,
    hash_field: str,
    required: bool,
) -> tuple[str, str, bytes] | None:
    storage_path = str(document.get(path_field) or "").strip()
    expected_hash = str(document.get(hash_field) or "").strip().lower()
    if not storage_path:
        if required:
            raise CanonicalAssetStorageError(f"{path_field} is missing")
        return None
    if len(expected_hash) != 64:
        raise CanonicalAssetStorageError(f"{hash_field} is missing or invalid")
    payload = await read_canonical_asset(storage_path)
    if not payload:
        raise CanonicalAssetStorageError(
            f"{path_field} could not be read from approved local storage"
        )
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_hash:
        raise CanonicalAssetStorageError(
            f"{path_field} does not match its immutable SHA-256"
        )
    return storage_path, expected_hash, payload


async def migrate(
    *,
    db_name: str,
    document_id: str,
    apply: bool,
) -> dict[str, Any]:
    if apply and not is_s3_enabled():
        raise CanonicalAssetStorageError("Private S3 storage is not available")

    client = AsyncIOMotorClient(settings.MONGODB_URL, serverSelectionTimeoutMS=10000)
    try:
        database = client[db_name]
        document = await database["documents"].find_one(
            {"document_id": document_id},
            projection={
                "document_id": 1,
                "exam_mode": 1,
                "file_path": 1,
                "sha256": 1,
                "filename": 1,
                "upload_id": 1,
                "answer_sheet_path": 1,
                "answer_sheet_sha256": 1,
                "answer_sheet_filename": 1,
                "answer_sheet_upload_id": 1,
            },
        )
        if document is None:
            raise CanonicalAssetStorageError("Document was not found")
        if str(document.get("exam_mode") or "").strip().lower() != "pcr":
            raise CanonicalAssetStorageError("Document is not a PCR paper")

        paper = await _verified_asset(
            document,
            path_field="file_path",
            hash_field="sha256",
            required=True,
        )
        solution = await _verified_asset(
            document,
            path_field="answer_sheet_path",
            hash_field="answer_sheet_sha256",
            required=False,
        )
        assert paper is not None

        report: dict[str, Any] = {
            "db_name": db_name,
            "document_id": document_id,
            "mode": "apply" if apply else "verify_only",
            "paper": {
                "source": paper[0],
                "sha256": paper[1],
                "verified": bool(paper[2]),
                "already_s3": paper[0].startswith("s3://"),
            },
            "solution": None,
            "database_updated": False,
        }
        if solution is not None:
            report["solution"] = {
                "source": solution[0],
                "sha256": solution[1],
                "verified": bool(solution[2]),
                "already_s3": solution[0].startswith("s3://"),
            }
        if not apply:
            return report

        new_paths: dict[str, str] = {}
        if not paper[0].startswith("s3://"):
            paper_transfer = await store_canonical_asset(
                data=paper[2],
                local_path=paper[0],
                upload_id=str(document.get("upload_id") or "legacy-paper"),
                tenant_db=db_name,
                document_id=document_id,
                artifact_kind="question-paper",
                filename=str(document.get("filename") or Path(paper[0]).name),
                content_type="application/pdf",
                sha256=paper[1],
            )
            new_paths["file_path"] = paper_transfer.storage_path
            report["paper"]["target"] = paper_transfer.storage_path
        if solution is not None and not solution[0].startswith("s3://"):
            solution_transfer = await store_canonical_asset(
                data=solution[2],
                local_path=solution[0],
                upload_id=str(
                    document.get("answer_sheet_upload_id") or "legacy-solution"
                ),
                tenant_db=db_name,
                document_id=document_id,
                artifact_kind="teacher-solution",
                filename=str(
                    document.get("answer_sheet_filename")
                    or Path(solution[0]).name
                ),
                content_type="application/pdf",
                sha256=solution[1],
            )
            new_paths["answer_sheet_path"] = solution_transfer.storage_path
            report["solution"]["target"] = solution_transfer.storage_path

        if not new_paths:
            report["database_updated"] = True
            return report

        compare_filter: dict[str, Any] = {
            "_id": document["_id"],
            "file_path": document.get("file_path"),
            "sha256": document.get("sha256"),
            "answer_sheet_path": document.get("answer_sheet_path"),
            "answer_sheet_sha256": document.get("answer_sheet_sha256"),
        }
        now = datetime.now(timezone.utc)
        update_fields: dict[str, Any] = {
            **new_paths,
            "is_s3": str(new_paths.get("file_path") or paper[0]).startswith(
                "s3://"
            ),
            "canonical_asset_storage_backend": "s3",
            "canonical_asset_migration": {
                "version": 1,
                "migrated_at": now,
                "previous_file_path": document.get("file_path"),
                "previous_answer_sheet_path": document.get("answer_sheet_path"),
                "paper_sha256": document.get("sha256"),
                "answer_sheet_sha256": document.get("answer_sheet_sha256"),
            },
            "updated_at": now,
        }
        result = await database["documents"].update_one(
            compare_filter,
            {"$set": update_fields},
        )
        if result.matched_count != 1:
            raise CanonicalAssetStorageError(
                "Document changed during migration; database paths were not updated"
            )
        report["database_updated"] = True
        return report
    finally:
        client.close()


async def _main() -> None:
    args = _args()
    report = await migrate(
        db_name=_validate_identifier(args.db_name, "Database name"),
        document_id=_validate_identifier(args.document_id, "Document id"),
        apply=bool(args.apply),
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(_main())
