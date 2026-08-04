"""Migrate scanned document and answer-key PDFs from local release storage.

Run a dry audit first, then apply the exact same tenant selection:

    python scripts/migrations/migrate_authoring_uploads_to_private_s3.py --dry-run
    python scripts/migrations/migrate_authoring_uploads_to_private_s3.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient


BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-name", action="append", help="Limit to one or more tenant DBs")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _text(value: Any) -> str:
    return str(value or "").strip()


async def _tenant_names(client: AsyncIOMotorClient, requested: Iterable[str]) -> list[str]:
    names = [name for name in requested if name]
    if names:
        return sorted(set(names))
    return sorted(name for name in await client.list_database_names() if name.startswith("skb_"))


async def _run(args: argparse.Namespace) -> None:
    load_dotenv(BACKEND_ROOT / ".env")
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise RuntimeError("MONGODB_URI is required")

    from core.upload_security.durable_authoring import (
        complete_authoring_promotion,
        stage_released_authoring_pdf,
    )
    from core.upload_security.storage import PrivateUploadStorage

    client = AsyncIOMotorClient(mongo_uri)
    storage = PrivateUploadStorage()
    counts: Counter[str] = Counter()
    fields = (
        ("file_path", "sha256", "filename", "upload_id", "source_document"),
        (
            "answer_sheet_path",
            "answer_sheet_sha256",
            "answer_sheet_filename",
            "answer_sheet_upload_id",
            "teacher_solution",
        ),
    )
    try:
        for db_name in await _tenant_names(client, args.db_name or []):
            db = client[db_name]
            async for document in db["documents"].find({}):
                document_id = _text(document.get("document_id") or document.get("_id"))
                for path_field, sha_field, filename_field, upload_field, role in fields:
                    source_path = _text(document.get(path_field))
                    if not source_path or source_path.startswith("s3://"):
                        continue
                    label = f"{db_name}:{document_id}:{role}"
                    try:
                        data = await storage.read_released_path(source_path)
                        actual_sha256 = hashlib.sha256(data).hexdigest()
                        expected_sha256 = _text(document.get(sha_field)).lower()
                        if not expected_sha256 or actual_sha256 != expected_sha256:
                            raise ValueError("stored SHA-256 is missing or does not match")
                        if not args.apply:
                            counts["ready_to_migrate"] += 1
                            print(f"READY {label}")
                            continue

                        promotion = await stage_released_authoring_pdf(
                            released_path=source_path,
                            expected_sha256=expected_sha256,
                            filename=_text(document.get(filename_field)) or f"{role}.pdf",
                            content_type="application/pdf",
                            tenant_db=db_name,
                            document_id=document_id,
                            artifact_role=role,
                            upload_id=_text(document.get(upload_field)),
                            data=data,
                            storage=storage,
                        )
                        result = await db["documents"].update_one(
                            {"_id": document["_id"], path_field: source_path},
                            {
                                "$set": {
                                    path_field: promotion.storage_uri,
                                    "storage_backend": "s3",
                                    "authoring_storage_migrated_at": datetime.now(timezone.utc),
                                }
                            },
                        )
                        if result.matched_count != 1:
                            raise RuntimeError("document changed while migration was running")
                        await complete_authoring_promotion(db, promotion, storage=storage)
                        counts["migrated"] += 1
                        print(f"MIGRATED {label}")
                    except Exception as exc:
                        counts["failed"] += 1
                        print(f"FAILED {label}: {str(exc)[:300]}")
        print(json.dumps(dict(sorted(counts.items())), sort_keys=True))
    finally:
        client.close()


if __name__ == "__main__":
    parsed = _args()
    if parsed.apply == parsed.dry_run:
        raise SystemExit("Pass exactly one of --dry-run or --apply")
    asyncio.run(_run(parsed))
