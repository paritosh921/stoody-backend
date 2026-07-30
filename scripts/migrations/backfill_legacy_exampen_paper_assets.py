"""Backfill immutable assets for every safe legacy PCR paper snapshot.

This is the required data migration for the immutable-paper upgrade. It never
uses an arbitrary current authoring file: each source must match the SHA-256
recorded by the finalized paper version before it is copied into the durable
private asset store.

Run a dry audit first, then apply the exact same selection:

    python scripts/migrations/backfill_legacy_exampen_paper_assets.py --dry-run
    python scripts/migrations/backfill_legacy_exampen_paper_assets.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient


BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

_CANONICAL_CONTEXT_VERSIONS = {
    "canonical-full-document-visual-v1",
    "canonical-full-document-visual-v2",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-name", action="append", help="Limit to one or more tenant DBs")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _text(value: Any) -> str:
    return str(value or "").strip()


def _local_upload_index(root: Path) -> Dict[str, str]:
    """Index released upload bytes by the scanner's immutable SHA-256 sidecar."""

    indexed: Dict[str, str] = {}
    if not root.is_dir():
        return indexed
    for metadata_path in root.rglob("*.metadata.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            digest = _text((payload.get("metadata") or {}).get("sha256")).lower()
        except (OSError, ValueError, TypeError):
            continue
        candidate = Path(str(metadata_path)[: -len(".metadata.json")])
        if digest and candidate.is_file():
            indexed.setdefault(digest, str(candidate))
    return indexed


def _source_or_recovered_copy(
    source: Any,
    digest: Any,
    local_uploads: Dict[str, str],
) -> Optional[str]:
    """Return a reachable original source, never a filename-based substitute."""

    source_text = _text(source)
    if source_text.startswith("s3://"):
        return source_text
    if source_text and Path(source_text).is_file():
        return source_text
    return local_uploads.get(_text(digest).lower())


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

    from config_async import settings
    from services.exampen_paper_service import migrate_legacy_paper_snapshot_assets

    client = AsyncIOMotorClient(mongo_uri)
    counts: Counter[str] = Counter()
    try:
        local_uploads = await asyncio.to_thread(
            _local_upload_index, Path(settings.UPLOAD_PRIVATE_LOCAL_DIR)
        )
        for db_name in await _tenant_names(client, args.db_name or []):
            db = client[db_name]
            cursor = db["exampen_paper_versions"].find(
                {
                    "paper_context.version": {"$in": sorted(_CANONICAL_CONTEXT_VERSIONS)},
                    "$or": [
                        {"paper_assets.question_paper": {"$exists": False}},
                        {"paper_assets.question_paper": None},
                    ],
                }
            )
            async for version in cursor:
                counts["candidates"] += 1
                document = await db["documents"].find_one(
                    {"document_id": version.get("document_id")}
                )
                label = f"{db_name}:{version.get('paper_version_id')}"
                if not document:
                    counts["unrecoverable"] += 1
                    print(f"UNRECOVERABLE {label}: source document is missing")
                    continue

                source_document = dict(document)
                source_document["file_path"] = _source_or_recovered_copy(
                    document.get("file_path"), document.get("sha256"), local_uploads
                )
                if _text(document.get("answer_sheet_path")):
                    source_document["answer_sheet_path"] = _source_or_recovered_copy(
                        document.get("answer_sheet_path"),
                        document.get("answer_sheet_sha256"),
                        local_uploads,
                    )
                if not _text(source_document.get("file_path")):
                    counts["unrecoverable"] += 1
                    print(f"UNRECOVERABLE {label}: question-paper bytes are unavailable")
                    continue
                if _text(document.get("answer_sheet_path")) and not _text(
                    source_document.get("answer_sheet_path")
                ):
                    counts["unrecoverable"] += 1
                    print(f"UNRECOVERABLE {label}: teacher-solution bytes are unavailable")
                    continue

                if not args.apply:
                    counts["ready_to_migrate"] += 1
                    print(f"READY {label}")
                    continue
                try:
                    migrated = await migrate_legacy_paper_snapshot_assets(
                        db,
                        source_document,
                        paper_version_id=_text(version.get("paper_version_id")),
                    )
                    assets = dict(migrated.get("paper_assets") or {})
                    await db["documents"].update_one(
                        {"document_id": document["document_id"]},
                        {
                            "$set": {
                                "exam_paper_assets": assets,
                                "exam_paper_assets_migrated_at": migrated.get(
                                    "paper_assets_migrated_at"
                                ),
                            }
                        },
                    )
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
