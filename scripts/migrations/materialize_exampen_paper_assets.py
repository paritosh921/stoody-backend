"""Pin a finalized PCR paper to durable content-addressed private storage.

Run this once for legacy paper versions that were finalized before immutable
paper assets were introduced. It verifies the source SHA-256 before upload and
never changes question or marking metadata.

Examples:
    python scripts/migrations/materialize_exampen_paper_assets.py \
      --db-name skb_indl-ciel-1001 --document-id phy019 --dry-run

    python scripts/migrations/materialize_exampen_paper_assets.py \
      --db-name skb_indl-ciel-1001 --document-id phy019 \
      --question-paper-source data/private_uploads/clean/.../paper.pdf --apply
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient


BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-name", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--paper-version-id")
    parser.add_argument("--question-paper-source")
    parser.add_argument("--teacher-solution-source")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _verify_source(path_text: str | None, expected_sha256: str | None, label: str) -> None:
    if not path_text or path_text.startswith("s3://"):
        print(f"{label}: source will be read from its stored URI")
        return
    path = Path(path_text)
    if not path.is_file():
        raise RuntimeError(f"{label}: source file does not exist: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and actual != expected_sha256:
        raise RuntimeError(f"{label}: SHA-256 does not match the document record")
    print(f"{label}: source hash verified ({actual})")


async def _run(args: argparse.Namespace) -> None:
    load_dotenv()
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise RuntimeError("MONGODB_URI is required")
    client = AsyncIOMotorClient(mongo_uri)
    try:
        db = client[args.db_name]
        document = await db["documents"].find_one({"document_id": args.document_id})
        if not document:
            raise RuntimeError("Document not found")
        version = await db["exampen_paper_versions"].find_one(
            {
                "document_id": args.document_id,
                **({"paper_version_id": args.paper_version_id} if args.paper_version_id else {}),
            }
        )
        if not version:
            raise RuntimeError("Finalized paper version not found")

        source_document = dict(document)
        if args.question_paper_source:
            source_document["file_path"] = args.question_paper_source
        if args.teacher_solution_source:
            source_document["answer_sheet_path"] = args.teacher_solution_source
        _verify_source(
            source_document.get("file_path"), source_document.get("sha256"), "question paper"
        )
        if source_document.get("answer_sheet_path"):
            _verify_source(
                source_document.get("answer_sheet_path"),
                source_document.get("answer_sheet_sha256"),
                "teacher solution",
            )
        if not args.apply:
            print("Dry run complete. Use --apply to upload and pin the immutable assets.")
            return
        if args.dry_run:
            raise RuntimeError("Choose either --apply or --dry-run, not both")

        from services.exampen_paper_service import migrate_legacy_paper_snapshot_assets

        migrated = await migrate_legacy_paper_snapshot_assets(
            db,
            source_document,
            paper_version_id=version["paper_version_id"],
        )
        assets = migrated.get("paper_assets") or {}
        await db["documents"].update_one(
            {"document_id": args.document_id},
            {
                "$set": {
                    "exam_paper_assets": assets,
                    "exam_paper_assets_migrated_at": migrated.get("paper_assets_migrated_at"),
                }
            },
        )
        print(
            "Materialized paper version "
            f"{migrated['paper_version_id']} with asset "
            f"{assets.get('question_paper', {}).get('asset_id')}"
        )
    finally:
        client.close()


if __name__ == "__main__":
    parsed = _args()
    if parsed.apply == parsed.dry_run:
        raise SystemExit("Pass exactly one of --dry-run or --apply")
    asyncio.run(_run(parsed))
