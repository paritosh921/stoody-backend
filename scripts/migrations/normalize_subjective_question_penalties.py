#!/usr/bin/env python3
"""Normalize legacy Subjective PCR penalties to zero.

The historical extractor defaulted every Test Series question to ``penalty=1``.
This migration corrects editable PCR authoring records only; immutable paper
versions remain untouched. The command is a dry-run unless the exact apply
confirmation is supplied.

Examples:

    python scripts/migrations/normalize_subjective_question_penalties.py \
        --tenant-db skb_indl-ciel-1001 --document-id test23

    python scripts/migrations/normalize_subjective_question_penalties.py \
        --tenant-db skb_indl-ciel-1001 --document-id test23 \
        --apply --confirm NORMALIZE_SUBJECTIVE_PENALTIES
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.database import DatabaseManager


CONFIRMATION_TOKEN = "NORMALIZE_SUBJECTIVE_PENALTIES"
MIGRATION_VERSION = "objective-only-negative-marking-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-db", help="Limit the audit to one tenant database")
    parser.add_argument("--document-id", help="Limit the audit to one PCR document")
    parser.add_argument("--apply", action="store_true", help="Apply the audited updates")
    parser.add_argument("--confirm", help=f"Required with --apply: {CONFIRMATION_TOKEN}")
    return parser


async def _tenant_names(manager: DatabaseManager, requested: str | None) -> list[str]:
    if requested:
        return [requested]
    master = await manager.get_master_db()
    if master is None:
        raise RuntimeError("Master database is unavailable")
    rows = await master["tenants"].find(
        {"db_name": {"$type": "string"}},
        {"_id": 0, "db_name": 1},
    ).to_list(length=None)
    return sorted({str(row["db_name"]) for row in rows if row.get("db_name")})


async def run(args: argparse.Namespace) -> int:
    if args.apply and args.confirm != CONFIRMATION_TOKEN:
        print(json.dumps({"error": "confirmation_required", "token": CONFIRMATION_TOKEN}))
        return 2

    manager = DatabaseManager()
    await manager.initialize()
    results: list[dict[str, Any]] = []
    try:
        for db_name in await _tenant_names(manager, args.tenant_db):
            tenant_db = await manager.get_tenant_db(db_name)
            if tenant_db is None:
                raise RuntimeError(f"Tenant database is unavailable: {db_name}")
            document_query: dict[str, Any] = {
                "exam_mode": "pcr",
                "exam_finalized": {"$ne": True},
            }
            if args.document_id:
                document_query["document_id"] = args.document_id
            documents = await tenant_db["documents"].find(
                document_query,
                {"_id": 0, "document_id": 1, "question_type": 1},
            ).to_list(length=None)
            if args.document_id and not documents:
                protected_document = await tenant_db["documents"].find_one(
                    {"document_id": args.document_id, "exam_mode": "pcr"},
                    {"_id": 0, "document_id": 1, "exam_finalized": 1},
                )
                if protected_document and protected_document.get("exam_finalized") is True:
                    results.append(
                        {
                            "db_name": db_name,
                            "document_id": args.document_id,
                            "candidate_questions": 0,
                            "status": "skipped_finalized",
                            "reason": "Immutable finalized papers are normalized at the read and grading boundaries",
                        }
                    )
            for document in documents:
                document_id = str(document.get("document_id") or "")
                subjective_type_filters: list[dict[str, Any]] = [
                    {"question_type": "subjective"}
                ]
                if document.get("question_type") == "subjective":
                    subjective_type_filters.append(
                        {"question_type": {"$in": [None, ""]}}
                    )
                question_filter: dict[str, Any] = {
                    "document_id": document_id,
                    "penalty": {"$ne": 0},
                    "$or": subjective_type_filters,
                }
                candidates = await tenant_db["questions"].count_documents(question_filter)
                result = {
                    "db_name": db_name,
                    "document_id": document_id,
                    "candidate_questions": candidates,
                    "status": "dry_run",
                }
                if args.apply and candidates:
                    update = await tenant_db["questions"].update_many(
                        question_filter,
                        {
                            "$set": {
                                "penalty": 0.0,
                                "penalty_contract_version": MIGRATION_VERSION,
                                "penalty_normalized_at": datetime.now(timezone.utc),
                            }
                        },
                    )
                    result.update(
                        {
                            "status": "applied",
                            "matched_questions": update.matched_count,
                            "modified_questions": update.modified_count,
                        }
                    )
                results.append(result)
        print(json.dumps({"mode": "apply" if args.apply else "dry_run", "results": results}, default=str, indent=2))
        return 0
    finally:
        await manager.close()


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
