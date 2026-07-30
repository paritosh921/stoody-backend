"""Dry-run-first migration for an incorrectly versioned objective PCR exam."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import certifi
from motor.motor_asyncio import AsyncIOMotorClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_async import MONGODB_URL
from services.exampen_objective_contract_migration import (
    migrate_objective_contract,
    plan_objective_contract_migration,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely migrate or upgrade one unpublished objective PCR cohort "
            "to the current answer-ledger contract"
        )
    )
    parser.add_argument("--db-name", required=True)
    parser.add_argument("--exam-id", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Commit the migration. Default is read-only dry-run.",
    )
    parser.add_argument(
        "--confirm-exam-id",
        help="Required with --apply and must exactly equal --exam-id.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    if args.apply and args.confirm_exam_id != args.exam_id:
        raise SystemExit(
            "--apply requires --confirm-exam-id matching --exam-id exactly"
        )

    if not MONGODB_URL:
        raise RuntimeError("MONGODB_URI is not configured")
    client = AsyncIOMotorClient(
        MONGODB_URL,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=10000,
        tls=True,
        tlsCAFile=certifi.where(),
    )
    try:
        await client.admin.command("ping")
        tenant_db = client[args.db_name]
        if args.apply:
            result = await migrate_objective_contract(tenant_db, args.exam_id)
        else:
            plan = await plan_objective_contract_migration(
                tenant_db,
                args.exam_id,
            )
            result = {
                "dry_run": True,
                "exam_id": plan["exam_id"],
                "eligible": plan["eligible"],
                "already_migrated": plan["already_migrated"],
                "current_paper_version_id": plan["current_paper_version_id"],
                "submission_count": len(plan.get("submission_ids") or []),
                "migration_mode": plan.get("migration_mode"),
                "target_contract_version": str(
                    (plan.get("target_context") or {}).get("version") or ""
                ),
            }
        print(json.dumps(result, indent=2, default=str))
        return 0
    finally:
        client.close()


def main() -> int:
    return asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
