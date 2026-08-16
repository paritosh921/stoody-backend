#!/usr/bin/env python3
"""Inspect or migrate one released legacy subjective PCR exam to v16.

Dry-run is the default.  Apply is deliberately limited to one explicit tenant
and exam, changes the complete submitted cohort, and queues durable pipeline-7
jobs.  It never calls the grading provider itself.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.database import DatabaseManager  # noqa: E402
from services.pcr_grading_contract_migration import (  # noqa: E402
    GradingContractMigrationError,
    LEGACY_TO_V16_CONFIRMATION_TOKEN,
    inspect_legacy_contracts,
    migrate_legacy_exam_to_v16,
)


CONFIRMATION_TOKEN = LEGACY_TO_V16_CONFIRMATION_TOKEN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run or apply a guarded legacy subjective PCR to v16 migration"
    )
    parser.add_argument("--tenant-db", help="Limit inspection to one tenant database")
    parser.add_argument("--exam-id", help="Limit inspection to one exam")
    parser.add_argument("--apply", action="store_true", help="Apply one eligible exam migration")
    parser.add_argument(
        "--confirm",
        help=f"Required with --apply: {CONFIRMATION_TOKEN}",
    )
    parser.add_argument(
        "--requested-by",
        default="operations:pcr-legacy-v16-migration",
        help="Audit actor",
    )
    return parser


async def _tenant_names(db_manager: DatabaseManager) -> list[str]:
    master_db = await db_manager.get_master_db()
    if master_db is None:
        raise RuntimeError("Master database is unavailable")
    rows = await master_db["tenants"].find(
        {"db_name": {"$type": "string"}}, {"_id": 0, "db_name": 1}
    ).to_list(length=None)
    return sorted({str(row["db_name"]) for row in rows if row.get("db_name")})


async def run(args: argparse.Namespace) -> int:
    if args.apply and args.confirm != CONFIRMATION_TOKEN:
        print(
            json.dumps(
                {
                    "error": "confirmation_required",
                    "detail": f"Pass --confirm {CONFIRMATION_TOKEN} with --apply",
                }
            )
        )
        return 2
    if args.apply and (not args.tenant_db or not args.exam_id):
        print(
            json.dumps(
                {
                    "error": "explicit_target_required",
                    "detail": "--apply requires both --tenant-db and --exam-id",
                }
            )
        )
        return 2

    db_manager = DatabaseManager()
    await db_manager.initialize()
    try:
        tenant_names = (
            [args.tenant_db] if args.tenant_db else await _tenant_names(db_manager)
        )
        plans: list[dict[str, Any]] = []
        for db_name in tenant_names:
            tenant_db = await db_manager.get_tenant_db(db_name)
            if tenant_db is None:
                raise RuntimeError(f"Tenant database is unavailable: {db_name}")
            plans.extend(
                await inspect_legacy_contracts(
                    tenant_db,
                    db_name=db_name,
                    exam_id=args.exam_id,
                )
            )

        if not args.apply:
            print(json.dumps({"mode": "dry-run", "plans": plans}, indent=2, default=str))
            return 0

        if len(plans) > 1:
            print(
                json.dumps(
                    {
                        "mode": "apply",
                        "error": "target_not_unique",
                        "detail": f"Expected at most one legacy exam plan, found {len(plans)}",
                        "plans": plans,
                    },
                    indent=2,
                    default=str,
                )
            )
            return 3
        if plans and not plans[0].get("eligible"):
            print(
                json.dumps(
                    {
                        "mode": "apply",
                        "error": "migration_blocked",
                        "plans": plans,
                    },
                    indent=2,
                    default=str,
                )
            )
            return 3

        # A resumable migration has already moved the exam contract to v16, so
        # it no longer appears in the legacy-only inspection query.  Calling
        # the guarded service directly for this one explicit target lets the
        # same command repair an applying/failed/complete-but-undrained run.
        # The service still rejects unrelated current-v16 and future contracts.

        tenant_db = await db_manager.get_tenant_db(args.tenant_db)
        if tenant_db is None:
            raise RuntimeError(f"Tenant database is unavailable: {args.tenant_db}")
        result = await migrate_legacy_exam_to_v16(
            tenant_db,
            db_name=args.tenant_db,
            exam_id=args.exam_id,
            requested_by=args.requested_by,
            confirmation_token=args.confirm,
        )
        print(json.dumps({"mode": "apply", "results": [result]}, indent=2, default=str))
        return 0
    except GradingContractMigrationError as exc:
        print(
            json.dumps(
                {
                    "mode": "apply" if args.apply else "dry-run",
                    "error": "migration_failed",
                    "detail": str(exc),
                },
                indent=2,
            )
        )
        return 3
    finally:
        await db_manager.close()


def main() -> int:
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
