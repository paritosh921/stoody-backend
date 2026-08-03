#!/usr/bin/env python3
"""Inspect or migrate obsolete PCR v5 grading contracts to evidence-graph v6.

The command is read-only unless both ``--apply`` and the exact confirmation
token are supplied.  Stop the ExamPen worker and beat services before applying;
restart them after the command completes so queued cohort jobs run on v6.
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

from core.database import DatabaseManager
from services.pcr_grading_contract_migration import (
    GradingContractMigrationError,
    inspect_v5_contracts,
    migrate_v5_exam_to_v6,
)


CONFIRMATION_TOKEN = "MIGRATE_PCR_V5_TO_V6"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run or apply a cohort-safe PCR v5 to v6 migration"
    )
    parser.add_argument("--tenant-db", help="Limit inspection to one tenant database")
    parser.add_argument("--exam-id", help="Limit inspection to one exam")
    parser.add_argument("--apply", action="store_true", help="Apply eligible migrations")
    parser.add_argument("--confirm", help=f"Required with --apply: {CONFIRMATION_TOKEN}")
    parser.add_argument(
        "--requested-by", default="operations:pcr-v5-v6-migration", help="Audit actor"
    )
    return parser


async def _tenant_names() -> list[str]:
    master_db = DatabaseManager.get_master_db()
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

    await DatabaseManager.initialize()
    try:
        tenant_names = [args.tenant_db] if args.tenant_db else await _tenant_names()
        plans: list[dict[str, Any]] = []
        for db_name in tenant_names:
            tenant_db = DatabaseManager.get_tenant_db(db_name)
            plans.extend(
                await inspect_v5_contracts(
                    tenant_db, db_name=db_name, exam_id=args.exam_id
                )
            )

        if not args.apply:
            print(json.dumps({"mode": "dry-run", "plans": plans}, default=str, indent=2))
            return 0

        results: list[dict[str, Any]] = []
        for plan in plans:
            if not plan["eligible"]:
                results.append({**plan, "status": "skipped"})
                continue
            tenant_db = DatabaseManager.get_tenant_db(plan["db_name"])
            try:
                results.append(
                    await migrate_v5_exam_to_v6(
                        tenant_db,
                        db_name=plan["db_name"],
                        exam_id=plan["exam_id"],
                        requested_by=args.requested_by,
                    )
                )
            except GradingContractMigrationError as exc:
                results.append(
                    {
                        "db_name": plan["db_name"],
                        "exam_id": plan["exam_id"],
                        "status": "failed",
                        "detail": str(exc),
                    }
                )
        print(json.dumps({"mode": "apply", "results": results}, default=str, indent=2))
        return 1 if any(row.get("status") == "failed" for row in results) else 0
    finally:
        await DatabaseManager.close()


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
