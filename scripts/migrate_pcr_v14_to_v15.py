#!/usr/bin/env python3
"""Inspect or apply the guarded PCR v14 -> v15 pipeline migration.

The command is dry-run by default. Applying requires ``--apply`` and the
exact confirmation token; no provider/model call is made by this command.
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
    V14_TO_V15_CONFIRMATION_TOKEN,
    inspect_v14_contracts,
    migrate_v14_exam_to_v15,
)

CONFIRMATION_TOKEN = V14_TO_V15_CONFIRMATION_TOKEN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run or apply a guarded PCR v14 to v15 pipeline migration"
    )
    parser.add_argument("--tenant-db", help="Limit inspection to one tenant database")
    parser.add_argument("--exam-id", help="Limit inspection to one exam")
    parser.add_argument("--apply", action="store_true", help="Apply eligible migrations")
    parser.add_argument("--confirm", help=f"Required with --apply: {CONFIRMATION_TOKEN}")
    parser.add_argument(
        "--requested-by",
        default="operations:pcr-v14-v15-migration",
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
        print(json.dumps({
            "error": "confirmation_required",
            "detail": f"Pass --confirm {CONFIRMATION_TOKEN} with --apply",
        }))
        return 2

    db_manager = DatabaseManager()
    await db_manager.initialize()
    try:
        tenant_names = [args.tenant_db] if args.tenant_db else await _tenant_names(db_manager)
        plans: list[dict[str, Any]] = []
        for db_name in tenant_names:
            tenant_db = await db_manager.get_tenant_db(db_name)
            if tenant_db is None:
                raise RuntimeError(f"Tenant database is unavailable: {db_name}")
            plans.extend(await inspect_v14_contracts(tenant_db, db_name=db_name, exam_id=args.exam_id))

        if not args.apply:
            print(json.dumps({"mode": "dry-run", "plans": plans}, default=str, indent=2))
            return 0

        results: list[dict[str, Any]] = []
        for plan in plans:
            if not plan["eligible"]:
                results.append({**plan, "status": "skipped"})
                continue
            tenant_db = await db_manager.get_tenant_db(plan["db_name"])
            if tenant_db is None:
                raise RuntimeError(f"Tenant database is unavailable: {plan['db_name']}")
            try:
                results.append(await migrate_v14_exam_to_v15(
                    tenant_db,
                    db_name=plan["db_name"],
                    exam_id=plan["exam_id"],
                    requested_by=args.requested_by,
                    confirmation_token=args.confirm,
                ))
            except GradingContractMigrationError as exc:
                results.append({**plan, "status": "failed", "detail": str(exc)})
        print(json.dumps({"mode": "apply", "results": results}, default=str, indent=2))
        return 1 if any(row.get("status") == "failed" for row in results) else 0
    finally:
        await db_manager.close()


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
