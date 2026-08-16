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
FLEET_CONFIRMATION_TOKEN = "MIGRATE_ELIGIBLE_PCR_LEGACY_FLEET_TO_V16"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run or apply a guarded legacy subjective PCR to v16 migration"
    )
    parser.add_argument("--tenant-db", help="Limit inspection to one tenant database")
    parser.add_argument("--exam-id", help="Limit inspection to one exam")
    apply_mode = parser.add_mutually_exclusive_group()
    apply_mode.add_argument(
        "--apply", action="store_true", help="Apply one explicitly targeted exam migration"
    )
    apply_mode.add_argument(
        "--apply-eligible",
        action="store_true",
        help="Discover and migrate a bounded batch of eligible exams across tenants",
    )
    parser.add_argument(
        "--confirm",
        help=f"Required with --apply: {CONFIRMATION_TOKEN}",
    )
    parser.add_argument(
        "--requested-by",
        default="operations:pcr-legacy-v16-migration",
        help="Audit actor",
    )
    parser.add_argument(
        "--max-exams",
        type=int,
        default=10,
        help="Maximum exams in one --apply-eligible batch (default: 10)",
    )
    parser.add_argument(
        "--max-submissions",
        type=int,
        default=100,
        help="Maximum submitted copies queued by one --apply-eligible batch (default: 100)",
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
    fleet_apply = bool(args.apply_eligible)
    apply_requested = bool(args.apply or fleet_apply)
    expected_confirmation = (
        FLEET_CONFIRMATION_TOKEN if fleet_apply else CONFIRMATION_TOKEN
    )
    if apply_requested and args.confirm != expected_confirmation:
        print(
            json.dumps(
                {
                    "error": "confirmation_required",
                    "detail": (
                        f"Pass --confirm {expected_confirmation} with the selected apply mode"
                    ),
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
    if fleet_apply and args.exam_id:
        print(
            json.dumps(
                {
                    "error": "fleet_target_invalid",
                    "detail": (
                        "--apply-eligible discovers exams automatically; use --tenant-db "
                        "to limit the fleet or --apply for one --exam-id"
                    ),
                }
            )
        )
        return 2
    if fleet_apply and (args.max_exams <= 0 or args.max_submissions <= 0):
        print(
            json.dumps(
                {
                    "error": "invalid_fleet_limit",
                    "detail": "--max-exams and --max-submissions must both be positive",
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
        scan_errors: list[dict[str, str]] = []
        for db_name in tenant_names:
            try:
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
            except Exception as exc:
                if args.tenant_db and not fleet_apply:
                    raise
                scan_errors.append(
                    {
                        "db_name": str(db_name),
                        "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                    }
                )

        if not apply_requested:
            print(
                json.dumps(
                    {"mode": "dry-run", "plans": plans, "scan_errors": scan_errors},
                    indent=2,
                    default=str,
                )
            )
            return 4 if scan_errors else 0

        if fleet_apply:
            eligible = sorted(
                (plan for plan in plans if plan.get("eligible")),
                key=lambda plan: (str(plan.get("db_name") or ""), str(plan.get("exam_id") or "")),
            )
            selected: list[dict[str, Any]] = []
            selected_submission_count = 0
            for plan in eligible:
                submission_count = int(plan.get("submission_count") or 0)
                if len(selected) >= args.max_exams:
                    continue
                if selected_submission_count + submission_count > args.max_submissions:
                    continue
                selected.append(plan)
                selected_submission_count += submission_count

            selected_keys = {
                (str(plan.get("db_name") or ""), str(plan.get("exam_id") or ""))
                for plan in selected
            }
            deferred = [
                plan
                for plan in eligible
                if (
                    str(plan.get("db_name") or ""),
                    str(plan.get("exam_id") or ""),
                )
                not in selected_keys
            ]
            results: list[dict[str, Any]] = []
            errors: list[dict[str, str]] = []
            for plan in selected:
                db_name = str(plan["db_name"])
                target_exam_id = str(plan["exam_id"])
                tenant_db = await db_manager.get_tenant_db(db_name)
                if tenant_db is None:
                    errors.append(
                        {
                            "db_name": db_name,
                            "exam_id": target_exam_id,
                            "error": "tenant database unavailable",
                        }
                    )
                    continue
                try:
                    results.append(
                        await migrate_legacy_exam_to_v16(
                            tenant_db,
                            db_name=db_name,
                            exam_id=target_exam_id,
                            requested_by=args.requested_by,
                            confirmation_token=CONFIRMATION_TOKEN,
                        )
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "db_name": db_name,
                            "exam_id": target_exam_id,
                            "error": f"{type(exc).__name__}: {str(exc)[:400]}",
                        }
                    )

            print(
                json.dumps(
                    {
                        "mode": "apply-eligible",
                        "eligible_exam_count": len(eligible),
                        "selected_exam_count": len(selected),
                        "selected_submission_count": selected_submission_count,
                        "deferred_exam_count": len(deferred),
                        "blocked_exam_count": sum(
                            1 for plan in plans if not plan.get("eligible")
                        ),
                        "scan_errors": scan_errors,
                        "results": results,
                        "errors": errors,
                    },
                    indent=2,
                    default=str,
                )
            )
            return 4 if errors or scan_errors else 0

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
                    "mode": "apply-eligible" if fleet_apply else "apply",
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
