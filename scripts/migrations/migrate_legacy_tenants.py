"""
Comprehensive legacy tenant migration for strict tenant model.

Combines three concerns into one idempotent script:
  A) Stamp admin_id on orphan documents across all 13 tenant-scoped collections
  B) Backfill enabled_features_v2 / subscription_tier / max_students / max_tutors
     on legacy tenants in skb_master
  C) Validate tenant integrity (required fields present)

Usage:
    # Dry-run all tenants
    python scripts/migrations/migrate_legacy_tenants.py --all --dry-run

    # Apply to all tenants
    python scripts/migrations/migrate_legacy_tenants.py --all

    # Single tenant
    python scripts/migrations/migrate_legacy_tenants.py --tenant-id CIEL-1001 --dry-run

    # Custom MongoDB URI
    python scripts/migrations/migrate_legacy_tenants.py --all --mongo-uri mongodb+srv://...
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# Must match core/tenant.py TENANT_SCOPED_COLLECTIONS exactly
TENANT_SCOPED_COLLECTIONS = [
    "students",
    "documents",
    "tutors",
    "questions",
    "question_attempts",
    "student_activity_log",
    "chat_sessions",
    "student_test_attempts",
    "assignments",
    "meetings",
    "notifications",
    "class_schedules",
    "smartboard_sessions",
]

# Required fields on every tenant document in skb_master.tenants
REQUIRED_TENANT_FIELDS = [
    "tenant_id",
    "institution_id",
    "db_name",
    "status",
    "assigned_superadmin_id",
]

# Default feature v2 structure for "core" tier
# Canonical tiers: core / advanced / max  (see core/tenant_features.py:14-21)
DEFAULT_SUBSCRIPTION_TIER = "core"
DEFAULT_MAX_STUDENTS = 100
DEFAULT_MAX_TUTORS = 10


def _build_default_features_v2() -> Dict[str, Any]:
    """Build a core-tier enabled_features_v2 document.

    This is a standalone version so the migration script doesn't import
    the running application code (which requires async runtime, config, etc.).
    Canonical tiers: core / advanced / max (see core/tenant_features.py).
    """
    # Core tier gets core features only
    core_features = [
        "admin_monitoring",
        "admin_student_db",
        "admin_tutor_db",
        "admin_content_management",
        "student_learning_mode",
        "student_exam_mode",
        "tutor_portal_access",
        "superadmin_inbox",
    ]
    advanced_features = [
        "admin_question_bank",
        "admin_leaderboard",
        "tutor_documents",
        "tutor_leaderboard",
        "tutor_online_class",
        "student_video_lessons",
        "student_leaderboard_view",
        "admin_registration_status_tools",
    ]
    max_features = [
        "student_ai_mentor",
        "tutor_analytics",
        "stoody_pen_capture",
        "smartboard_core_dummy",
        "smartboard_live_session_dummy",
        "smartboard_token_dummy",
    ]

    # Core tier: core=True, advanced=False, max=False
    effective = {}
    for key in core_features:
        effective[key] = True
    for key in advanced_features:
        effective[key] = False
    for key in max_features:
        effective[key] = False

    return {
        "version": 2,
        "tier": "core",
        "overrides": {},
        "effective": effective,
    }


# ---------------------------------------------------------------------------
# Part A: Stamp admin_id on orphan documents
# ---------------------------------------------------------------------------

def fix_orphan_admin_ids(
    tenant_db,
    admin_oid: ObjectId,
    db_name: str,
    dry_run: bool,
) -> Dict[str, int]:
    """Find and fix documents missing admin_id in a tenant database.

    Returns dict of collection_name -> count of fixed documents.
    """
    results: Dict[str, int] = {}

    orphan_filter = {
        "$or": [
            {"admin_id": {"$exists": False}},
            {"admin_id": None},
        ]
    }
    string_filter = {"admin_id": {"$type": "string"}}

    for coll_name in TENANT_SCOPED_COLLECTIONS:
        collection = tenant_db[coll_name]
        fixed = 0

        # Fix missing admin_id
        orphan_count = collection.count_documents(orphan_filter)
        if orphan_count > 0:
            if dry_run:
                print(f"  [{db_name}] {coll_name}: {orphan_count} orphans — WOULD FIX")
            else:
                result = collection.update_many(
                    orphan_filter,
                    {"$set": {"admin_id": admin_oid}},
                )
                fixed += result.modified_count
                print(f"  [{db_name}] {coll_name}: {result.modified_count} orphans fixed")

        # Fix string admin_id -> ObjectId
        string_count = collection.count_documents(string_filter)
        if string_count > 0:
            if dry_run:
                print(f"  [{db_name}] {coll_name}: {string_count} string admin_id — WOULD CONVERT")
            else:
                converted = 0
                for doc in collection.find(string_filter, {"admin_id": 1}):
                    try:
                        oid = ObjectId(doc["admin_id"])
                        collection.update_one(
                            {"_id": doc["_id"]},
                            {"$set": {"admin_id": oid}},
                        )
                        converted += 1
                    except Exception:
                        pass
                fixed += converted
                if converted:
                    print(f"  [{db_name}] {coll_name}: {converted} string admin_id converted to ObjectId")

        if orphan_count == 0 and string_count == 0:
            total = collection.count_documents({})
            if total > 0:
                print(f"  [{db_name}] {coll_name}: OK ({total} docs, all have admin_id)")

        results[coll_name] = fixed if not dry_run else orphan_count + string_count

    return results


def resolve_admin_for_tenant(tenant_db, db_name: str) -> Optional[ObjectId]:
    """Find the primary admin in a tenant database.

    Priority: master_admin role > any admin > None
    """
    # Try master_admin first
    admin = tenant_db["admins"].find_one({"role": "master_admin"})
    if admin:
        return admin["_id"]

    # Any admin
    admin = tenant_db["admins"].find_one({})
    if admin:
        print(f"  [{db_name}] WARNING: No master_admin found, using first admin: {admin.get('email')}")
        return admin["_id"]

    print(f"  [{db_name}] ERROR: No admin found in database — skipping orphan fix")
    return None


# ---------------------------------------------------------------------------
# Part B: Backfill tenant metadata in skb_master
# ---------------------------------------------------------------------------

def backfill_tenant_features(
    master_db,
    tenant_doc: Dict[str, Any],
    dry_run: bool,
) -> bool:
    """Backfill missing v2 feature fields on a tenant document.

    Returns True if update was needed.
    """
    tenant_id = tenant_doc.get("tenant_id") or str(tenant_doc["_id"])
    updates: Dict[str, Any] = {}

    if not tenant_doc.get("enabled_features_v2"):
        updates["enabled_features_v2"] = _build_default_features_v2()

    if not tenant_doc.get("subscription_tier"):
        updates["subscription_tier"] = DEFAULT_SUBSCRIPTION_TIER

    if not tenant_doc.get("max_students"):
        updates["max_students"] = DEFAULT_MAX_STUDENTS

    if not tenant_doc.get("max_tutors"):
        updates["max_tutors"] = DEFAULT_MAX_TUTORS

    if not updates:
        return False

    field_names = ", ".join(updates.keys())
    if dry_run:
        print(f"  [master] tenant {tenant_id}: WOULD backfill {field_names}")
    else:
        updates["updated_at"] = datetime.utcnow()
        master_db["tenants"].update_one(
            {"_id": tenant_doc["_id"]},
            {"$set": updates},
        )
        print(f"  [master] tenant {tenant_id}: backfilled {field_names}")

    return True


# ---------------------------------------------------------------------------
# Part C: Validate tenant integrity
# ---------------------------------------------------------------------------

def validate_tenant_integrity(
    tenant_doc: Dict[str, Any],
) -> List[str]:
    """Check that a tenant document has all required fields.

    Returns list of issues (empty if valid).
    """
    issues = []

    for field in REQUIRED_TENANT_FIELDS:
        value = tenant_doc.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            issues.append(f"missing '{field}'")

    status = tenant_doc.get("status", "")
    valid_statuses = {"pending", "verification", "approved", "active", "suspended", "rejected"}
    if status not in valid_statuses:
        issues.append(f"invalid status '{status}'")

    return issues


# Fields that can be missing on pending/verification tenants (assigned at approval)
DEFERRABLE_FIELDS_FOR_PENDING = {"tenant_id", "institution_id", "db_name"}


def quarantine_tenant(
    master_db,
    tenant_doc: Dict[str, Any],
    issues: List[str],
    dry_run: bool,
) -> bool:
    """Quarantine an active/approved tenant with integrity issues.

    Sets status to 'suspended' with a quarantine_reason so the super-admin
    can review and correct the record before re-activating.

    Pending/verification tenants are NOT quarantined — they are expected to
    have incomplete fields until approved.

    Returns True if the tenant was quarantined.
    """
    tenant_id = tenant_doc.get("tenant_id") or tenant_doc.get("requested_tenant_id") or str(tenant_doc["_id"])
    current_status = tenant_doc.get("status", "")

    # Pending/verification tenants naturally lack tenant_id/db_name — don't quarantine
    if current_status in ("pending", "verification"):
        non_deferrable = [i for i in issues if not any(f in i for f in DEFERRABLE_FIELDS_FOR_PENDING)]
        if not non_deferrable:
            print(f"  [{tenant_id}] Pending tenant — deferrable issues, not quarantined")
            return False
        # Only quarantine pending tenants if they have non-deferrable issues
        issues = non_deferrable

    # Already suspended/rejected — don't re-quarantine
    if current_status in ("suspended", "rejected"):
        print(f"  [{tenant_id}] Already {current_status} — issues noted but not re-quarantined")
        return False

    reason = f"Migration integrity check failed: {'; '.join(issues)}"

    if dry_run:
        print(f"  [{tenant_id}] WOULD QUARANTINE (suspend): {reason}")
    else:
        master_db["tenants"].update_one(
            {"_id": tenant_doc["_id"]},
            {
                "$set": {
                    "status": "suspended",
                    "quarantine_reason": reason,
                    "quarantined_at": datetime.utcnow(),
                    "pre_quarantine_status": current_status,
                    "updated_at": datetime.utcnow(),
                },
            },
        )
        print(f"  [{tenant_id}] QUARANTINED (was {current_status}): {reason}")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_tenant(
    client: MongoClient,
    master_db,
    tenant_doc: Dict[str, Any],
    dry_run: bool,
) -> Dict[str, Any]:
    """Process a single tenant: integrity check + quarantine, feature backfill, orphan fix."""
    tenant_id = tenant_doc.get("tenant_id") or tenant_doc.get("requested_tenant_id") or str(tenant_doc["_id"])
    db_name = tenant_doc.get("db_name")
    status = tenant_doc.get("status", "unknown")
    result = {
        "tenant_id": tenant_id,
        "status": status,
        "integrity_issues": [],
        "quarantined": False,
        "features_backfilled": False,
        "orphans_fixed": 0,
        "skipped": False,
    }

    print(f"\n--- Tenant: {tenant_id} (status={status}, db={db_name}) ---")

    # C) Integrity check — quarantine if critical issues on active/approved tenants
    issues = validate_tenant_integrity(tenant_doc)
    result["integrity_issues"] = issues
    if issues:
        print(f"  INTEGRITY: {', '.join(issues)}")
        quarantined = quarantine_tenant(master_db, tenant_doc, issues, dry_run)
        result["quarantined"] = quarantined
        if quarantined:
            # Don't proceed with migration on quarantined tenants — super-admin must fix first
            result["skipped"] = True
            return result

    # B) Feature backfill (always run on master, even for pending tenants)
    result["features_backfilled"] = backfill_tenant_features(master_db, tenant_doc, dry_run)

    # A) Orphan fix (only for tenants that have a database)
    if not db_name:
        print(f"  SKIP orphan fix: no db_name assigned yet")
        result["skipped"] = True
        return result

    tenant_db = client[db_name]
    admin_oid = resolve_admin_for_tenant(tenant_db, db_name)

    if admin_oid is None:
        result["skipped"] = True
        return result

    orphan_results = fix_orphan_admin_ids(tenant_db, admin_oid, db_name, dry_run)
    result["orphans_fixed"] = sum(orphan_results.values())

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Comprehensive legacy tenant migration for strict tenant model"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Process all tenants")
    group.add_argument("--tenant-id", help="Process a single tenant by tenant_id")
    parser.add_argument(
        "--mongo-uri",
        default=None,
        help="MongoDB URI (defaults to MONGODB_URI env)",
    )
    parser.add_argument(
        "--master-db",
        default=None,
        help="Master DB name (defaults to MONGODB_DB_MASTER env or skb_master)",
    )
    parser.add_argument(
        "--include-pending",
        action="store_true",
        help="Also process pending/unapproved tenants (default: only active/approved)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to database",
    )
    args = parser.parse_args()

    mongo_uri = args.mongo_uri or os.getenv("MONGODB_URI")
    master_db_name = args.master_db or os.getenv("MONGODB_DB_MASTER", "skb_master")

    if not mongo_uri:
        print("ERROR: MONGODB_URI is not set. Use --mongo-uri or set MONGODB_URI env.")
        return 1

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=30000, connectTimeoutMS=30000)

    try:
        master_db = client[master_db_name]

        # Verify connection
        client.admin.command("ping")
        print(f"Connected to MongoDB. Master DB: {master_db_name}")

        # Build tenant query
        if args.tenant_id:
            tenant_key = args.tenant_id.strip().upper()
            tenant_query = {
                "$or": [
                    {"tenant_id": tenant_key},
                    {"institution_id": tenant_key},
                    {"requested_tenant_id": tenant_key},
                ]
            }
        elif args.include_pending:
            tenant_query = {}
        else:
            tenant_query = {"status": {"$in": ["active", "approved"]}}

        tenants = list(master_db["tenants"].find(tenant_query))
        total = len(tenants)

        if total == 0:
            print("No tenants matched the query.")
            if not args.include_pending:
                print("Tip: use --include-pending to include non-active tenants.")
            return 0

        print(f"\nFound {total} tenant(s) to process.")
        if args.dry_run:
            print("MODE: DRY RUN (no changes will be made)\n")
        else:
            print("MODE: LIVE (changes will be written)\n")

        print("=" * 70)

        # Process each tenant
        results: List[Dict[str, Any]] = []
        for tenant_doc in tenants:
            result = process_tenant(client, master_db, tenant_doc, args.dry_run)
            results.append(result)

        # Summary
        print("\n" + "=" * 70)
        print("MIGRATION SUMMARY")
        print("=" * 70)

        integrity_failures = [r for r in results if r["integrity_issues"]]
        quarantined = [r for r in results if r.get("quarantined")]
        features_updated = [r for r in results if r["features_backfilled"]]
        orphans_fixed_total = sum(r["orphans_fixed"] for r in results)
        skipped = [r for r in results if r["skipped"]]

        print(f"  Tenants processed:     {total}")
        print(f"  Features backfilled:   {len(features_updated)}")
        print(f"  Orphan docs fixed:     {orphans_fixed_total}")
        print(f"  Quarantined:           {len(quarantined)}")
        print(f"  Skipped (no db/admin): {len(skipped)}")
        print(f"  Integrity issues:      {len(integrity_failures)}")
        print(f"  Dry run:               {args.dry_run}")

        if quarantined:
            print("\nQUARANTINED tenants (suspended — super-admin must review):")
            for r in quarantined:
                print(f"  - {r['tenant_id']} (was {r['status']}): {', '.join(r['integrity_issues'])}")

        if integrity_failures:
            non_quarantined_issues = [r for r in integrity_failures if not r.get("quarantined")]
            if non_quarantined_issues:
                print("\nTenants with integrity issues (not quarantined — pending/verification):")
                for r in non_quarantined_issues:
                    print(f"  - {r['tenant_id']} ({r['status']}): {', '.join(r['integrity_issues'])}")

        if skipped:
            non_quarantined_skipped = [r for r in skipped if not r.get("quarantined")]
            if non_quarantined_skipped:
                print("\nSkipped tenants (no db or admin):")
                for r in non_quarantined_skipped:
                    print(f"  - {r['tenant_id']} (status={r['status']})")

        print("\nDone.")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
