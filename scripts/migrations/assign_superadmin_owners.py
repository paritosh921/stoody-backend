"""
Backfill tenant ownership for super-admin workflow.

Assignment priority:
1) tenants.authorization_code_used -> super_admins.authorization_code
2) optional manual mapping file (tenant_id/requested_tenant_id/admin_email -> superadmin email/auth code)
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Optional

from dotenv import load_dotenv
from pymongo import MongoClient


def normalize_code(value: str) -> str:
    return value.strip().upper()


def normalize_email(value: str) -> str:
    return value.strip().lower()


def load_mappings(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Mapping file must be a JSON object.")
    return {str(k): str(v) for k, v in data.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Assign super-admin owners to unassigned tenants.")
    parser.add_argument("--mongo-uri", default=None, help="MongoDB URI (defaults to MONGODB_URI env)")
    parser.add_argument("--master-db", default=None, help="Master DB name (defaults to MONGODB_DB_MASTER env)")
    parser.add_argument(
        "--mapping-file",
        default=None,
        help="JSON map: tenant_id/requested_tenant_id/admin_email -> superadmin email or auth code",
    )
    parser.add_argument(
        "--include-non-pending",
        action="store_true",
        help="Include all statuses (default only processes pending tenants).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing updates.")
    args = parser.parse_args()

    load_dotenv()
    mongo_uri = args.mongo_uri or os.getenv("MONGODB_URI")
    master_db_name = args.master_db or os.getenv("MONGODB_DB_MASTER", "skb_master")

    if not mongo_uri:
        print("ERROR: MONGODB_URI is not set.")
        return 1

    mappings = load_mappings(args.mapping_file)
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=30000, connectTimeoutMS=30000)
    try:
        master_db = client[master_db_name]
        super_admins = list(master_db["super_admins"].find({"is_active": True}))

        by_code = {}
        by_email = {}
        for admin in super_admins:
            code = admin.get("authorization_code")
            email = admin.get("email")
            if code:
                by_code[normalize_code(code)] = admin
            if email:
                by_email[normalize_email(email)] = admin

        query = {
            "$or": [
                {"assigned_superadmin_id": {"$exists": False}},
                {"assigned_superadmin_id": None},
            ]
        }
        if not args.include_non_pending:
            query["status"] = "pending"

        tenants = list(master_db["tenants"].find(query))
        print(f"Found {len(tenants)} unassigned tenant(s) to evaluate.")

        assigned_count = 0
        unresolved = []

        for tenant in tenants:
            tenant_id = tenant.get("tenant_id") or str(tenant.get("_id"))
            admin_email = tenant.get("admin_email") or ""
            chosen_admin = None
            chosen_code = None
            reason = None

            auth_code_used = tenant.get("authorization_code_used")
            if auth_code_used:
                code = normalize_code(auth_code_used)
                chosen_admin = by_code.get(code)
                if chosen_admin:
                    chosen_code = code
                    reason = "authorization_code_used"

            if not chosen_admin and mappings:
                lookup_keys = [tenant.get("tenant_id"), tenant.get("requested_tenant_id"), tenant.get("admin_email")]
                mapped_value = None
                for key in lookup_keys:
                    if key and key in mappings:
                        mapped_value = mappings[key]
                        break

                if mapped_value:
                    mapped_code = normalize_code(mapped_value)
                    mapped_email = normalize_email(mapped_value)
                    chosen_admin = by_code.get(mapped_code) or by_email.get(mapped_email)
                    if chosen_admin:
                        if mapped_code in by_code:
                            chosen_code = mapped_code
                        elif chosen_admin.get("authorization_code"):
                            chosen_code = normalize_code(chosen_admin["authorization_code"])
                        reason = "manual_mapping"

            if not chosen_admin:
                unresolved.append(
                    {
                        "tenant": tenant_id,
                        "admin_email": admin_email,
                        "authorization_code_used": tenant.get("authorization_code_used"),
                    }
                )
                continue

            update_payload = {
                "assigned_superadmin_id": chosen_admin["_id"],
                "updated_at": datetime.utcnow(),
            }
            if chosen_code and not tenant.get("authorization_code_used"):
                update_payload["authorization_code_used"] = chosen_code

            if not args.dry_run:
                master_db["tenants"].update_one(
                    {"_id": tenant["_id"]},
                    {"$set": update_payload},
                )

            assigned_count += 1
            print(
                f"ASSIGNED tenant={tenant_id} admin_email={admin_email} "
                f"super_admin={chosen_admin.get('email')} reason={reason}"
            )

        print(f"\nSummary: assigned={assigned_count}, unresolved={len(unresolved)}, dry_run={args.dry_run}")

        if unresolved:
            print("Unresolved tenants:")
            for item in unresolved:
                print(
                    f"- tenant={item['tenant']} admin_email={item['admin_email']} "
                    f"authorization_code_used={item['authorization_code_used']}"
                )
            print("Tip: provide --mapping-file for unresolved legacy entries.")

        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
