"""
Manage super-admin lifecycle: suspend, activate, deactivate, delete, status, migrate.

Usage:
    python manage_superadmin.py --email sa@example.com --action status
    python manage_superadmin.py --email sa@example.com --action suspend --reason "Non-payment"
    python manage_superadmin.py --email sa@example.com --action activate
    python manage_superadmin.py --email sa@example.com --action deactivate --reason "Contract ended"
    python manage_superadmin.py --email sa@example.com --action delete --force
    python manage_superadmin.py --action migrate   # one-time: backfill status field on all SAs
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# Allow importing tui.db from the scripts directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tui.db import DB

load_dotenv()


def show_status(db: DB, email: str) -> None:
    sa = db.get_superadmin_by_email(email)
    if not sa:
        print(f"Super-admin not found: {email}")
        sys.exit(1)
    status = db._derive_sa_status(sa)
    tenants = db.list_tenants_for_superadmin(sa["_id"])
    print(f"Email:    {sa['email']}")
    print(f"Name:     {sa.get('name', '')}")
    print(f"Status:   {status}")
    print(f"Tenants:  {len(tenants)}")
    if sa.get("suspended_at"):
        print(f"Suspended at:     {sa['suspended_at']}")
        print(f"Suspended reason: {sa.get('suspended_reason', '')}")
    if sa.get("deactivated_at"):
        print(f"Deactivated at:     {sa['deactivated_at']}")
        print(f"Deactivated reason: {sa.get('deactivated_reason', '')}")


def confirm_action(action: str, email: str, force: bool) -> bool:
    if force:
        return True
    answer = input(f"Are you sure you want to {action} super-admin {email}? [y/N] ")
    return answer.strip().lower() == "y"


def run_migrate(db: DB) -> None:
    """Backfill status field on all existing SA docs based on is_active."""
    coll = db.master["super_admins"]
    docs = list(coll.find({}))
    updated = 0
    for doc in docs:
        if "status" in doc:
            continue
        is_active = doc.get("is_active", True)
        new_status = "active" if is_active else "deactivated"
        update: dict = {"$set": {"status": new_status}}
        # Ensure is_active field exists for backward compat
        if "is_active" not in doc:
            update["$set"]["is_active"] = True
        coll.update_one({"_id": doc["_id"]}, update)
        updated += 1
        print(f"  {doc.get('email', '?')}: is_active={is_active} -> status={new_status}")
    print(f"Migration complete. Updated {updated}/{len(docs)} super-admin(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage super-admin lifecycle")
    parser.add_argument("--email", help="Super-admin email (required except for --action migrate)")
    parser.add_argument(
        "--action",
        required=True,
        choices=["suspend", "activate", "deactivate", "delete", "status", "migrate"],
        help="Action to perform",
    )
    parser.add_argument("--reason", default="", help="Reason for suspend/deactivate")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    db = DB()

    if args.action == "migrate":
        run_migrate(db)
        db.close()
        return

    if not args.email:
        print("Error: --email is required for this action.")
        sys.exit(1)

    if args.action == "status":
        show_status(db, args.email)
        db.close()
        return

    sa = db.get_superadmin_by_email(args.email)
    if not sa:
        print(f"Super-admin not found: {args.email}")
        db.close()
        sys.exit(1)

    if not confirm_action(args.action, args.email, args.force):
        print("Aborted.")
        db.close()
        return

    if args.action == "suspend":
        result = db.suspend_superadmin(sa["_id"], args.reason)
        print(f"Suspended. {result['tenants_affected']} tenant(s) platform-suspended.")
    elif args.action == "activate":
        result = db.activate_superadmin(sa["_id"])
        print(f"Activated. {result['tenants_affected']} tenant(s) platform access restored.")
    elif args.action == "deactivate":
        result = db.deactivate_superadmin(sa["_id"], args.reason)
        print(f"Deactivated. {result['tenants_affected']} tenant(s) platform-suspended.")
    elif args.action == "delete":
        result = db.delete_superadmin(sa["_id"])
        print(f"Deleted. {result['tenants_orphaned']} tenant(s) orphaned.")

    db.close()


if __name__ == "__main__":
    main()
