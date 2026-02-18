"""
Assign unassigned tenants to a super-admin.

Usage:
    python assign_tenants.py --superadmin-email sa@example.com --all-unassigned
    python assign_tenants.py --superadmin-email sa@example.com --all-unassigned --include-all-statuses
"""

import argparse
import os
import sys

from dotenv import load_dotenv

# Allow importing tui.db from the scripts directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tui.db import DB

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign tenants to a super-admin")
    parser.add_argument("--superadmin-email", required=True, help="Super-admin email")
    parser.add_argument(
        "--all-unassigned",
        action="store_true",
        required=True,
        help="Assign all unassigned tenants",
    )
    parser.add_argument(
        "--include-all-statuses",
        action="store_true",
        help="Include tenants of any status (default: only active/approved/pending)",
    )
    args = parser.parse_args()

    db = DB()

    sa = db.get_superadmin_by_email(args.superadmin_email)
    if not sa:
        print(f"Super-admin not found: {args.superadmin_email}")
        db.close()
        sys.exit(1)

    print(f"Super-admin: {sa.get('name', '')} ({sa['email']})")
    print(f"Include all statuses: {args.include_all_statuses}")

    count = db.assign_all_tenants_to_superadmin(
        sa["_id"],
        include_all_statuses=args.include_all_statuses,
    )
    print(f"Assigned {count} tenant(s) to {sa['email']}.")

    db.close()


if __name__ == "__main__":
    main()
