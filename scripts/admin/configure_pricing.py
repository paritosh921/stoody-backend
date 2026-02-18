"""
Get or set per-super-admin pricing configuration.

Usage:
  python configure_pricing.py --superadmin-email admin@example.com --action get
  python configure_pricing.py --superadmin-email admin@example.com --action set --core-student-monthly 1.0
  python configure_pricing.py --superadmin-email admin@example.com --action set --sa-fee-monthly 150 --billing-cycle annual
"""

import argparse
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from pymongo import MongoClient

CURRENCY_MAP = {"USD": "$", "EUR": "\u20ac", "INR": "\u20b9"}

DEFAULT_PRICING = {
    "currency": "USD",
    "currency_symbol": "$",
    "tiers": {
        "core": {"student_monthly": 0.50, "student_annual": 5.00,
                 "tutor_monthly": 2.00, "tutor_annual": 20.00,
                 "admin_monthly": 10.00, "admin_annual": 100.00},
        "advanced": {"student_monthly": 1.00, "student_annual": 10.00,
                     "tutor_monthly": 4.00, "tutor_annual": 40.00,
                     "admin_monthly": 15.00, "admin_annual": 150.00},
        "max": {"student_monthly": 2.00, "student_annual": 20.00,
                "tutor_monthly": 8.00, "tutor_annual": 80.00,
                "admin_monthly": 25.00, "admin_annual": 250.00},
    },
    "superadmin_fee": {"monthly": 100.00, "annual": 1000.00},
    "billing_cycle": "monthly",
    "billing_day": 1,
    "notes": "",
}

TIER_ROLES = ["student", "tutor", "admin"]
PERIODS = ["monthly", "annual"]
TIER_NAMES = ["core", "advanced", "max"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure per-super-admin pricing in MongoDB.")
    parser.add_argument("--superadmin-email", required=True, help="Email of the super-admin")
    parser.add_argument("--action", required=True, choices=["get", "set"], help="Action to perform")
    parser.add_argument("--currency", default=None, choices=["USD", "EUR", "INR"], help="Currency code")

    # Per-tier, per-role rate args
    for tier in TIER_NAMES:
        for role in TIER_ROLES:
            for period in PERIODS:
                parser.add_argument(
                    f"--{tier}-{role}-{period}",
                    type=float, default=None,
                    help=f"{tier.capitalize()} tier {role} {period} rate",
                )

    # Super-admin fee
    parser.add_argument("--sa-fee-monthly", type=float, default=None, help="Super-admin monthly fee")
    parser.add_argument("--sa-fee-annual", type=float, default=None, help="Super-admin annual fee")

    # Billing
    parser.add_argument("--billing-cycle", default=None, choices=["monthly", "annual"], help="Billing cycle")
    parser.add_argument("--billing-day", type=int, default=None, help="Billing day (1-28)")
    parser.add_argument("--notes", default=None, help="Optional notes")
    parser.add_argument("--mongo-uri", default=None, help="MongoDB URI (defaults to MONGODB_URI env)")
    parser.add_argument("--master-db", default=None, help="Master DB name (defaults to MONGODB_DB_MASTER env)")
    args = parser.parse_args()

    load_dotenv()
    mongo_uri = args.mongo_uri or os.getenv("MONGODB_URI")
    master_db_name = args.master_db or os.getenv("MONGODB_DB_MASTER", "skb_master")

    if not mongo_uri:
        print("ERROR: MONGODB_URI is not set.")
        return 1

    if args.billing_day is not None and not (1 <= args.billing_day <= 28):
        print("ERROR: --billing-day must be between 1 and 28.")
        return 1

    email = args.superadmin_email.strip().lower()

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=30000, connectTimeoutMS=30000)
    try:
        master_db = client[master_db_name]

        # Look up super-admin
        sa = master_db["super_admins"].find_one({"email": email})
        if not sa:
            print(f"ERROR: No super-admin found with email '{email}'.")
            return 1

        sa_id = sa["_id"]
        pricing_col = master_db["superadmin_pricing"]
        pricing_col.create_index([("superadmin_id", 1)], unique=True, name="uniq_superadmin_pricing")

        if args.action == "get":
            doc = pricing_col.find_one({"superadmin_id": sa_id})
            if doc:
                doc.pop("_id", None)
                doc["superadmin_id"] = str(doc["superadmin_id"])
                for key in ("created_at", "updated_at"):
                    if isinstance(doc.get(key), datetime):
                        doc[key] = doc[key].isoformat()
                print(json.dumps(doc, indent=2, default=str))
            else:
                print("No pricing document found. Defaults will be used:")
                print(json.dumps(DEFAULT_PRICING, indent=2))
            return 0

        # action == "set"
        now = datetime.utcnow()
        update_fields: dict = {"updated_at": now}

        if args.currency:
            update_fields["currency"] = args.currency
            update_fields["currency_symbol"] = CURRENCY_MAP.get(args.currency, args.currency)

        # Build tiers from individual args (deep merge with existing)
        existing = pricing_col.find_one({"superadmin_id": sa_id})
        existing_tiers = (existing or {}).get("tiers", dict(DEFAULT_PRICING["tiers"]))
        # Ensure each tier dict exists
        for t in TIER_NAMES:
            if t not in existing_tiers:
                existing_tiers[t] = dict(DEFAULT_PRICING["tiers"].get(t, {}))

        tiers_changed = False
        for tier in TIER_NAMES:
            for role in TIER_ROLES:
                for period in PERIODS:
                    arg_name = f"{tier}_{role}_{period}"
                    val = getattr(args, arg_name, None)
                    if val is not None:
                        existing_tiers[tier][f"{role}_{period}"] = val
                        tiers_changed = True
        if tiers_changed:
            update_fields["tiers"] = existing_tiers

        # Super-admin fee
        existing_fee = (existing or {}).get("superadmin_fee", dict(DEFAULT_PRICING["superadmin_fee"]))
        fee_changed = False
        if args.sa_fee_monthly is not None:
            existing_fee["monthly"] = args.sa_fee_monthly
            fee_changed = True
        if args.sa_fee_annual is not None:
            existing_fee["annual"] = args.sa_fee_annual
            fee_changed = True
        if fee_changed:
            update_fields["superadmin_fee"] = existing_fee

        if args.billing_cycle is not None:
            update_fields["billing_cycle"] = args.billing_cycle
        if args.billing_day is not None:
            update_fields["billing_day"] = args.billing_day
        if args.notes is not None:
            update_fields["notes"] = args.notes

        if len(update_fields) <= 1:
            print("ERROR: No fields to update. Provide at least one --flag.")
            return 1

        # Remove legacy flat-rate fields if present
        unset_fields = {}
        if existing:
            for old_key in ("tier_rates", "flat_per_student", "flat_per_tutor", "flat_per_admin", "superadmin_base_fee"):
                if old_key in existing:
                    unset_fields[old_key] = ""

        update_op = {
            "$set": update_fields,
            "$setOnInsert": {"created_at": now, "superadmin_id": sa_id},
        }
        if unset_fields:
            update_op["$unset"] = unset_fields

        result = pricing_col.update_one(
            {"superadmin_id": sa_id},
            update_op,
            upsert=True,
        )

        action = "created" if result.upserted_id else "updated"
        print(f"SUCCESS: Pricing {action} for {email}")

        # Show current state
        doc = pricing_col.find_one({"superadmin_id": sa_id})
        if doc:
            doc.pop("_id", None)
            doc["superadmin_id"] = str(doc["superadmin_id"])
            for key in ("created_at", "updated_at"):
                if isinstance(doc.get(key), datetime):
                    doc[key] = doc[key].isoformat()
            print(json.dumps(doc, indent=2, default=str))
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
