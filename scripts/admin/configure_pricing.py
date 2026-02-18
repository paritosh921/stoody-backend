"""
Get or set per-super-admin pricing configuration.

Usage:
  python configure_pricing.py --superadmin-email admin@example.com --action get
  python configure_pricing.py --superadmin-email admin@example.com --action set --currency INR --per-student 10
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
    "tier_rates": {"core": 50.0, "advanced": 120.0, "max": 250.0, "custom": 200.0},
    "flat_per_student": 0.50,
    "flat_per_tutor": 2.00,
    "flat_per_admin": 10.00,
    "superadmin_base_fee": 100.00,
    "notes": "",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure per-super-admin pricing in MongoDB.")
    parser.add_argument("--superadmin-email", required=True, help="Email of the super-admin")
    parser.add_argument("--action", required=True, choices=["get", "set"], help="Action to perform")
    parser.add_argument("--currency", default=None, choices=["USD", "EUR", "INR"], help="Currency code")
    parser.add_argument("--tier-core", type=float, default=None, help="Core tier monthly rate")
    parser.add_argument("--tier-advanced", type=float, default=None, help="Advanced tier monthly rate")
    parser.add_argument("--tier-max", type=float, default=None, help="Max tier monthly rate")
    parser.add_argument("--tier-custom", type=float, default=None, help="Custom tier monthly rate")
    parser.add_argument("--per-student", type=float, default=None, help="Per-student monthly surcharge")
    parser.add_argument("--per-tutor", type=float, default=None, help="Per-tutor monthly surcharge")
    parser.add_argument("--per-admin", type=float, default=None, help="Per-admin monthly surcharge")
    parser.add_argument("--base-fee", type=float, default=None, help="Super-admin base fee")
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

        # Build tier_rates from individual args
        existing = pricing_col.find_one({"superadmin_id": sa_id})
        existing_rates = (existing or {}).get("tier_rates", dict(DEFAULT_PRICING["tier_rates"]))
        tier_changed = False
        if args.tier_core is not None:
            existing_rates["core"] = args.tier_core
            tier_changed = True
        if args.tier_advanced is not None:
            existing_rates["advanced"] = args.tier_advanced
            tier_changed = True
        if args.tier_max is not None:
            existing_rates["max"] = args.tier_max
            tier_changed = True
        if args.tier_custom is not None:
            existing_rates["custom"] = args.tier_custom
            tier_changed = True
        if tier_changed:
            update_fields["tier_rates"] = existing_rates

        if args.per_student is not None:
            update_fields["flat_per_student"] = args.per_student
        if args.per_tutor is not None:
            update_fields["flat_per_tutor"] = args.per_tutor
        if args.per_admin is not None:
            update_fields["flat_per_admin"] = args.per_admin
        if args.base_fee is not None:
            update_fields["superadmin_base_fee"] = args.base_fee
        if args.notes is not None:
            update_fields["notes"] = args.notes

        if len(update_fields) <= 1:
            print("ERROR: No fields to update. Provide at least one --flag.")
            return 1

        result = pricing_col.update_one(
            {"superadmin_id": sa_id},
            {"$set": update_fields, "$setOnInsert": {"created_at": now, "superadmin_id": sa_id}},
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
