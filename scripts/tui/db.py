"""
Synchronous MongoDB helper for the Textual TUI.
All methods use pymongo (sync) and are called from Textual @work(thread=True) workers.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

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


class DB:
    """Sync MongoDB helper wrapping pymongo."""

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        master_db_name: Optional[str] = None,
    ):
        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI", "")
        self.master_db_name = master_db_name or os.getenv("MONGODB_DB_MASTER", "skb_master")
        self._client: Optional[MongoClient] = None

    @property
    def client(self) -> MongoClient:
        if self._client is None:
            self._client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
            )
        return self._client

    @property
    def master(self):
        return self.client[self.master_db_name]

    def close(self):
        if self._client:
            self._client.close()
            self._client = None

    # ---- Super-admins ----

    def list_superadmins(self) -> List[Dict[str, Any]]:
        rows = list(self.master["super_admins"].find({}))
        for r in rows:
            r["_id"] = str(r["_id"])
            # count tenants assigned
            r["tenant_count"] = self.master["tenants"].count_documents(
                {"assigned_superadmin_id": ObjectId(r["_id"])}
            )
        return rows

    def get_superadmin_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        doc = self.master["super_admins"].find_one({"email": email.strip().lower()})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    # ---- Pricing ----

    def get_pricing(self, superadmin_id: str) -> Dict[str, Any]:
        doc = self.master["superadmin_pricing"].find_one(
            {"superadmin_id": ObjectId(superadmin_id)}
        )
        if doc:
            result = {**DEFAULT_PRICING}
            for key in DEFAULT_PRICING:
                if key in doc and doc[key] is not None:
                    result[key] = doc[key]
            return result
        return {**DEFAULT_PRICING}

    def upsert_pricing(self, superadmin_id: str, fields: Dict[str, Any]) -> bool:
        now = datetime.utcnow()
        fields["updated_at"] = now
        if "currency" in fields and "currency_symbol" not in fields:
            fields["currency_symbol"] = CURRENCY_MAP.get(fields["currency"], fields["currency"])
        self.master["superadmin_pricing"].create_index(
            [("superadmin_id", 1)], unique=True, name="uniq_superadmin_pricing"
        )
        result = self.master["superadmin_pricing"].update_one(
            {"superadmin_id": ObjectId(superadmin_id)},
            {
                "$set": fields,
                "$setOnInsert": {"created_at": now, "superadmin_id": ObjectId(superadmin_id)},
            },
            upsert=True,
        )
        return result.upserted_id is not None or result.modified_count > 0

    # ---- Tenants ----

    def list_tenants_for_superadmin(self, superadmin_id: str) -> List[Dict[str, Any]]:
        tenants = list(
            self.master["tenants"].find(
                {"assigned_superadmin_id": ObjectId(superadmin_id)}
            )
        )
        for t in tenants:
            t["_id"] = str(t["_id"])
        return tenants

    def get_tenant_user_counts(self, db_name: str) -> Dict[str, int]:
        try:
            tdb = self.client[db_name]
            return {
                "students": tdb["students"].count_documents({}),
                "tutors": tdb["tutors"].count_documents({}),
                "admins": tdb["admins"].count_documents({}),
            }
        except Exception:
            return {"students": 0, "tutors": 0, "admins": 0}

    # ---- Cost computation ----

    def compute_costs_for_superadmin(self, superadmin_id: str) -> Dict[str, Any]:
        pricing = self.get_pricing(superadmin_id)
        tier_rates = pricing["tier_rates"]
        per_student = pricing["flat_per_student"]
        per_tutor = pricing["flat_per_tutor"]
        per_admin = pricing["flat_per_admin"]
        base_fee = pricing["superadmin_base_fee"]

        tenants = list(
            self.master["tenants"].find({
                "assigned_superadmin_id": ObjectId(superadmin_id),
                "status": {"$in": ["active", "approved"]},
            })
        )

        tenant_costs = []
        total_tenants_cost = 0.0

        for t in tenants:
            fv2 = t.get("enabled_features_v2") or {}
            tier = fv2.get("tier", "core")
            flat_fee = tier_rates.get(tier, tier_rates.get("core", 50.0))

            counts = {"students": 0, "tutors": 0, "admins": 0}
            db_name = t.get("db_name")
            if db_name:
                counts = self.get_tenant_user_counts(db_name)

            s_cost = counts["students"] * per_student
            t_cost = counts["tutors"] * per_tutor
            a_cost = counts["admins"] * per_admin
            total = flat_fee + s_cost + t_cost + a_cost
            total_tenants_cost += total

            tenant_costs.append({
                "tenant_id": str(t["_id"]),
                "institution_name": t.get("institution_name", ""),
                "tier": tier,
                "status": t.get("status", ""),
                "flat_fee": round(flat_fee, 2),
                "student_count": counts["students"],
                "tutor_count": counts["tutors"],
                "admin_count": counts["admins"],
                "student_surcharge": round(s_cost, 2),
                "tutor_surcharge": round(t_cost, 2),
                "admin_surcharge": round(a_cost, 2),
                "total_cost": round(total, 2),
            })

        return {
            "pricing": pricing,
            "tenant_costs": tenant_costs,
            "total_tenants_cost": round(total_tenants_cost, 2),
            "superadmin_base_fee": round(base_fee, 2),
            "total_platform_cost": round(total_tenants_cost + base_fee, 2),
        }

    # ---- Aggregate stats ----

    def get_aggregate_stats(self) -> Dict[str, Any]:
        sa_count = self.master["super_admins"].count_documents({})
        tenant_count = self.master["tenants"].count_documents({})
        active_count = self.master["tenants"].count_documents({"status": "active"})

        total_students = 0
        total_tutors = 0
        for t in self.master["tenants"].find({"status": {"$in": ["active", "approved"]}, "db_name": {"$exists": True}}):
            db_name = t.get("db_name")
            if db_name:
                counts = self.get_tenant_user_counts(db_name)
                total_students += counts["students"]
                total_tutors += counts["tutors"]

        return {
            "superadmin_count": sa_count,
            "tenant_count": tenant_count,
            "active_tenants": active_count,
            "total_students": total_students,
            "total_tutors": total_tutors,
        }
