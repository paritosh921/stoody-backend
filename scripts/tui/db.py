"""
Synchronous MongoDB helper for the Textual TUI.
All methods use pymongo (sync) and are called from Textual @work(thread=True) workers.
"""

import calendar
import io
import json
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from scripts.admin.create_superadmin import (
    ensure_indexes,
    generate_temp_password,
    normalize_email,
    resolve_unique_auth_code,
)

load_dotenv()

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


# ---- Module-level helpers ----

def _convert_legacy_pricing(doc):
    """Convert a legacy pricing document (with tier_rates) to the new tiered format."""
    per_student = doc.get("flat_per_student", 0.50)
    per_tutor = doc.get("flat_per_tutor", 2.00)
    per_admin = doc.get("flat_per_admin", 10.00)
    base_fee = doc.get("superadmin_base_fee", 100.00)
    return {
        "currency": doc.get("currency", "USD"),
        "currency_symbol": doc.get("currency_symbol", "$"),
        "tiers": {
            "core": {"student_monthly": per_student, "student_annual": per_student * 10,
                     "tutor_monthly": per_tutor, "tutor_annual": per_tutor * 10,
                     "admin_monthly": per_admin, "admin_annual": per_admin * 10},
            "advanced": {"student_monthly": per_student * 2, "student_annual": per_student * 20,
                         "tutor_monthly": per_tutor * 2, "tutor_annual": per_tutor * 20,
                         "admin_monthly": per_admin * 1.5, "admin_annual": per_admin * 15},
            "max": {"student_monthly": per_student * 4, "student_annual": per_student * 40,
                    "tutor_monthly": per_tutor * 4, "tutor_annual": per_tutor * 40,
                    "admin_monthly": per_admin * 2.5, "admin_annual": per_admin * 25},
        },
        "superadmin_fee": {"monthly": base_fee, "annual": base_fee * 10},
        "billing_cycle": "monthly",
        "billing_day": 1,
        "notes": doc.get("notes", ""),
    }


def _compute_billing_period(now, cycle, billing_day):
    """Return (period_start, period_end, next_due) datetimes for the current billing period."""
    if cycle == "annual":
        year = now.year if now.month > 1 or now.day >= billing_day else now.year - 1
        period_start = datetime(year, 1, min(billing_day, 28))
        period_end = datetime(year, 12, 31, 23, 59, 59)
        next_due = datetime(year + 1, 1, min(billing_day, 28))
    else:
        day = min(billing_day, calendar.monthrange(now.year, now.month)[1])
        if now.day >= day:
            period_start = datetime(now.year, now.month, day)
            if now.month == 12:
                next_month_year, next_month = now.year + 1, 1
            else:
                next_month_year, next_month = now.year, now.month + 1
            next_day = min(billing_day, calendar.monthrange(next_month_year, next_month)[1])
            period_end = datetime(next_month_year, next_month, next_day) - timedelta(seconds=1)
            next_due = datetime(next_month_year, next_month, next_day)
        else:
            if now.month == 1:
                prev_year, prev_month = now.year - 1, 12
            else:
                prev_year, prev_month = now.year, now.month - 1
            prev_day = min(billing_day, calendar.monthrange(prev_year, prev_month)[1])
            period_start = datetime(prev_year, prev_month, prev_day)
            period_end = datetime(now.year, now.month, day) - timedelta(seconds=1)
            next_due = datetime(now.year, now.month, day)
    return period_start, period_end, next_due


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
        self._s3_client = None
        self._backend_root = Path(__file__).resolve().parents[2]

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
        self._s3_client = None

    # ---- Super-admins ----

    @staticmethod
    def _derive_sa_status(doc: Dict[str, Any]) -> str:
        """Derive status from doc, falling back to is_active for legacy docs."""
        if "status" in doc:
            return doc["status"]
        return "active" if doc.get("is_active", True) else "deactivated"

    def list_superadmins(self) -> List[Dict[str, Any]]:
        rows = list(self.master["super_admins"].find({}))
        for r in rows:
            r["_id"] = str(r["_id"])
            r["status"] = self._derive_sa_status(r)
            r["password_reset_requested"] = bool(r.get("password_reset_requested", False))
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

    def get_superadmin_by_id(self, sa_id: str) -> Optional[Dict[str, Any]]:
        doc = self.master["super_admins"].find_one({"_id": ObjectId(sa_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
            doc["status"] = self._derive_sa_status(doc)
        return doc

    def create_superadmin(
        self, email: str, name: str, authorization_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new super-admin. Reuses logic from scripts.admin.create_superadmin."""
        from passlib.context import CryptContext

        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        col = self.master["super_admins"]
        ensure_indexes(col)

        email = normalize_email(email)
        if "@" not in email:
            raise ValueError("Invalid email format.")

        if col.find_one({"email": email}):
            raise ValueError(f"Super-admin with email '{email}' already exists.")

        auth_code = resolve_unique_auth_code(col, authorization_code)
        temp_password = generate_temp_password()
        now = datetime.utcnow()

        insert_doc = {
            "email": email,
            "name": name.strip(),
            "password_hash": pwd_context.hash(temp_password),
            "temp_password": temp_password,
            "role": "super_admin",
            "permissions": ["all"],
            "is_active": True,
            "status": "active",
            "authorization_code": auth_code,
            "requires_password_change": True,
            "password_reset_requested": False,
            "password_changed_at": None,
            "created_at": now,
            "updated_at": now,
            "two_fa": {
                "enabled": False,
                "required": True,
                "secret_enc": None,
                "temp_secret_enc": None,
                "verified_at": None,
                "last_verified_at": None,
            },
        }

        result = col.insert_one(insert_doc)
        return {
            "admin_id": str(result.inserted_id),
            "email": email,
            "name": name.strip(),
            "authorization_code": auth_code,
            "temporary_password": temp_password,
        }

    def reset_superadmin_password(self, sa_id: str) -> Dict[str, Any]:
        """Issue a new temporary password and clear any pending reset request."""
        from passlib.context import CryptContext

        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        temp_password = generate_temp_password()
        now = datetime.utcnow()

        result = self.master["super_admins"].update_one(
            {"_id": ObjectId(sa_id)},
            {
                "$set": {
                    "password_hash": pwd_context.hash(temp_password),
                    "temp_password": temp_password,
                    "requires_password_change": True,
                    "password_changed_at": None,
                    "password_reset_requested": False,
                    "updated_at": now,
                },
                "$unset": {
                    "password_reset_requested_at": "",
                },
            },
        )
        if result.matched_count == 0:
            raise ValueError("Super-admin not found.")

        return {
            "temporary_password": temp_password,
            "requires_password_change": True,
        }

    # ---- Super-admin lifecycle management ----

    def _cascade_platform_suspended(self, sa_id: str, suspended: bool) -> int:
        """Set platform_suspended on all tenants assigned to a super-admin."""
        now = datetime.utcnow()
        update: Dict[str, Any] = {"$set": {"platform_suspended": suspended}}
        if suspended:
            update["$set"]["platform_suspended_at"] = now
        else:
            update["$unset"] = {"platform_suspended_at": ""}
        result = self.master["tenants"].update_many(
            {"assigned_superadmin_id": ObjectId(sa_id)},
            update,
        )
        return result.modified_count

    def suspend_superadmin(self, sa_id: str, reason: str = "") -> Dict[str, Any]:
        """Suspend a super-admin and cascade platform_suspended to their tenants."""
        now = datetime.utcnow()
        self.master["super_admins"].update_one(
            {"_id": ObjectId(sa_id)},
            {"$set": {
                "status": "suspended",
                "is_active": False,
                "suspended_at": now,
                "suspended_reason": reason or None,
            }},
        )
        affected = self._cascade_platform_suspended(sa_id, True)
        return {"status": "suspended", "tenants_affected": affected}

    def activate_superadmin(self, sa_id: str) -> Dict[str, Any]:
        """Reactivate a super-admin and lift platform_suspended from their tenants."""
        self.master["super_admins"].update_one(
            {"_id": ObjectId(sa_id)},
            {"$set": {"status": "active", "is_active": True},
             "$unset": {"suspended_at": "", "suspended_reason": "",
                        "deactivated_at": "", "deactivated_reason": ""}},
        )
        affected = self._cascade_platform_suspended(sa_id, False)
        return {"status": "active", "tenants_affected": affected}

    def deactivate_superadmin(self, sa_id: str, reason: str = "") -> Dict[str, Any]:
        """Deactivate a super-admin and cascade platform_suspended to their tenants."""
        now = datetime.utcnow()
        self.master["super_admins"].update_one(
            {"_id": ObjectId(sa_id)},
            {"$set": {
                "status": "deactivated",
                "is_active": False,
                "deactivated_at": now,
                "deactivated_reason": reason or None,
            }},
        )
        affected = self._cascade_platform_suspended(sa_id, True)
        return {"status": "deactivated", "tenants_affected": affected}

    def delete_superadmin(self, sa_id: str) -> Dict[str, Any]:
        """Delete a super-admin, orphan their tenants (set platform_suspended)."""
        # Orphan tenants: clear assignment and suspend platform access
        result = self.master["tenants"].update_many(
            {"assigned_superadmin_id": ObjectId(sa_id)},
            {"$unset": {"assigned_superadmin_id": ""},
             "$set": {"platform_suspended": True, "platform_suspended_at": datetime.utcnow()}},
        )
        tenants_orphaned = result.modified_count
        self.master["super_admins"].delete_one({"_id": ObjectId(sa_id)})
        return {"deleted": True, "tenants_orphaned": tenants_orphaned}

    def assign_all_tenants_to_superadmin(
        self, sa_id: str, include_all_statuses: bool = False
    ) -> int:
        """Assign unassigned tenants to a super-admin. Returns count updated."""
        query: Dict[str, Any] = {
            "$or": [
                {"assigned_superadmin_id": {"$exists": False}},
                {"assigned_superadmin_id": None},
            ]
        }
        if not include_all_statuses:
            query["status"] = {"$in": ["active", "approved", "pending"]}
        result = self.master["tenants"].update_many(
            query,
            {"$set": {"assigned_superadmin_id": ObjectId(sa_id)}},
        )
        return result.modified_count

    # ---- Pricing ----

    def get_pricing(self, superadmin_id: str) -> Dict[str, Any]:
        doc = self.master["superadmin_pricing"].find_one(
            {"superadmin_id": ObjectId(superadmin_id)}
        )
        if doc:
            # Legacy detection: old docs have tier_rates but not tiers
            if "tier_rates" in doc and "tiers" not in doc:
                return _convert_legacy_pricing(doc)
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
        tiers = pricing["tiers"]
        sa_fee_data = pricing["superadmin_fee"]
        cycle = pricing.get("billing_cycle", "monthly")
        suffix = "monthly" if cycle == "monthly" else "annual"

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
            tier_rates = tiers.get(tier, tiers.get("core", DEFAULT_PRICING["tiers"]["core"]))

            counts = {"students": 0, "tutors": 0, "admins": 0}
            db_name = t.get("db_name")
            if db_name:
                counts = self.get_tenant_user_counts(db_name)

            s_cost = round(counts["students"] * tier_rates.get(f"student_{suffix}", 0), 2)
            t_cost = round(counts["tutors"] * tier_rates.get(f"tutor_{suffix}", 0), 2)
            a_cost = round(counts["admins"] * tier_rates.get(f"admin_{suffix}", 0), 2)
            total = round(s_cost + t_cost + a_cost, 2)
            total_tenants_cost += total

            tenant_costs.append({
                "tenant_id": str(t["_id"]),
                "institution_name": t.get("institution_name", ""),
                "tier": tier,
                "status": t.get("status", ""),
                "student_count": counts["students"],
                "tutor_count": counts["tutors"],
                "admin_count": counts["admins"],
                "student_cost": s_cost,
                "tutor_cost": t_cost,
                "admin_cost": a_cost,
                "total_cost": total,
            })

        sa_fee = sa_fee_data.get(suffix, sa_fee_data.get("monthly", 100.0))

        return {
            "pricing": pricing,
            "billing_cycle": cycle,
            "tenant_costs": tenant_costs,
            "total_tenants_cost": round(total_tenants_cost, 2),
            "superadmin_fee": round(sa_fee, 2),
            "total_platform_cost": round(total_tenants_cost + sa_fee, 2),
        }

    # ---- Payments & Billing ----

    def list_payments(self, superadmin_id: str, skip: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
        payments = list(
            self.master["superadmin_payments"].find(
                {"superadmin_id": ObjectId(superadmin_id)}
            ).sort("payment_date", -1).skip(skip).limit(limit)
        )
        for p in payments:
            p["_id"] = str(p["_id"])
            p["superadmin_id"] = str(p["superadmin_id"])
            for key in ("payment_date", "period_start", "period_end", "created_at"):
                if isinstance(p.get(key), datetime):
                    p[key] = p[key].isoformat()
        return payments

    def record_payment(
        self,
        superadmin_id: str,
        amount: float,
        payment_method: str = "bank_transfer",
        reference: str = "",
        notes: str = "",
        payment_date: Optional[datetime] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        recorded_by: str = "",
    ) -> str:
        now = datetime.utcnow()
        doc = {
            "superadmin_id": ObjectId(superadmin_id),
            "amount": amount,
            "currency": self.get_pricing(superadmin_id).get("currency", "USD"),
            "payment_date": payment_date or now,
            "payment_method": payment_method,
            "reference": reference,
            "period_start": period_start,
            "period_end": period_end,
            "notes": notes,
            "recorded_by": recorded_by,
            "created_at": now,
        }
        result = self.master["superadmin_payments"].insert_one(doc)
        return str(result.inserted_id)

    def get_payment_totals(
        self,
        superadmin_id: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        match: Dict[str, Any] = {"superadmin_id": ObjectId(superadmin_id)}
        if period_start or period_end:
            date_filter: Dict[str, Any] = {}
            if period_start:
                date_filter["$gte"] = period_start
            if period_end:
                date_filter["$lte"] = period_end
            match["payment_date"] = date_filter
        pipeline = [
            {"$match": match},
            {"$group": {"_id": None, "total": {"$sum": "$amount"}, "count": {"$sum": 1}}},
        ]
        result = list(self.master["superadmin_payments"].aggregate(pipeline))
        if result:
            return {"total": result[0]["total"], "count": result[0]["count"]}
        return {"total": 0.0, "count": 0}

    def get_billing_summary(self, superadmin_id: str) -> Dict[str, Any]:
        pricing = self.get_pricing(superadmin_id)
        costs = self.compute_costs_for_superadmin(superadmin_id)
        cycle = pricing.get("billing_cycle", "monthly")
        billing_day = pricing.get("billing_day", 1)
        now = datetime.utcnow()
        period_start, period_end, next_due = _compute_billing_period(now, cycle, billing_day)

        paid_period = self.get_payment_totals(superadmin_id, period_start, period_end)
        paid_all = self.get_payment_totals(superadmin_id)

        current_cost = costs["total_platform_cost"]
        return {
            "billing_cycle": cycle,
            "billing_day": billing_day,
            "period_start": period_start.strftime("%Y-%m-%d"),
            "period_end": period_end.strftime("%Y-%m-%d"),
            "current_period_cost": current_cost,
            "paid_this_period": round(paid_period["total"], 2),
            "balance_due": round(current_cost - paid_period["total"], 2),
            "total_paid_all_time": round(paid_all["total"], 2),
            "next_due_date": next_due.strftime("%Y-%m-%d"),
            "currency": pricing.get("currency", "USD"),
            "currency_symbol": pricing.get("currency_symbol", "$"),
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

    # ---- Desktop diagnostics (agent upload bundles) ----

    def _get_s3_client(self):
        if self._s3_client is not None:
            return self._s3_client

        use_s3 = os.getenv("USE_S3_STORAGE", "false").lower() == "true"
        if not use_s3:
            return None

        bucket = os.getenv("S3_BUCKET_NAME", "")
        key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
        key_secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        region = os.getenv("AWS_REGION", "ap-south-1")
        if not bucket or not key_id or not key_secret:
            return None

        try:
            import boto3  # type: ignore

            self._s3_client = boto3.client(
                "s3",
                aws_access_key_id=key_id,
                aws_secret_access_key=key_secret,
                region_name=region,
            )
        except Exception:
            self._s3_client = None
        return self._s3_client

    def _download_storage_bytes(self, storage_path: str) -> Optional[bytes]:
        if not storage_path:
            return None
        path = storage_path.strip()
        if path.startswith("s3://"):
            # Format: s3://bucket/key
            no_scheme = path.replace("s3://", "", 1)
            if "/" not in no_scheme:
                return None
            bucket, key = no_scheme.split("/", 1)
            client = self._get_s3_client()
            if not client:
                return None
            try:
                obj = client.get_object(Bucket=bucket, Key=key)
                return obj["Body"].read()
            except Exception:
                return None

        local_path = Path(path)
        if not local_path.is_absolute():
            local_path = self._backend_root / local_path
        if not local_path.exists():
            return None
        try:
            return local_path.read_bytes()
        except Exception:
            return None

    def _delete_storage_path(self, storage_path: str) -> bool:
        """Delete a storage object from S3/local path (best effort)."""
        if not storage_path:
            return True
        path = storage_path.strip()
        if path.startswith("s3://"):
            no_scheme = path.replace("s3://", "", 1)
            if "/" not in no_scheme:
                return False
            bucket, key = no_scheme.split("/", 1)
            client = self._get_s3_client()
            if not client:
                return False
            try:
                client.delete_object(Bucket=bucket, Key=key)
                return True
            except Exception:
                return False

        local_path = Path(path)
        if not local_path.is_absolute():
            local_path = self._backend_root / local_path
        try:
            if local_path.exists():
                local_path.unlink()
            return True
        except Exception:
            return False

    @staticmethod
    def _safe_json_load(text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return text

    @staticmethod
    def _truncate_text(text: str, max_chars: int = 5000) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... [truncated]"

    def list_diagnostics_reports(self, limit: int = 300) -> List[Dict[str, Any]]:
        """List diagnostics reports across all tenant DBs."""
        tenants = list(
            self.master["tenants"].find(
                {"db_name": {"$exists": True, "$ne": None}},
                {
                    "db_name": 1,
                    "tenant_id": 1,
                    "institution_name": 1,
                    "organization": 1,
                },
            )
        )

        rows: List[Dict[str, Any]] = []
        for tenant in tenants:
            db_name = tenant.get("db_name")
            if not db_name:
                continue
            tenant_id = tenant.get("tenant_id")
            tenant_name = tenant.get("institution_name") or tenant.get("organization") or ""
            try:
                docs = list(
                    self.client[db_name]["desktop_diagnostics"]
                    .find(
                        {},
                        {
                            "ticket_id": 1,
                            "package_id": 1,
                            "user_id": 1,
                            "username": 1,
                            "uploaded_at": 1,
                            "size_bytes": 1,
                            "app_version": 1,
                            "pen_mac": 1,
                            "storage_path": 1,
                        },
                    )
                    .sort("uploaded_at", -1)
                    .limit(200)
                )
            except Exception:
                continue

            for doc in docs:
                rows.append(
                    {
                        "_id": str(doc.get("_id")),
                        "db_name": db_name,
                        "tenant_id": tenant_id,
                        "tenant_name": tenant_name,
                        "ticket_id": doc.get("ticket_id", ""),
                        "package_id": doc.get("package_id", ""),
                        "user_id": doc.get("user_id", ""),
                        "username": doc.get("username", ""),
                        "uploaded_at": doc.get("uploaded_at"),
                        "size_bytes": int(doc.get("size_bytes", 0) or 0),
                        "app_version": doc.get("app_version", ""),
                        "pen_mac": doc.get("pen_mac", ""),
                        "storage_path": doc.get("storage_path", ""),
                    }
                )

        rows.sort(key=lambda r: r.get("uploaded_at") or datetime.min, reverse=True)
        return rows[: max(1, int(limit))]

    def get_diagnostics_report_details(self, report_id: str, db_name: str) -> Dict[str, Any]:
        """Fetch report metadata and inspect ZIP contents in-memory only."""
        tenant_db = self.client[db_name]
        doc = tenant_db["desktop_diagnostics"].find_one({"_id": ObjectId(report_id)})
        if not doc:
            raise ValueError("Diagnostics report not found")

        storage_path = doc.get("storage_path", "")
        raw = self._download_storage_bytes(storage_path)
        if not raw:
            raise ValueError("Unable to load diagnostics archive from storage path")

        detail: Dict[str, Any] = {
            "ticket_id": doc.get("ticket_id", ""),
            "package_id": doc.get("package_id", ""),
            "db_name": db_name,
            "tenant_user": doc.get("user_id", ""),
            "username": doc.get("username", ""),
            "uploaded_at": doc.get("uploaded_at"),
            "size_bytes": int(doc.get("size_bytes", 0) or 0),
            "storage_path": storage_path,
            "zip_size_bytes": len(raw),
            "entries": [],
            "manifest": {},
            "system": {},
            "runtime_state": {},
            "errors_summary": {},
            "user_note": {},
        }

        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            names = sorted(zf.namelist())
            detail["entries"] = names

            def _read_json(name: str) -> Any:
                if name not in names:
                    return {}
                text = zf.read(name).decode("utf-8", errors="ignore")
                return self._safe_json_load(text)

            detail["manifest"] = _read_json("manifest.json")
            detail["system"] = _read_json("system.json")
            detail["runtime_state"] = _read_json("runtime_state.json")
            detail["errors_summary"] = _read_json("errors_summary.json")
            detail["user_note"] = _read_json("user_note.json")

            # Keep console trace compact for TUI readability.
            trace = _read_json("console_trace.json")
            if isinstance(trace, list):
                detail["console_trace_tail"] = trace[-40:]
            else:
                detail["console_trace_tail"] = trace

        pretty = []
        pretty.append(f"Ticket: {detail['ticket_id']}")
        pretty.append(f"Package: {detail['package_id']}")
        pretty.append(f"DB: {detail['db_name']}")
        pretty.append(f"User: {detail['tenant_user']} ({detail['username']})")
        pretty.append(f"Uploaded At: {detail['uploaded_at']}")
        pretty.append(f"Reported Size: {detail['size_bytes']} bytes")
        pretty.append(f"Archive Size: {detail['zip_size_bytes']} bytes")
        pretty.append(f"Storage: {detail['storage_path']}")
        pretty.append("")
        pretty.append(f"Archive Entries ({len(detail['entries'])}):")
        for name in detail["entries"]:
            pretty.append(f"  - {name}")
        pretty.append("")
        pretty.append("User Note:")
        pretty.append(self._truncate_text(json.dumps(detail.get("user_note", {}), indent=2, default=str), 1500))
        pretty.append("")
        pretty.append("Manifest:")
        pretty.append(self._truncate_text(json.dumps(detail.get("manifest", {}), indent=2, default=str), 3000))
        pretty.append("")
        pretty.append("System:")
        pretty.append(self._truncate_text(json.dumps(detail.get("system", {}), indent=2, default=str), 3000))
        pretty.append("")
        pretty.append("Runtime State:")
        pretty.append(self._truncate_text(json.dumps(detail.get("runtime_state", {}), indent=2, default=str), 5000))
        pretty.append("")
        pretty.append("Errors Summary:")
        pretty.append(self._truncate_text(json.dumps(detail.get("errors_summary", {}), indent=2, default=str), 5000))
        pretty.append("")
        pretty.append("Console Trace (tail):")
        pretty.append(self._truncate_text(json.dumps(detail.get("console_trace_tail", {}), indent=2, default=str), 5000))

        detail["pretty_text"] = "\n".join(pretty)
        return detail

    def list_desktop_bug_reports(self, limit: int = 400) -> List[Dict[str, Any]]:
        """List desktop Help-tab bug reports across all tenant DBs."""
        tenants = list(
            self.master["tenants"].find(
                {"db_name": {"$exists": True, "$ne": None}},
                {
                    "db_name": 1,
                    "tenant_id": 1,
                    "institution_name": 1,
                    "organization": 1,
                },
            )
        )

        rows: List[Dict[str, Any]] = []
        for tenant in tenants:
            db_name = tenant.get("db_name")
            if not db_name:
                continue
            tenant_id = tenant.get("tenant_id")
            tenant_name = tenant.get("institution_name") or tenant.get("organization") or ""
            try:
                docs = list(
                    self.client[db_name]["desktop_bug_reports"]
                    .find(
                        {},
                        {
                            "ticket_id": 1,
                            "title": 1,
                            "description": 1,
                            "user_id": 1,
                            "username": 1,
                            "created_at": 1,
                            "app_version": 1,
                            "pen_mac": 1,
                            "machine": 1,
                            "os_info": 1,
                            "pen_connected": 1,
                            "attachment_count": 1,
                            "attachment_total_bytes": 1,
                        },
                    )
                    .sort("created_at", -1)
                    .limit(300)
                )
            except Exception:
                continue

            for doc in docs:
                rows.append(
                    {
                        "_id": str(doc.get("_id")),
                        "db_name": db_name,
                        "tenant_id": tenant_id,
                        "tenant_name": tenant_name,
                        "ticket_id": doc.get("ticket_id", ""),
                        "title": doc.get("title", ""),
                        "description": doc.get("description", ""),
                        "user_id": doc.get("user_id", ""),
                        "username": doc.get("username", ""),
                        "created_at": doc.get("created_at"),
                        "app_version": doc.get("app_version", ""),
                        "pen_mac": doc.get("pen_mac", ""),
                        "machine": doc.get("machine", ""),
                        "os_info": doc.get("os_info", ""),
                        "pen_connected": bool(doc.get("pen_connected", False)),
                        "attachment_count": int(doc.get("attachment_count", 0) or 0),
                        "attachment_total_bytes": int(doc.get("attachment_total_bytes", 0) or 0),
                    }
                )

        rows.sort(key=lambda r: r.get("created_at") or datetime.min, reverse=True)
        return rows[: max(1, int(limit))]

    def get_desktop_bug_report_details(self, report_id: str, db_name: str) -> Dict[str, Any]:
        """Fetch one desktop bug report document and format readable details."""
        tenant_db = self.client[db_name]
        doc = tenant_db["desktop_bug_reports"].find_one({"_id": ObjectId(report_id)})
        if not doc:
            raise ValueError("Desktop bug report not found")

        detail: Dict[str, Any] = {
            "ticket_id": doc.get("ticket_id", ""),
            "db_name": db_name,
            "user_id": doc.get("user_id", ""),
            "username": doc.get("username", ""),
            "created_at": doc.get("created_at"),
            "app_version": doc.get("app_version", ""),
            "pen_mac": doc.get("pen_mac", ""),
            "pen_connected": bool(doc.get("pen_connected", False)),
            "machine": doc.get("machine", ""),
            "os_info": doc.get("os_info", ""),
            "title": doc.get("title", ""),
            "description": doc.get("description", ""),
            "reported_at_client": doc.get("reported_at_client"),
            "attachments": doc.get("attachments") or [],
        }

        pretty: List[str] = []
        pretty.append(f"Ticket: {detail['ticket_id']}")
        pretty.append(f"DB: {detail['db_name']}")
        pretty.append(f"User: {detail['user_id']} ({detail['username']})")
        pretty.append(f"Created At: {detail['created_at']}")
        pretty.append(f"Reported At (Client): {detail.get('reported_at_client')}")
        pretty.append(f"App Version: {detail['app_version']}")
        pretty.append(f"Pen MAC: {detail['pen_mac']}")
        pretty.append(f"Pen Connected: {detail['pen_connected']}")
        pretty.append(f"Machine: {detail['machine']}")
        pretty.append(f"OS Info: {detail['os_info']}")
        pretty.append(f"Attachments: {len(detail['attachments'])}")
        for item in detail["attachments"]:
            pretty.append(
                f"  - {item.get('filename', '')} | {item.get('content_type', '')} | "
                f"{int(item.get('size_bytes', 0) or 0)} bytes"
            )
            pretty.append(f"    storage: {item.get('storage_path', '')}")
        pretty.append("")
        pretty.append("Title:")
        pretty.append(self._truncate_text(str(detail["title"]), 1000))
        pretty.append("")
        pretty.append("Description:")
        pretty.append(self._truncate_text(str(detail["description"]), 12000))

        detail["pretty_text"] = "\n".join(pretty)
        return detail

    def delete_diagnostics_report(self, report_id: str, db_name: str) -> Dict[str, Any]:
        """Delete one diagnostics record and its archive object."""
        tenant_db = self.client[db_name]
        doc = tenant_db["desktop_diagnostics"].find_one({"_id": ObjectId(report_id)})
        if not doc:
            return {"deleted": False, "reason": "not_found"}
        storage_deleted = self._delete_storage_path(str(doc.get("storage_path", "")))
        tenant_db["desktop_diagnostics"].delete_one({"_id": ObjectId(report_id)})
        return {"deleted": True, "storage_deleted": storage_deleted}

    def delete_all_diagnostics_reports(self) -> Dict[str, Any]:
        """Delete all diagnostics records and archive objects across all tenant DBs."""
        deleted_docs = 0
        deleted_storage = 0
        tenants = list(self.master["tenants"].find({"db_name": {"$exists": True, "$ne": None}}, {"db_name": 1}))
        for tenant in tenants:
            db_name = tenant.get("db_name")
            if not db_name:
                continue
            coll = self.client[db_name]["desktop_diagnostics"]
            docs = list(coll.find({}, {"_id": 1, "storage_path": 1}))
            for doc in docs:
                deleted_docs += 1
                if self._delete_storage_path(str(doc.get("storage_path", ""))):
                    deleted_storage += 1
            coll.delete_many({})
        return {"deleted_docs": deleted_docs, "deleted_storage": deleted_storage}

    def delete_desktop_bug_report(self, report_id: str, db_name: str) -> Dict[str, Any]:
        """Delete one desktop bug report and all referenced attachments."""
        tenant_db = self.client[db_name]
        doc = tenant_db["desktop_bug_reports"].find_one({"_id": ObjectId(report_id)})
        if not doc:
            return {"deleted": False, "reason": "not_found"}
        deleted_storage = 0
        for item in (doc.get("attachments") or []):
            if self._delete_storage_path(str(item.get("storage_path", ""))):
                deleted_storage += 1
        tenant_db["desktop_bug_reports"].delete_one({"_id": ObjectId(report_id)})
        return {"deleted": True, "deleted_storage": deleted_storage}

    def delete_all_desktop_bug_reports(self) -> Dict[str, Any]:
        """Delete all desktop bug reports and attachment objects across all tenants."""
        deleted_docs = 0
        deleted_storage = 0
        tenants = list(self.master["tenants"].find({"db_name": {"$exists": True, "$ne": None}}, {"db_name": 1}))
        for tenant in tenants:
            db_name = tenant.get("db_name")
            if not db_name:
                continue
            coll = self.client[db_name]["desktop_bug_reports"]
            docs = list(coll.find({}, {"_id": 1, "attachments": 1}))
            for doc in docs:
                deleted_docs += 1
                for item in (doc.get("attachments") or []):
                    if self._delete_storage_path(str(item.get("storage_path", ""))):
                        deleted_storage += 1
            coll.delete_many({})
        return {"deleted_docs": deleted_docs, "deleted_storage": deleted_storage}
