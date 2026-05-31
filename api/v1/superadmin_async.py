"""
Super Admin API for Stoody Platform
Handles tenant management, registration approval, and feature flags
"""

import calendar
import logging
import os
import re
import secrets
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

import jwt
import pyotp
from bson import ObjectId
from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

from config_async import settings
from core.auth import AuthManager
from core.database import DatabaseManager
from core.token_blacklist import revoke_user_session
from core.tenant_features import (
    LEGACY_DEFAULT_TENANT_FEATURES,
    build_enabled_features_v2,
    export_legacy_features,
    get_feature_catalog,
)

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings for super admin (separate from regular auth)
SUPERADMIN_JWT_SECRET = getattr(settings, "SUPERADMIN_JWT_SECRET", None) or getattr(settings, "JWT_SECRET_KEY", "")
SUPERADMIN_JWT_ALGORITHM = "HS256"
SUPERADMIN_JWT_EXPIRATION_HOURS = 24

# Temp token + 2FA settings
APP_SECRET_KEY = getattr(settings, "JWT_SECRET_KEY", SUPERADMIN_JWT_SECRET)
TEMP_TOKEN_MAX_AGE_SECONDS = 600
TOTP_ISSUER = "Stoody Super Admin"
TOTP_ENC_KEY = os.getenv("TOTP_ENC_KEY", "")
AUTHORIZATION_CODE_PATTERN = re.compile(r"^[A-Z0-9]{6}$")
INSTITUTION_ID_PATTERN = re.compile(r"^[A-Z]{4}-[0-9]{4}$")


# ============ PYDANTIC MODELS ============

class SuperAdminLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)


class SuperAdminAuthResponse(BaseModel):
    success: bool = True
    next: Literal["RESET_PASSWORD", "SETUP_2FA", "OTP", "DONE"]
    temp_token: Optional[str] = None
    access_token: Optional[str] = None
    admin: Dict[str, Any]
    requires_password_change: bool = False
    message: Optional[str] = None


class SuperAdminPasswordChangeRequest(BaseModel):
    temp_token: str
    old_password: str = Field(..., min_length=6)
    new_password: str = Field(..., min_length=8)


class SuperAdminPasswordResetRequest(BaseModel):
    email: EmailStr


class SuperAdminPasswordResetRequestResponse(BaseModel):
    success: bool = True
    message: str


class TempTokenRequest(BaseModel):
    temp_token: str


class OTPVerifyRequest(BaseModel):
    temp_token: str
    otp: str = Field(..., min_length=6, max_length=6)


class ApproveTenantRequest(BaseModel):
    institution_id: str = Field(..., min_length=9, max_length=9, pattern=r'^[A-Z]{4}-[0-9]{4}$')
    notes: Optional[str] = None
    features: Optional[Dict[str, bool]] = None


class RejectTenantRequest(BaseModel):
    reason: str = Field(..., min_length=10)


class SuspendTenantRequest(BaseModel):
    reason: str = Field(..., min_length=10)


class UpdateFeaturesRequest(BaseModel):
    features: Dict[str, bool]


class UpdateFeaturesV2Request(BaseModel):
    tier: Literal["core", "advanced", "max", "custom"] = "core"
    overrides: Dict[str, bool] = Field(default_factory=dict)


class UpdateLimitsRequest(BaseModel):
    max_students: Optional[int] = None
    max_tutors: Optional[int] = None
    subscription_tier: Optional[str] = None


class ResetPasswordRequest(BaseModel):
    # Legacy clients may still send this field, but super-admin resets now
    # generate the replacement password server-side and return it once.
    new_password: Optional[str] = Field(default=None, min_length=8)


class UpdateTenantIdRequest(BaseModel):
    institution_id: str = Field(..., min_length=9, max_length=9, pattern=r'^[A-Z]{4}-[0-9]{4}$')


class SendMessageRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1)
    priority: Optional[str] = Field(default="normal", pattern=r'^(low|normal|high|urgent)$')


class SetTenantPricingRequest(BaseModel):
    price: float = Field(..., ge=0)
    notes: Optional[str] = None


class DeleteTenantRequest(BaseModel):
    confirmation: str = Field(..., description="Must match institution name to confirm deletion")


class TierRates(BaseModel):
    student_monthly: float = Field(0, ge=0)
    student_annual: float = Field(0, ge=0)
    tutor_monthly: float = Field(0, ge=0)
    tutor_annual: float = Field(0, ge=0)
    admin_monthly: float = Field(0, ge=0)
    admin_annual: float = Field(0, ge=0)


class SuperAdminFee(BaseModel):
    monthly: float = Field(100.0, ge=0)
    annual: float = Field(1000.0, ge=0)


class SuperAdminPricingUpdate(BaseModel):
    currency: Optional[str] = Field(None, pattern=r'^[A-Z]{3}$')
    currency_symbol: Optional[str] = Field(None, max_length=5)
    tiers: Optional[Dict[str, TierRates]] = None
    superadmin_fee: Optional[SuperAdminFee] = None
    billing_cycle: Optional[Literal["monthly", "annual"]] = None
    billing_day: Optional[int] = Field(None, ge=1, le=28)
    notes: Optional[str] = None


class RecordPaymentRequest(BaseModel):
    amount: float = Field(..., gt=0)
    payment_date: Optional[str] = None
    payment_method: str = Field("bank_transfer")
    reference: str = Field("")
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    notes: str = Field("")


CURRENCY_MAP = {"USD": "$", "EUR": "\u20ac", "INR": "\u20b9"}

DEFAULT_SUPERADMIN_PRICING = {
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


class MarkNotificationsReadRequest(BaseModel):
    message_ids: Optional[List[str]] = None
    tenant_id: Optional[str] = None


# ============ HELPERS ============

async def get_database(request: Request) -> DatabaseManager:
    return request.app.state.db


async def get_auth_manager(request: Request) -> Optional[AuthManager]:
    return getattr(request.app.state, "auth", None)


def _model_dump(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def convert_objectids(obj):
    """Recursively convert ObjectId fields to strings"""
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, dict):
        return {k: convert_objectids(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_objectids(item) for item in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _convert_legacy_pricing(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old flat-rate pricing doc to new tiered structure."""
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


async def _get_pricing(master_db, admin_id: str) -> Dict[str, Any]:
    """Read pricing with legacy fallback."""
    doc = await master_db["superadmin_pricing"].find_one(
        {"superadmin_id": ObjectId(admin_id)}
    )
    if doc:
        doc.pop("_id", None)
        doc.pop("superadmin_id", None)
        # Legacy format detection: has tier_rates (flat dict) but no tiers
        if "tier_rates" in doc and "tiers" not in doc:
            return _convert_legacy_pricing(doc)
        # New format
        result = {**DEFAULT_SUPERADMIN_PRICING}
        for key in DEFAULT_SUPERADMIN_PRICING:
            if key in doc and doc[key] is not None:
                result[key] = doc[key]
        return result
    return {**DEFAULT_SUPERADMIN_PRICING}


def _compute_billing_period(now: datetime, cycle: str, billing_day: int):
    """Returns (period_start, period_end, next_due_date)."""
    if cycle == "annual":
        year = now.year if now.month > 1 or now.day >= billing_day else now.year - 1
        period_start = datetime(year, 1, min(billing_day, 28))
        period_end = datetime(year, 12, 31, 23, 59, 59)
        next_due = datetime(year + 1, 1, min(billing_day, 28))
    else:
        # monthly
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


async def _compute_total_cost(master_db, db, admin_id: str, pricing: Dict[str, Any]) -> Dict[str, Any]:
    """Shared cost computation for platform-costs and billing-summary."""
    tiers = pricing["tiers"]
    sa_fee_data = pricing["superadmin_fee"]
    cycle = pricing.get("billing_cycle", "monthly")
    suffix = "monthly" if cycle == "monthly" else "annual"

    tenants = await master_db["tenants"].find({
        "assigned_superadmin_id": ObjectId(admin_id),
        "status": {"$in": ["active", "approved"]},
    }).to_list(length=1000)

    tenant_costs = []
    total_tenants_cost = 0.0

    for tenant in tenants:
        features_v2 = _tenant_features_v2_from_doc(tenant)
        tier = features_v2.get("tier", "core")
        tier_rates = tiers.get(tier, tiers.get("core", DEFAULT_SUPERADMIN_PRICING["tiers"]["core"]))

        student_count = 0
        tutor_count = 0
        admin_count = 0
        db_name = tenant.get("db_name")
        if db_name:
            try:
                tenant_db = await db.get_tenant_db(db_name)
                if tenant_db:
                    student_count = await tenant_db["students"].count_documents({})
                    tutor_count = await tenant_db["tutors"].count_documents({})
                    admin_count = await tenant_db["admins"].count_documents({})
            except Exception:
                pass

        student_cost = round(student_count * tier_rates.get(f"student_{suffix}", 0), 2)
        tutor_cost = round(tutor_count * tier_rates.get(f"tutor_{suffix}", 0), 2)
        admin_cost = round(admin_count * tier_rates.get(f"admin_{suffix}", 0), 2)
        total_cost = round(student_cost + tutor_cost + admin_cost, 2)
        total_tenants_cost += total_cost

        tenant_costs.append({
            "tenant_id": str(tenant["_id"]),
            "institution_name": tenant.get("institution_name", ""),
            "tier": tier,
            "student_count": student_count,
            "tutor_count": tutor_count,
            "admin_count": admin_count,
            "student_cost": student_cost,
            "tutor_cost": tutor_cost,
            "admin_cost": admin_cost,
            "total_cost": total_cost,
        })

    sa_fee = sa_fee_data.get(suffix, sa_fee_data.get("monthly", 100.0))

    return {
        "tenant_costs": tenant_costs,
        "total_tenants_cost": round(total_tenants_cost, 2),
        "superadmin_fee": round(sa_fee, 2),
        "total_platform_cost": round(total_tenants_cost + sa_fee, 2),
    }


def _sanitize_legacy_features(raw: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    sanitized = dict(LEGACY_DEFAULT_TENANT_FEATURES)
    if not isinstance(raw, dict):
        return sanitized
    for key in LEGACY_DEFAULT_TENANT_FEATURES.keys():
        if key in raw:
            sanitized[key] = bool(raw.get(key))
    return sanitized


def _sanitize_v2_overrides(raw: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): bool(value) for key, value in raw.items()}


def _tenant_features_v2_from_doc(tenant: Dict[str, Any]) -> Dict[str, Any]:
    return build_enabled_features_v2(
        tenant.get("enabled_features_v2"),
        tenant.get("enabled_features"),
    )


def _feature_diff(before_effective: Dict[str, bool], after_effective: Dict[str, bool]) -> List[Dict[str, Any]]:
    diff: List[Dict[str, Any]] = []
    for key in sorted(set(before_effective.keys()) | set(after_effective.keys())):
        before_value = bool(before_effective.get(key))
        after_value = bool(after_effective.get(key))
        if before_value != after_value:
            diff.append({
                "key": key,
                "from": before_value,
                "to": after_value,
            })
    return diff


def _get_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(APP_SECRET_KEY)


def create_temp_token(admin_id: str, purpose: str) -> str:
    return _get_serializer().dumps({
        "uid": admin_id,
        "purpose": purpose,
        "type": "superadmin",
    })


def verify_temp_token(token: str, expected_purpose: str) -> Dict[str, Any]:
    try:
        payload = _get_serializer().loads(token, max_age=TEMP_TOKEN_MAX_AGE_SECONDS)
        if payload.get("type") != "superadmin":
            raise HTTPException(status_code=401, detail="Invalid token type")
        if payload.get("purpose") != expected_purpose:
            raise HTTPException(status_code=400, detail="Invalid token purpose")
        return payload
    except SignatureExpired:
        raise HTTPException(status_code=401, detail="Session expired. Please login again.")
    except BadSignature:
        raise HTTPException(status_code=401, detail="Invalid session token")


def _get_fernet() -> Optional[Fernet]:
    if not TOTP_ENC_KEY:
        logger.warning("TOTP_ENC_KEY not set - 2FA secrets stored unencrypted")
        return None
    try:
        return Fernet(TOTP_ENC_KEY.encode())
    except Exception as e:
        logger.error("Invalid TOTP_ENC_KEY: %s", e)
        return None


def encrypt_secret(secret: str) -> str:
    fernet = _get_fernet()
    if not fernet:
        return secret
    return fernet.encrypt(secret.encode()).decode()


def decrypt_secret(encrypted: str) -> str:
    fernet = _get_fernet()
    if not fernet:
        return encrypted
    try:
        return fernet.decrypt(encrypted.encode()).decode()
    except InvalidToken:
        raise HTTPException(status_code=500, detail="2FA configuration error")


def normalize_institution_id(institution_id: str) -> str:
    return institution_id.strip().upper()


def derive_tenant_id(institution_id: str) -> str:
    normalized = normalize_institution_id(institution_id)
    if not INSTITUTION_ID_PATTERN.match(normalized):
        raise HTTPException(status_code=400, detail="Institution ID must match XXXX-0000 format")
    return normalized


def build_db_name(institution_id: str) -> str:
    normalized = normalize_institution_id(institution_id)
    return f"skb_{normalized.lower()}"


def create_superadmin_token(admin_id: str, email: str) -> str:
    payload = {
        "sub": admin_id,
        "email": email,
        "type": "superadmin",
        "exp": datetime.utcnow() + timedelta(hours=SUPERADMIN_JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SUPERADMIN_JWT_SECRET, algorithm=SUPERADMIN_JWT_ALGORITHM)


def _derive_sa_status(doc: Dict[str, Any]) -> str:
    """Derive status from doc, falling back to is_active for legacy docs."""
    if "status" in doc:
        return doc["status"]
    return "active" if doc.get("is_active", True) else "deactivated"


def build_admin_response(admin_doc: Dict[str, Any]) -> Dict[str, Any]:
    two_fa = admin_doc.get("two_fa") or {}
    auth_code = (admin_doc.get("authorization_code") or "").upper()
    if auth_code and not AUTHORIZATION_CODE_PATTERN.match(auth_code):
        logger.warning("Super admin %s has invalid authorization code format", admin_doc.get("email"))

    sa_status = _derive_sa_status(admin_doc)
    return {
        "_id": str(admin_doc["_id"]),
        "email": admin_doc["email"],
        "name": admin_doc.get("name", ""),
        "role": admin_doc.get("role", "super_admin"),
        "permissions": admin_doc.get("permissions", ["all"]),
        "is_active": sa_status == "active",
        "status": sa_status,
        "requires_password_change": bool(admin_doc.get("requires_password_change", False)),
        "two_fa": {
            "enabled": bool(two_fa.get("enabled", False)),
            "required": bool(two_fa.get("required", True)),
        },
        "authorization_code": auth_code,
    }


async def get_master_db_or_503(db: DatabaseManager):
    master_db = await db.get_master_db()
    if master_db is None:
        raise HTTPException(status_code=503, detail="Master database unavailable")
    return master_db


async def get_superadmin_by_id_or_401(master_db, admin_id: str) -> Dict[str, Any]:
    try:
        admin_oid = ObjectId(admin_id)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    admin = await master_db["super_admins"].find_one({"_id": admin_oid})
    if not admin:
        raise HTTPException(status_code=401, detail="Admin not found or inactive")

    sa_status = _derive_sa_status(admin)
    if sa_status == "suspended":
        raise HTTPException(status_code=403, detail="Account suspended")
    if sa_status == "deactivated":
        raise HTTPException(status_code=403, detail="Account deactivated")
    return admin


def ensure_tenant_owned_by_admin(tenant: Dict[str, Any], admin_id: str) -> None:
    assigned = tenant.get("assigned_superadmin_id")
    if not assigned:
        raise HTTPException(
            status_code=403,
            detail="Tenant is not assigned to any super admin. Run migration assignment first."
        )
    if str(assigned) != admin_id:
        raise HTTPException(status_code=403, detail="Access denied for this tenant")


async def get_tenant_for_admin_or_error(master_db, tenant_id: str, admin_id: str) -> Dict[str, Any]:
    try:
        tenant = await master_db["tenants"].find_one({"_id": ObjectId(tenant_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tenant ID")

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    ensure_tenant_owned_by_admin(tenant, admin_id)
    return tenant


def generate_admin_reset_password(length: int = 22) -> str:
    """Generate a copyable high-entropy password for super-admin resets."""
    alphabet = string.ascii_letters + string.digits + "-_"
    return "".join(secrets.choice(alphabet) for _ in range(length))


async def get_tenant_master_admin_or_error(tenant_db, tenant: Dict[str, Any]) -> Dict[str, Any]:
    admin_email = tenant.get("admin_email") or (tenant.get("pending_admin") or {}).get("email")
    queries: List[Dict[str, Any]] = []
    if admin_email:
        queries.extend([
            {"email": admin_email, "role": "master_admin"},
            {"email": admin_email},
        ])
    queries.append({"role": "master_admin"})

    for query in queries:
        master_admin = await tenant_db["admins"].find_one(query)
        if master_admin:
            return master_admin

    raise HTTPException(status_code=400, detail="Master admin account not found for this tenant")


async def best_effort_revoke_admin_sessions(auth_manager: Optional[AuthManager], admin_id: str) -> None:
    """Try to force old sessions out without failing the completed reset."""
    if auth_manager is None:
        logger.warning("Auth manager unavailable; skipping session revocation for admin %s", admin_id)
        return

    try:
        await auth_manager.invalidate_user_session(admin_id)
    except Exception as e:
        logger.warning("Failed to invalidate cached admin session for %s: %s", admin_id, e)

    try:
        await revoke_user_session(auth_manager.cache_manager, admin_id)
    except Exception as e:
        logger.warning("Failed to set admin session revocation for %s: %s", admin_id, e)


async def ensure_tenant_indexes(tenant_db) -> None:
    await tenant_db["students"].create_index([("username_lower", 1)], unique=True, sparse=True, name="uniq_students_username_lower")
    await tenant_db["students"].create_index([("username", 1)], unique=True, name="uniq_students_username")
    await tenant_db["students"].create_index([("email", 1)], sparse=True, name="idx_students_email")
    await tenant_db["tutors"].create_index([("username_lower", 1)], unique=True, sparse=True, name="uniq_tutors_username_lower")
    await tenant_db["tutors"].create_index([("username", 1)], unique=True, name="uniq_tutors_username")
    await tenant_db["tutors"].create_index([("tutor_id", 1)], unique=True, name="uniq_tutors_tutor_id")
    await tenant_db["admins"].create_index([("email", 1)], unique=True, name="uniq_admins_email")
    await tenant_db["strokes"].create_index([("user_id", 1), ("timestamp", -1)], name="idx_strokes_user_ts")
    await tenant_db["smartboard_sessions"].create_index([("session_id", 1)], unique=True, name="uniq_smartboard_session_id")
    await tenant_db["smartboard_sessions"].create_index([("tutor_id", 1), ("status", 1)], name="idx_smartboard_tutor_status")


async def verify_superadmin_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: DatabaseManager = Depends(get_database),
) -> Dict[str, Any]:
    """Verify super admin JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SUPERADMIN_JWT_SECRET, algorithms=[SUPERADMIN_JWT_ALGORITHM])

        if payload.get("type") != "superadmin":
            raise HTTPException(status_code=401, detail="Invalid token type")

        admin_id = payload.get("sub")
        if not admin_id:
            raise HTTPException(status_code=401, detail="Invalid token")

        master_db = await get_master_db_or_503(db)
        admin_doc = await get_superadmin_by_id_or_401(master_db, admin_id)

        return {
            "admin_id": str(admin_doc["_id"]),
            "email": admin_doc["email"],
            "name": admin_doc.get("name", ""),
            "role": admin_doc.get("role", "super_admin"),
            "permissions": admin_doc.get("permissions", ["all"]),
        }

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


async def _complete_superadmin_login(master_db, admin_doc: Dict[str, Any]) -> SuperAdminAuthResponse:
    await master_db["super_admins"].update_one(
        {"_id": admin_doc["_id"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )
    token = create_superadmin_token(str(admin_doc["_id"]), admin_doc["email"])
    return SuperAdminAuthResponse(
        next="DONE",
        access_token=token,
        admin=build_admin_response(admin_doc),
        requires_password_change=False,
    )


# ============ AUTH ENDPOINTS ============

@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "superadmin"}


@router.post("/login", response_model=SuperAdminAuthResponse)
async def superadmin_login(
    request: SuperAdminLoginRequest,
    db: DatabaseManager = Depends(get_database),
):
    master_db = await get_master_db_or_503(db)

    admin = await master_db["super_admins"].find_one({"email": request.email})

    if not admin or not pwd_context.verify(request.password, admin.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    sa_status = _derive_sa_status(admin)
    if sa_status == "suspended":
        raise HTTPException(status_code=403, detail="Account suspended. Contact platform administrator.")
    if sa_status == "deactivated":
        raise HTTPException(status_code=403, detail="Account deactivated. Contact platform administrator.")

    requires_password_change = bool(admin.get("requires_password_change", False))
    if requires_password_change:
        return SuperAdminAuthResponse(
            next="RESET_PASSWORD",
            temp_token=create_temp_token(str(admin["_id"]), "RESET_PASSWORD"),
            admin=build_admin_response(admin),
            requires_password_change=True,
            message="Password change required before access.",
        )

    two_fa = admin.get("two_fa") or {}
    two_fa_required = bool(two_fa.get("required", True))
    two_fa_enabled = bool(two_fa.get("enabled", False))

    if two_fa_required and not two_fa_enabled:
        return SuperAdminAuthResponse(
            next="SETUP_2FA",
            temp_token=create_temp_token(str(admin["_id"]), "SETUP_2FA"),
            admin=build_admin_response(admin),
            requires_password_change=False,
            message="2FA setup required.",
        )

    if two_fa_enabled:
        return SuperAdminAuthResponse(
            next="OTP",
            temp_token=create_temp_token(str(admin["_id"]), "OTP"),
            admin=build_admin_response(admin),
            requires_password_change=False,
            message="Enter authenticator OTP.",
        )

    return await _complete_superadmin_login(master_db, admin)


@router.post("/password/request-reset", response_model=SuperAdminPasswordResetRequestResponse)
async def request_superadmin_password_reset(
    request: SuperAdminPasswordResetRequest,
    db: DatabaseManager = Depends(get_database),
):
    master_db = await get_master_db_or_503(db)
    email = request.email.strip().lower()
    admin = await master_db["super_admins"].find_one({"email": email})
    if admin:
        await master_db["super_admins"].update_one(
            {"_id": admin["_id"]},
            {
                "$set": {
                    "password_reset_requested": True,
                    "password_reset_requested_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                }
            },
        )

    return SuperAdminPasswordResetRequestResponse(
        message="If the account exists, the password reset request has been recorded for platform review."
    )


@router.post("/password/change", response_model=SuperAdminAuthResponse)
async def change_superadmin_password(
    request: SuperAdminPasswordChangeRequest,
    db: DatabaseManager = Depends(get_database),
):
    payload = verify_temp_token(request.temp_token, "RESET_PASSWORD")
    admin_id = payload.get("uid")
    if not admin_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    master_db = await get_master_db_or_503(db)
    admin_doc = await get_superadmin_by_id_or_401(master_db, admin_id)

    if not pwd_context.verify(request.old_password, admin_doc.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    new_hash = pwd_context.hash(request.new_password)
    await master_db["super_admins"].update_one(
        {"_id": admin_doc["_id"]},
        {
            "$set": {
                "password_hash": new_hash,
                "requires_password_change": False,
                "password_changed_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "password_reset_requested": False,
            },
            "$unset": {
                "temp_password": "",
                "password_reset_requested_at": "",
            },
        }
    )

    admin_doc = await get_superadmin_by_id_or_401(master_db, admin_id)
    two_fa = admin_doc.get("two_fa") or {}
    two_fa_required = bool(two_fa.get("required", True))
    two_fa_enabled = bool(two_fa.get("enabled", False))

    if two_fa_required and not two_fa_enabled:
        return SuperAdminAuthResponse(
            next="SETUP_2FA",
            temp_token=create_temp_token(str(admin_doc["_id"]), "SETUP_2FA"),
            admin=build_admin_response(admin_doc),
            requires_password_change=False,
            message="Password updated. 2FA setup required.",
        )

    if two_fa_enabled:
        return SuperAdminAuthResponse(
            next="OTP",
            temp_token=create_temp_token(str(admin_doc["_id"]), "OTP"),
            admin=build_admin_response(admin_doc),
            requires_password_change=False,
            message="Password updated. Enter authenticator OTP.",
        )

    return await _complete_superadmin_login(master_db, admin_doc)


@router.post("/2fa/setup/start")
async def superadmin_2fa_setup_start(
    request: TempTokenRequest,
    db: DatabaseManager = Depends(get_database),
):
    payload = verify_temp_token(request.temp_token, "SETUP_2FA")
    admin_id = payload.get("uid")
    if not admin_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    master_db = await get_master_db_or_503(db)
    admin_doc = await get_superadmin_by_id_or_401(master_db, admin_id)

    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)
    otpauth_url = totp.provisioning_uri(name=admin_doc["email"], issuer_name=TOTP_ISSUER)

    await master_db["super_admins"].update_one(
        {"_id": admin_doc["_id"]},
        {
            "$set": {
                "two_fa.temp_secret_enc": encrypt_secret(secret),
                "two_fa.setup_started_at": datetime.utcnow(),
                "two_fa.required": True,
            }
        }
    )

    return {
        "success": True,
        "otpauth_url": otpauth_url,
        "setup_key": secret,
        "message": "Scan QR URL or enter setup key in Google Authenticator.",
    }


@router.post("/2fa/setup/verify", response_model=SuperAdminAuthResponse)
async def superadmin_2fa_setup_verify(
    request: OTPVerifyRequest,
    db: DatabaseManager = Depends(get_database),
):
    payload = verify_temp_token(request.temp_token, "SETUP_2FA")
    admin_id = payload.get("uid")
    if not admin_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    master_db = await get_master_db_or_503(db)
    admin_doc = await get_superadmin_by_id_or_401(master_db, admin_id)

    temp_secret_enc = (admin_doc.get("two_fa") or {}).get("temp_secret_enc")
    if not temp_secret_enc:
        raise HTTPException(status_code=400, detail="2FA setup not started")

    secret = decrypt_secret(temp_secret_enc)
    totp = pyotp.TOTP(secret)
    if not totp.verify(request.otp.strip(), valid_window=1):
        raise HTTPException(status_code=401, detail="Invalid verification code")

    await master_db["super_admins"].update_one(
        {"_id": admin_doc["_id"]},
        {
            "$set": {
                "two_fa.enabled": True,
                "two_fa.required": True,
                "two_fa.secret_enc": encrypt_secret(secret),
                "two_fa.temp_secret_enc": None,
                "two_fa.verified_at": datetime.utcnow(),
                "two_fa.setup_started_at": None,
                "two_fa.last_verified_at": datetime.utcnow(),
            }
        }
    )

    admin_doc = await get_superadmin_by_id_or_401(master_db, admin_id)
    return await _complete_superadmin_login(master_db, admin_doc)


@router.post("/2fa/verify-otp", response_model=SuperAdminAuthResponse)
async def superadmin_2fa_verify_otp(
    request: OTPVerifyRequest,
    db: DatabaseManager = Depends(get_database),
):
    payload = verify_temp_token(request.temp_token, "OTP")
    admin_id = payload.get("uid")
    if not admin_id:
        raise HTTPException(status_code=401, detail="Invalid session")

    master_db = await get_master_db_or_503(db)
    admin_doc = await get_superadmin_by_id_or_401(master_db, admin_id)

    secret_enc = (admin_doc.get("two_fa") or {}).get("secret_enc")
    if not secret_enc:
        raise HTTPException(status_code=400, detail="2FA is not enabled for this account")

    secret = decrypt_secret(secret_enc)
    totp = pyotp.TOTP(secret)
    if not totp.verify(request.otp.strip(), valid_window=1):
        raise HTTPException(status_code=401, detail="Invalid verification code")

    await master_db["super_admins"].update_one(
        {"_id": admin_doc["_id"]},
        {"$set": {"two_fa.last_verified_at": datetime.utcnow()}}
    )

    admin_doc = await get_superadmin_by_id_or_401(master_db, admin_id)
    return await _complete_superadmin_login(master_db, admin_doc)


@router.post("/setup")
async def first_time_setup_disabled():
    raise HTTPException(
        status_code=410,
        detail="Super-admin setup via API is disabled. Use backend provisioning script instead."
    )


# ============ DASHBOARD ENDPOINTS ============

@router.get("/dashboard/stats")
async def get_dashboard_stats(
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    master_db = await get_master_db_or_503(db)

    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    owner_filter = {"assigned_superadmin_id": ObjectId(admin["admin_id"])}

    status_counts = await master_db["tenants"].aggregate([
        {"$match": owner_filter},
        {"$group": {"_id": "$status", "count": {"$sum": 1}}},
    ]).to_list(length=100)
    status_map = {item["_id"]: item["count"] for item in status_counts}

    week_count = await master_db["tenants"].count_documents({
        **owner_filter,
        "created_at": {"$gte": week_ago}
    })
    month_count = await master_db["tenants"].count_documents({
        **owner_filter,
        "created_at": {"$gte": month_ago}
    })

    total = sum(status_map.values())
    pending_count = status_map.get("pending", 0) + status_map.get("verification", 0)

    return {
        "total_tenants": total,
        "pending_registrations": pending_count,
        "active_tenants": status_map.get("active", 0),
        "suspended_tenants": status_map.get("suspended", 0),
        "rejected_registrations": status_map.get("rejected", 0),
        "total_students": 0,
        "total_tutors": 0,
        "registrations_this_week": week_count,
        "registrations_this_month": month_count,
    }


# ============ TENANT ENDPOINTS ============

@router.get("/tenants")
async def get_tenants(
    status: Optional[str] = None,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    master_db = await get_master_db_or_503(db)

    query: Dict[str, Any] = {"assigned_superadmin_id": ObjectId(admin["admin_id"])}
    if status:
        if status == "pending":
            query["status"] = {"$in": ["pending", "verification"]}
        else:
            query["status"] = status

    tenants = await master_db["tenants"].find(query).sort("created_at", -1).to_list(length=1000)

    result = []
    for tenant in tenants:
        tenant = convert_objectids(tenant)
        if "pending_admin" in tenant and tenant["pending_admin"]:
            tenant["pending_admin"].pop("password_hash", None)
        result.append(tenant)

    return result


@router.get("/features/catalog")
async def get_features_catalog(
    admin: Dict = Depends(verify_superadmin_token),
):
    _ = admin  # authenticated super-admin context
    return {
        "success": True,
        "version": 2,
        "categories": ["core", "advanced", "max"],
        "features": get_feature_catalog(),
    }


@router.get("/tenants/{tenant_id}/features-v2")
async def get_tenant_features_v2(
    tenant_id: str,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    master_db = await get_master_db_or_503(db)
    tenant = await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])
    features_v2 = _tenant_features_v2_from_doc(tenant)
    return {
        "success": True,
        "tenant_id": tenant_id,
        "features_v2": features_v2,
    }


@router.put("/tenants/{tenant_id}/features-v2")
async def update_tenant_features_v2(
    tenant_id: str,
    request: UpdateFeaturesV2Request,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    master_db = await get_master_db_or_503(db)
    tenant = await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    before_v2 = _tenant_features_v2_from_doc(tenant)
    after_v2 = build_enabled_features_v2(
        {
            "tier": request.tier,
            "overrides": _sanitize_v2_overrides(request.overrides),
        },
        tenant.get("enabled_features"),
    )
    changed_features = _feature_diff(before_v2.get("effective", {}), after_v2.get("effective", {}))

    feature_action = {
        "action": "feature_changed",
        "by": admin["email"],
        "at": datetime.utcnow(),
        "changes": after_v2.get("effective", {}),
        "details": {
            "tier_before": before_v2.get("tier"),
            "tier_after": after_v2.get("tier"),
            "changed_features": changed_features,
        },
        "notes": (
            f"Tier changed from {before_v2.get('tier')} to {after_v2.get('tier')}"
            if before_v2.get("tier") != after_v2.get("tier")
            else "Feature overrides updated"
        ),
    }

    legacy_export = export_legacy_features(after_v2.get("effective", {}))

    await master_db["tenants"].update_one(
        {"_id": ObjectId(tenant_id)},
        {
            "$set": {
                "enabled_features_v2": after_v2,
                "enabled_features": legacy_export,
            },
            "$push": {"approval_history": feature_action},
        },
    )

    return {
        "success": True,
        "tenant_id": tenant_id,
        "features_v2": after_v2,
        "changed_features": changed_features,
    }


@router.get("/tenants/{tenant_id}")
async def get_tenant_by_id(
    tenant_id: str,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    master_db = await get_master_db_or_503(db)
    tenant = await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    if tenant.get("status") == "pending":
        verification_at = datetime.utcnow()
        verification_action = {
            "action": "verification_started",
            "by": admin["email"],
            "at": verification_at,
            "notes": "Application opened for review",
        }
        update_result = await master_db["tenants"].update_one(
            {"_id": tenant["_id"], "status": "pending"},
            {
                "$set": {
                    "status": "verification",
                    "verification_started_at": verification_at,
                },
                "$push": {
                    "approval_history": verification_action,
                },
            }
        )
        if update_result.modified_count == 1:
            tenant["status"] = "verification"
            tenant["verification_started_at"] = verification_at
            history = tenant.get("approval_history") or []
            history.append(verification_action)
            tenant["approval_history"] = history

    tenant = convert_objectids(tenant)
    if "pending_admin" in tenant and tenant["pending_admin"]:
        tenant["pending_admin"].pop("password_hash", None)

    return tenant


@router.get("/tenants/{tenant_id}/files")
async def get_tenant_files(
    tenant_id: str,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    master_db = await get_master_db_or_503(db)
    await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    files = await master_db["tenant_application_files"].find({
        "tenant_request_id": ObjectId(tenant_id)
    }).to_list(length=100)

    return [convert_objectids(file) for file in files]


@router.post("/tenants/{tenant_id}/approve")
async def approve_tenant(
    tenant_id: str,
    request: ApproveTenantRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    master_db = await get_master_db_or_503(db)
    tenant = await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    if tenant["status"] not in ["pending", "verification"]:
        raise HTTPException(status_code=400, detail="Tenant is not in pending or verification status")

    normalized_institution_id = normalize_institution_id(request.institution_id)
    if not INSTITUTION_ID_PATTERN.match(normalized_institution_id):
        raise HTTPException(status_code=400, detail="Institution ID must match XXXX-0000 format")

    required_fields = {
        "institution_name": tenant.get("institution_name"),
        "organization": tenant.get("organization"),
        "admin_email": tenant.get("admin_email"),
        "admin_full_name": tenant.get("admin_full_name"),
        "contact_email": tenant.get("contact_email"),
    }
    missing = [field for field, value in required_fields.items() if not (value and str(value).strip())]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot approve tenant. Missing required fields: {', '.join(missing)}"
        )

    new_tenant_id = derive_tenant_id(normalized_institution_id)
    db_name = build_db_name(normalized_institution_id)

    existing = await master_db["tenants"].find_one({
        "_id": {"$ne": ObjectId(tenant_id)},
        "$or": [
            {"institution_id": normalized_institution_id},
            {"tenant_id": new_tenant_id},
            {"db_name": db_name},
        ]
    })
    if existing:
        raise HTTPException(status_code=400, detail="Institution ID or tenant ID already in use")

    default_features = _sanitize_legacy_features(request.features)
    default_features_v2 = build_enabled_features_v2(None, default_features)

    approval_action = {
        "action": "approved",
        "by": admin["email"],
        "at": datetime.utcnow(),
        "notes": request.notes,
    }

    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Tenant database unavailable")

    pending_admin = tenant.get("pending_admin") or {}
    pending_password_hash = pending_admin.get("password_hash")
    if not pending_password_hash:
        raise HTTPException(status_code=400, detail="Pending admin credentials missing")

    admin_email = tenant.get("admin_email")
    existing_admin = await tenant_db["admins"].find_one({"email": admin_email})
    if not existing_admin:
        admin_doc = {
            "email": admin_email,
            "password_hash": pending_password_hash,
            "full_name": pending_admin.get("full_name") or tenant.get("admin_full_name"),
            "role": "master_admin",
            "permissions": [],
            "is_active": True,
            "created_at": datetime.utcnow(),
            "created_by": None,
            "two_fa": pending_admin.get("two_fa") or {
                "enabled": False,
                "required": True,
                "secret_enc": None,
                "verified_at": None,
            }
        }
        await tenant_db["admins"].insert_one(admin_doc)

    await ensure_tenant_indexes(tenant_db)

    update_result = await master_db["tenants"].update_one(
        {"_id": ObjectId(tenant_id)},
        {
            "$set": {
                "status": "approved",
                "tenant_id": new_tenant_id,
                "db_name": db_name,
                "institution_id": normalized_institution_id,
                "subdomain": None,
                "approved_at": datetime.utcnow(),
                "enabled_features": default_features,
                "enabled_features_v2": default_features_v2,
                "max_students": 100,
                "max_tutors": 10,
                "subscription_tier": "standard",
            },
            "$push": {
                "approval_history": approval_action,
            }
        }
    )

    if update_result.modified_count != 1:
        raise HTTPException(status_code=500, detail="Failed to approve tenant")

    return {"success": True, "tenant_id": new_tenant_id, "institution_id": normalized_institution_id}


@router.post("/tenants/{tenant_id}/reject")
async def reject_tenant(
    tenant_id: str,
    request: RejectTenantRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    master_db = await get_master_db_or_503(db)
    await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    rejection_action = {
        "action": "rejected",
        "by": admin["email"],
        "at": datetime.utcnow(),
        "notes": request.reason,
    }

    await master_db["tenants"].update_one(
        {"_id": ObjectId(tenant_id)},
        {
            "$set": {
                "status": "rejected",
                "rejected_at": datetime.utcnow(),
                "rejection_reason": request.reason,
            },
            "$push": {
                "approval_history": rejection_action,
            }
        }
    )

    return {"success": True}


@router.post("/tenants/{tenant_id}/activate")
async def activate_tenant(
    tenant_id: str,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    master_db = await get_master_db_or_503(db)
    tenant = await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    if tenant["status"] not in ["approved", "suspended"]:
        raise HTTPException(status_code=400, detail="Tenant must be approved or suspended to activate")

    activation_action = {
        "action": "activated",
        "by": admin["email"],
        "at": datetime.utcnow(),
    }

    await master_db["tenants"].update_one(
        {"_id": ObjectId(tenant_id)},
        {
            "$set": {"status": "active"},
            "$push": {"approval_history": activation_action}
        }
    )

    return {"success": True}


@router.post("/tenants/{tenant_id}/suspend")
async def suspend_tenant(
    tenant_id: str,
    request: SuspendTenantRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    master_db = await get_master_db_or_503(db)
    await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    suspend_action = {
        "action": "suspended",
        "by": admin["email"],
        "at": datetime.utcnow(),
        "notes": request.reason,
    }

    await master_db["tenants"].update_one(
        {"_id": ObjectId(tenant_id)},
        {
            "$set": {"status": "suspended"},
            "$push": {"approval_history": suspend_action}
        }
    )

    return {"success": True}


@router.put("/tenants/{tenant_id}/features")
async def update_tenant_features(
    tenant_id: str,
    request: UpdateFeaturesRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    master_db = await get_master_db_or_503(db)
    tenant = await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    features_data = _sanitize_legacy_features(request.features)
    before_v2 = _tenant_features_v2_from_doc(tenant)
    after_v2 = build_enabled_features_v2(None, features_data)
    changed_features = _feature_diff(before_v2.get("effective", {}), after_v2.get("effective", {}))

    feature_action = {
        "action": "feature_changed",
        "by": admin["email"],
        "at": datetime.utcnow(),
        "changes": after_v2.get("effective", {}),
        "details": {
            "tier_before": before_v2.get("tier"),
            "tier_after": after_v2.get("tier"),
            "changed_features": changed_features,
            "source": "legacy_endpoint",
        },
    }

    await master_db["tenants"].update_one(
        {"_id": ObjectId(tenant_id)},
        {
            "$set": {
                "enabled_features": features_data,
                "enabled_features_v2": after_v2,
            },
            "$push": {"approval_history": feature_action}
        }
    )

    return {"success": True, "features_v2": after_v2, "changed_features": changed_features}


@router.put("/tenants/{tenant_id}/limits")
async def update_tenant_limits(
    tenant_id: str,
    request: UpdateLimitsRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    master_db = await get_master_db_or_503(db)
    await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    update_fields: Dict[str, Any] = {}
    if request.max_students is not None:
        update_fields["max_students"] = request.max_students
    if request.max_tutors is not None:
        update_fields["max_tutors"] = request.max_tutors
    if request.subscription_tier is not None:
        update_fields["subscription_tier"] = request.subscription_tier

    if update_fields:
        await master_db["tenants"].update_one(
            {"_id": ObjectId(tenant_id)},
            {"$set": update_fields}
        )

    return {"success": True}


@router.post("/tenants/{tenant_id}/reset-admin-password")
async def reset_tenant_admin_password(
    tenant_id: str,
    request: ResetPasswordRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
    auth_manager: Optional[AuthManager] = Depends(get_auth_manager),
):
    master_db = await get_master_db_or_503(db)
    tenant = await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    generated_password = generate_admin_reset_password()
    new_password_hash = (
        auth_manager.get_password_hash(generated_password)
        if auth_manager is not None
        else pwd_context.hash(generated_password)
    )
    now = datetime.utcnow()

    if tenant["status"] == "pending" and tenant.get("pending_admin"):
        await master_db["tenants"].update_one(
            {"_id": ObjectId(tenant_id)},
            {
                "$set": {
                    "pending_admin.password_hash": new_password_hash,
                    "pending_admin.password_reset_by_superadmin_at": now,
                },
                "$push": {
                    "approval_history": {
                        "action": "password_reset",
                        "by": admin["email"],
                        "at": now,
                        "notes": "Password reset by super admin",
                    }
                }
            }
        )
        return {"success": True, "generated_password": generated_password}

    if tenant["status"] in ["active", "approved"] and tenant.get("db_name"):
        tenant_db = await db.get_tenant_db(tenant["db_name"])
        if tenant_db:
            master_admin = await get_tenant_master_admin_or_error(tenant_db, tenant)
            await tenant_db["admins"].update_one(
                {"_id": master_admin["_id"]},
                {
                    "$set": {
                        "password_hash": new_password_hash,
                        "password_reset_by_superadmin_at": now,
                        "password_reset_by_superadmin_id": admin["admin_id"],
                    }
                }
            )
            await best_effort_revoke_admin_sessions(auth_manager, str(master_admin["_id"]))

            await master_db["tenants"].update_one(
                {"_id": ObjectId(tenant_id)},
                {
                    "$push": {
                        "approval_history": {
                            "action": "password_reset",
                            "by": admin["email"],
                            "at": now,
                            "notes": "Password reset by super admin",
                        }
                    }
                }
            )
            return {"success": True, "generated_password": generated_password}

    raise HTTPException(status_code=400, detail="Unable to reset password for this tenant")


@router.post("/tenants/{tenant_id}/reset-admin-2fa")
async def reset_tenant_admin_2fa(
    tenant_id: str,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
    auth_manager: Optional[AuthManager] = Depends(get_auth_manager),
):
    master_db = await get_master_db_or_503(db)
    tenant = await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])
    now = datetime.utcnow()

    if tenant["status"] == "pending" and tenant.get("pending_admin"):
        await master_db["tenants"].update_one(
            {"_id": ObjectId(tenant_id)},
            {
                "$set": {
                    "pending_admin.two_fa.enabled": False,
                    "pending_admin.two_fa.required": False,
                    "pending_admin.two_fa.secret_enc": None,
                    "pending_admin.two_fa.temp_secret_enc": None,
                    "pending_admin.two_fa.reset_by_superadmin_at": now,
                    "pending_admin.two_fa.reset_by_superadmin_id": admin["admin_id"],
                },
                "$push": {
                    "approval_history": {
                        "action": "two_fa_reset",
                        "by": admin["email"],
                        "at": now,
                        "notes": "2FA reset by super admin",
                    }
                }
            }
        )
        return {
            "success": True,
            "message": "Admin 2FA reset successfully. The admin can enable 2FA again from settings.",
        }

    if tenant["status"] in ["active", "approved"] and tenant.get("db_name"):
        tenant_db = await db.get_tenant_db(tenant["db_name"])
        if tenant_db:
            master_admin = await get_tenant_master_admin_or_error(tenant_db, tenant)
            await tenant_db["admins"].update_one(
                {"_id": master_admin["_id"]},
                {
                    "$set": {
                        "two_fa.enabled": False,
                        "two_fa.required": False,
                        "two_fa.secret_enc": None,
                        "two_fa.temp_secret_enc": None,
                        "two_fa.setup_started_at": None,
                        "two_fa.last_verified_at": None,
                        "two_fa.reset_by_superadmin_at": now,
                        "two_fa.reset_by_superadmin_id": admin["admin_id"],
                    }
                }
            )
            await best_effort_revoke_admin_sessions(auth_manager, str(master_admin["_id"]))

            await master_db["tenants"].update_one(
                {"_id": ObjectId(tenant_id)},
                {
                    "$push": {
                        "approval_history": {
                            "action": "two_fa_reset",
                            "by": admin["email"],
                            "at": now,
                            "notes": "2FA reset by super admin",
                        }
                    }
                }
            )
            return {
                "success": True,
                "message": "Admin 2FA reset successfully. The admin can enable 2FA again from settings.",
            }

    raise HTTPException(status_code=400, detail="Unable to reset 2FA for this tenant")


@router.put("/tenants/{tenant_id}/institution-id")
async def update_tenant_institution_id(
    tenant_id: str,
    request: UpdateTenantIdRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    master_db = await get_master_db_or_503(db)
    tenant = await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    new_id = normalize_institution_id(request.institution_id)
    new_tenant_id = derive_tenant_id(new_id)

    existing = await master_db["tenants"].find_one({
        "_id": {"$ne": ObjectId(tenant_id)},
        "$or": [
            {"institution_id": new_id},
            {"tenant_id": new_tenant_id},
        ]
    })
    if existing:
        raise HTTPException(status_code=400, detail="Institution ID or tenant ID already in use by another tenant")

    old_id = tenant.get("institution_id")

    update_action = {
        "action": "id_changed",
        "by": admin["email"],
        "at": datetime.utcnow(),
        "notes": f"Institution ID changed from {old_id} to {new_id}",
        "changes": {"old_id": old_id, "new_id": new_id}
    }

    update_fields = {
        "institution_id": new_id,
        "tenant_id": new_tenant_id,
    }
    if not tenant.get("db_name"):
        update_fields["db_name"] = build_db_name(new_id)

    await master_db["tenants"].update_one(
        {"_id": ObjectId(tenant_id)},
        {
            "$set": update_fields,
            "$push": {"approval_history": update_action}
        }
    )

    return {"success": True, "institution_id": new_id, "tenant_id": new_tenant_id}


@router.post("/tenants/{tenant_id}/messages")
async def send_message_to_tenant(
    tenant_id: str,
    subject: str = Form(..., min_length=1, max_length=200),
    message: str = Form(..., min_length=1),
    priority: str = Form("normal", pattern=r'^(low|normal|high|urgent)$'),
    attachments: List[UploadFile] = File(default=[]),
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    from utils.message_attachments import upload_message_attachments

    master_db = await get_master_db_or_503(db)
    tenant = await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    # Upload attachments to S3
    attachment_metadata = await upload_message_attachments(
        [f for f in attachments if f.filename]
    )

    message_doc = {
        "tenant_id": ObjectId(tenant_id),
        "superadmin_id": ObjectId(admin["admin_id"]),
        "authorization_code": (tenant.get("authorization_code_used") or "").strip().upper() or None,
        "from_admin": admin["email"],
        "from_name": admin["name"],
        "to_email": (tenant.get("admin_email") or "").strip().lower(),
        "subject": subject,
        "message": message,
        "priority": priority,
        "direction": "superadmin_to_admin",
        "attachments": attachment_metadata,
        "created_at": datetime.utcnow(),
        "read": False,
        "read_at": None,
    }

    result = await master_db["superadmin_messages"].insert_one(message_doc)

    return {
        "success": True,
        "message_id": str(result.inserted_id)
    }


@router.get("/tenants/{tenant_id}/messages")
async def get_tenant_messages(
    tenant_id: str,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    master_db = await get_master_db_or_503(db)
    await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    messages = await master_db["superadmin_messages"].find({
        "tenant_id": ObjectId(tenant_id),
        "superadmin_id": ObjectId(admin["admin_id"]),
    }).sort("created_at", -1).to_list(length=100)

    # Auto-mark unread admin-to-superadmin messages as read
    unread_ids = [
        msg["_id"] for msg in messages
        if msg.get("direction") == "admin_to_superadmin" and not msg.get("read")
    ]
    if unread_ids:
        await master_db["superadmin_messages"].update_many(
            {"_id": {"$in": unread_ids}},
            {"$set": {"read": True, "read_at": datetime.utcnow()}},
        )
        for msg in messages:
            if msg["_id"] in unread_ids:
                msg["read"] = True
                msg["read_at"] = datetime.utcnow()

    return [convert_objectids(msg) for msg in messages]


@router.delete("/tenants/{tenant_id}/messages/{message_id}")
async def delete_tenant_message(
    tenant_id: str,
    message_id: str,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    master_db = await get_master_db_or_503(db)
    await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    try:
        message_oid = ObjectId(message_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid message ID")

    result = await master_db["superadmin_messages"].delete_one({
        "_id": message_oid,
        "tenant_id": ObjectId(tenant_id),
        "superadmin_id": ObjectId(admin["admin_id"]),
    })

    if result.deleted_count != 1:
        raise HTTPException(status_code=404, detail="Message not found")

    return {"success": True, "deleted_message_id": message_id}


@router.get("/tenants/{tenant_id}/messages/{message_id}/attachments/{attachment_index}")
async def download_tenant_message_attachment(
    tenant_id: str,
    message_id: str,
    attachment_index: int,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Download an attachment from a tenant message (super-admin access)."""
    import base64
    from fastapi.responses import Response as RawResponse
    from utils.s3_storage import download_file as s3_download_file

    master_db = await get_master_db_or_503(db)
    await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    try:
        message_oid = ObjectId(message_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid message ID")

    msg = await master_db["superadmin_messages"].find_one({
        "_id": message_oid,
        "tenant_id": ObjectId(tenant_id),
    })
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")

    attachments = msg.get("attachments") or []
    if attachment_index < 0 or attachment_index >= len(attachments):
        raise HTTPException(status_code=404, detail="Attachment not found")

    att = attachments[attachment_index]
    filename = att.get("filename", "attachment")
    content_type = att.get("content_type", "application/octet-stream")

    # S3-stored attachment (new format)
    if att.get("storage_path"):
        file_data = await s3_download_file(att["storage_path"])
        if file_data is None:
            raise HTTPException(status_code=404, detail="Attachment file not found in storage")
        return RawResponse(
            content=file_data,
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # Legacy base64-encoded attachment
    if att.get("data_base64"):
        file_data = base64.b64decode(att["data_base64"])
        return RawResponse(
            content=file_data,
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    raise HTTPException(status_code=404, detail="Attachment has no file data")


@router.delete("/tenants/{tenant_id}")
async def delete_tenant(
    tenant_id: str,
    request: DeleteTenantRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    master_db = await get_master_db_or_503(db)
    tenant = await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    if request.confirmation.lower() != tenant["institution_name"].lower():
        raise HTTPException(status_code=400, detail="Confirmation text does not match institution name")

    logger.warning(
        "Super admin %s deleting tenant %s (%s)",
        admin["email"],
        tenant_id,
        tenant["institution_name"],
    )

    await master_db["tenant_application_files"].delete_many({
        "tenant_request_id": ObjectId(tenant_id)
    })

    await master_db["superadmin_messages"].delete_many({
        "tenant_id": ObjectId(tenant_id)
    })

    await master_db["tenants"].delete_one({"_id": ObjectId(tenant_id)})

    return {"success": True, "deleted": tenant["institution_name"]}


# ============ NOTIFICATIONS ============


@router.get("/notifications/unread")
async def get_unread_notifications(
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Return unread admin-to-superadmin messages for the notification bell."""
    master_db = await get_master_db_or_503(db)

    pipeline = [
        {
            "$match": {
                "superadmin_id": ObjectId(admin["admin_id"]),
                "direction": "admin_to_superadmin",
                "read": False,
            }
        },
        {"$sort": {"created_at": -1}},
        {"$limit": 50},
        {
            "$lookup": {
                "from": "tenants",
                "localField": "tenant_id",
                "foreignField": "_id",
                "as": "tenant_info",
            }
        },
        {
            "$project": {
                "message_id": {"$toString": "$_id"},
                "tenant_id": {"$toString": "$tenant_id"},
                "tenant_name": {"$arrayElemAt": ["$tenant_info.institution_name", 0]},
                "from_email": "$from_admin",
                "subject": 1,
                "created_at": 1,
            }
        },
    ]

    notifications = await master_db["superadmin_messages"].aggregate(pipeline).to_list(length=50)

    for n in notifications:
        n.pop("_id", None)
        if isinstance(n.get("created_at"), datetime):
            n["created_at"] = n["created_at"].isoformat()

    return {
        "unread_count": len(notifications),
        "notifications": notifications,
    }


@router.post("/notifications/mark-read")
async def mark_notifications_read(
    request: MarkNotificationsReadRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Mark notifications as read by message IDs or by tenant ID."""
    master_db = await get_master_db_or_503(db)

    query: Dict[str, Any] = {
        "superadmin_id": ObjectId(admin["admin_id"]),
        "direction": "admin_to_superadmin",
        "read": False,
    }

    if request.message_ids:
        try:
            oids = [ObjectId(mid) for mid in request.message_ids]
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid message ID format")
        query["_id"] = {"$in": oids}
    elif request.tenant_id:
        try:
            query["tenant_id"] = ObjectId(request.tenant_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid tenant ID format")
    else:
        raise HTTPException(status_code=400, detail="Provide message_ids or tenant_id")

    result = await master_db["superadmin_messages"].update_many(
        query,
        {"$set": {"read": True, "read_at": datetime.utcnow()}},
    )

    return {"success": True, "marked_count": result.modified_count}


# ============ TENANT STATS ============


@router.get("/tenants/{tenant_id}/stats")
async def get_tenant_stats(
    tenant_id: str,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Get detailed usage stats for a specific tenant."""
    master_db = await get_master_db_or_503(db)
    tenant = await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    db_name = tenant.get("db_name")
    if not db_name:
        raise HTTPException(status_code=400, detail="Tenant database not provisioned yet")

    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Could not connect to tenant database")

    now = datetime.utcnow()
    thirty_days_ago = now - timedelta(days=30)
    seven_days_ago = now - timedelta(days=7)

    # Gather counts in parallel via individual queries
    students_total = await tenant_db["students"].count_documents({})
    students_active_7d = await tenant_db["students"].count_documents(
        {"last_login": {"$gte": seven_days_ago}}
    )
    students_active_30d = await tenant_db["students"].count_documents(
        {"last_login": {"$gte": thirty_days_ago}}
    )

    tutors_total = await tenant_db["tutors"].count_documents({})
    tutors_active_7d = await tenant_db["tutors"].count_documents(
        {"last_login": {"$gte": seven_days_ago}}
    )
    tutors_active_30d = await tenant_db["tutors"].count_documents(
        {"last_login": {"$gte": thirty_days_ago}}
    )

    admins_total = await tenant_db["admins"].count_documents({})

    documents_count = await tenant_db["documents"].count_documents({})
    questions_count = await tenant_db["questions"].count_documents({})

    # Paper count (papers collection or distinct paper_id in questions)
    papers_count = 0
    try:
        papers_count = len(await tenant_db["questions"].distinct("paper_id"))
    except Exception:
        pass

    # Activity counts
    practice_total = 0
    practice_30d = 0
    try:
        practice_total = await tenant_db["practice_attempts"].count_documents({})
        practice_30d = await tenant_db["practice_attempts"].count_documents(
            {"created_at": {"$gte": thirty_days_ago}}
        )
    except Exception:
        pass

    mcq_total = 0
    mcq_30d = 0
    try:
        mcq_total = await tenant_db["test_attempts"].count_documents({})
        mcq_30d = await tenant_db["test_attempts"].count_documents(
            {"submitted_at": {"$gte": thirty_days_ago}}
        )
    except Exception:
        pass

    chat_total = 0
    chat_30d = 0
    try:
        chat_total = await tenant_db["chat_sessions"].count_documents({})
        chat_30d = await tenant_db["chat_sessions"].count_documents(
            {"created_at": {"$gte": thirty_days_ago}}
        )
    except Exception:
        pass

    smartboard_total = 0
    smartboard_30d = 0
    try:
        smartboard_total = await tenant_db["smartboard_sessions"].count_documents({})
        smartboard_30d = await tenant_db["smartboard_sessions"].count_documents(
            {"created_at": {"$gte": thirty_days_ago}}
        )
    except Exception:
        pass

    pen_total = 0
    pen_30d = 0
    try:
        pen_total = await tenant_db["strokes"].count_documents({})
        pen_30d = await tenant_db["strokes"].count_documents(
            {"timestamp": {"$gte": thirty_days_ago}}
        )
    except Exception:
        pass

    # Feature usage mapping
    features_v2 = build_enabled_features_v2(
        tenant.get("enabled_features_v2"),
        tenant.get("enabled_features"),
    )
    effective = features_v2.get("effective", {})

    feature_usage = {
        "learning_mode": {
            "enabled": bool(effective.get("student_learning_mode")),
            "active": practice_30d > 0,
        },
        "exam_mode": {
            "enabled": bool(effective.get("student_exam_mode")),
            "active": mcq_30d > 0,
        },
        "ai_mentor": {
            "enabled": bool(effective.get("student_ai_mentor")),
            "active": chat_30d > 0,
        },
        "online_class": {
            "enabled": bool(effective.get("tutor_online_class")),
            "active": False,  # Would need meetings collection check
        },
        "pen_capture": {
            "enabled": bool(effective.get("stoody_pen_capture")),
            "active": pen_30d > 0,
        },
        "paper_builder": {
            "enabled": bool(effective.get("admin_question_bank")),
            "active": questions_count > 0,
        },
        "video_lessons": {
            "enabled": bool(effective.get("student_video_lessons")),
            "active": False,  # Would need video views tracking
        },
    }

    max_students = tenant.get("max_students") or 0
    max_tutors = tenant.get("max_tutors") or 0
    student_util = round((students_total / max_students * 100), 1) if max_students > 0 else 0
    tutor_util = round((tutors_total / max_tutors * 100), 1) if max_tutors > 0 else 0

    enabled_count = sum(1 for v in effective.values() if v)
    total_count = len(effective)

    return {
        "tenant_id": str(tenant["_id"]),
        "institution_name": tenant.get("institution_name", ""),
        "snapshot_at": now.isoformat(),
        "limits": {
            "max_students": max_students,
            "max_tutors": max_tutors,
            "current_students": students_total,
            "current_tutors": tutors_total,
            "student_utilization_pct": student_util,
            "tutor_utilization_pct": tutor_util,
        },
        "users": {
            "students": {
                "total": students_total,
                "active_7d": students_active_7d,
                "active_30d": students_active_30d,
            },
            "tutors": {
                "total": tutors_total,
                "active_7d": tutors_active_7d,
                "active_30d": tutors_active_30d,
            },
            "admins": {"total": admins_total},
        },
        "content": {
            "documents": documents_count,
            "questions": questions_count,
            "papers": papers_count,
        },
        "activity": {
            "practice_attempts": {"total": practice_total, "last_30d": practice_30d},
            "mcq_attempts": {"total": mcq_total, "last_30d": mcq_30d},
            "chat_sessions": {"total": chat_total, "last_30d": chat_30d},
            "smartboard_sessions": {"total": smartboard_total, "last_30d": smartboard_30d},
            "pen_strokes": {"total": pen_total, "last_30d": pen_30d},
        },
        "feature_usage": feature_usage,
        "subscription_tier": features_v2.get("tier", "core"),
        "enabled_features_count": enabled_count,
        "total_features_count": total_count,
    }


@router.get("/stats/overview")
async def get_stats_overview(
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Aggregate stats overview across all tenants assigned to this super admin."""
    master_db = await get_master_db_or_503(db)

    tenants = await master_db["tenants"].find({
        "assigned_superadmin_id": ObjectId(admin["admin_id"]),
        "status": {"$in": ["active", "approved"]},
    }).to_list(length=500)

    total_students = 0
    total_tutors = 0
    top_by_students: List[Dict[str, Any]] = []
    top_by_activity: List[Dict[str, Any]] = []

    thirty_days_ago = datetime.utcnow() - timedelta(days=30)

    # Feature adoption tracking
    feature_keys = [
        "student_ai_mentor", "admin_question_bank", "tutor_online_class",
        "stoody_pen_capture", "student_video_lessons", "student_leaderboard_view",
    ]
    feature_enabled: Dict[str, int] = {k: 0 for k in feature_keys}
    feature_active: Dict[str, int] = {k: 0 for k in feature_keys}

    for tenant in tenants:
        db_name = tenant.get("db_name")
        if not db_name:
            continue

        try:
            tenant_db = await db.get_tenant_db(db_name)
            if tenant_db is None:
                continue

            s_count = await tenant_db["students"].count_documents({})
            t_count = await tenant_db["tutors"].count_documents({})
            total_students += s_count
            total_tutors += t_count

            attempts_30d = 0
            try:
                practice_30d = await tenant_db["practice_attempts"].count_documents(
                    {"created_at": {"$gte": thirty_days_ago}}
                )
                mcq_30d = await tenant_db["test_attempts"].count_documents(
                    {"submitted_at": {"$gte": thirty_days_ago}}
                )
                attempts_30d = practice_30d + mcq_30d
            except Exception:
                pass

            tenant_name = tenant.get("institution_name", "Unknown")
            tid = str(tenant["_id"])

            top_by_students.append({"tenant_id": tid, "name": tenant_name, "students": s_count})
            top_by_activity.append({"tenant_id": tid, "name": tenant_name, "attempts_30d": attempts_30d})

            # Feature adoption
            features_v2 = build_enabled_features_v2(
                tenant.get("enabled_features_v2"),
                tenant.get("enabled_features"),
            )
            effective = features_v2.get("effective", {})
            for fk in feature_keys:
                if effective.get(fk):
                    feature_enabled[fk] += 1
                    # Rough activity check
                    if fk == "student_ai_mentor" and attempts_30d > 0:
                        feature_active[fk] += 1
                    elif fk == "admin_question_bank":
                        try:
                            q = await tenant_db["questions"].count_documents({})
                            if q > 0:
                                feature_active[fk] += 1
                        except Exception:
                            pass
                    elif fk in ("tutor_online_class", "student_video_lessons", "student_leaderboard_view"):
                        if attempts_30d > 0:
                            feature_active[fk] += 1
                    elif fk == "stoody_pen_capture":
                        try:
                            sc = await tenant_db["strokes"].count_documents(
                                {"timestamp": {"$gte": thirty_days_ago}}
                            )
                            if sc > 0:
                                feature_active[fk] += 1
                        except Exception:
                            pass
        except Exception as e:
            logger.warning("Failed to gather stats for tenant %s: %s", tenant.get("institution_name"), e)
            continue

    top_by_students.sort(key=lambda x: x["students"], reverse=True)
    top_by_activity.sort(key=lambda x: x["attempts_30d"], reverse=True)

    total_active = len(tenants)
    feature_adoption = {}
    for fk in feature_keys:
        enabled = feature_enabled[fk]
        active = feature_active[fk]
        pct = round(active / enabled * 100, 1) if enabled > 0 else 0
        feature_adoption[fk] = {
            "enabled_count": enabled,
            "active_count": active,
            "pct": pct,
        }

    return {
        "total_active_tenants": total_active,
        "total_students": total_students,
        "total_tutors": total_tutors,
        "top_tenants_by_students": top_by_students[:5],
        "top_tenants_by_activity": top_by_activity[:5],
        "feature_adoption": feature_adoption,
    }


# ============ ACCOUNTS & PLATFORM COSTS ============


@router.get("/accounts-overview")
async def get_accounts_overview(
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Return all tenants for this super-admin with account overview data."""
    master_db = await get_master_db_or_503(db)

    tenants = await master_db["tenants"].find({
        "assigned_superadmin_id": ObjectId(admin["admin_id"]),
        "status": {"$in": ["active", "approved", "suspended"]},
    }).to_list(length=1000)

    accounts = []
    for tenant in tenants:
        features_v2 = _tenant_features_v2_from_doc(tenant)
        accounts.append({
            "tenant_id": str(tenant["_id"]),
            "institution_name": tenant.get("institution_name", ""),
            "institution_id": tenant.get("institution_id"),
            "status": tenant.get("status"),
            "subscription_tier": features_v2.get("tier", "core"),
            "max_students": tenant.get("max_students", 0),
            "max_tutors": tenant.get("max_tutors", 0),
            "admin_set_price": tenant.get("admin_set_price", 0),
            "admin_email": tenant.get("admin_email", ""),
        })

    return {"accounts": accounts}


@router.put("/tenants/{tenant_id}/pricing")
async def set_tenant_pricing(
    tenant_id: str,
    request: SetTenantPricingRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Set the price charged to a tenant by the super-admin distributor."""
    master_db = await get_master_db_or_503(db)
    await get_tenant_for_admin_or_error(master_db, tenant_id, admin["admin_id"])

    update_fields: Dict[str, Any] = {
        "admin_set_price": request.price,
    }
    if request.notes:
        update_fields["pricing_notes"] = request.notes

    await master_db["tenants"].update_one(
        {"_id": ObjectId(tenant_id)},
        {"$set": update_fields},
    )

    return {"success": True, "tenant_id": tenant_id, "price": request.price}


@router.get("/pricing")
async def get_superadmin_pricing(
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Get pricing configuration for the current super-admin."""
    master_db = await get_master_db_or_503(db)
    pricing = await _get_pricing(master_db, admin["admin_id"])
    # Attach timestamps from the raw doc
    raw_doc = await master_db["superadmin_pricing"].find_one(
        {"superadmin_id": ObjectId(admin["admin_id"])}
    )
    if raw_doc:
        pricing["created_at"] = raw_doc.get("created_at", "").isoformat() if isinstance(raw_doc.get("created_at"), datetime) else str(raw_doc.get("created_at", ""))
        pricing["updated_at"] = raw_doc.get("updated_at", "").isoformat() if isinstance(raw_doc.get("updated_at"), datetime) else str(raw_doc.get("updated_at", ""))
    else:
        pricing["created_at"] = None
        pricing["updated_at"] = None
    return pricing


@router.put("/pricing")
async def update_superadmin_pricing(
    request: SuperAdminPricingUpdate,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Upsert pricing configuration for the current super-admin."""
    master_db = await get_master_db_or_503(db)
    now = datetime.utcnow()

    update_fields: Dict[str, Any] = {"updated_at": now}
    data = _model_dump(request)

    # Deep-merge tiers: read existing, overlay provided
    if data.get("tiers") is not None:
        existing_pricing = await _get_pricing(master_db, admin["admin_id"])
        merged_tiers = {**existing_pricing.get("tiers", {})}
        for tier_name, rates in data["tiers"].items():
            if tier_name not in merged_tiers:
                merged_tiers[tier_name] = {}
            if isinstance(rates, dict):
                merged_tiers[tier_name] = {**merged_tiers[tier_name], **rates}
            else:
                merged_tiers[tier_name] = {**merged_tiers[tier_name], **_model_dump(rates)}
        update_fields["tiers"] = merged_tiers
    if data.get("superadmin_fee") is not None:
        fee = data["superadmin_fee"]
        update_fields["superadmin_fee"] = _model_dump(fee) if hasattr(fee, "model_dump") or hasattr(fee, "dict") else fee
    if data.get("currency") is not None:
        update_fields["currency"] = data["currency"]
    if data.get("currency_symbol") is not None:
        update_fields["currency_symbol"] = data["currency_symbol"]
    if data.get("billing_cycle") is not None:
        update_fields["billing_cycle"] = data["billing_cycle"]
    if data.get("billing_day") is not None:
        update_fields["billing_day"] = data["billing_day"]
    if data.get("notes") is not None:
        update_fields["notes"] = data["notes"]

    # Auto-resolve currency_symbol from CURRENCY_MAP if currency is set but symbol is not
    if "currency" in update_fields and "currency_symbol" not in update_fields:
        update_fields["currency_symbol"] = CURRENCY_MAP.get(update_fields["currency"], update_fields["currency"])

    # Remove old flat-rate fields if they exist (migration cleanup)
    remove_fields = {}
    existing_doc = await master_db["superadmin_pricing"].find_one(
        {"superadmin_id": ObjectId(admin["admin_id"])}
    )
    if existing_doc:
        for old_key in ("tier_rates", "flat_per_student", "flat_per_tutor", "flat_per_admin", "superadmin_base_fee"):
            if old_key in existing_doc:
                remove_fields[old_key] = ""

    await master_db["superadmin_pricing"].create_index(
        [("superadmin_id", 1)], unique=True, name="uniq_superadmin_pricing"
    )

    update_op: Dict[str, Any] = {
        "$set": update_fields,
        "$setOnInsert": {"created_at": now, "superadmin_id": ObjectId(admin["admin_id"])},
    }
    if remove_fields:
        update_op["$unset"] = remove_fields

    result = await master_db["superadmin_pricing"].update_one(
        {"superadmin_id": ObjectId(admin["admin_id"])},
        update_op,
        upsert=True,
    )

    return {
        "success": True,
        "upserted": result.upserted_id is not None,
        "modified": result.modified_count > 0,
    }


@router.get("/platform-costs")
async def get_platform_costs(
    db_manager: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Calculate platform costs for all active tenants based on per-SA pricing."""
    master_db = await get_master_db_or_503(db_manager)
    pricing = await _get_pricing(master_db, admin["admin_id"])
    cost_data = await _compute_total_cost(master_db, db_manager, admin["admin_id"], pricing)
    cycle = pricing.get("billing_cycle", "monthly")

    return {
        "platform_pricing": {
            "currency": pricing["currency"],
            "currency_symbol": pricing["currency_symbol"],
            "tiers": pricing["tiers"],
            "superadmin_fee": pricing["superadmin_fee"],
            "billing_cycle": cycle,
            "billing_day": pricing.get("billing_day", 1),
            "notes": pricing.get("notes", ""),
        },
        "billing_cycle": cycle,
        "tenant_costs": cost_data["tenant_costs"],
        "total_tenants_cost": cost_data["total_tenants_cost"],
        "superadmin_fee": cost_data["superadmin_fee"],
        "total_platform_cost": cost_data["total_platform_cost"],
        "period_label": cycle,
    }


# ============ PAYMENTS & BILLING ============


@router.post("/payments")
async def record_payment(
    request: RecordPaymentRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Record a payment made by this super-admin."""
    master_db = await get_master_db_or_503(db)
    now = datetime.utcnow()

    payment_date = now
    if request.payment_date:
        try:
            payment_date = datetime.fromisoformat(request.payment_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid payment_date format (use ISO 8601)")

    period_start = None
    period_end = None
    if request.period_start:
        try:
            period_start = datetime.fromisoformat(request.period_start)
        except ValueError:
            pass
    if request.period_end:
        try:
            period_end = datetime.fromisoformat(request.period_end)
        except ValueError:
            pass

    payment_doc = {
        "superadmin_id": ObjectId(admin["admin_id"]),
        "amount": request.amount,
        "currency": (await _get_pricing(master_db, admin["admin_id"])).get("currency", "USD"),
        "payment_date": payment_date,
        "payment_method": request.payment_method,
        "reference": request.reference,
        "period_start": period_start,
        "period_end": period_end,
        "notes": request.notes,
        "recorded_by": admin["email"],
        "created_at": now,
    }

    result = await master_db["superadmin_payments"].insert_one(payment_doc)
    return {"success": True, "payment_id": str(result.inserted_id)}


@router.get("/payments")
async def list_payments(
    skip: int = 0,
    limit: int = 50,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """List payments for this super-admin, newest first."""
    master_db = await get_master_db_or_503(db)

    payments = await master_db["superadmin_payments"].find(
        {"superadmin_id": ObjectId(admin["admin_id"])}
    ).sort("payment_date", -1).skip(skip).limit(limit).to_list(length=limit)

    total = await master_db["superadmin_payments"].count_documents(
        {"superadmin_id": ObjectId(admin["admin_id"])}
    )

    return {
        "payments": [convert_objectids(p) for p in payments],
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/billing-summary")
async def get_billing_summary(
    db_manager: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Current billing period summary: cost, paid, balance, next due date."""
    master_db = await get_master_db_or_503(db_manager)
    pricing = await _get_pricing(master_db, admin["admin_id"])
    cost_data = await _compute_total_cost(master_db, db_manager, admin["admin_id"], pricing)

    cycle = pricing.get("billing_cycle", "monthly")
    billing_day = pricing.get("billing_day", 1)
    now = datetime.utcnow()

    period_start, period_end, next_due = _compute_billing_period(now, cycle, billing_day)

    # Sum payments in current period
    paid_pipeline = [
        {"$match": {
            "superadmin_id": ObjectId(admin["admin_id"]),
            "payment_date": {"$gte": period_start, "$lte": period_end},
        }},
        {"$group": {"_id": None, "total": {"$sum": "$amount"}}},
    ]
    paid_result = await master_db["superadmin_payments"].aggregate(paid_pipeline).to_list(1)
    paid_this_period = paid_result[0]["total"] if paid_result else 0.0

    # Total paid all time
    all_time_pipeline = [
        {"$match": {"superadmin_id": ObjectId(admin["admin_id"])}},
        {"$group": {"_id": None, "total": {"$sum": "$amount"}}},
    ]
    all_time_result = await master_db["superadmin_payments"].aggregate(all_time_pipeline).to_list(1)
    total_paid_all_time = all_time_result[0]["total"] if all_time_result else 0.0

    current_period_cost = cost_data["total_platform_cost"]
    balance_due = round(current_period_cost - paid_this_period, 2)

    return {
        "billing_cycle": cycle,
        "billing_day": billing_day,
        "period_start": period_start.strftime("%Y-%m-%d"),
        "period_end": period_end.strftime("%Y-%m-%d"),
        "current_period_cost": current_period_cost,
        "paid_this_period": round(paid_this_period, 2),
        "balance_due": balance_due,
        "total_paid_all_time": round(total_paid_all_time, 2),
        "next_due_date": next_due.strftime("%Y-%m-%d"),
        "currency": pricing.get("currency", "USD"),
        "currency_symbol": pricing.get("currency_symbol", "$"),
    }


# ============================================================================
# EXAMPEN HUB PROVISIONING & GATE ADMIN (Tasks 6-7)
# ============================================================================

# Hub provisioning is handled exclusively by hub_ops_async.py at
# POST /api/v1/hubs/provision (admin JWT auth). The super-admin's role
# is code generation only (POST /api/v1/superadmin/evalpen/hubs/provision-code).
# The competing hub_provision_router that was here has been removed to
# eliminate the dual-implementation conflict.  The variable is set to
# None so that existing imports in main_async.py do not break.
hub_provision_router = None


# ---- Pydantic models for ExamPen endpoints ----

class HubProvisionCodeRequest(BaseModel):
    institution_id: str = Field(..., min_length=9, max_length=9, pattern=r'^[A-Z]{4}-[0-9]{4}$')
    note: Optional[str] = None


class GateConfigUpdateRequest(BaseModel):
    daily_token_limit: Optional[int] = Field(None, ge=1)
    weekly_token_limit: Optional[int] = Field(None, ge=1)
    monthly_token_limit: Optional[int] = Field(None, ge=1)


# ---- Task 6: Hub Provisioning ----

@router.post("/evalpen/hubs/provision-code")
async def generate_hub_provision_code(
    request: HubProvisionCodeRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Generate a single-use provisioning code for a hub to register against a tenant."""
    master_db = await get_master_db_or_503(db)

    institution_id = normalize_institution_id(request.institution_id)

    # Verify the tenant exists
    tenant = await master_db["tenants"].find_one({
        "institution_id": institution_id,
        "status": {"$in": ["active", "approved"]},
    })
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found or not active")

    # Generate 12-char alphanumeric code
    code = secrets.token_urlsafe(9)[:12].upper()
    expires_at = datetime.utcnow() + timedelta(hours=24)

    await master_db["exampen_hub_provision_codes"].insert_one({
        "code": code,
        "institution_id": institution_id,
        "note": request.note,
        "created_by": admin["admin_id"],
        "created_at": datetime.utcnow(),
        "expires_at": expires_at,
        "used": False,
    })

    return {
        "code": code,
        "expires_at": expires_at.isoformat(),
        "institution_id": institution_id,
    }


@router.get("/evalpen/hubs")
async def list_provisioned_hubs(
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """List all provisioned ExamPen hubs."""
    master_db = await get_master_db_or_503(db)

    hubs_cursor = master_db["exampen_hubs"].find({}).sort("provisioned_at", -1)
    hubs = await hubs_cursor.to_list(length=1000)

    return {
        "hubs": [
            {
                "hub_id": h["hub_id"],
                "institution_id": h["institution_id"],
                "status": h.get("status", "unknown"),
                "last_seen_at": h["last_seen_at"].isoformat() if h.get("last_seen_at") else None,
                "provisioned_at": h["provisioned_at"].isoformat() if h.get("provisioned_at") else None,
            }
            for h in hubs
        ]
    }


# ---- Task 7: Super-admin Gate Admin ----

async def _get_exampen_tenants(master_db) -> List[Dict[str, Any]]:
    """Return tenants whose effective feature state enables ExamPen."""
    tenants = await master_db["tenants"].find({
        "status": {"$in": ["active", "approved"]},
    }).to_list(length=1000)
    return [
        tenant
        for tenant in tenants
        if _tenant_features_v2_from_doc(tenant).get("effective", {}).get("exampen", False)
    ]


@router.get("/evalpen/gate/tenants")
async def list_gate_tenants(
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """List tenants with ExamPen enabled and their LLM gate config."""
    master_db = await get_master_db_or_503(db)
    tenants = await _get_exampen_tenants(master_db)

    result = []
    for tenant in tenants:
        institution_id = tenant.get("institution_id", "")
        db_name = tenant.get("db_name", "")

        gate_config = None
        if db_name:
            try:
                tenant_db = await db.get_tenant_db(db_name)
                if tenant_db is not None:
                    config_doc = await tenant_db["llm_gate_config"].find_one({"_id": "gate_config"})
                    if config_doc:
                        config_doc.pop("_id", None)
                        gate_config = convert_objectids(config_doc)
            except Exception:
                logger.warning("Failed to fetch gate config for tenant %s", db_name)

        result.append({
            "institution_id": institution_id,
            "db_name": db_name,
            "gate_config": gate_config,
        })

    return {"tenants": result}


@router.put("/evalpen/gate/tenants/{institution_id}/config")
async def update_gate_config(
    institution_id: str,
    request: GateConfigUpdateRequest,
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Override LLM gate config for a specific ExamPen-enabled tenant."""
    master_db = await get_master_db_or_503(db)
    institution_id = normalize_institution_id(institution_id)

    tenant = await master_db["tenants"].find_one({
        "institution_id": institution_id,
        "status": {"$in": ["active", "approved"]},
    })
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found or not active")

    db_name = tenant.get("db_name")
    if not db_name:
        raise HTTPException(status_code=404, detail="Tenant has no database configured")

    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(status_code=503, detail="Could not connect to tenant database")

    # Build the $set payload from provided fields only
    update_fields: Dict[str, Any] = {"updated_at": datetime.utcnow()}
    if request.daily_token_limit is not None:
        update_fields["daily_token_limit"] = request.daily_token_limit
    if request.weekly_token_limit is not None:
        update_fields["weekly_token_limit"] = request.weekly_token_limit
    if request.monthly_token_limit is not None:
        update_fields["monthly_token_limit"] = request.monthly_token_limit

    await tenant_db["llm_gate_config"].update_one(
        {"_id": "gate_config"},
        {"$set": update_fields},
        upsert=True,
    )

    return {
        "success": True,
        "institution_id": institution_id,
        "updated_fields": {k: v for k, v in update_fields.items() if k != "updated_at"},
    }


@router.get("/evalpen/gate/usage/aggregate")
async def aggregate_gate_usage(
    db: DatabaseManager = Depends(get_database),
    admin: Dict = Depends(verify_superadmin_token),
):
    """Cross-tenant usage summary for all ExamPen-enabled tenants."""
    master_db = await get_master_db_or_503(db)
    tenants = await _get_exampen_tenants(master_db)

    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    total_tokens_today = 0
    total_cost_today_usd = 0.0
    by_tenant = []

    for tenant in tenants:
        institution_id = tenant.get("institution_id", "")
        db_name = tenant.get("db_name", "")

        tenant_tokens = 0
        tenant_cost = 0.0

        if db_name:
            try:
                tenant_db = await db.get_tenant_db(db_name)
                if tenant_db is not None:
                    pipeline = [
                        {"$match": {"called_at": {"$gte": today_start}}},
                        {"$group": {
                            "_id": None,
                            "total_tokens": {"$sum": "$total_tokens"},
                            "total_cost_usd": {"$sum": "$estimated_cost_usd"},
                        }},
                    ]
                    results = await tenant_db["llm_token_usage_log"].aggregate(
                        pipeline
                    ).to_list(length=1)
                    if results:
                        tenant_tokens = int(results[0].get("total_tokens", 0))
                        tenant_cost = float(results[0].get("total_cost_usd", 0.0))
            except Exception:
                logger.warning(
                    "Failed to aggregate gate usage for tenant %s", db_name
                )

        total_tokens_today += tenant_tokens
        total_cost_today_usd += tenant_cost

        by_tenant.append({
            "institution_id": institution_id,
            "db_name": db_name,
            "tokens_today": tenant_tokens,
            "cost_today_usd": round(tenant_cost, 6),
        })

    return {
        "total_tokens_today": total_tokens_today,
        "total_cost_today_usd": round(total_cost_today_usd, 6),
        "by_tenant": by_tenant,
    }
