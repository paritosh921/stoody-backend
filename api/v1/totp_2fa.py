"""
TOTP Two-Factor Authentication Plugin for Stoody
Google Authenticator compatible 2FA using PyOTP

This module provides:
- TOTP secret generation and QR code provisioning
- 2FA setup flow for first-time users
- OTP verification for returning users
- Rate limiting to prevent brute force attacks
- Encrypted secret storage

Flow:
1. User logs in with username/password
2. If 2FA not enabled but required -> SETUP_2FA (show QR + instructions)
3. If 2FA enabled -> OTP (verify 6-digit code)
4. After verification -> Access token (6h expiry)
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import pyotp
from cryptography.fernet import Fernet, InvalidToken
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from fastapi import APIRouter, Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from bson import ObjectId

from core.database import DatabaseManager
from core.auth import AuthManager
from core.tenant_registry import (
    get_tenant_by_subdomain,
    get_tenant_by_tenant_id,
    normalize_tenant_id,
)
from core.tenant_features import merge_tenant_features, is_feature_enabled
from config_async import settings, JWT_SECRET_KEY

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()
limiter = Limiter(key_func=get_remote_address)

# ============================================================================
# Configuration
# ============================================================================

# TOTP encryption key - MUST be set in production via environment variable
# Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
TOTP_ENC_KEY = os.getenv("TOTP_ENC_KEY", "")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", JWT_SECRET_KEY)

# 6-hour session expiry for 2FA users (in minutes)
TWO_FA_TOKEN_EXPIRE_MINUTES = 360  # 6 hours

# Temp token expiry (for the time between password verification and OTP entry)
TEMP_TOKEN_MAX_AGE_SECONDS = 600  # 10 minutes

# TOTP issuer name (appears in Google Authenticator)
TOTP_ISSUER = "Stoody"

TENANT_ID_PATTERN = r'^[A-Za-z]{4}-?[0-9]{4}$'

# Legacy accounts exempt from 2FA (old system compatibility)
# These accounts will use regular login without 2FA requirement
TWO_FA_EXEMPT_EMAILS = [
    "cielknowledge@gmail.com",  # Legacy admin account
]


# ============================================================================
# Encryption helpers
# ============================================================================

def _get_fernet() -> Optional[Fernet]:
    """Get Fernet cipher for encrypting/decrypting TOTP secrets"""
    if not TOTP_ENC_KEY:
        logger.warning("TOTP_ENC_KEY not set - 2FA secrets will not be encrypted!")
        return None
    try:
        return Fernet(TOTP_ENC_KEY.encode())
    except Exception as e:
        logger.error(f"Invalid TOTP_ENC_KEY: {e}")
        return None


def encrypt_secret(secret: str) -> str:
    """Encrypt a TOTP secret for storage"""
    fernet = _get_fernet()
    if fernet:
        return fernet.encrypt(secret.encode()).decode()
    return secret  # Fallback to plaintext (NOT recommended for production)


def decrypt_secret(encrypted: str) -> str:
    """Decrypt a stored TOTP secret"""
    fernet = _get_fernet()
    if fernet:
        try:
            return fernet.decrypt(encrypted.encode()).decode()
        except InvalidToken:
            logger.error("Failed to decrypt TOTP secret - invalid token")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="2FA configuration error"
            )
    return encrypted  # Assume plaintext if no encryption key


# ============================================================================
# Temp token helpers (for secure handoff between password and OTP steps)
# ============================================================================

def _get_serializer() -> URLSafeTimedSerializer:
    """Get serializer for creating temp tokens"""
    return URLSafeTimedSerializer(APP_SECRET_KEY)


def create_temp_token(
    user_id: str,
    user_type: str,
    purpose: str,
    tenant_id: Optional[str] = None,
    db_name: Optional[str] = None,
    institution_id: Optional[str] = None,
    enabled_features: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a short-lived temp token for 2FA flow.
    Purpose can be 'SETUP' or 'OTP'
    """
    return _get_serializer().dumps({
        "uid": user_id,
        "user_type": user_type,
        "purpose": purpose,
        "tenant_id": tenant_id,
        "db_name": db_name,
        "institution_id": institution_id,
        "enabled_features": enabled_features,
    })


def verify_temp_token(token: str, expected_purpose: str) -> Dict[str, Any]:
    """
    Verify and decode a temp token.
    Raises HTTPException if invalid or expired.
    """
    try:
        payload = _get_serializer().loads(token, max_age=TEMP_TOKEN_MAX_AGE_SECONDS)
        if payload.get("purpose") != expected_purpose:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token purpose"
            )
        return payload
    except SignatureExpired:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please log in again."
        )
    except BadSignature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session token"
        )


# ============================================================================
# Pydantic Request/Response Models
# ============================================================================

class LoginRequest(BaseModel):
    """Login request with username/email and password"""
    username: str = Field(..., min_length=1, description="Username or email")
    password: str = Field(..., min_length=1)
    user_type: str = Field(default="admin", description="'admin' or 'tutor'")
    tenant_id: Optional[str] = Field(None, pattern=TENANT_ID_PATTERN)


class TempTokenRequest(BaseModel):
    """Request containing a temp token"""
    temp_token: str


class OTPVerifyRequest(BaseModel):
    """Request to verify OTP code"""
    temp_token: str
    otp: str = Field(..., min_length=6, max_length=6)


class LoginResponse(BaseModel):
    """Response from login endpoint"""
    success: bool
    next: str  # 'DONE', 'SETUP_2FA', or 'OTP'
    temp_token: Optional[str] = None  # For 2FA flows
    access_token: Optional[str] = None  # When login complete
    user: Optional[Dict[str, Any]] = None


class SetupStartResponse(BaseModel):
    """Response from 2FA setup start"""
    success: bool
    otpauth_url: str
    message: str = "Scan the QR code with Google Authenticator"


class SetupVerifyResponse(BaseModel):
    """Response from 2FA setup verification"""
    success: bool
    access_token: str
    user: Dict[str, Any]
    message: str = "2FA enabled successfully"


class OTPVerifyResponse(BaseModel):
    """Response from OTP verification"""
    success: bool
    access_token: str
    user: Dict[str, Any]


class TwoFAStatusResponse(BaseModel):
    """Response with 2FA status"""
    success: bool
    two_fa_enabled: bool
    two_fa_required: bool


# ============================================================================
# Dependency Injection
# ============================================================================

async def get_database(request: Request) -> DatabaseManager:
    return request.app.state.db


async def get_auth_manager(request: Request) -> AuthManager:
    return request.app.state.auth


# ============================================================================
# Helper Functions
# ============================================================================

def _get_request_subdomain(request: Request) -> Optional[str]:
    return getattr(request.state, "subdomain", None)

async def _resolve_tenant_for_auth(
    db: DatabaseManager,
    request: Request,
    tenant_id: Optional[str],
    require_active: bool = True,
) -> Dict[str, Any]:
    subdomain = _get_request_subdomain(request)
    tenant = None
    if subdomain:
        tenant = await get_tenant_by_subdomain(db, subdomain, include_inactive=not require_active)
    if not tenant and tenant_id:
        tenant = await get_tenant_by_tenant_id(
            db,
            tenant_id,
            include_inactive=not require_active
        )
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant not found for this request"
        )
    if require_active and tenant.get("status") != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant is not active"
        )
    return tenant

async def _get_tenant_db_or_503(db: DatabaseManager, tenant: Dict[str, Any]):
    db_name = tenant.get("db_name")
    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant database not available"
        )
    return tenant_db

async def get_user_by_id(tenant_db, user_id: str, user_type: str) -> Optional[Dict]:
    """Get user by ID and type from tenant DB"""
    collection = "admins" if user_type == "admin" else "tutors"
    try:
        return await tenant_db[collection].find_one({"_id": ObjectId(user_id)})
    except Exception as e:
        logger.error(f"Error fetching user: {e}")
        return None


async def update_user_2fa(tenant_db, user_id: str, user_type: str,
                          update_data: Dict[str, Any]) -> bool:
    """Update user's 2FA fields in tenant DB"""
    collection = "admins" if user_type == "admin" else "tutors"
    try:
        result = await tenant_db[collection].update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"Error updating user 2FA: {e}")
        return False


def _build_user_response(user: Dict[str, Any], user_id: str, user_type: str,
                         payload: Optional[Dict[str, Any]] = None,
                         tenant: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a complete user response dict, including tutor-specific fields when applicable."""
    response = {
        "user_id": user_id,
        "user_type": user_type,
        "email": user.get("email"),
        "username": user.get("username"),
        "full_name": user.get("name") or user.get("full_name"),
        "subdomain": user.get("subdomain"),
        "tenant_id": (payload or {}).get("tenant_id") or (tenant or {}).get("tenant_id"),
        "db_name": (payload or {}).get("db_name") or (tenant or {}).get("db_name"),
        "institution_id": (payload or {}).get("institution_id") or (tenant or {}).get("institution_id"),
        "enabled_features": merge_tenant_features(
            (payload or {}).get("enabled_features") or (tenant or {}).get("enabled_features")
        ),
    }

    if user_type == "admin":
        response["admin_role"] = user.get("role", "master_admin")
        response["permissions"] = user.get("permissions") or []
    elif user_type == "tutor":
        # Use the custom tutor_id from the document, not the MongoDB _id
        response["tutor_id"] = user.get("tutor_id") or user_id
        response["can_edit_students"] = user.get("can_edit_students", False)
        response["standards"] = user.get("standards") or []
        response["sections"] = user.get("sections") or []
        response["subjects"] = user.get("subjects") or []
        response["plan_types"] = user.get("plan_types") or []
        response["teaching_assignments"] = user.get("teaching_assignments") or []
        response["requires_password_change"] = user.get("requires_password_change", False)

    return response


def _build_user_token_data(user: Dict[str, Any], user_id: str, user_type: str,
                           payload: Optional[Dict[str, Any]] = None,
                           tenant: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build token data dict for creating access tokens."""
    data = {
        "user_id": user_id,
        "user_type": user_type,
        "email": user.get("email"),
        "username": user.get("username"),
        "full_name": user.get("name") or user.get("full_name"),
        "subdomain": user.get("subdomain"),
        "tenant_id": (payload or {}).get("tenant_id") or (tenant or {}).get("tenant_id"),
        "db_name": (payload or {}).get("db_name") or (tenant or {}).get("db_name"),
        "institution_id": (payload or {}).get("institution_id") or (tenant or {}).get("institution_id"),
        "enabled_features": merge_tenant_features(
            (payload or {}).get("enabled_features") or (tenant or {}).get("enabled_features")
        ),
    }

    if user_type == "admin":
        data["admin_id"] = user_id
        data["admin_role"] = user.get("role", "master_admin")
        data["permissions"] = user.get("permissions") or []
    elif user_type == "tutor":
        data["admin_id"] = str(user.get("created_by", ""))
        # Use the custom tutor_id from the document, not the MongoDB _id
        data["tutor_id"] = user.get("tutor_id") or user_id

    return data


def create_6h_access_token(auth_manager: AuthManager, user_data: Dict[str, Any]) -> str:
    """Create an access token with 6-hour expiry for 2FA users"""
    token_data = {
        "sub": str(user_data.get("user_id") or user_data.get("_id")),
        "user_type": user_data.get("user_type"),
        "email": user_data.get("email"),
        "username": user_data.get("username"),
        "admin_id": user_data.get("admin_id"),
        "subdomain": user_data.get("subdomain"),
        "tutor_id": user_data.get("tutor_id"),
        "tenant_id": user_data.get("tenant_id"),
        "db_name": user_data.get("db_name"),
        "institution_id": user_data.get("institution_id"),
        "admin_role": user_data.get("admin_role"),
        "permissions": user_data.get("permissions"),
        "enabled_features": user_data.get("enabled_features"),
    }
    return auth_manager.create_access_token(
        token_data, 
        expires_delta=timedelta(minutes=TWO_FA_TOKEN_EXPIRE_MINUTES)
    )


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/login-2fa", response_model=LoginResponse)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def login_with_2fa(
    request: Request,
    login_data: LoginRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Login endpoint with 2FA support.
    
    Returns one of:
    - next='DONE' + access_token: Login complete (2FA not required)
    - next='SETUP_2FA' + temp_token: User needs to set up 2FA
    - next='OTP' + temp_token: User needs to enter OTP
    """
    try:
        user_type = login_data.user_type.lower()
        tenant_id = normalize_tenant_id(login_data.tenant_id) if login_data.tenant_id else None
        tenant = await _resolve_tenant_for_auth(db, request, tenant_id)
        tenant_db = await _get_tenant_db_or_503(db, tenant)
        enabled_features = merge_tenant_features(tenant.get("enabled_features"))
        
        if user_type == "admin":
            # Authenticate admin
            admin_data = await auth_manager.authenticate_admin(
                login_data.username, login_data.password, db, tenant_db
            )
            if not admin_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )
            user = await tenant_db["admins"].find_one({"email": login_data.username})
        elif user_type == "tutor":
            if not is_feature_enabled(enabled_features, "tutor_panel"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Tutor panel is disabled for this institution"
                )
            # Authenticate tutor
            tutor_data = await auth_manager.authenticate_tutor(
                login_data.username, login_data.password, db, tenant_db
            )
            if not tutor_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password"
                )
            normalized_username = login_data.username.strip()
            username_lower = normalized_username.lower()
            user = await tenant_db["tutors"].find_one({"username_lower": username_lower})
            if not user:
                user = await tenant_db["tutors"].find_one(
                    {"username": normalized_username},
                    collation={"locale": "en", "strength": 2}
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user_type. Must be 'admin' or 'tutor'"
            )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        user_id = str(user["_id"])
        user_email = (user.get("email") or "").lower()  # Handle None email
        two_fa = user.get("two_fa", {})
        
        # Check if account is exempt from 2FA (legacy accounts)
        is_exempt = user_email in [e.lower() for e in TWO_FA_EXEMPT_EMAILS]
        
        # Check 2FA status
        two_fa_enabled = two_fa.get("enabled", False)
        two_fa_required = two_fa.get("required", True) and not is_exempt  # Exempt accounts skip 2FA
        
        if two_fa_required and not two_fa_enabled:
            # User needs to set up 2FA
            temp_token = create_temp_token(
                user_id,
                user_type,
                "SETUP",
                tenant_id=tenant.get("tenant_id"),
                db_name=tenant.get("db_name"),
                institution_id=tenant.get("institution_id"),
                enabled_features=enabled_features,
            )
            return LoginResponse(
                success=True,
                next="SETUP_2FA",
                temp_token=temp_token
            )
        
        if two_fa_enabled and not is_exempt:
            # User needs to enter OTP
            temp_token = create_temp_token(
                user_id,
                user_type,
                "OTP",
                tenant_id=tenant.get("tenant_id"),
                db_name=tenant.get("db_name"),
                institution_id=tenant.get("institution_id"),
                enabled_features=enabled_features,
            )
            return LoginResponse(
                success=True,
                next="OTP",
                temp_token=temp_token
            )
        
        # 2FA not required and not enabled - direct login
        user_data = _build_user_token_data(user, user_id, user_type, tenant=tenant)
        
        session_data = await auth_manager.create_user_session(user_data)
        
        # Build complete user response with role-specific fields
        user_response = _build_user_response(user, user_id, user_type, tenant=tenant)
        
        return LoginResponse(
            success=True,
            next="DONE",
            access_token=session_data["access_token"],
            user=user_response
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"2FA login error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/setup/start", response_model=SetupStartResponse)
@limiter.limit("10/minute")
async def start_2fa_setup(
    request: Request,
    data: TempTokenRequest,
    db: DatabaseManager = Depends(get_database)
):
    """
    Start 2FA setup process.
    Generates a new TOTP secret and returns the otpauth URL for QR code generation.
    """
    try:
        payload = verify_temp_token(data.temp_token, "SETUP")
        user_id = payload["uid"]
        user_type = payload["user_type"]

        tenant_db = None
        if payload.get("db_name"):
            tenant_db = await db.get_tenant_db(payload.get("db_name"))
        if tenant_db is None:
            tenant = await _resolve_tenant_for_auth(
                db,
                request,
                payload.get("tenant_id"),
                require_active=False
            )
            tenant_db = await _get_tenant_db_or_503(db, tenant)

        user = await get_user_by_id(tenant_db, user_id, user_type)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Generate new TOTP secret
        secret = pyotp.random_base32()
        
        # Create otpauth URL for QR code
        label = user.get("email") or user.get("username") or "user"
        totp = pyotp.TOTP(secret)
        otpauth_url = totp.provisioning_uri(name=label, issuer_name=TOTP_ISSUER)
        
        # Store temp secret (encrypted)
        await update_user_2fa(tenant_db, user_id, user_type, {
            "two_fa.temp_secret_enc": encrypt_secret(secret),
            "two_fa.setup_started_at": datetime.utcnow()
        })
        
        return SetupStartResponse(
            success=True,
            otpauth_url=otpauth_url
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"2FA setup start error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start 2FA setup"
        )


@router.post("/setup/verify", response_model=SetupVerifyResponse)
@limiter.limit("5/minute")  # Rate limit to prevent brute force
async def verify_2fa_setup(
    request: Request,
    data: OTPVerifyRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Complete 2FA setup by verifying the first OTP code.
    If successful, enables 2FA and returns an access token.
    """
    try:
        payload = verify_temp_token(data.temp_token, "SETUP")
        user_id = payload["uid"]
        user_type = payload["user_type"]

        tenant_db = None
        if payload.get("db_name"):
            tenant_db = await db.get_tenant_db(payload.get("db_name"))
        if tenant_db is None:
            tenant = await _resolve_tenant_for_auth(
                db,
                request,
                payload.get("tenant_id"),
                require_active=False
            )
            tenant_db = await _get_tenant_db_or_503(db, tenant)

        user = await get_user_by_id(tenant_db, user_id, user_type)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get temp secret
        temp_secret_enc = user.get("two_fa", {}).get("temp_secret_enc")
        if not temp_secret_enc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA setup not started. Please start setup first."
            )
        
        secret = decrypt_secret(temp_secret_enc)
        
        # Verify OTP
        totp = pyotp.TOTP(secret)
        if not totp.verify(data.otp.strip(), valid_window=1):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid verification code. Please try again."
            )
        
        # Enable 2FA
        await update_user_2fa(tenant_db, user_id, user_type, {
            "two_fa.enabled": True,
            "two_fa.required": True,
            "two_fa.secret_enc": encrypt_secret(secret),
            "two_fa.temp_secret_enc": None,
            "two_fa.verified_at": datetime.utcnow(),
            "two_fa.setup_started_at": None
        })
        
        # Create access token with 6h expiry
        user_data = _build_user_token_data(user, user_id, user_type, payload=payload)
        access_token = create_6h_access_token(auth_manager, user_data)
        
        # Build complete user response with role-specific fields
        user_response = _build_user_response(user, user_id, user_type, payload=payload)
        
        logger.info(f"2FA enabled for {user_type} {user_id}")
        
        return SetupVerifyResponse(
            success=True,
            access_token=access_token,
            user=user_response
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"2FA setup verify error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify 2FA setup"
        )


@router.post("/verify-otp", response_model=OTPVerifyResponse)
@limiter.limit("5/minute")  # Rate limit to prevent brute force
async def verify_otp(
    request: Request,
    data: OTPVerifyRequest,
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Verify OTP for login.
    Called after password verification when 2FA is enabled.
    """
    try:
        payload = verify_temp_token(data.temp_token, "OTP")
        user_id = payload["uid"]
        user_type = payload["user_type"]

        tenant_db = None
        if payload.get("db_name"):
            tenant_db = await db.get_tenant_db(payload.get("db_name"))
        if tenant_db is None:
            tenant = await _resolve_tenant_for_auth(
                db,
                request,
                payload.get("tenant_id"),
                require_active=False
            )
            tenant_db = await _get_tenant_db_or_503(db, tenant)

        user = await get_user_by_id(tenant_db, user_id, user_type)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get secret
        secret_enc = user.get("two_fa", {}).get("secret_enc")
        if not secret_enc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA not enabled for this account"
            )
        
        secret = decrypt_secret(secret_enc)
        
        # Verify OTP
        totp = pyotp.TOTP(secret)
        if not totp.verify(data.otp.strip(), valid_window=1):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid verification code"
            )
        
        # Update last login
        collection = "admins" if user_type == "admin" else "tutors"
        await tenant_db[collection].update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"last_login": datetime.utcnow(), "two_fa.last_verified_at": datetime.utcnow()}}
        )
        
        # Create access token with 6h expiry
        user_data = _build_user_token_data(user, user_id, user_type, payload=payload)
        access_token = create_6h_access_token(auth_manager, user_data)
        
        # Build complete user response with role-specific fields
        user_response = _build_user_response(user, user_id, user_type, payload=payload)
        
        return OTPVerifyResponse(
            success=True,
            access_token=access_token,
            user=user_response
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OTP verify error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify OTP"
        )


@router.get("/status", response_model=TwoFAStatusResponse)
async def get_2fa_status(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Get current user's 2FA status.
    Requires authentication.
    """
    try:
        # Verify token
        token = credentials.credentials
        user_data = await auth_manager.verify_token_and_get_user(token)
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication"
            )
        
        user_id = user_data.get("user_id")
        user_type = user_data.get("user_type")

        tenant_db = await db.get_tenant_db(user_data.get("db_name"))
        if tenant_db is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant context required"
            )

        user = await get_user_by_id(tenant_db, user_id, user_type)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        two_fa = user.get("two_fa", {})
        
        return TwoFAStatusResponse(
            success=True,
            two_fa_enabled=two_fa.get("enabled", False),
            two_fa_required=two_fa.get("required", True)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"2FA status error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get 2FA status"
        )


@router.post("/admin/reset/{target_user_id}")
async def admin_reset_2fa(
    request: Request,
    target_user_id: str,
    target_user_type: str = "tutor",
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Admin endpoint to reset a user's 2FA.
    This allows the user to set up 2FA again on their next login.
    """
    try:
        # Verify admin token
        token = credentials.credentials
        admin_data = await auth_manager.verify_token_and_get_user(token)
        
        if not admin_data or admin_data.get("user_type") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )

        tenant_db = await db.get_tenant_db(admin_data.get("db_name"))
        if tenant_db is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant context required"
            )

        # Reset target user's 2FA
        success = await update_user_2fa(tenant_db, target_user_id, target_user_type.lower(), {
            "two_fa.enabled": False,
            "two_fa.secret_enc": None,
            "two_fa.temp_secret_enc": None,
            "two_fa.required": True,  # Still require 2FA on next login
            "two_fa.reset_by": admin_data.get("user_id"),
            "two_fa.reset_at": datetime.utcnow()
        })
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reset 2FA"
            )
        
        logger.info(f"2FA reset for {target_user_type} {target_user_id} by admin {admin_data.get('user_id')}")
        
        return {
            "success": True,
            "message": f"2FA has been reset for user. They will need to set up 2FA on next login."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin reset 2FA error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset 2FA"
        )


@router.post("/disable")
async def disable_2fa(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Disable 2FA for current user (admin only).
    Note: This should be used with caution in production.
    """
    try:
        # Verify token
        token = credentials.credentials
        user_data = await auth_manager.verify_token_and_get_user(token)

        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication"
            )

        # Only allow admins to disable 2FA
        if user_data.get("user_type") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can disable 2FA"
            )

        user_id = user_data.get("user_id")
        user_type = user_data.get("user_type")

        tenant_db = await db.get_tenant_db(user_data.get("db_name"))
        if tenant_db is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant context required"
            )

        await update_user_2fa(tenant_db, user_id, user_type, {
            "two_fa.enabled": False,
            "two_fa.required": False,
            "two_fa.secret_enc": None,
            "two_fa.disabled_at": datetime.utcnow()
        })

        logger.info(f"2FA disabled for {user_type} {user_id}")

        return {
            "success": True,
            "message": "2FA has been disabled for your account"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Disable 2FA error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to disable 2FA"
        )


class ResetSelfRequest(BaseModel):
    """Request to reset own 2FA with OTP verification"""
    otp: str = Field(..., min_length=6, max_length=6)


@router.post("/reset-self")
@limiter.limit("3/minute")  # Strict rate limit for security
async def reset_own_2fa(
    request: Request,
    data: ResetSelfRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: DatabaseManager = Depends(get_database),
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """
    Reset own 2FA after verifying current OTP.

    This allows users to reconfigure their authenticator app by:
    1. Verifying their current OTP code (proves they have access to authenticator)
    2. Clearing the existing 2FA secret
    3. User will be prompted to set up 2FA again on next login

    Security: Requires valid current OTP to prevent unauthorized resets.
    """
    try:
        # Verify token
        token = credentials.credentials
        user_data = await auth_manager.verify_token_and_get_user(token)

        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication"
            )

        user_id = user_data.get("user_id")
        user_type = user_data.get("user_type")

        tenant_db = await db.get_tenant_db(user_data.get("db_name"))
        if tenant_db is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant context required"
            )

        user = await get_user_by_id(tenant_db, user_id, user_type)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Check if 2FA is enabled
        two_fa = user.get("two_fa", {})
        if not two_fa.get("enabled"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA is not enabled for this account"
            )

        # Get current secret and verify OTP
        secret_enc = two_fa.get("secret_enc")
        if not secret_enc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="2FA configuration error"
            )

        secret = decrypt_secret(secret_enc)
        totp = pyotp.TOTP(secret)

        if not totp.verify(data.otp.strip(), valid_window=1):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid verification code. Please enter your current authenticator code."
            )

        # OTP verified - reset 2FA so user can set up again
        await update_user_2fa(tenant_db, user_id, user_type, {
            "two_fa.enabled": False,
            "two_fa.secret_enc": None,
            "two_fa.temp_secret_enc": None,
            "two_fa.required": True,  # Still require 2FA on next login
            "two_fa.reset_at": datetime.utcnow(),
            "two_fa.reset_reason": "self_reset"
        })

        logger.info(f"2FA self-reset by {user_type} {user_id}")

        return {
            "success": True,
            "message": "2FA has been reset. You will need to set up 2FA again on your next login."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Self-reset 2FA error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset 2FA"
        )
