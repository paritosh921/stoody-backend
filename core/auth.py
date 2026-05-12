"""
Async Authentication Manager for SkillBot
JWT-based authentication with caching and rate limiting
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from passlib.context import CryptContext
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time

from config_async import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    PASSWORD_MAX_LENGTH,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory cache for min_token_issued_at (avoids DB hit on every request)
# ---------------------------------------------------------------------------
_min_iat_cache: Dict[str, Any] = {"value": None, "fetched_at": 0.0}
_MIN_IAT_CACHE_TTL = 60  # seconds

# bcrypt 72-byte password limit
BCRYPT_MAX_BYTES = 72

class AuthManager:
    """Async authentication manager with JWT and caching"""

    def __init__(self):
        # Use bcrypt directly to avoid passlib compatibility issues
        try:
            import bcrypt
            self.bcrypt = bcrypt
            self.use_bcrypt = True
        except Exception:
            self.use_bcrypt = False
            # Fallback to passlib with pbkdf2
            self.pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

        self.security = HTTPBearer()
        self.cache_manager = None  # Will be injected

    async def initialize(self):
        """Initialize auth manager"""
        logger.info("✅ Auth manager initialized")

    def set_cache_manager(self, cache_manager):
        """Set cache manager for session caching"""
        self.cache_manager = cache_manager

    def set_db_manager(self, db_manager):
        """Set database manager (needed for deploy-time session invalidation)"""
        self.db_manager = db_manager

    async def _get_min_token_issued_at(self) -> Optional[float]:
        """
        Return the min_token_issued_at timestamp (epoch seconds).
        Cached in-memory for 60s to avoid a DB round-trip on every request.
        """
        global _min_iat_cache
        now = time.time()
        if (
            _min_iat_cache["value"] is not None
            and now - _min_iat_cache["fetched_at"] < _MIN_IAT_CACHE_TTL
        ):
            return _min_iat_cache["value"]

        db_manager = getattr(self, "db_manager", None)
        if db_manager is None:
            return _min_iat_cache["value"]  # stale is better than crash

        try:
            master_db = await db_manager.get_master_db()
            if master_db is None:
                return _min_iat_cache["value"]
            doc = await master_db["system_config"].find_one(
                {"key": "min_token_issued_at"}
            )
            if doc and doc.get("value") is not None:
                _min_iat_cache["value"] = doc["value"]
            else:
                _min_iat_cache["value"] = None
            _min_iat_cache["fetched_at"] = now
        except Exception as e:
            logger.warning("Failed to fetch min_token_issued_at: %s", e)
            # Keep stale value; don't block auth on DB hiccup

        return _min_iat_cache["value"]

    # Password utilities
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify password against hash.

        Note: bcrypt has a 72-byte limit. Passwords longer than 72 bytes
        (when UTF-8 encoded) are silently truncated. This is a known
        limitation of bcrypt.
        """
        try:
            if not plain_password or not hashed_password:
                return False

            if self.use_bcrypt:
                # Use bcrypt directly
                password_bytes = plain_password.encode('utf-8')

                # Log warning if password was truncated (for debugging only)
                if len(password_bytes) > BCRYPT_MAX_BYTES:
                    logger.debug(
                        f"Password exceeds {BCRYPT_MAX_BYTES} bytes "
                        f"({len(password_bytes)} bytes) - truncating for verification"
                    )
                    password_bytes = password_bytes[:BCRYPT_MAX_BYTES]

                hash_bytes = hashed_password.encode('utf-8')
                return self.bcrypt.checkpw(password_bytes, hash_bytes)
            else:
                # Fallback to passlib
                return self.pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification failed: {str(e)}")
            return False

    def get_password_hash(self, password: str, warn_on_truncation: bool = True) -> str:
        """
        Generate password hash using bcrypt.

        IMPORTANT: bcrypt has a 72-byte limit for passwords. If the password
        exceeds this limit when encoded as UTF-8, it will be truncated.
        This is a security consideration - users should be warned about
        this limitation at the API level.

        Args:
            password: Plain text password to hash
            warn_on_truncation: Log warning if password is truncated (default True)

        Returns:
            Hashed password string
        """
        if self.use_bcrypt:
            # Encode password to bytes
            password_bytes = password.encode('utf-8')

            # Check if truncation will occur
            if len(password_bytes) > BCRYPT_MAX_BYTES:
                if warn_on_truncation:
                    logger.warning(
                        f"Password exceeds bcrypt {BCRYPT_MAX_BYTES}-byte limit "
                        f"({len(password_bytes)} bytes). "
                        "Password will be truncated. Consider validating password "
                        "length at the API level to warn users."
                    )
                password_bytes = password_bytes[:BCRYPT_MAX_BYTES]

            salt = self.bcrypt.gensalt(rounds=12)  # 12 rounds for good security/performance balance
            return self.bcrypt.hashpw(password_bytes, salt).decode('utf-8')
        else:
            # Fallback to passlib
            return self.pwd_context.hash(password)

    # JWT token utilities
    def create_access_token(self, data: Dict[str, Any],
                           expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })

        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return encoded_jwt

    def decode_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode and verify JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

            # Check token type
            if payload.get("type") != "access":
                return None

            # Check expiration
            if payload.get("exp", 0) < time.time():
                return None

            return payload

        except ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except InvalidTokenError as e:
            logger.warning(f"JWT decode error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected token decode error: {str(e)}")
            return None

    # Session management with caching
    async def create_user_session(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create user session with caching"""
        user_id = str(user_data.get("user_id") or user_data.get("id") or user_data.get("_id"))
        user_type = user_data.get("user_type", "student")

        # Create JWT token - include admin_id and subdomain for multi-tenancy
        token_data = {
            "sub": user_id,
            "user_type": user_type,
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
            "enabled_features_v2": user_data.get("enabled_features_v2"),
        }

        access_token = self.create_access_token(token_data)

        # Cache session data
        session_data = {
            "user_id": user_id,
            "user_type": user_type,
            "email": user_data.get("email"),
            "username": user_data.get("username"),
            "full_name": user_data.get("full_name"),
            "admin_id": user_data.get("admin_id"),
            "subdomain": user_data.get("subdomain"),
            "tutor_id": user_data.get("tutor_id"),
            "tenant_id": user_data.get("tenant_id"),
            "db_name": user_data.get("db_name"),
            "institution_id": user_data.get("institution_id"),
            "admin_role": user_data.get("admin_role"),
            "permissions": user_data.get("permissions"),
            "enabled_features": user_data.get("enabled_features"),
            "enabled_features_v2": user_data.get("enabled_features_v2"),
            # Content scoping fields (mainly for tutors)
            "standards": user_data.get("standards"),
            "sections": user_data.get("sections"),
            "subjects": user_data.get("subjects"),
            "plan_types": user_data.get("plan_types"),
            "teaching_assignments": user_data.get("teaching_assignments"),
            "can_edit_students": user_data.get("can_edit_students"),
            "created_at": time.time()
        }

        if self.cache_manager:
            await self.cache_manager.cache_user_session(
                user_id, session_data, JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
            )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user": session_data
        }

    async def get_user_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user session from cache"""
        if not self.cache_manager:
            return None

        return await self.cache_manager.get_user_session(user_id)

    async def invalidate_user_session(self, user_id: str) -> bool:
        """Invalidate user session"""
        if not self.cache_manager:
            return True

        return await self.cache_manager.delete(f"session:{user_id}", "auth")

    async def verify_token_and_get_user(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify token and get user data with caching"""
        try:
            # Decode token
            payload = self.decode_access_token(token)
            if not payload:
                return None

            user_id = payload.get("sub")
            if not user_id:
                return None

            # ── Deploy-time session invalidation ──
            # If a min_token_issued_at timestamp exists in skb_master.system_config,
            # reject any JWT whose iat predates it.
            token_iat = payload.get("iat")
            if token_iat is not None and payload.get("user_type") != "hub":
                min_iat = await self._get_min_token_issued_at()
                if min_iat is not None and token_iat < min_iat:
                    logger.info(
                        "Token rejected: iat %s < min_token_issued_at %s",
                        token_iat, min_iat,
                    )
                    return None

            # Check cached session first
            if self.cache_manager:
                cached_session = await self.get_user_session(user_id)
                if cached_session:
                    return cached_session

            # Return basic user data from token
            return {
                "user_id": user_id,
                "user_type": payload.get("user_type"),
                "email": payload.get("email"),
                "username": payload.get("username"),
                "admin_id": payload.get("admin_id"),
                "subdomain": payload.get("subdomain"),
                "tutor_id": payload.get("tutor_id"),
                "hub_id": payload.get("hub_id"),
                "tenant_id": payload.get("tenant_id"),
                "db_name": payload.get("db_name"),
                "institution_id": payload.get("institution_id"),
                "scopes": payload.get("scopes"),
                "type": payload.get("type"),
                "admin_role": payload.get("admin_role"),
                "permissions": payload.get("permissions"),
                "enabled_features": payload.get("enabled_features"),
                "enabled_features_v2": payload.get("enabled_features_v2"),
            }

        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            return None

    # Authorization decorators and utilities
    def require_auth(self, required_role: Optional[str] = None):
        """Decorator to require authentication"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # This would be used with FastAPI dependencies
                return await func(*args, **kwargs)
            return wrapper
        return decorator

    def check_user_permissions(self, user_data: Dict[str, Any],
                             required_role: str = None,
                             resource_owner_id: str = None) -> bool:
        """Check if user has required permissions"""
        try:
            user_type = user_data.get("user_type", "student")
            user_id = user_data.get("user_id")

            # Admin can access everything
            if user_type == "admin":
                return True

            # Check specific role requirement
            if required_role:
                if user_type != required_role:
                    return False

            # Check resource ownership
            if resource_owner_id:
                if str(user_id) != str(resource_owner_id):
                    return False

            return True

        except Exception as e:
            logger.error(f"Permission check failed: {str(e)}")
            return False

    # Rate limiting for auth endpoints
    async def check_auth_rate_limit(self, identifier: str, action: str = "login") -> tuple[bool, int]:
        """Check authentication rate limits"""
        if not self.cache_manager:
            return True, 10

        # Different limits for different actions
        limits = {
            "login": (10, 300),  # 10 attempts per 5 minutes
            "register": (5, 600),  # 5 attempts per 10 minutes
            "password_reset": (3, 600)  # 3 attempts per 10 minutes
        }

        limit, window = limits.get(action, (10, 300))
        key = f"auth:{action}:{identifier}"

        return await self.cache_manager.rate_limit_check(key, limit, window)

    # Utility methods for different user types
    async def authenticate_admin(self, email: str, password: str,
                               db_manager, db_override=None) -> Optional[Dict[str, Any]]:
        """Authenticate admin user"""
        try:
            # Check rate limit
            allowed, remaining = await self.check_auth_rate_limit(email, "login")
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many login attempts. Please try again later."
                )

            # Find admin user in database
            if db_override is not None:
                admin_data = await db_override["admins"].find_one(
                    {"email": email, "is_active": True}
                )
            else:
                admin_data = await db_manager.mongo_find_one(
                    "admins",
                    {"email": email, "is_active": True}
                )

            if not admin_data:
                return None

            # Verify password
            if not self.verify_password(password, admin_data["password_hash"]):
                return None

            # Update last login
            if db_override is not None:
                await db_override["admins"].update_one(
                    {"_id": admin_data["_id"]},
                    {"$set": {"last_login": datetime.utcnow()}}
                )
            else:
                await db_manager.mongo_update_one(
                    "admins",
                    {"_id": admin_data["_id"]},
                    {"$set": {"last_login": datetime.utcnow()}}
                )

            return {
                "user_id": str(admin_data["_id"]),
                "admin_id": str(admin_data["_id"]),  # Add admin_id for consistency
                "email": admin_data["email"],
                "full_name": admin_data.get("full_name", "Admin"),
                "user_type": "admin",
                "subdomain": admin_data.get("subdomain")
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Admin authentication failed: {str(e)}")
            return None

    async def authenticate_student(self, username: str, password: str,
                                 db_manager) -> Optional[Dict[str, Any]]:
        """Authenticate student user"""
        try:
            normalized_username = username.strip()
            username_lower = normalized_username.lower()
            # Check rate limit
            allowed, remaining = await self.check_auth_rate_limit(username_lower, "login")
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many login attempts. Please try again later."
                )

            # Find student user in database
            student_data = await db_manager.mongo_find_one(
                "students",
                {"username_lower": username_lower, "is_active": True}
            )
            if not student_data:
                student_data = await db_manager.mongo_find_one(
                    "students",
                    {"username": normalized_username, "is_active": True},
                    collation={"locale": "en", "strength": 2}
                )

            if not student_data:
                return None

            # Verify password
            if not self.verify_password(password, student_data["password_hash"]):
                return None

            # Update last login
            await db_manager.mongo_update_one(
                "students",
                {"_id": student_data["_id"]},
                {"$set": {"last_login": datetime.utcnow()}}
            )

            return {
                "id": str(student_data["_id"]),
                "username": student_data["username"],
                "full_name": student_data.get("full_name", username),
                "email": student_data.get("email"),
                "user_type": "student"
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Student authentication failed: {str(e)}")
            return None

    async def authenticate_tutor(self, username: str, password: str,
                                 db_manager, db_override=None) -> Optional[Dict[str, Any]]:
        """Authenticate tutor user"""
        try:
            normalized_username = username.strip()
            username_lower = normalized_username.lower()
            # Check rate limit
            allowed, _ = await self.check_auth_rate_limit(username_lower, "login")
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many login attempts. Please try again later."
                )

            # Find tutor in database
            if db_override is not None:
                tutor_data = await db_override["tutors"].find_one(
                    {"username_lower": username_lower, "is_active": True}
                )
                if not tutor_data:
                    tutor_data = await db_override["tutors"].find_one(
                        {"username": normalized_username, "is_active": True},
                        collation={"locale": "en", "strength": 2}
                    )
            else:
                tutor_data = await db_manager.mongo_find_one(
                    "tutors",
                    {"username_lower": username_lower, "is_active": True}
                )
                if not tutor_data:
                    tutor_data = await db_manager.mongo_find_one(
                        "tutors",
                        {"username": normalized_username, "is_active": True},
                        collation={"locale": "en", "strength": 2}
                    )

            if not tutor_data:
                return None

            # Verify password
            if not self.verify_password(password, tutor_data.get("password_hash", "")):
                return None

            # Update last login
            if db_override is not None:
                await db_override["tutors"].update_one(
                    {"_id": tutor_data["_id"]},
                    {"$set": {"last_login": datetime.utcnow()}}
                )
            else:
                await db_manager.mongo_update_one(
                    "tutors",
                    {"_id": tutor_data["_id"]},
                    {"$set": {"last_login": datetime.utcnow()}}
                )

            return {
                "user_id": str(tutor_data["_id"]),
                "tutor_id": tutor_data.get("tutor_id"),
                "username": tutor_data.get("username"),
                "email": tutor_data.get("email"),
                "full_name": tutor_data.get("name"),
                "user_type": "tutor",
                # Store admin context for multi-tenant filtering
                "admin_id": str(tutor_data.get("created_by")) if tutor_data.get("created_by") else None,
                # Content scoping fields
                "standards": tutor_data.get("standards", []),
                "sections": tutor_data.get("sections", []),
                "subjects": tutor_data.get("subjects", []),
                "plan_types": tutor_data.get("plan_types", []),
                "teaching_assignments": tutor_data.get("teaching_assignments", []),
                "can_edit_students": tutor_data.get("can_edit_students", False),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Tutor authentication failed: {str(e)}")
            return None

    # Security utilities
    def generate_secure_password(self, length: int = 12) -> str:
        """Generate a secure random password"""
        import secrets
        import string

        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    async def change_password(self, user_id: str, old_password: str,
                            new_password: str, user_type: str,
                            db_manager) -> bool:
        """Change user password"""
        try:
            collection_name = "admins" if user_type == "admin" else "students"

            # Get current user data
            user_data = await db_manager.mongo_find_one(
                collection_name,
                {"_id": user_id}
            )

            if not user_data:
                return False

            # Verify old password
            if not self.verify_password(old_password, user_data["password_hash"]):
                return False

            # Hash new password
            new_password_hash = self.get_password_hash(new_password)

            # Update password
            result = await db_manager.mongo_update_one(
                collection_name,
                {"_id": user_id},
                {
                    "$set": {
                        "password_hash": new_password_hash,
                        "password_changed_at": datetime.utcnow()
                    }
                }
            )

            # Invalidate user sessions
            await self.invalidate_user_session(user_id)

            return result

        except Exception as e:
            logger.error(f"Password change failed: {str(e)}")
            return False
