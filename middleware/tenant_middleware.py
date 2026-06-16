"""
Tenant Middleware for FastAPI

This middleware automatically sets the tenant context from the authenticated user
at the start of each request, ensuring all database operations are tenant-scoped.

USAGE:
1. Add the middleware to FastAPI app:
   app.add_middleware(TenantMiddleware)

2. Use get_tenant_db dependency in routes:
   @router.get("/students")
   async def get_students(tenant_db: TenantAwareDB = Depends(get_tenant_db)):
       return await tenant_db.find("students", {})
"""

import logging
from typing import Optional, Dict, Any
from fastapi import Request, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from core.tenant import TenantContext, TenantAwareDB, TenantContextError, TenantIsolationError
from core.database import DatabaseManager
from core.cookie_auth import COOKIE_NAME as AUTH_COOKIE_NAME, cookie_auth_manager
from core.tenant_features import (
    build_enabled_features_v2,
    merge_tenant_features,
    required_feature_for_path,
)
from api.v1.auth_async import get_database, get_current_user
from config_async import MONGODB_DB_STOODY

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


class TenantMiddleware(BaseHTTPMiddleware):
    """
    Middleware that sets tenant context from JWT token.

    This runs before route handlers and ensures TenantContext
    is always set for authenticated requests.
    """

    # Paths that don't require tenant context
    EXEMPT_PATHS = {
        "/health",
        "/healthz",
        "/alb-health",
        "/api/health",
        "/api/v1/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/api/v1/auth/student/login",
        "/api/v1/auth/register",
        "/api/v1/auth/admin/register",
        "/api/v1/auth/student/password-reset/request",
        "/api/v1/auth/student/password-reset/complete",
        "/api/v1/auth/tutor/password-reset/request",
        "/api/v1/auth/tutor/password-reset/complete",
        "/api/v1/auth/admin/password-reset/request",
        "/api/v1/auth/admin/password-reset/complete",
        "/api/v1/auth/student/forgot-password",
        "/api/v1/auth/verify",
        "/api/v1/auth/tenant/check-availability",
        "/api/v1/auth/tenant/lookup",
        "/api/v1/auth/admin/registration-status-auth",
        "/api/v1/auth/admin/registration-status-message",
        "/api/v1/auth/admin/registration-status",
    }

    # Path prefixes that should be exempt (super admin uses different JWT)
    EXEMPT_PREFIXES = {
        "/api/v1/superadmin",
    }

    # Uploaded documents are served only through authenticated API routes.
    # Direct static URLs reveal stored PDFs after logout and bypass document ACLs.
    PROTECTED_STATIC_PREFIXES = (
        "/uploads/documents",
    )

    # Static/resource prefixes — served without tenant context (images, uploads, etc.)
    STATIC_PREFIXES = (
        "/images/",
        "/uploads/",
        "/static/",
        "/favicon",
    )

    async def dispatch(self, request: Request, call_next) -> Response:
        # Clear any stale tenant context
        TenantContext.clear()

        # Skip tenant context for exact exempt paths (no prefix matching — prevents accidental exemptions)
        path = request.url.path
        if path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Skip tenant context for exempt path prefixes (super admin uses separate JWT)
        if any(path.startswith(prefix) for prefix in self.EXEMPT_PREFIXES):
            return await call_next(request)

        if any(path.startswith(prefix) for prefix in self.PROTECTED_STATIC_PREFIXES):
            logger.warning("Blocked direct protected upload access: %s", path)
            return Response(
                content='{"detail":"Not found"}',
                status_code=status.HTTP_404_NOT_FOUND,
                media_type="application/json",
            )

        # Static resources (images, uploads) served without tenant context
        if any(path.startswith(prefix) for prefix in self.STATIC_PREFIXES):
            return await call_next(request)

        # Try to extract JWT from Authorization header first, then check
        # the auth cookie.  Browser-initiated requests such as
        # <img src="..."> do not carry the Authorization header but *do*
        # send cookies automatically, so the cookie path is essential for
        # tenant-scoped resources served to the browser directly.
        _needs_cookie_bridge = False
        _bridge_token = None

        try:
            token = None
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                # If authenticated via header but no cookie exists yet,
                # we'll set the cookie on the response so that future
                # browser-initiated requests (img, link, etc.) carry it.
                if not request.cookies.get(AUTH_COOKIE_NAME):
                    _needs_cookie_bridge = True
                    _bridge_token = token

            if not token:
                token = request.cookies.get(AUTH_COOKIE_NAME)

            if token:
                auth_manager = request.app.state.auth
                user_data = await auth_manager.verify_token_and_get_user(token)

                if user_data:
                    is_tenant_user = (
                        user_data.get("user_type") in {"admin", "tutor", "student"}
                        and user_data.get("tenant_id")
                    )
                    # For role-aware feature gating, include device context.
                    # Smartboard devices use "smartboard" as a pseudo-role for
                    # OCR/notes path resolution.
                    effective_role = user_data.get("user_type") or ""
                    if user_data.get("device") == "smartboard":
                        effective_role = "smartboard"
                    required_feature = required_feature_for_path(path, effective_role)

                    # Fetch tenant doc for all authenticated tenant users
                    # (needed for platform_suspended check + feature gating)
                    tenant_doc = None
                    if is_tenant_user:
                        try:
                            master_db = await request.app.state.db.get_master_db()
                            tenant_key = str(user_data.get("tenant_id", "")).strip().upper()
                            if master_db is not None and tenant_key:
                                tenant_doc = await master_db["tenants"].find_one(
                                    {"$or": [{"tenant_id": tenant_key}, {"institution_id": tenant_key}]},
                                    {"enabled_features": 1, "enabled_features_v2": 1,
                                     "platform_suspended": 1, "status": 1},
                                )
                        except Exception as tenant_load_error:
                            logger.debug(f"Could not fetch tenant doc from master DB: {tenant_load_error}")

                    # Block requests if platform access is suspended by super-admin lifecycle
                    if tenant_doc and tenant_doc.get("platform_suspended"):
                        return Response(
                            content='{"detail":"Platform access suspended. Contact your service provider."}',
                            status_code=status.HTTP_403_FORBIDDEN,
                            media_type="application/json",
                        )

                    # Feature gating check
                    if required_feature and is_tenant_user:
                        enabled_features = (tenant_doc or {}).get("enabled_features")
                        enabled_features_v2 = (tenant_doc or {}).get("enabled_features_v2")

                        merged_features = merge_tenant_features(
                            enabled_features or user_data.get("enabled_features"),
                            enabled_features_v2 or user_data.get("enabled_features_v2"),
                        )
                        user_data["enabled_features"] = merged_features
                        user_data["enabled_features_v2"] = build_enabled_features_v2(
                            enabled_features_v2 or user_data.get("enabled_features_v2"),
                            enabled_features or user_data.get("enabled_features"),
                        )

                        if not merged_features.get(required_feature, True):
                            return Response(
                                content=f'{{"detail":"Feature \\"{required_feature}\\" is disabled by super admin"}}',
                                status_code=status.HTTP_403_FORBIDDEN,
                                media_type="application/json",
                            )

                    admin_id = user_data.get("admin_id")
                    user_type = user_data.get("user_type")
                    is_b2c = user_data.get("is_b2c") or user_type in ("b2c_user", "b2c_admin")
                    db_name = user_data.get("db_name") or (MONGODB_DB_STOODY if is_b2c else None)
                    if not db_name and not is_b2c:
                        # STRICT MODE: All authenticated tenant requests must
                        # have db_name. Reject regardless of auth method.
                        logger.warning(
                            "Tenant db_name missing for user %s (type=%s, path=%s) — rejecting request",
                            user_data.get("user_id"), user_data.get("user_type"), path,
                        )
                        return Response(
                            content='{"detail":"Tenant database missing. Please log in again with tenant ID."}',
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            media_type="application/json",
                        )
                    else:
                        if user_data.get("user_type") == "admin":
                            admin_id = admin_id or user_data.get("user_id")

                        TenantContext.set(
                            admin_id=admin_id,
                            user_type=user_type,
                            user_id=user_data.get("user_id"),
                            tutor_id=user_data.get("tutor_id"),
                            db_name=db_name,
                            tenant_id=user_data.get("tenant_id"),
                            institution_id=user_data.get("institution_id"),
                        )
                        logger.debug(f"Tenant context set for user {user_data.get('user_id')}, admin_id={admin_id}")

        except Exception as e:
            logger.debug(f"Could not set tenant context from token: {e}")

        try:
            response = await call_next(request)

            # Bridge: if the user authenticated via header but has no
            # cookie yet, set one so that subsequent browser-initiated
            # requests (e.g. <img> tags) can resolve the tenant context.
            if _needs_cookie_bridge and _bridge_token:
                try:
                    cookie_auth_manager.set_auth_cookie(response, _bridge_token)
                except Exception:
                    pass  # non-critical; don't break the response

            return response
        except TenantContextError as e:
            logger.error(f"Tenant context error: {e}")
            return Response(
                content=f'{{"detail": "Tenant context required: {str(e)}"}}',
                status_code=status.HTTP_403_FORBIDDEN,
                media_type="application/json"
            )
        except TenantIsolationError as e:
            logger.error(f"SECURITY: Tenant isolation violation: {e}")
            return Response(
                content='{"detail": "Access denied"}',
                status_code=status.HTTP_403_FORBIDDEN,
                media_type="application/json"
            )
        finally:
            # Always clear context after request
            TenantContext.clear()


async def get_tenant_db(
    db: DatabaseManager = Depends(get_database),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> TenantAwareDB:
    """
    FastAPI dependency that provides a tenant-aware database wrapper.

    This ensures:
    1. User is authenticated
    2. Tenant context is set from user
    3. All database operations are automatically tenant-scoped

    Usage in routes:
        @router.get("/students")
        async def get_students(tenant_db: TenantAwareDB = Depends(get_tenant_db)):
            # This automatically filters by current admin's tenant
            return await tenant_db.find("students", {"grade": "10"})
    """
    # Extract admin_id based on user type
    user_type = current_user.get("user_type")
    is_b2c = current_user.get("is_b2c") or user_type in ("b2c_user", "b2c_admin")
    db_name = current_user.get("db_name") or (MONGODB_DB_STOODY if is_b2c else None)
    if not db_name and not is_b2c:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant database missing. Please log in again with tenant ID.",
        )

    if user_type == "admin":
        admin_id = current_user.get("admin_id") or current_user.get("user_id")
    elif user_type == "tutor":
        admin_id = current_user.get("admin_id")
    elif user_type == "student":
        admin_id = current_user.get("admin_id")
    else:
        admin_id = current_user.get("admin_id")

    # Set tenant context
    TenantContext.set(
        admin_id=admin_id,
        user_type=user_type,
        user_id=current_user.get("user_id"),
        tutor_id=current_user.get("tutor_id"),
        db_name=db_name,
        tenant_id=current_user.get("tenant_id"),
        institution_id=current_user.get("institution_id"),
    )

    # Return tenant-aware database wrapper
    return TenantAwareDB(db)


async def get_tenant_db_optional(
    db: DatabaseManager = Depends(get_database),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> TenantAwareDB:
    """
    FastAPI dependency for optional tenant context (public endpoints).

    If authenticated, sets tenant context.
    If not authenticated, returns tenant_db without context (global access only).
    """
    if credentials:
        try:
            from api.v1.auth_async import get_auth_manager
            # Would need to inject auth_manager here - simplified for example
            pass
        except Exception:
            pass

    return TenantAwareDB(db)


def set_tenant_from_user(current_user: Dict[str, Any]):
    """
    Helper function to set tenant context from user dict.
    Use this when you can't use the dependency injection.

    Usage:
        set_tenant_from_user(current_user)
        # Now tenant context is set
    """
    user_type = current_user.get("user_type")

    if user_type == "admin":
        admin_id = current_user.get("admin_id") or current_user.get("user_id")
    else:
        admin_id = current_user.get("admin_id")

    TenantContext.set(
        admin_id=admin_id,
        user_type=user_type,
        user_id=current_user.get("user_id"),
        tutor_id=current_user.get("tutor_id"),
        db_name=current_user.get("db_name"),
        tenant_id=current_user.get("tenant_id"),
        institution_id=current_user.get("institution_id"),
    )
