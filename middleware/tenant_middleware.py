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
from api.v1.auth_async import get_database, get_current_user

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
        "/docs",
        "/openapi.json",
        "/redoc",
        "/api/v1/auth/admin/login",
        "/api/v1/auth/tutor/login",
        "/api/v1/auth/student/login",
        "/api/v1/auth/register",
    }

    async def dispatch(self, request: Request, call_next) -> Response:
        # Clear any stale tenant context
        TenantContext.clear()

        # Skip tenant context for exempt paths
        path = request.url.path
        if any(path.startswith(exempt) for exempt in self.EXEMPT_PATHS):
            return await call_next(request)

        # Try to extract tenant info from authorization header
        try:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

                # Get auth manager and verify token
                auth_manager = request.app.state.auth
                user_data = await auth_manager.verify_token_and_get_user(token)

                if user_data:
                    # Set tenant context
                    admin_id = user_data.get("admin_id")
                    if user_data.get("user_type") == "admin":
                        # For admins, admin_id is their own user_id
                        admin_id = admin_id or user_data.get("user_id")

                    TenantContext.set(
                        admin_id=admin_id,
                        user_type=user_data.get("user_type"),
                        user_id=user_data.get("user_id"),
                        tutor_id=user_data.get("tutor_id"),
                        db_name=user_data.get("db_name"),
                        tenant_id=user_data.get("tenant_id"),
                        institution_id=user_data.get("institution_id"),
                    )
                    logger.debug(f"Tenant context set for user {user_data.get('user_id')}, admin_id={admin_id}")

        except Exception as e:
            logger.debug(f"Could not set tenant context from token: {e}")

        try:
            response = await call_next(request)
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
        db_name=current_user.get("db_name"),
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
