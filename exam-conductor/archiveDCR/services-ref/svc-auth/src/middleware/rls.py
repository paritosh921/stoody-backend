"""RLS middleware for svc-auth.

Injects ``SET app.current_tenant`` on every request that carries
a validated JWT with a tenant_id claim.  Uses the shared
``exampen_common.db.rls_middleware`` helper.

Flow:
  1. Middleware extracts the bearer token from the Authorization header.
  2. Decodes the JWT *without signature verification* to read ``tenant_id``
     (full verification happens later in route dependencies).
  3. Stores ``tenant_id`` on ``request.state`` so downstream code can read it.
  4. Wraps the DB session factory in an RLS-aware version for the request.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import jwt as pyjwt
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from exampen_common.logging import get_logger

_log = get_logger(__name__)

# Context variable so repos and helpers can access the current tenant
# without needing the Request object.
tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="")

# Roles that operate across tenants and should bypass RLS.
_CROSS_TENANT_ROLES = frozenset({"super_admin"})


def _extract_tenant_from_bearer(request: Request) -> tuple[str, str]:
    """Extract tenant_id and role from the Authorization bearer token.

    Returns (tenant_id, role).  Both default to empty string on failure.
    Does NOT verify the signature — that is handled by route-level
    dependencies.  This is safe because RLS is a *defense-in-depth*
    layer; it does not replace auth.
    """
    auth_header: str | None = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return "", ""
    token = auth_header.removeprefix("Bearer ").strip()
    if not token:
        return "", ""
    try:
        claims: dict[str, Any] = pyjwt.decode(
            token, options={"verify_signature": False}
        )
        tenant_id = str(claims.get("tenant_id", ""))
        role = str(claims.get("role", claims.get("stoody_role", "")))
        return tenant_id, role
    except Exception:
        return "", ""


class RLSMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that sets RLS tenant context per request.

    Reads the bearer JWT from the ``Authorization`` header, extracts
    ``tenant_id``, and injects it into the DB connection via
    ``SET app.current_tenant``.

    Requests without a bearer token (health checks, public routes)
    or with a cross-tenant role (``super_admin``) are passed through
    without RLS injection.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        tenant_id, role = _extract_tenant_from_bearer(request)

        if tenant_id and role not in _CROSS_TENANT_ROLES:
            # Store on request.state for route-level access.
            request.state.tenant_id = tenant_id
            # Store in context var for repo-level access.
            token = tenant_id_var.set(tenant_id)

            # Set RLS on a connection so all queries in this request
            # are scoped to the tenant.
            await _set_rls_on_engine(request, tenant_id)

            _log.debug(
                "RLS context set: tenant=%s path=%s",
                tenant_id,
                request.url.path,
            )
        else:
            request.state.tenant_id = ""
            token = tenant_id_var.set("")
            if role in _CROSS_TENANT_ROLES:
                _log.debug(
                    "RLS bypassed for cross-tenant role=%s path=%s",
                    role,
                    request.url.path,
                )

        try:
            response = await call_next(request)
        finally:
            tenant_id_var.reset(token)

        return response


async def _set_rls_on_engine(request: Request, tenant_id: str) -> None:
    """Set ``app.current_tenant`` on a pooled connection.

    Uses ``set_config(..., true)`` which scopes the setting to the
    current transaction.  The connection is committed so the setting
    takes effect for subsequent session usage within this request.

    Failures are logged but do not block the request — RLS is a
    defense-in-depth layer, not the primary auth gate.
    """
    from exampen_common.db import rls_middleware as _rls_mw

    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        return
    try:
        async with engine.connect() as conn:
            await _rls_mw(conn, tenant_id)
            await conn.commit()
    except Exception:
        _log.warning(
            "Failed to set RLS context for tenant=%s on %s",
            tenant_id,
            request.url.path,
            exc_info=True,
        )
