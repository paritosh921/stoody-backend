"""
EvalPen Usage API — LLM gate budget and usage endpoints.

Exposes the shared LLM gate's usage/config surface as REST endpoints
for admin users.

Endpoints (LLM_GATE_SPEC.md §10, eval-usage.openapi.yaml):
    GET  /current  — Current budget status (daily/weekly/monthly windows)
    GET  /history  — Historical usage rollups with optional filters
    PUT  /config   — Update gate configuration (admin-only)

Spec authority  : new-docs/architecture/LLM_GATE_SPEC.md §10
Wire format     : new-docs/api/eval-usage.openapi.yaml
Ownership       : LLM gate reads only (STATE_OWNERSHIP_MAP.md) — config
                  write delegated to LLMGate.update_config()
Hard constraints: C1 (MongoDB only), C6 (OpenAPI spec owns wire format)
"""

from __future__ import annotations

import importlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

# ---------------------------------------------------------------------------
# Import from exam-conductor (hyphenated directory — requires importlib)
# ---------------------------------------------------------------------------
_llm_gate_mod = importlib.import_module("exam-conductor.llm_gate")

LLMGate = _llm_gate_mod.LLMGate
GateConfig = _llm_gate_mod.GateConfig
CurrentUsage = _llm_gate_mod.CurrentUsage
UsageWindow = _llm_gate_mod.UsageWindow
TokenUsageRollup = _llm_gate_mod.TokenUsageRollup
PeriodType = _llm_gate_mod.PeriodType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
# Mounted by main_async.py (SWM-008) at prefix "{API_V1_PREFIX}/evalpen/usage"
# so endpoint paths here are relative: /current, /history, /config.
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _require_admin(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Dependency that enforces admin-only access.

    Rejects students, tutors, and any non-admin user type with 403.
    """
    user_type = current_user.get("user_type", "")
    if user_type not in ("admin", "master_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required for gate usage endpoints",
        )
    return current_user


async def _get_tenant_db(
    db: DatabaseManager,
    current_user: Dict[str, Any],
):
    """
    Resolve the tenant Motor database from the authenticated user's JWT
    claims.  Returns an ``AsyncIOMotorDatabase`` instance.
    """
    db_name = current_user.get("db_name")
    if not db_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing tenant database context",
        )
    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant database not available",
        )
    return tenant_db


async def _get_gate(
    db: DatabaseManager,
    current_user: Dict[str, Any],
) -> LLMGate:
    """
    Instantiate and initialise an ``LLMGate`` for the current tenant.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    gate = LLMGate(tenant_db)
    await gate.initialize()
    return gate


# ---------------------------------------------------------------------------
# GET /current — Current gate usage and remaining headroom
# ---------------------------------------------------------------------------

@router.get("/current")
async def get_current_usage(
    current_user: Dict[str, Any] = Depends(_require_admin),
    db: DatabaseManager = Depends(get_database),
):
    """
    Return current budget status with daily, weekly, and monthly usage
    windows.

    Response shape matches ``CurrentUsage`` in eval-usage.openapi.yaml.
    """
    gate = await _get_gate(db, current_user)
    usage = await gate.current_usage()
    return usage


# ---------------------------------------------------------------------------
# GET /history — Historical usage rollups
# ---------------------------------------------------------------------------

@router.get("/history")
async def get_usage_history(
    period_type: Optional[str] = Query(
        None,
        description="Filter by period type: daily, weekly, or monthly",
    ),
    since: Optional[str] = Query(
        None,
        description="ISO-8601 date string — only return rollups starting on or after this date",
    ),
    current_user: Dict[str, Any] = Depends(_require_admin),
    db: DatabaseManager = Depends(get_database),
):
    """
    Return historical usage rollups, optionally filtered by period_type
    and/or since date.

    Response shape: ``{ items: UsageRollup[] }`` per eval-usage.openapi.yaml.
    """
    gate = await _get_gate(db, current_user)

    # Parse optional period_type filter
    pt: Optional[PeriodType] = None
    if period_type is not None:
        period_type_lower = period_type.strip().lower()
        try:
            pt = PeriodType(period_type_lower)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid period_type: '{period_type}'. Must be one of: daily, weekly, monthly",
            )

    # Parse optional since filter
    since_dt: Optional[datetime] = None
    if since is not None:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid since date: '{since}'. Must be ISO-8601 format",
            )

    rollups: List[TokenUsageRollup] = await gate.repo.list_rollups(
        period_type=pt,
        since=since_dt,
    )

    # Serialize to match eval-usage.openapi.yaml UsageRollup schema
    items = []
    for r in rollups:
        items.append({
            "period_type": r.period_type.value,
            "period_start": r.period_start.strftime("%Y-%m-%d"),
            "total_tokens": r.total_tokens,
            "total_cost_usd": r.total_cost_usd,
        })

    return {"items": items}


# ---------------------------------------------------------------------------
# PUT /config — Update gate configuration (admin-only)
# ---------------------------------------------------------------------------

@router.put("/config")
async def update_gate_config(
    config: GateConfig = Body(...),
    current_user: Dict[str, Any] = Depends(_require_admin),
    db: DatabaseManager = Depends(get_database),
):
    """
    Update the tenant's LLM gate configuration.

    Request body matches ``GateConfig`` in eval-usage.openapi.yaml:
    all fields are optional and nullable (null = unlimited).
    """
    gate = await _get_gate(db, current_user)
    await gate.update_config(config)

    logger.info(
        "Gate config updated by admin=%s tenant=%s",
        current_user.get("user_id", "unknown"),
        current_user.get("db_name", "unknown"),
    )

    return {"status": "ok"}
