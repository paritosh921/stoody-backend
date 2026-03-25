"""
EvalPen Invigilator Code Generation API — generates 6-char alphanumeric
invigilator codes for supervised exam sessions.

Provides endpoints for admins/tutors to:
  1. Generate a unique invigilator code tied to a specific exam

Architecture:
    Pure MongoDB + secrets — no exam-conductor imports needed.

Ownership Declaration:
    - Writes:  exampen_invigilator_codes (tenant DB)
    - Reads from: exampen_invigilator_codes (collision check)

Hard constraints:
    - C1: MongoDB only
    - Codes stored in tenant DB, NOT master DB
"""

from __future__ import annotations

import logging
import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CODE_ALPHABET = string.ascii_uppercase + string.digits
_CODE_LENGTH = 6
_CODE_TTL_HOURS = 24
_MAX_COLLISION_RETRIES = 1


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

def require_admin_or_tutor(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Dependency: require admin or tutor role for invigilator operations."""
    allowed = {"admin", "tutor", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or tutor access required for invigilator operations",
        )
    return current_user


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateCodeRequest(BaseModel):
    """Request body for generating an invigilator code."""

    exam_id: str = Field(
        ...,
        min_length=1,
        description="ID of the exam to generate an invigilator code for",
    )


class GenerateCodeResponse(BaseModel):
    """Response for a successfully generated invigilator code."""

    code: str = Field(..., description="6-char uppercase alphanumeric code")
    exam_id: str = Field(..., description="Associated exam ID")
    expires_at: str = Field(..., description="ISO-8601 expiration timestamp")


# ---------------------------------------------------------------------------
# Helper: resolve tenant DB
# ---------------------------------------------------------------------------

async def _get_tenant_db(
    db: DatabaseManager,
    current_user: Dict[str, Any],
) -> Any:
    """Resolve the tenant database from the authenticated user's JWT claims."""
    db_name = current_user.get("db_name")
    if not db_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context missing from authentication token",
        )
    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant database not available",
        )
    return tenant_db


# ---------------------------------------------------------------------------
# Helper: generate code
# ---------------------------------------------------------------------------

def _generate_code() -> str:
    """Generate a 6-character uppercase alphanumeric code."""
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LENGTH))


# ---------------------------------------------------------------------------
# Helper: ensure unique index
# ---------------------------------------------------------------------------

_index_ensured = False


async def _ensure_code_index(collection) -> None:
    """Create a unique index on the `code` field (idempotent, once per process)."""
    global _index_ensured
    if _index_ensured:
        return
    await collection.create_index("code", unique=True)
    _index_ensured = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/generate-code",
    response_model=GenerateCodeResponse,
    summary="Generate a 6-char invigilator code for an exam",
    responses={
        403: {"description": "Insufficient permissions"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def generate_invigilator_code(
    body: GenerateCodeRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> GenerateCodeResponse:
    """Generate a unique 6-character alphanumeric invigilator code.

    The code is stored in the tenant DB collection
    ``exampen_invigilator_codes`` with a 24-hour TTL from generation time.
    If a code collision occurs (extremely unlikely), one retry is attempted.
    """
    tenant_db = await _get_tenant_db(db, current_user)

    try:
        collection = tenant_db["exampen_invigilator_codes"]
        await _ensure_code_index(collection)

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=_CODE_TTL_HOURS)
        generated_by = current_user.get("user_id", "unknown")

        code = _generate_code()

        doc = {
            "code": code,
            "exam_id": body.exam_id,
            "generated_by": generated_by,
            "generated_at": now,
            "expires_at": expires_at,
            "used": False,
        }

        # Attempt insert; retry once on duplicate key (collision)
        for attempt in range(_MAX_COLLISION_RETRIES + 1):
            try:
                await collection.insert_one(doc)
                break
            except Exception as insert_exc:
                # Check for duplicate key error (MongoDB error code 11000)
                if (
                    hasattr(insert_exc, "code")
                    and insert_exc.code == 11000
                    and attempt < _MAX_COLLISION_RETRIES
                ):
                    logger.warning(
                        "Invigilator code collision on '%s', retrying (attempt %d)",
                        code,
                        attempt + 1,
                    )
                    code = _generate_code()
                    doc["code"] = code
                    continue
                raise

        logger.info(
            "Invigilator code '%s' generated for exam %s by %s",
            code,
            body.exam_id,
            generated_by,
        )

        return GenerateCodeResponse(
            code=code,
            exam_id=body.exam_id,
            expires_at=expires_at.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to generate invigilator code for exam %s: %s",
            body.exam_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate invigilator code",
        )
