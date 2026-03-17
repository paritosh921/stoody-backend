"""
Pen Settings API — GET/PUT user Fn button action preferences.

Stores only action IDs in the user document under `pen_settings.fn_actions`.
- Tenant students/tutors: reads/writes against tenant DB collections.
- B2C users: reads/writes against the dedicated B2C `users` collection.
"""

import logging
from typing import Any, Dict, List

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from api.v1.auth_async import get_current_user
from core.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Action validation constants (must match frontend fnActions.ts)
# ---------------------------------------------------------------------------

ALL_ACTIONS = {"CYCLE_COLOR", "TOGGLE_ERASER", "TOGGLE_RULED", "LOCK_QUESTION", "SHARE_SCREEN"}

ROLE_ALLOWED_ACTIONS: Dict[str, set] = {
    "student": {"CYCLE_COLOR", "TOGGLE_ERASER", "TOGGLE_RULED", "SHARE_SCREEN"},
    "b2c_user": {"CYCLE_COLOR", "TOGGLE_ERASER", "TOGGLE_RULED", "SHARE_SCREEN"},
    "tutor": ALL_ACTIONS,
}

DEFAULT_FN_ACTIONS: Dict[str, List[str]] = {
    "student": ["CYCLE_COLOR", "TOGGLE_ERASER", "TOGGLE_RULED", "SHARE_SCREEN"],
    "b2c_user": ["CYCLE_COLOR", "TOGGLE_ERASER", "TOGGLE_RULED", "SHARE_SCREEN"],
    "tutor": ["CYCLE_COLOR", "TOGGLE_ERASER", "TOGGLE_RULED", "LOCK_QUESTION", "SHARE_SCREEN"],
}


async def get_database(request: Request) -> DatabaseManager:
    return request.app.state.db

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_role(user_type: str) -> str:
    """Normalize user_type to a role key."""
    if user_type in ("student", "b2c_user", "tutor"):
        return user_type
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"Pen settings not available for role '{user_type}'",
    )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PenSettingsResponse(BaseModel):
    fn_actions: List[str]


class PenSettingsUpdate(BaseModel):
    fn_actions: List[str] = Field(..., min_length=1, max_length=10)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=PenSettingsResponse)
async def get_pen_settings(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    user_type = current_user.get("user_type", "")
    role = _normalize_role(user_type)
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")

    # B2C users live in the dedicated B2C database, not a tenant DB
    if role == "b2c_user":
        doc = await db.b2c_find_one(
            "users",
            {"_id": ObjectId(user_id)},
            {"pen_settings": 1},
        )
    else:
        db_name = current_user.get("db_name")
        if not db_name:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
        tenant_db = db.client[db_name]
        collection_name = "tutors" if role == "tutor" else "students"
        doc = await tenant_db[collection_name].find_one(
            {"_id": ObjectId(user_id)},
            {"pen_settings": 1},
        )

    if doc and doc.get("pen_settings", {}).get("fn_actions"):
        return PenSettingsResponse(fn_actions=doc["pen_settings"]["fn_actions"])

    # Return defaults for this role
    return PenSettingsResponse(fn_actions=DEFAULT_FN_ACTIONS.get(role, DEFAULT_FN_ACTIONS["student"]))


@router.put("", response_model=PenSettingsResponse)
async def update_pen_settings(
    body: PenSettingsUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    user_type = current_user.get("user_type", "")
    role = _normalize_role(user_type)
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")

    allowed = ROLE_ALLOWED_ACTIONS.get(role, set())
    invalid = [a for a in body.fn_actions if a not in allowed]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Actions not allowed for role '{role}': {invalid}",
        )

    # Check for duplicates
    if len(body.fn_actions) != len(set(body.fn_actions)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Duplicate actions are not allowed",
        )

    update_op = {"$set": {"pen_settings.fn_actions": body.fn_actions}}

    if role == "b2c_user":
        # Use the collection directly to check matched_count (not modified_count).
        # The db.b2c_update_one() wrapper returns False on no-op updates (same data),
        # which would incorrectly 404 when the user saves an unchanged action list.
        b2c_collection = await db.get_b2c_collection("users")
        if b2c_collection is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="B2C database unavailable",
            )
        result = await b2c_collection.update_one(
            {"_id": ObjectId(user_id)},
            update_op,
        )
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User document not found",
            )
    else:
        db_name = current_user.get("db_name")
        if not db_name:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
        tenant_db = db.client[db_name]
        collection_name = "tutors" if role == "tutor" else "students"
        result = await tenant_db[collection_name].update_one(
            {"_id": ObjectId(user_id)},
            update_op,
        )
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User document not found",
            )

    return PenSettingsResponse(fn_actions=body.fn_actions)
