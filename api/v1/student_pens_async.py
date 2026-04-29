"""Student-facing pen-binding API.

Mirrors `admin_pens_async` but scoped to the calling student via the
student JWT — so the mobile app (and any future student-side web flow)
has a way to read its own pen list and auto-bind a fresh pen on first
connect, without needing the device-token machinery the desktop agent
uses against the agent backend.

All reads/writes hit the SAME tenant `pens` collection that the agent
server (`stoody-ble-agent/server`) writes during desktop registration.
That collection is the single source of truth for "which pen belongs to
which student", regardless of whether the student first registered from
the desktop, the mobile app, or an admin pre-assignment.

Deletion is intentionally NOT exposed here. Per slice #9, students can
never unbind their own pen — only an admin can, via
`DELETE /api/v1/admin/pens/{mac}`.
"""
from __future__ import annotations

import logging
import re
import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from api.v1.admin_async import get_tenant_db_or_403
from api.v1.admin_pens_async import DEFAULT_PEN_LIMIT, _student_pen_limit
from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()


_MAC_PATTERN = re.compile(r"^[0-9A-F]{2}(:[0-9A-F]{2}){5}$")


def _normalize_mac(value: str) -> str:
    mac = (value or "").strip().upper()
    if not _MAC_PATTERN.match(mac):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="pen_mac must be 6 hex octets separated by colons (e.g. AA:BB:CC:DD:EE:01)",
        )
    return mac


def _require_student(current_user: Dict[str, Any]) -> str:
    """Ensure caller is a student and return their `username`.

    Username is what the agent server writes as `pens.user_id`, so we
    use it as the join key into the bindings collection.
    """
    if (current_user.get("user_type") or "").lower() != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only students can use this endpoint.",
        )
    username = current_user.get("username")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Student token is missing username.",
        )
    return username


class StudentPenOut(BaseModel):
    pen_id: Optional[str] = None
    pen_mac: str
    pen_name: Optional[str] = None
    status: str
    last_registered_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class StudentPenListResponse(BaseModel):
    pens: List[StudentPenOut]
    current_count: int
    allowed_pen_count: int


class StudentRegisterPenRequest(BaseModel):
    pen_mac: str = Field(..., description="BLE MAC, AA:BB:CC:DD:EE:01")
    pen_name: str = Field(..., min_length=1, max_length=64)

    @field_validator("pen_mac")
    @classmethod
    def _check_mac(cls, v: str) -> str:
        return _normalize_mac(v)


class StudentRegisterPenResponse(BaseModel):
    pen_id: str
    pen_mac: str
    pen_name: str
    status: str


async def _resolve_limit(tenant_db, username: str) -> int:
    student_doc = await tenant_db["students"].find_one(
        {"username": username},
        {"allowed_pen_count": 1},
    )
    return _student_pen_limit(student_doc or {})


def _serialize(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pen_id": p.get("pen_id"),
        "pen_mac": p.get("pen_mac"),
        "pen_name": p.get("pen_name"),
        "status": p.get("status", "unknown"),
        "last_registered_at": p.get("last_registered_at"),
        "created_at": p.get("created_at"),
    }


@router.get("/pens")
async def student_list_my_pens(
    db: DatabaseManager = Depends(get_database),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return the calling student's active bindings + per-student policy.

    Mobile clients call this on screen focus to know:
      * which MAC(s) to allow connecting to;
      * the per-student `allowed_pen_count` so they can show "1/1 used"
        and gate the auto-bind path.
    """
    username = _require_student(current_user)
    tenant_db = await get_tenant_db_or_403(db, current_user)

    pens = await tenant_db["pens"].find(
        {"user_id": username, "status": "active"}
    ).sort("last_registered_at", -1).to_list(length=200)

    limit = await _resolve_limit(tenant_db, username)
    return {
        "pens": [_serialize(p) for p in pens],
        "current_count": len(pens),
        "allowed_pen_count": limit,
    }


@router.post("/pens", status_code=status.HTTP_200_OK)
async def student_register_my_pen(
    payload: StudentRegisterPenRequest,
    db: DatabaseManager = Depends(get_database),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Bind a pen to the calling student.

    Same rules as the agent server's `register_pen`:
      * Cross-user uniqueness — another student holding this MAC active → 409.
      * Per-student limit — `current >= allowed_pen_count` → 409 (with a
        hint to ask the admin to raise the limit). Re-registering the
        SAME mac is always allowed and never counts toward the limit.
    """
    username = _require_student(current_user)
    tenant_db = await get_tenant_db_or_403(db, current_user)
    pens = tenant_db["pens"]
    mac = payload.pen_mac

    conflict = await pens.find_one(
        {"pen_mac": mac, "status": "active", "user_id": {"$ne": username}},
        {"_id": 1},
    )
    if conflict:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Pen already registered to another student in this tenant.",
        )

    same_mac_already_active = await pens.find_one(
        {"user_id": username, "pen_mac": mac, "status": "active"},
        {"_id": 1},
    )
    if not same_mac_already_active:
        active_count = await pens.count_documents(
            {"user_id": username, "status": "active"}
        )
        limit = await _resolve_limit(tenant_db, username)
        if active_count >= limit:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Pen limit reached ({active_count}/{limit}). "
                    "Ask your admin to raise your pen limit, or to unbind one of your existing pens."
                ),
            )

    pen_id = f"pen_{secrets.token_urlsafe(8)}"
    now = datetime.utcnow()
    await pens.update_one(
        {"user_id": username, "pen_mac": mac},
        {
            "$setOnInsert": {
                "pen_id": pen_id,
                "user_id": username,
                "created_at": now,
            },
            "$set": {
                "pen_name": payload.pen_name,
                "status": "active",
                "updated_at": now,
                "last_registered_at": now,
            },
        },
        upsert=True,
    )

    fresh = await pens.find_one({"user_id": username, "pen_mac": mac})
    return {
        "pen_id": (fresh or {}).get("pen_id", pen_id),
        "pen_mac": mac,
        "pen_name": payload.pen_name,
        "status": "registered",
    }
