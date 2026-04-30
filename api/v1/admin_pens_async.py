"""Admin pen-binding API.

Reads (and selectively writes) the same `pens` collection that the BLE agent
backend (`stoody-ble-agent/server`) maintains. Both services target the same
tenant Mongo database (`skb_<institution_id>`), so admins can:

  * See which student each registered pen belongs to.
  * Hard-unbind a pen (e.g. lost, transferred, theft).
  * Pre-assign a MAC to a student before the student first connects.

The pen documents written by the agent server have:
    pen_id, user_id (= student username), pen_mac, pen_name, status,
    created_at, updated_at, last_registered_at, deregistered_at?

Cross-user uniqueness for active bindings is enforced both here (admin
pre-assignment) and on the agent backend (`register_pen`). The
production-ready guarantee comes from a partial unique index — see
`scripts/migrations/add_pen_binding_index.py` (slice #7).
"""
from __future__ import annotations

import logging
import re
import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from bson import ObjectId

from api.v1.admin_async import (
    get_tenant_db_or_403,
    require_admin_or_tutor,
)
from api.v1.auth_async import get_database
from core.database import DatabaseManager
from utils.tutor_scoping import build_tutor_class_criteria

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


class PenBindingOut(BaseModel):
    pen_id: Optional[str] = None
    pen_mac: str
    pen_name: Optional[str] = None
    status: str
    user_id: Optional[str] = None
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    last_registered_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    deregistered_at: Optional[datetime] = None


class AdminAssignPenRequest(BaseModel):
    pen_mac: str = Field(..., description="BLE MAC, AA:BB:CC:DD:EE:01")
    pen_name: str = Field(..., min_length=1, max_length=64)

    @field_validator("pen_mac")
    @classmethod
    def _check_mac(cls, v: str) -> str:
        return _normalize_mac(v)


class AdminSetPenLimitRequest(BaseModel):
    allowed_pen_count: int = Field(
        ...,
        description="How many active pen bindings this student may hold. Must be >= 1.",
    )


DEFAULT_PEN_LIMIT = 1


def _student_pen_limit(student_doc: Dict[str, Any]) -> int:
    raw = (student_doc or {}).get("allowed_pen_count")
    if isinstance(raw, int) and raw > 0:
        return raw
    return DEFAULT_PEN_LIMIT


async def _attach_student_metadata(
    tenant_db, pen_docs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Annotate each pen with the student name + student_id when resolvable.

    Joins on `students.username == pens.user_id`. Tutor- or admin-owned pens
    are returned with student fields blank rather than dropped, so admins
    notice non-student bindings instead of silently hiding them.
    """
    usernames = sorted({p.get("user_id") for p in pen_docs if p.get("user_id")})
    student_lookup: Dict[str, Dict[str, Any]] = {}
    if usernames:
        # Tenant DB collection name matches the agent backend's expectation.
        cursor = tenant_db["students"].find(
            {"username": {"$in": usernames}},
            {"username": 1, "student_id": 1, "name": 1, "full_name": 1},
        )
        async for doc in cursor:
            student_lookup[doc["username"]] = doc

    out: List[Dict[str, Any]] = []
    for p in pen_docs:
        username = p.get("user_id")
        student_doc = student_lookup.get(username) if username else None
        out.append(
            {
                "pen_id": p.get("pen_id"),
                "pen_mac": p.get("pen_mac"),
                "pen_name": p.get("pen_name"),
                "status": p.get("status", "unknown"),
                "user_id": username,
                "student_id": (student_doc or {}).get("student_id"),
                "student_name": (student_doc or {}).get("full_name")
                or (student_doc or {}).get("name"),
                "last_registered_at": p.get("last_registered_at"),
                "created_at": p.get("created_at"),
                "deregistered_at": p.get("deregistered_at"),
            }
        )
    return out


async def _resolve_student_username(tenant_db, student_id: str) -> str:
    """Look up a student's username from a stable student_id."""
    student = await tenant_db["students"].find_one(
        {"student_id": student_id},
        {"username": 1},
    )
    if not student or not student.get("username"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student {student_id} not found in this tenant",
        )
    return student["username"]


def _actor_is_admin(current_user: Dict[str, Any]) -> bool:
    return (current_user.get("user_type") or "").lower() in {"admin", "b2c_admin"}


async def _get_tutor_doc_or_403(tenant_db, current_user: Dict[str, Any]) -> Dict[str, Any]:
    if (current_user.get("user_type") or "").lower() != "tutor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or tutor access required",
        )
    if not current_user.get("can_edit_students"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tutor does not have permission to manage students",
        )
    tutor_id = current_user.get("tutor_id")
    if not tutor_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tutor identity missing from token",
        )
    tutor_doc = await tenant_db["tutors"].find_one({"tutor_id": tutor_id})
    if not tutor_doc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tutor not found in this tenant",
        )
    return tutor_doc


async def _get_tutor_visible_student_usernames(
    tenant_db,
    current_user: Dict[str, Any],
) -> set[str]:
    tutor_doc = await _get_tutor_doc_or_403(tenant_db, current_user)
    tutor_id = current_user.get("tutor_id")
    usernames: set[str] = set()

    assigned_student_ids = [s for s in (tutor_doc.get("assigned_student_ids") or []) if s]
    if assigned_student_ids:
        cursor = tenant_db["students"].find(
            {"student_id": {"$in": assigned_student_ids}},
            {"username": 1},
        )
        async for doc in cursor:
            username = doc.get("username")
            if username:
                usernames.add(username)

    cursor = tenant_db["students"].find(
        {"teacher_ids": {"$in": [tutor_id]}},
        {"username": 1},
    )
    async for doc in cursor:
        username = doc.get("username")
        if username:
            usernames.add(username)

    admin_oid = None
    raw_admin_id = current_user.get("admin_id")
    if raw_admin_id:
        try:
            admin_oid = ObjectId(raw_admin_id)
        except Exception:
            admin_oid = None
    class_query = build_tutor_class_criteria(tutor_doc, admin_oid)
    if class_query:
        cursor = tenant_db["students"].find(class_query, {"username": 1})
        async for doc in cursor:
            username = doc.get("username")
            if username:
                usernames.add(username)

    return usernames


async def _resolve_student_username_for_actor(
    tenant_db,
    current_user: Dict[str, Any],
    student_id: str,
) -> str:
    username = await _resolve_student_username(tenant_db, student_id)
    if _actor_is_admin(current_user):
        return username

    visible_usernames = await _get_tutor_visible_student_usernames(tenant_db, current_user)
    if username not in visible_usernames:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tutor can only manage pens for assigned students",
        )
    return username


async def _require_pen_manage_access_for_mac(
    tenant_db,
    current_user: Dict[str, Any],
    pen_mac: str,
) -> None:
    if _actor_is_admin(current_user):
        return
    visible_usernames = await _get_tutor_visible_student_usernames(tenant_db, current_user)
    binding = await tenant_db["pens"].find_one(
        {"pen_mac": pen_mac, "status": "active"},
        {"user_id": 1},
    )
    if not binding:
        return
    if binding.get("user_id") not in visible_usernames:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tutor can only manage pens for assigned students",
        )


@router.get("/pens", response_model=List[PenBindingOut])
async def list_all_pens_in_tenant(
    pen_status: str = Query("active", pattern="^(active|deregistered|all)$"),
    db: DatabaseManager = Depends(get_database),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
) -> List[PenBindingOut]:
    """List pen bindings in the tenant.

    Admins see all tenant bindings. Tutors with `can_edit_students`
    only see pens belonging to students in their visible scope.
    """
    tenant_db = await get_tenant_db_or_403(db, current_user)
    query: Dict[str, Any] = {} if pen_status == "all" else {"status": pen_status}
    if not _actor_is_admin(current_user):
        visible_usernames = sorted(await _get_tutor_visible_student_usernames(tenant_db, current_user))
        if not visible_usernames:
            return []
        query["user_id"] = {"$in": visible_usernames}
    pens = await tenant_db["pens"].find(query).sort("last_registered_at", -1).to_list(length=2000)
    annotated = await _attach_student_metadata(tenant_db, pens)
    return [PenBindingOut(**row) for row in annotated]


@router.get(
    "/students/{student_id}/pens",
    response_model=List[PenBindingOut],
)
async def list_pens_for_student(
    student_id: str,
    pen_status: str = Query("active", pattern="^(active|deregistered|all)$"),
    db: DatabaseManager = Depends(get_database),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
) -> List[PenBindingOut]:
    tenant_db = await get_tenant_db_or_403(db, current_user)
    username = await _resolve_student_username_for_actor(tenant_db, current_user, student_id)
    query: Dict[str, Any] = {"user_id": username}
    if pen_status != "all":
        query["status"] = pen_status
    pens = await tenant_db["pens"].find(query).sort("last_registered_at", -1).to_list(length=200)
    annotated = await _attach_student_metadata(tenant_db, pens)
    return [PenBindingOut(**row) for row in annotated]


@router.delete("/pens/{pen_mac}")
async def admin_unbind_pen(
    pen_mac: str,
    db: DatabaseManager = Depends(get_database),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
) -> Dict[str, Any]:
    """Hard-unbind a pen — flips status to 'deregistered'.

    The agent server's `register_pen` will then accept a re-claim from any
    student in the tenant. Audit trail is preserved by keeping the row.
    """
    mac = _normalize_mac(pen_mac)
    tenant_db = await get_tenant_db_or_403(db, current_user)
    await _require_pen_manage_access_for_mac(tenant_db, current_user, mac)
    result = await tenant_db["pens"].update_one(
        {"pen_mac": mac, "status": "active"},
        {
            "$set": {
                "status": "deregistered",
                "deregistered_at": datetime.utcnow(),
                "deregistered_by_admin": str(current_user.get("user_id")),
                "deregistered_by_user_type": str(current_user.get("user_type") or ""),
            }
        },
    )
    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active binding found for pen {mac}",
        )
    return {"pen_mac": mac, "status": "deregistered"}


@router.post(
    "/students/{student_id}/pens",
    response_model=PenBindingOut,
    status_code=status.HTTP_201_CREATED,
)
async def admin_assign_pen(
    student_id: str,
    payload: AdminAssignPenRequest,
    db: DatabaseManager = Depends(get_database),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
) -> PenBindingOut:
    """Pre-assign a pen MAC to a student.

    Mirrors the agent backend's uniqueness rule: the same MAC may not be
    actively bound to two different students within a tenant. If the
    student already had this pen, returns the existing row with an updated
    name (idempotent). If a different student holds the active binding,
    returns 409.
    """
    tenant_db = await get_tenant_db_or_403(db, current_user)
    username = await _resolve_student_username_for_actor(tenant_db, current_user, student_id)
    mac = payload.pen_mac
    pens = tenant_db["pens"]

    conflict = await pens.find_one(
        {"pen_mac": mac, "status": "active", "user_id": {"$ne": username}},
        {"_id": 1, "user_id": 1},
    )
    if conflict:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Pen {mac} is already bound to another student in this tenant",
        )

    # Per-student pen-limit. Mirror the agent server: count active for
    # this student, compare to `students.allowed_pen_count` (default 1).
    # Reassigning the SAME mac is always a rename and never counts.
    same_mac_already_active = await pens.find_one(
        {"user_id": username, "pen_mac": mac, "status": "active"},
        {"_id": 1},
    )
    if not same_mac_already_active:
        active_count = await pens.count_documents(
            {"user_id": username, "status": "active"}
        )
        student_doc = await tenant_db["students"].find_one(
            {"student_id": student_id}, {"allowed_pen_count": 1}
        )
        limit = _student_pen_limit(student_doc or {})
        if active_count >= limit:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Student {student_id} has reached their pen limit "
                    f"({active_count}/{limit}). Either raise the limit via "
                    f"PATCH /admin/students/{student_id}/pen-limit, or unbind an "
                    f"existing pen first."
                ),
            )

    now = datetime.utcnow()
    pen_id = f"pen_{secrets.token_urlsafe(8)}"
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
                "assigned_by_admin": str(current_user.get("user_id")),
                "assigned_by_user_type": str(current_user.get("user_type") or ""),
            },
        },
        upsert=True,
    )

    fresh = await pens.find_one({"user_id": username, "pen_mac": mac})
    annotated = await _attach_student_metadata(tenant_db, [fresh])
    return PenBindingOut(**annotated[0])


@router.patch("/students/{student_id}/pen-limit")
async def admin_set_pen_limit(
    student_id: str,
    payload: AdminSetPenLimitRequest,
    db: DatabaseManager = Depends(get_database),
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
) -> Dict[str, Any]:
    """Update a student's `allowed_pen_count`.

    The agent backend reads this on every `register_pen` and `list_pens`
    call, so the change takes effect on the student's next interaction —
    no agent restart needed. Lowering the limit below the current bound
    count is permitted; existing bindings stay until an admin removes
    them via DELETE /admin/pens/{mac}.
    """
    if payload.allowed_pen_count < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="allowed_pen_count must be >= 1. Use DELETE /admin/pens/{mac} to unbind a specific pen.",
        )
    tenant_db = await get_tenant_db_or_403(db, current_user)
    await _resolve_student_username_for_actor(tenant_db, current_user, student_id)
    result = await tenant_db["students"].update_one(
        {"student_id": student_id},
        {"$set": {"allowed_pen_count": payload.allowed_pen_count}},
    )
    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Student {student_id} not found in this tenant",
        )
    return {"student_id": student_id, "allowed_pen_count": payload.allowed_pen_count}
