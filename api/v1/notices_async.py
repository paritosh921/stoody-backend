"""
Notice system for Stoody platform.

Allows teachers to send notices to their students and admins to send
notices to teachers, students, classes, or the entire school.
Notices create notifications via the existing notification infrastructure.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from bson import ObjectId

from fastapi import APIRouter, Request, HTTPException, Depends, status, Query, Body
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database
from api.v1.notifications_async import create_notifications_batch

logger = logging.getLogger(__name__)

router = APIRouter(tags=["notices"])
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AudienceSpec(BaseModel):
    type: str = Field(..., description="individual | class | school")
    recipient_type: str = Field("student", description="student | tutor | all")
    recipient_ids: Optional[List[str]] = None
    grades: Optional[List[str]] = None
    sections: Optional[List[str]] = None


class CreateNoticeRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1, max_length=5000)
    priority: str = Field("normal", description="normal | important | urgent")
    audience: AudienceSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _resolve_recipients(
    db: DatabaseManager,
    audience: AudienceSpec,
    current_user: Dict[str, Any],
) -> tuple[List[str], List[str]]:
    """
    Resolve audience spec into (student_ids, tutor_ids).

    For tutors, restricts audience to their own assigned students and
    teaching assignment classes. Admins have no restrictions within tenant.
    """
    user_type = current_user.get("user_type", "")
    student_ids: List[str] = []
    tutor_ids: List[str] = []

    if audience.type == "individual":
        # Direct IDs provided
        ids = audience.recipient_ids or []
        if audience.recipient_type == "student":
            # Validate tutor scope
            if user_type == "tutor":
                allowed = await _get_tutor_student_ids(db, current_user)
                student_ids = [sid for sid in ids if sid in allowed]
            else:
                student_ids = ids
        elif audience.recipient_type == "tutor":
            if user_type == "tutor":
                raise HTTPException(400, "Teachers cannot send notices to other teachers")
            tutor_ids = ids
        elif audience.recipient_type == "all":
            if user_type == "tutor":
                raise HTTPException(400, "Teachers cannot send school-wide notices")
            # Split: treat as student IDs (admin would use class/school for mixed)
            student_ids = ids

    elif audience.type == "class":
        grades = audience.grades or []
        sections = audience.sections or []
        if not grades:
            raise HTTPException(400, "grades required for class targeting")

        # If tutor, verify they teach these grades/sections
        if user_type == "tutor":
            allowed_classes = await _get_tutor_classes(db, current_user)
            for g in grades:
                for s in (sections or [""]):
                    if (g, s) not in allowed_classes and (g, "") not in allowed_classes:
                        raise HTTPException(
                            403,
                            f"You don't teach class {g}{'-' + s if s else ''}"
                        )

        # Resolve students in these classes
        if audience.recipient_type in ("student", "all"):
            filt: Dict[str, Any] = {"grade": {"$in": grades}}
            if sections:
                filt["section"] = {"$in": sections}
            students = await db.mongo_find("students", filt, projection={"_id": 1})
            student_ids = [str(s["_id"]) for s in students]

        # Resolve tutors teaching these classes
        if audience.recipient_type in ("tutor", "all"):
            if user_type == "tutor":
                raise HTTPException(400, "Teachers cannot send notices to other teachers")
            tutors = await db.mongo_find("tutors", {}, projection={"_id": 1, "teaching_assignments": 1})
            for t in tutors:
                for ta in (t.get("teaching_assignments") or []):
                    if ta.get("standard") in grades:
                        if not sections or any(s in (ta.get("sections") or []) for s in sections):
                            tutor_ids.append(str(t["_id"]))
                            break

    elif audience.type == "school":
        if user_type == "tutor":
            raise HTTPException(400, "Teachers cannot send school-wide notices")

        if audience.recipient_type in ("student", "all"):
            students = await db.mongo_find("students", {}, projection={"_id": 1})
            student_ids = [str(s["_id"]) for s in students]
        if audience.recipient_type in ("tutor", "all"):
            tutors = await db.mongo_find("tutors", {}, projection={"_id": 1})
            tutor_ids = [str(t["_id"]) for t in tutors]

    else:
        raise HTTPException(400, f"Invalid audience type: {audience.type}")

    # Deduplicate
    student_ids = list(set(student_ids))
    tutor_ids = list(set(tutor_ids))

    return student_ids, tutor_ids


async def _get_tutor_student_ids(db: DatabaseManager, current_user: Dict[str, Any]) -> set:
    """Get the set of student IDs a tutor is allowed to target."""
    user_id = current_user.get("user_id") or current_user.get("id")
    tutor_doc = await db.mongo_find_one("tutors", {"_id": ObjectId(user_id)})
    if not tutor_doc:
        return set()

    allowed = set()

    # Directly assigned students
    for sid in (tutor_doc.get("assigned_student_ids") or []):
        allowed.add(sid)

    # Students matching teaching assignments (grade + section)
    for ta in (tutor_doc.get("teaching_assignments") or []):
        grade = ta.get("standard")
        sections = ta.get("sections") or []
        if not grade:
            continue
        filt: Dict[str, Any] = {"grade": grade}
        if sections:
            filt["section"] = {"$in": sections}
        students = await db.mongo_find("students", filt, projection={"_id": 1})
        for s in students:
            allowed.add(str(s["_id"]))

    return allowed


async def _get_tutor_classes(db: DatabaseManager, current_user: Dict[str, Any]) -> set:
    """Get set of (grade, section) tuples the tutor teaches."""
    user_id = current_user.get("user_id") or current_user.get("id")
    tutor_doc = await db.mongo_find_one("tutors", {"_id": ObjectId(user_id)})
    if not tutor_doc:
        return set()

    classes = set()
    for ta in (tutor_doc.get("teaching_assignments") or []):
        grade = ta.get("standard", "")
        for sec in (ta.get("sections") or [""]):
            classes.add((grade, sec))
    return classes


async def _resolve_sender_name(
    db: DatabaseManager,
    current_user: Dict[str, Any],
    user_type: str,
    user_id: str,
) -> str:
    """
    Resolve a human-readable sender name.
    Admin → school name from school_settings, fallback to admin doc name.
    Tutor → tutor name from tutors collection.
    """
    if user_type == "admin":
        # Try school_settings → school_info.school_name
        try:
            settings = await db.mongo_find_one("school_settings", {})
            school_name = (settings or {}).get("school_info", {}).get("school_name", "")
            if school_name:
                return school_name
        except Exception:
            pass
        # Fallback: admin doc name
        try:
            admin_doc = await db.mongo_find_one("admins", {"_id": ObjectId(user_id)})
            if admin_doc:
                name = admin_doc.get("full_name") or admin_doc.get("name") or ""
                if name:
                    return name
        except Exception:
            pass
        return current_user.get("username") or current_user.get("email") or "School Admin"

    if user_type == "tutor":
        try:
            tutor_doc = await db.mongo_find_one("tutors", {"_id": ObjectId(user_id)})
            if tutor_doc:
                name = tutor_doc.get("name") or tutor_doc.get("full_name") or ""
                if name:
                    return name
        except Exception:
            pass
        return current_user.get("username") or current_user.get("email") or "Teacher"

    return current_user.get("name") or current_user.get("username") or "Unknown"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("")
@limiter.limit("30/minute")
async def create_notice(
    request: Request,
    payload: CreateNoticeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Create and send a notice. Available to admins and tutors."""
    user_type = current_user.get("user_type", "")
    if user_type not in ("admin", "tutor"):
        raise HTTPException(403, "Only admins and teachers can create notices")

    if payload.priority not in ("normal", "important", "urgent"):
        raise HTTPException(400, "priority must be normal, important, or urgent")

    # Resolve recipients
    student_ids, tutor_ids = await _resolve_recipients(db, payload.audience, current_user)
    all_recipients = student_ids + tutor_ids

    if not all_recipients:
        raise HTTPException(400, "No recipients found for the specified audience")

    user_id = str(
        current_user.get("user_id")
        or current_user.get("id")
        or ""
    )
    admin_id = current_user.get("admin_id")

    # Resolve sender display name
    user_name = await _resolve_sender_name(db, current_user, user_type, user_id)
    if user_type == "admin":
        admin_id = admin_id or current_user.get("user_id") or current_user.get("id")

    # Ensure ObjectId
    if admin_id and not isinstance(admin_id, ObjectId):
        try:
            admin_id = ObjectId(str(admin_id))
        except Exception:
            pass

    now = datetime.utcnow()

    # Build audience summary for display
    audience_label = await _build_audience_label(
        db, payload.audience, student_ids, tutor_ids,
    )

    # Store the notice document
    notice_doc = {
        "admin_id": admin_id,
        "title": payload.title,
        "body": payload.body,
        "priority": payload.priority,
        "created_by": user_id,
        "created_by_name": user_name,
        "created_by_type": user_type,
        "audience": {
            "type": payload.audience.type,
            "recipient_type": payload.audience.recipient_type,
            "recipient_ids": payload.audience.recipient_ids,
            "grades": payload.audience.grades,
            "sections": payload.audience.sections,
        },
        "audience_label": audience_label,
        "resolved_recipients": all_recipients,
        "recipient_count": len(all_recipients),
        "created_at": now,
    }

    result = await db.mongo_insert_one("notices", notice_doc)
    if not result:
        raise HTTPException(500, "Failed to create notice")

    notice_id = str(result)

    # Create notifications for all recipients via existing infrastructure
    preview = payload.body[:100] + ("..." if len(payload.body) > 100 else "")
    metadata = {"notice_id": notice_id, "priority": payload.priority}

    # Send to students
    if student_ids:
        await create_notifications_batch(
            db=db,
            admin_id=admin_id,
            recipient_ids=student_ids,
            notif_type="notice",
            category="notice",
            title=payload.title,
            message=preview,
            metadata=metadata,
            created_by=user_id,
            created_by_name=user_name,
            recipient_type="student",
        )

    # Send to tutors
    if tutor_ids:
        await create_notifications_batch(
            db=db,
            admin_id=admin_id,
            recipient_ids=tutor_ids,
            notif_type="notice",
            category="notice",
            title=payload.title,
            message=preview,
            metadata=metadata,
            created_by=user_id,
            created_by_name=user_name,
            recipient_type="tutor",
        )

    logger.info(
        f"Notice created | id={notice_id} by={user_name} "
        f"recipients={len(all_recipients)} priority={payload.priority}"
    )

    return {
        "success": True,
        "notice_id": notice_id,
        "recipient_count": len(all_recipients),
        "audience_label": audience_label,
    }


async def _build_audience_label(
    db: DatabaseManager,
    audience: AudienceSpec,
    student_ids: List[str],
    tutor_ids: List[str],
) -> str:
    """Human-readable audience label for display."""
    n_students = len(student_ids)
    n_tutors = len(tutor_ids)

    if audience.type == "school":
        parts = []
        if n_students:
            parts.append(f"{n_students} student{'s' if n_students != 1 else ''}")
        if n_tutors:
            parts.append(f"{n_tutors} teacher{'s' if n_tutors != 1 else ''}")
        return "Entire School — " + ", ".join(parts) if parts else "Entire School"

    if audience.type == "class":
        grades = audience.grades or []
        sections = audience.sections or []
        class_str = ", ".join(
            f"Class {g}" + (f"-{','.join(sections)}" if sections else "")
            for g in grades
        )
        return class_str + f" ({n_students + n_tutors} recipients)"

    if audience.type == "individual":
        total = n_students + n_tutors
        # For small groups, show actual names
        if total <= 5:
            names = []
            if student_ids:
                try:
                    oids = [ObjectId(sid) for sid in student_ids if ObjectId.is_valid(sid)]
                    if oids:
                        docs = await db.mongo_find(
                            "students", {"_id": {"$in": oids}},
                            projection={"name": 1},
                        )
                        names.extend(d.get("name", "Student") for d in docs)
                except Exception:
                    pass
            if tutor_ids:
                try:
                    oids = [ObjectId(tid) for tid in tutor_ids if ObjectId.is_valid(tid)]
                    if oids:
                        docs = await db.mongo_find(
                            "tutors", {"_id": {"$in": oids}},
                            projection={"name": 1},
                        )
                        names.extend(d.get("name", "Teacher") for d in docs)
                except Exception:
                    pass
            if names:
                return ", ".join(names)
        return f"{total} individual{'s' if total != 1 else ''}"

    return f"{n_students + n_tutors} recipients"


@router.get("/sent")
@limiter.limit("60/minute")
async def get_sent_notices(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=50),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """List notices created by the current user (admin or tutor)."""
    user_type = current_user.get("user_type", "")
    if user_type not in ("admin", "tutor"):
        raise HTTPException(403, "Only admins and teachers can view sent notices")

    user_id = str(
        current_user.get("user_id")
        or current_user.get("id")
        or ""
    )

    filt = {"created_by": user_id}
    skip_n = (page - 1) * limit

    total = await db.mongo_count("notices", filt)
    docs = await db.mongo_find(
        "notices", filt,
        sort=[("created_at", -1)],
        skip=skip_n,
        limit=limit,
    )

    notices = []
    for d in docs:
        created_at = d.get("created_at")
        notices.append({
            "id": str(d["_id"]),
            "title": d.get("title", ""),
            "body": d.get("body", ""),
            "priority": d.get("priority", "normal"),
            "audience_label": d.get("audience_label", ""),
            "audience": d.get("audience", {}),
            "recipient_count": d.get("recipient_count", 0),
            "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or ""),
        })

    return {"notices": notices, "total": total, "page": page, "limit": limit}


@router.get("/received")
@limiter.limit("60/minute")
async def get_received_notices(
    request: Request,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=50),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """List notices sent to the current user (student or tutor)."""
    user_id = str(
        current_user.get("user_id")
        or current_user.get("id")
        or ""
    )

    filt = {"resolved_recipients": user_id}
    skip_n = (page - 1) * limit

    total = await db.mongo_count("notices", filt)
    docs = await db.mongo_find(
        "notices", filt,
        sort=[("created_at", -1)],
        skip=skip_n,
        limit=limit,
    )

    notices = []
    for d in docs:
        created_at = d.get("created_at")
        notices.append({
            "id": str(d["_id"]),
            "title": d.get("title", ""),
            "body": d.get("body", ""),
            "priority": d.get("priority", "normal"),
            "created_by_name": d.get("created_by_name", ""),
            "created_by_type": d.get("created_by_type", ""),
            "audience_label": d.get("audience_label", ""),
            "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or ""),
        })

    return {"notices": notices, "total": total, "page": page, "limit": limit}


@router.get("/{notice_id}")
@limiter.limit("60/minute")
async def get_notice_detail(
    request: Request,
    notice_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Get full notice detail. Accessible to creator or any recipient."""
    try:
        oid = ObjectId(notice_id)
    except Exception:
        raise HTTPException(400, "Invalid notice ID")

    doc = await db.mongo_find_one("notices", {"_id": oid})
    if not doc:
        raise HTTPException(404, "Notice not found")

    # Access check: creator or recipient
    user_id = str(
        current_user.get("user_id")
        or current_user.get("id")
        or ""
    )
    is_creator = doc.get("created_by") == user_id
    is_recipient = user_id in (doc.get("resolved_recipients") or [])
    is_admin = current_user.get("user_type") == "admin"

    if not (is_creator or is_recipient or is_admin):
        raise HTTPException(403, "Access denied")

    created_at = doc.get("created_at")
    return {
        "id": str(doc["_id"]),
        "title": doc.get("title", ""),
        "body": doc.get("body", ""),
        "priority": doc.get("priority", "normal"),
        "created_by_name": doc.get("created_by_name", ""),
        "created_by_type": doc.get("created_by_type", ""),
        "audience_label": doc.get("audience_label", ""),
        "recipient_count": doc.get("recipient_count", 0),
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or ""),
    }


@router.delete("/{notice_id}")
@limiter.limit("30/minute")
async def delete_notice(
    request: Request,
    notice_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Delete a notice. Only the creator can delete."""
    try:
        oid = ObjectId(notice_id)
    except Exception:
        raise HTTPException(400, "Invalid notice ID")

    user_id = str(
        current_user.get("user_id")
        or current_user.get("id")
        or ""
    )

    doc = await db.mongo_find_one("notices", {"_id": oid})
    if not doc:
        raise HTTPException(404, "Notice not found")

    if doc.get("created_by") != user_id:
        raise HTTPException(403, "Only the creator can delete a notice")

    await db.mongo_delete_one("notices", {"_id": oid})
    logger.info(f"Notice deleted | id={notice_id} by={user_id}")

    return {"success": True}


# ---------------------------------------------------------------------------
# Tutor audience helpers — exposed for the frontend to populate dropdowns
# ---------------------------------------------------------------------------

@router.get("/audience/my-students")
@limiter.limit("30/minute")
async def get_my_students(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Return list of students the current tutor can send notices to."""
    user_type = current_user.get("user_type", "")
    if user_type == "tutor":
        allowed_ids = await _get_tutor_student_ids(db, current_user)
        if not allowed_ids:
            return {"students": []}
        # Fetch names
        students = await db.mongo_find(
            "students",
            {"_id": {"$in": [ObjectId(sid) for sid in allowed_ids if ObjectId.is_valid(sid)]}},
            projection={"_id": 1, "name": 1, "student_id": 1, "grade": 1, "section": 1},
        )
        return {
            "students": [
                {
                    "id": str(s["_id"]),
                    "name": s.get("name", ""),
                    "student_id": s.get("student_id", ""),
                    "grade": s.get("grade", ""),
                    "section": s.get("section", ""),
                }
                for s in students
            ]
        }
    elif user_type == "admin":
        # Admin sees all students
        students = await db.mongo_find(
            "students", {},
            projection={"_id": 1, "name": 1, "student_id": 1, "grade": 1, "section": 1},
        )
        return {
            "students": [
                {
                    "id": str(s["_id"]),
                    "name": s.get("name", ""),
                    "student_id": s.get("student_id", ""),
                    "grade": s.get("grade", ""),
                    "section": s.get("section", ""),
                }
                for s in students
            ]
        }
    return {"students": []}


@router.get("/audience/my-classes")
@limiter.limit("30/minute")
async def get_my_classes(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Return classes the tutor teaches or all classes for admin."""
    user_type = current_user.get("user_type", "")

    if user_type == "tutor":
        classes = await _get_tutor_classes(db, current_user)
        return {
            "classes": [
                {"grade": g, "section": s}
                for g, s in sorted(classes)
                if g  # skip empty
            ]
        }
    elif user_type == "admin":
        # Derive unique grade+section combos from students
        students = await db.mongo_find(
            "students", {},
            projection={"grade": 1, "section": 1},
        )
        classes_set = set()
        for s in students:
            g = s.get("grade", "")
            sec = s.get("section", "")
            if g:
                classes_set.add((g, sec))
        return {
            "classes": [
                {"grade": g, "section": s}
                for g, s in sorted(classes_set)
            ]
        }
    return {"classes": []}


@router.get("/audience/tutors")
@limiter.limit("30/minute")
async def get_all_tutors(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    """Return all tutors for admin notice targeting."""
    user_type = current_user.get("user_type", "")
    if user_type != "admin":
        raise HTTPException(403, "Only admins can list tutors for notices")

    tutors = await db.mongo_find(
        "tutors", {},
        projection={"_id": 1, "name": 1, "tutor_id": 1, "standards": 1, "sections": 1},
    )
    return {
        "tutors": [
            {
                "id": str(t["_id"]),
                "name": t.get("name", ""),
                "tutor_id": t.get("tutor_id", ""),
                "standards": t.get("standards", []),
                "sections": t.get("sections", []),
            }
            for t in tutors
        ]
    }
