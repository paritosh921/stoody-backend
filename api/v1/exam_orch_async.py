"""
ExamPen Exam Orchestration API — conducted exam lifecycle management.

Handles:
  - Exam create / list / view
  - Lifecycle transitions: draft -> armed -> in_progress -> collection_closed -> uploading -> ready_for_eval
  - Hub assignment / unassignment
  - Upload progress tracking

Architecture:
    IMPLEMENTATION_PLAN.md §UP-001
    architecture/DUAL_MODE_ARCHITECTURE.md §3
    integration/HUB_DEPLOYMENT_SPEC.md

Ownership Declaration:
    - Writes:  exampen_exams (exam lifecycle, hub assignments)
    - Reads from: exampen_exams, evalpen_submissions (progress rollup)
    - Never writes to: documents, practice persistence

Hard constraints:
    - C1: MongoDB only
    - C5: Ownership boundaries — backend is single writable owner for exam lifecycle
    - Lifecycle transitions are strictly ordered (no skipping states)

API authority:
    new-docs/api/exam-orch.openapi.yaml (to be created)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIFECYCLE_STATES = (
    "draft",
    "armed",
    "in_progress",
    "collection_closed",
    "uploading",
    "ready_for_eval",
)

LIFECYCLE_TRANSITIONS = {
    "draft": {"armed"},
    "armed": {"in_progress"},
    "in_progress": {"collection_closed"},
    "collection_closed": {"uploading"},
    "uploading": {"ready_for_eval"},
}


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

def require_admin_or_tutor(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Require admin or tutor role."""
    allowed = {"admin", "tutor", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or tutor access required for exam operations",
        )
    return current_user


def require_admin(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Require admin role only."""
    allowed = {"admin", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


# ---------------------------------------------------------------------------
# Tenant DB helper
# ---------------------------------------------------------------------------

async def _get_tenant_db(
    db: DatabaseManager,
    current_user: Dict[str, Any],
) -> Any:
    db_name = current_user.get("db_name")
    if not db_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context missing from token",
        )
    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant database not available",
        )
    return tenant_db


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class HubAssignment(BaseModel):
    hub_id: str
    hub_name: Optional[str] = None
    assigned_at: Optional[str] = None
    session_started_at: Optional[str] = None
    session_ended_at: Optional[str] = None


class ExamCreateRequest(BaseModel):
    exam_id: str = Field(..., min_length=1, description="Unique exam identifier")
    exam_type: Optional[str] = Field(
        None,
        description="dcr or pcr. When omitted, derived from prepared document's exam_mode.",
    )
    prepared_document_id: Optional[str] = Field(None, description="Linked prepared document")
    roster: Optional[List[str]] = Field(default_factory=list, description="Student IDs")
    duration_minutes: Optional[int] = Field(None, ge=1)
    hub_assignments: Optional[List[HubAssignment]] = Field(default_factory=list)


class ExamDetailResponse(BaseModel):
    exam_id: str
    title: Optional[str] = None
    exam_type: str
    lifecycle_state: str
    prepared_document_id: Optional[str] = None
    roster: List[str]
    duration_minutes: Optional[int] = None
    hub_assignments: List[HubAssignment]
    created_by: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    admin_id: Optional[str] = None
    teacher_ids: List[str] = Field(default_factory=list)
    created_by_tutor_id: Optional[str] = None


class ExamListResponse(BaseModel):
    items: List[ExamDetailResponse]
    total: int


class LifecycleTransitionRequest(BaseModel):
    target_state: str = Field(..., description=f"Must be one of: {LIFECYCLE_STATES}")


class AssignHubRequest(BaseModel):
    hub_id: str
    hub_name: Optional[str] = None


class UnassignHubRequest(BaseModel):
    hub_id: str


class UploadProgressResponse(BaseModel):
    exam_id: str
    lifecycle_state: str
    total_expected: int
    total_received: int
    total_acknowledged: int
    by_hub: Dict[str, Dict[str, Any]]
    by_student: Dict[str, Dict[str, Any]]


# ---------------------------------------------------------------------------
# Helper: build exam document
# ---------------------------------------------------------------------------

def _build_exam_doc(
    exam_id: str,
    exam_type: str,
    current_user: Dict[str, Any],
    title: Optional[str] = None,
    prepared_document_id: Optional[str] = None,
    roster: Optional[List[str]] = None,
    duration_minutes: Optional[int] = None,
    admin_id: Optional[str] = None,
    teacher_ids: Optional[List[str]] = None,
    created_by_tutor_id: Optional[str] = None,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "exam_id": exam_id,
        "title": title,
        "exam_type": exam_type,
        "lifecycle_state": "draft",
        "prepared_document_id": prepared_document_id,
        "roster": roster or [],
        "duration_minutes": duration_minutes,
        "hub_assignments": [],
        "created_by": current_user.get("user_id", "unknown"),
        "created_at": now,
        "updated_at": now,
        "admin_id": admin_id,
        "teacher_ids": teacher_ids or [],
        "created_by_tutor_id": created_by_tutor_id,
    }


# ---------------------------------------------------------------------------
# Helper: doc to response
# ---------------------------------------------------------------------------

def _doc_to_response(doc: Dict[str, Any]) -> ExamDetailResponse:
    def _fmt(v):
        if hasattr(v, "isoformat"):
            return v.isoformat()
        if v is not None:
            return str(v)
        return None

    hub_assignments = []
    for ha in doc.get("hub_assignments", []):
        hub_assignments.append(HubAssignment(
            hub_id=ha.get("hub_id", ""),
            hub_name=ha.get("hub_name"),
            assigned_at=_fmt(ha.get("assigned_at")),
            session_started_at=_fmt(ha.get("session_started_at")),
            session_ended_at=_fmt(ha.get("session_ended_at")),
        ))

    raw_teacher_ids = doc.get("teacher_ids") or []
    if not isinstance(raw_teacher_ids, list):
        raw_teacher_ids = []
    teacher_ids = [str(t) for t in raw_teacher_ids]

    return ExamDetailResponse(
        exam_id=doc.get("exam_id", ""),
        title=_fmt(doc.get("title")),
        exam_type=doc.get("exam_type", ""),
        lifecycle_state=doc.get("lifecycle_state", "draft"),
        prepared_document_id=doc.get("prepared_document_id"),
        roster=doc.get("roster", []),
        duration_minutes=doc.get("duration_minutes"),
        hub_assignments=hub_assignments,
        created_by=doc.get("created_by", ""),
        created_at=_fmt(doc.get("created_at")),
        updated_at=_fmt(doc.get("updated_at")),
        admin_id=_fmt(doc.get("admin_id")),
        teacher_ids=teacher_ids,
        created_by_tutor_id=_fmt(doc.get("created_by_tutor_id")),
    )


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

_indexes_ensured = False


async def _ensure_indexes(collection) -> None:
    global _indexes_ensured
    if _indexes_ensured:
        return
    await collection.create_index("exam_id", unique=True)
    await collection.create_index("lifecycle_state")
    await collection.create_index("prepared_document_id")
    await collection.create_index("created_by")
    await collection.create_index("created_by_tutor_id")
    await collection.create_index("admin_id")
    _indexes_ensured = True


# ---------------------------------------------------------------------------
# Prepared document / ownership helpers
# ---------------------------------------------------------------------------

def _current_tutor_id(current_user: Dict[str, Any]) -> Optional[str]:
    """Return the tutor id for the caller, falling back to user_id.

    Tutors may carry their tutor id in either the ``tutor_id`` or
    ``user_id`` claim depending on how the session was issued
    (see core/auth.py authenticate_tutor). Admins always return None.
    """
    if (current_user.get("user_type") or "").lower() != "tutor":
        return None
    tutor_id = current_user.get("tutor_id") or current_user.get("user_id")
    return str(tutor_id) if tutor_id is not None else None


async def _load_prepared_document(
    tenant_db: Any,
    prepared_document_id: str,
) -> Dict[str, Any]:
    """Fetch a prepared document and validate it is ready to back an exam.

    Raises:
        HTTPException 404: document not found in tenant documents collection
        HTTPException 400: document lacks exam_mode or is not finalized
    """
    doc = await tenant_db["documents"].find_one(
        {"document_id": prepared_document_id}
    )
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prepared document {prepared_document_id} not found",
        )

    exam_mode = doc.get("exam_mode")
    if exam_mode not in ("dcr", "pcr"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Prepared document has no valid exam_mode "
                f"({exam_mode!r}); cannot create exam."
            ),
        )

    if not doc.get("exam_finalized"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Prepared document is not finalized. "
                "Finalize the document before creating an exam."
            ),
        )

    return doc


def _prepared_document_duration_minutes(doc: Dict[str, Any]) -> Optional[int]:
    """Return the canonical exam duration from prepared-document metadata."""
    for field in ("duration_minutes", "total_minutes"):
        raw = doc.get(field)
        if raw is None:
            continue
        if isinstance(raw, bool):
            continue
        if isinstance(raw, int):
            value = raw
        elif isinstance(raw, str) and raw.strip().isdigit():
            value = int(raw.strip())
        else:
            continue
        if value > 0:
            return value
    return None


def _is_tutor_admin_role(current_user: Dict[str, Any]) -> bool:
    role = (current_user.get("user_type") or "").lower()
    return role in ("admin", "b2c_admin")


def _is_exam_visible_to_tutor(
    exam_doc: Dict[str, Any], tutor_id: str
) -> bool:
    """Tutor visibility rule (matches prepared-document teacher_ids model).

    Visible iff:
      - tutor created the exam (created_by_tutor_id matches), OR
      - tutor is listed in teacher_ids, OR
      - teacher_ids is empty / None / missing (open to all tutors)
    """
    if exam_doc.get("created_by_tutor_id") == tutor_id:
        return True

    teacher_ids = exam_doc.get("teacher_ids")
    if not teacher_ids:
        return True
    if isinstance(teacher_ids, list) and tutor_id in teacher_ids:
        return True
    return False


def _require_tutor_visibility(
    exam_doc: Dict[str, Any], current_user: Dict[str, Any]
) -> None:
    """Raise 403 if the calling tutor cannot see this exam.

    Admin/b2c_admin bypass. Must be called AFTER the exam doc is fetched
    (so 404 still fires for unknown exam_id).
    """
    if _is_tutor_admin_role(current_user):
        return
    tutor_id = _current_tutor_id(current_user)
    if tutor_id is None:
        # Non-tutor, non-admin caller already filtered by auth dep, but
        # be defensive.
        return
    if not _is_exam_visible_to_tutor(exam_doc, tutor_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Exam is not visible to this tutor",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conducted exam",
    responses={
        400: {"description": "Invalid request"},
        403: {"description": "Insufficient permissions"},
        409: {"description": "Exam ID already exists"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def create_exam(
    body: ExamCreateRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> ExamDetailResponse:
    """Create a new conducted exam record.

    The exam starts in ``draft`` lifecycle state. Use PATCH /exams/{exam_id}/lifecycle
    to transition through states. Use POST /exams/{exam_id}/hubs to assign hubs.

    Ownership derivation:
        - exam_type ← document.exam_mode when prepared_document_id is supplied
          (document is canonical; caller-supplied exam_type is ignored in that
          case). Otherwise ← body.exam_type.
        - admin_id ← document.admin_id when a prepared document is supplied
          and has one. Otherwise ← current_user.user_id for admin/b2c_admin,
          or ← current_user.admin_id for tutors. Prepared-document value
          remains highest-precedence.
        - teacher_ids ← document.teacher_ids when prepared_document_id is
          supplied.
        - created_by_tutor_id ← current_user.tutor_id (or user_id) for tutors.
        - Tutors may NOT create an exam from a prepared document whose
          non-empty teacher_ids does not include them (403).
    """
    tenant_db = await _get_tenant_db(db, current_user)

    derived_admin_id: Optional[str] = None
    derived_teacher_ids: Optional[List[str]] = None
    derived_exam_type: Optional[str] = body.exam_type
    derived_title: Optional[str] = None
    derived_duration_minutes: Optional[int] = body.duration_minutes

    if body.prepared_document_id:
        doc = await _load_prepared_document(tenant_db, body.prepared_document_id)
        # Document is canonical for exam_type when a prepared document is
        # linked. Caller-supplied exam_type is ignored to keep the prepared
        # document the single source of truth for the exam mode.
        derived_exam_type = doc.get("exam_mode")
        raw_title = doc.get("title")
        if raw_title is not None:
            derived_title = str(raw_title)
        derived_duration_minutes = _prepared_document_duration_minutes(doc)
        raw_admin = doc.get("admin_id")
        if raw_admin is not None:
            derived_admin_id = str(raw_admin)
        raw_teachers = doc.get("teacher_ids") or []
        if isinstance(raw_teachers, list):
            derived_teacher_ids = [str(t) for t in raw_teachers]
        else:
            derived_teacher_ids = []

    # Tutor safety: a tutor may only create an exam from a prepared document
    # whose teacher_ids is empty/open or explicitly lists the tutor. Admins
    # and b2c_admin bypass this check.
    created_by_tutor_id = _current_tutor_id(current_user)
    if (
        created_by_tutor_id is not None
        and body.prepared_document_id is not None
        and derived_teacher_ids
        and created_by_tutor_id not in derived_teacher_ids
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Prepared document is not assigned to this tutor",
        )

    # Fallback admin_id when no prepared document supplied one. Prepared-
    # document admin_id remains highest-precedence (set above).
    if derived_admin_id is None:
        user_type = (current_user.get("user_type") or "").lower()
        if user_type in ("admin", "b2c_admin"):
            uid = current_user.get("user_id")
            if uid:
                derived_admin_id = str(uid)
        elif user_type == "tutor":
            aid = current_user.get("admin_id")
            if aid:
                derived_admin_id = str(aid)

    if derived_exam_type not in ("dcr", "pcr"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="exam_type must be 'dcr' or 'pcr'",
        )

    collection = tenant_db["exampen_exams"]
    await _ensure_indexes(collection)

    doc = _build_exam_doc(
        exam_id=body.exam_id,
        exam_type=derived_exam_type,
        current_user=current_user,
        title=derived_title,
        prepared_document_id=body.prepared_document_id,
        roster=body.roster,
        duration_minutes=derived_duration_minutes,
        admin_id=derived_admin_id,
        teacher_ids=derived_teacher_ids,
        created_by_tutor_id=created_by_tutor_id,
    )

    try:
        await collection.insert_one(doc)
    except Exception as exc:
        if hasattr(exc, "code") and exc.code == 11000:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Exam {body.exam_id} already exists",
            )
        raise

    logger.info(
        "Exam %s created as %s by %s",
        body.exam_id,
        derived_exam_type,
        current_user.get("user_id"),
    )
    return _doc_to_response(doc)


@router.get(
    "",
    summary="List conducted exams visible to current user",
    responses={
        403: {"description": "Insufficient permissions"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def list_exams(
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
    lifecycle_filter: Optional[str] = None,
) -> ExamListResponse:
    """List all conducted exams for the current tenant.

    Admins see all exams. Tutors see exams they created or have roster access to.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    collection = tenant_db["exampen_exams"]

    query: Dict[str, Any] = {}
    if lifecycle_filter:
        if lifecycle_filter not in LIFECYCLE_STATES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid lifecycle state: {lifecycle_filter}",
            )
        query["lifecycle_state"] = lifecycle_filter

    # Tutor visibility: only exams they created, are listed in teacher_ids,
    # or that have empty/missing teacher_ids.
    tutor_id = _current_tutor_id(current_user)
    if tutor_id is not None:
        query["$or"] = [
            {"created_by_tutor_id": tutor_id},
            {"teacher_ids": tutor_id},
            {"teacher_ids": []},
            {"teacher_ids": None},
            {"teacher_ids": {"$exists": False}},
        ]

    cursor = collection.find(query).sort("created_at", -1)
    docs = await cursor.to_list(length=200)

    items = [_doc_to_response(d) for d in docs]
    total = len(items)

    logger.info(
        "Listed %d exams for user %s (lifecycle=%s)",
        total,
        current_user.get("user_id"),
        lifecycle_filter,
    )
    return ExamListResponse(items=items, total=total)


@router.get(
    "/{exam_id}",
    summary="Get conducted exam detail",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_exam(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> ExamDetailResponse:
    """Get full detail for one conducted exam."""
    tenant_db = await _get_tenant_db(db, current_user)
    collection = tenant_db["exampen_exams"]

    doc = await collection.find_one({"exam_id": exam_id})
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {exam_id} not found",
        )

    _require_tutor_visibility(doc, current_user)

    return _doc_to_response(doc)


@router.patch(
    "/{exam_id}/lifecycle",
    summary="Transition exam lifecycle state",
    responses={
        400: {"description": "Invalid transition or exam_type"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def transition_lifecycle(
    exam_id: str,
    body: LifecycleTransitionRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> ExamDetailResponse:
    """Transition exam to a new lifecycle state.

    Valid transitions (strictly ordered):
      draft -> armed -> in_progress -> collection_closed -> uploading -> ready_for_eval

    Cannot skip states. Only authorized roles can transition to certain states.
    """
    if body.target_state not in LIFECYCLE_STATES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid target_state. Must be one of: {LIFECYCLE_STATES}",
        )

    tenant_db = await _get_tenant_db(db, current_user)
    collection = tenant_db["exampen_exams"]

    doc = await collection.find_one({"exam_id": exam_id})
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {exam_id} not found",
        )

    _require_tutor_visibility(doc, current_user)

    current_state = doc.get("lifecycle_state", "draft")

    # Check strict ordering
    allowed_next = LIFECYCLE_TRANSITIONS.get(current_state, set())
    if body.target_state not in allowed_next:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid transition from '{current_state}' to '{body.target_state}'. "
                   f"Allowed next state: {allowed_next or 'none (final state)'}",
        )

    now = datetime.now(timezone.utc)
    update: Dict[str, Any] = {
        "$set": {
            "lifecycle_state": body.target_state,
            "updated_at": now,
        }
    }

    if body.target_state == "in_progress":
        update["$set"]["started_at"] = now
    elif body.target_state == "collection_closed":
        update["$set"]["collection_closed_at"] = now
    elif body.target_state == "ready_for_eval":
        update["$set"]["ready_for_eval_at"] = now

    await collection.update_one({"exam_id": exam_id}, update)

    updated_doc = await collection.find_one({"exam_id": exam_id})
    logger.info(
        "Exam %s transitioned %s -> %s by %s",
        exam_id,
        current_state,
        body.target_state,
        current_user.get("user_id"),
    )
    return _doc_to_response(updated_doc)


@router.post(
    "/{exam_id}/hubs",
    summary="Assign a hub to an exam",
    responses={
        400: {"description": "Invalid request"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        409: {"description": "Hub already assigned"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def assign_hub(
    exam_id: str,
    body: AssignHubRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> ExamDetailResponse:
    """Assign a registered ExamPen hub to this exam."""
    tenant_db = await _get_tenant_db(db, current_user)
    collection = tenant_db["exampen_exams"]

    doc = await collection.find_one({"exam_id": exam_id})
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {exam_id} not found",
        )

    _require_tutor_visibility(doc, current_user)

    current_state = doc.get("lifecycle_state", "draft")
    if current_state != "draft":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Hub assignments can only be changed while exam is in draft state. "
                   f"Current state: {current_state}",
        )

    existing = doc.get("hub_assignments", [])
    if any(ha.get("hub_id") == body.hub_id for ha in existing):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Hub {body.hub_id} is already assigned to this exam",
        )

    now = datetime.now(timezone.utc)
    new_assignment = {
        "hub_id": body.hub_id,
        "hub_name": body.hub_name,
        "assigned_at": now,
        "session_started_at": None,
        "session_ended_at": None,
    }

    await collection.update_one(
        {"exam_id": exam_id},
        {
            "$push": {"hub_assignments": new_assignment},
            "$set": {"updated_at": now},
        },
    )

    updated_doc = await collection.find_one({"exam_id": exam_id})
    logger.info(
        "Hub %s assigned to exam %s by %s",
        body.hub_id,
        exam_id,
        current_user.get("user_id"),
    )
    return _doc_to_response(updated_doc)


@router.delete(
    "/{exam_id}/hubs/{hub_id}",
    summary="Unassign a hub from an exam",
    responses={
        400: {"description": "Hub not assigned to this exam"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def unassign_hub(
    exam_id: str,
    hub_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> ExamDetailResponse:
    """Unassign a hub from this exam."""
    tenant_db = await _get_tenant_db(db, current_user)
    collection = tenant_db["exampen_exams"]

    doc = await collection.find_one({"exam_id": exam_id})
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {exam_id} not found",
        )

    _require_tutor_visibility(doc, current_user)

    current_state = doc.get("lifecycle_state", "draft")
    if current_state != "draft":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Hub assignments can only be changed while exam is in draft state. "
                   f"Current state: {current_state}",
        )

    existing = doc.get("hub_assignments", [])
    if not any(ha.get("hub_id") == hub_id for ha in existing):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Hub {hub_id} is not assigned to this exam",
        )

    now = datetime.now(timezone.utc)
    await collection.update_one(
        {"exam_id": exam_id},
        {
            "$pull": {"hub_assignments": {"hub_id": hub_id}},
            "$set": {"updated_at": now},
        },
    )

    updated_doc = await collection.find_one({"exam_id": exam_id})
    logger.info(
        "Hub %s unassigned from exam %s by %s",
        hub_id,
        exam_id,
        current_user.get("user_id"),
    )
    return _doc_to_response(updated_doc)


@router.get(
    "/{exam_id}/progress",
    summary="Get upload progress for an exam",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_upload_progress(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> UploadProgressResponse:
    """Get per-hub and per-student upload progress for this exam.

    Reads from evalpen_submissions to compute received/acknowledged counts.
    """
    tenant_db = await _get_tenant_db(db, current_user)

    exam_col = tenant_db["exampen_exams"]
    doc = await exam_col.find_one({"exam_id": exam_id})
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {exam_id} not found",
        )

    _require_tutor_visibility(doc, current_user)

    # Aggregate submissions for this exam
    pipeline = [
        {"$match": {"exam_id": exam_id}},
        {
            "$group": {
                "_id": {
                    "student_id": "$student_id",
                    "hub_id": {"$ifNull": ["$hub_id", "$pen_mac"]},
                },
                "received": {"$sum": 1},
                "acknowledged": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$upload_status", "acknowledged"]},
                            1,
                            0,
                        ]
                    }
                },
            }
        },
    ]
    cursor = tenant_db["evalpen_submissions"].aggregate(pipeline)
    agg_results = await cursor.to_list(length=1000)

    roster = set(doc.get("roster", []))
    hub_ids = {ha.get("hub_id") for ha in doc.get("hub_assignments", [])}

    by_hub: Dict[str, Dict[str, Any]] = {hid: {"received": 0, "acknowledged": 0} for hid in hub_ids}
    by_student: Dict[str, Dict[str, Any]] = {sid: {"received": 0, "acknowledged": 0} for sid in roster}

    for r in agg_results:
        key = r["_id"]
        sid = key.get("student_id", "")
        hid = key.get("hub_id", "")
        received = r.get("received", 0)
        acknowledged = r.get("acknowledged", 0)

        if sid in by_student:
            by_student[sid]["received"] += received
            by_student[sid]["acknowledged"] += acknowledged

        if hid in by_hub:
            by_hub[hid]["received"] += received
            by_hub[hid]["acknowledged"] += acknowledged

    total_expected = len(roster)
    total_received = sum(v["received"] for v in by_student.values())
    total_acknowledged = sum(v["acknowledged"] for v in by_student.values())

    return UploadProgressResponse(
        exam_id=exam_id,
        lifecycle_state=doc.get("lifecycle_state", "draft"),
        total_expected=total_expected,
        total_received=total_received,
        total_acknowledged=total_acknowledged,
        by_hub=by_hub,
        by_student=by_student,
    )
