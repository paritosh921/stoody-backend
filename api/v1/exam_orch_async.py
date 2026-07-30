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
import re
import uuid
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

PEN_MAC_PATTERN = re.compile(r"^[0-9A-Fa-f]{2}(:[0-9A-Fa-f]{2}){5}$")
MAX_PEN_BINDINGS = 256
CAPTURE_MODES = {"pen", "camera", "hybrid"}
DEFAULT_PCR_CAMERA_MAX_PAGES = 40
ACTIVE_COLLECTION_STATES = {"draft", "armed", "in_progress"}


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
    # Kept only so older clients do not fail schema validation.  The server
    # always generates the real session id; a client-controlled exam id used
    # to allow a finalized paper and captured submissions to diverge.
    exam_id: Optional[str] = Field(
        None,
        min_length=1,
        description="Deprecated client request label; not used as the session id.",
    )
    exam_type: Optional[str] = Field(
        None,
        description="dcr or pcr. When omitted, derived from prepared document's exam_mode.",
    )
    prepared_document_id: Optional[str] = Field(None, description="Linked prepared document")
    request_id: Optional[str] = Field(
        None,
        min_length=1,
        max_length=128,
        description="Optional idempotency key supplied by the setup client.",
    )
    roster: Optional[List[str]] = Field(default_factory=list, description="Student IDs")
    pen_bindings: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Mapping of pen MAC address to student ID for ExamPen capture",
    )
    duration_minutes: Optional[int] = Field(None, ge=1)
    hub_assignments: Optional[List[HubAssignment]] = Field(default_factory=list)
    capture_mode: str = Field(
        "pen",
        description="pen, camera, or hybrid. Camera is PCR-only.",
    )
    student_self_submission_enabled: bool = Field(
        False,
        description="Allow rostered students to submit photographed/scanned PCR copies from the web portal.",
    )
    student_submission_max_pages: int = Field(
        20,
        ge=1,
        le=50,
        description="Maximum rendered/uploaded answer pages a student may submit.",
    )


class ExamDetailResponse(BaseModel):
    exam_id: str
    title: Optional[str] = None
    exam_type: str
    lifecycle_state: str
    prepared_document_id: Optional[str] = None
    roster: List[str]
    pen_bindings: Dict[str, str] = Field(default_factory=dict)
    duration_minutes: Optional[int] = None
    hub_assignments: List[HubAssignment]
    created_by: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    admin_id: Optional[str] = None
    teacher_ids: List[str] = Field(default_factory=list)
    created_by_tutor_id: Optional[str] = None
    paper_version_id: Optional[str] = None
    paper_content_hash: Optional[str] = None
    capture_mode: str = "pen"
    student_self_submission_enabled: bool = False
    student_submission_max_pages: int = 20
    session_request_id: Optional[str] = None


class ExamListResponse(BaseModel):
    items: List[ExamDetailResponse]
    total: int


class LifecycleTransitionRequest(BaseModel):
    target_state: str = Field(..., description=f"Must be one of: {LIFECYCLE_STATES}")


class ExamSetupUpdateRequest(BaseModel):
    """Mutable capture configuration, allowed only while a session is draft."""
    roster: Optional[List[str]] = None
    pen_bindings: Optional[Dict[str, str]] = None
    capture_mode: Optional[str] = None
    student_self_submission_enabled: Optional[bool] = None
    student_submission_max_pages: Optional[int] = Field(None, ge=1, le=50)


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
    absent_student_ids: List[str] = Field(default_factory=list)


class PreflightCheck(BaseModel):
    id: str
    label: str
    ready: bool
    detail: str


class ExamPreflightResponse(BaseModel):
    exam_id: str
    lifecycle_state: str
    ready_to_arm: bool
    checks: List[PreflightCheck]


class MarkAbsentRequest(BaseModel):
    student_id: str = Field(..., min_length=1)
    note: Optional[str] = Field(None, max_length=500)


class ProcessingJobReprocessRequest(BaseModel):
    """Optional audit note recorded when staff rerun an answer-copy check."""

    reason: Optional[str] = Field(None, max_length=500)


class ProcessingJobResponse(BaseModel):
    job_id: str
    submission_id: str
    student_id: Optional[str] = None
    status: str
    attempts: int = 0
    last_error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    finished_at: Optional[str] = None
    reprocess_count: int = 0
    reprocess_requested_at: Optional[str] = None
    reprocess_requested_by: Optional[str] = None
    segmentation: Dict[str, Any] = Field(default_factory=dict)
    evaluation: Dict[str, Any] = Field(default_factory=dict)


class ProcessingJobListResponse(BaseModel):
    exam_id: str
    items: List[ProcessingJobResponse]


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
    pen_bindings: Optional[Dict[str, str]] = None,
    duration_minutes: Optional[int] = None,
    admin_id: Optional[str] = None,
    teacher_ids: Optional[List[str]] = None,
    created_by_tutor_id: Optional[str] = None,
    paper_version_id: Optional[str] = None,
    paper_content_hash: Optional[str] = None,
    capture_mode: str = "pen",
    student_self_submission_enabled: bool = False,
    student_submission_max_pages: int = 20,
    session_request_id: Optional[str] = None,
    legacy_requested_exam_id: Optional[str] = None,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "exam_id": exam_id,
        "title": title,
        "exam_type": exam_type,
        "lifecycle_state": "draft",
        "prepared_document_id": prepared_document_id,
        "roster": roster or [],
        "pen_bindings": _safe_string_dict(pen_bindings),
        "duration_minutes": duration_minutes,
        "hub_assignments": [],
        "created_by": current_user.get("user_id", "unknown"),
        "created_at": now,
        "updated_at": now,
        "admin_id": admin_id,
        "teacher_ids": teacher_ids or [],
        "created_by_tutor_id": created_by_tutor_id,
        "paper_version_id": paper_version_id,
        "paper_content_hash": paper_content_hash,
        "capture_mode": capture_mode,
        "student_self_submission_enabled": student_self_submission_enabled,
        "student_submission_max_pages": student_submission_max_pages,
        "session_request_id": session_request_id,
        "legacy_requested_exam_id": legacy_requested_exam_id,
        "session_snapshot_status": "pending",
        "absent_student_ids": [],
    }


def _normalize_pen_bindings(
    pen_bindings: Optional[Dict[str, str]],
    roster: Optional[List[str]],
) -> Dict[str, str]:
    """Validate and normalize pen MAC -> student mappings for an exam."""
    if not pen_bindings:
        return {}
    if len(pen_bindings) > MAX_PEN_BINDINGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"pen_bindings cannot exceed {MAX_PEN_BINDINGS} entries",
        )

    roster_set = {str(student_id) for student_id in (roster or [])}
    normalized: Dict[str, str] = {}
    for raw_mac, raw_student_id in pen_bindings.items():
        mac = str(raw_mac or "").strip().upper()
        student_id = str(raw_student_id or "").strip()
        if not PEN_MAC_PATTERN.fullmatch(mac):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid pen MAC in pen_bindings: {raw_mac}",
            )
        if not student_id or student_id not in roster_set:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "pen_bindings values must reference students in the exam roster"
                ),
            )
        normalized[mac] = student_id
    return normalized


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
        pen_bindings=_safe_string_dict(doc.get("pen_bindings")),
        duration_minutes=doc.get("duration_minutes"),
        hub_assignments=hub_assignments,
        created_by=doc.get("created_by", ""),
        created_at=_fmt(doc.get("created_at")),
        updated_at=_fmt(doc.get("updated_at")),
        admin_id=_fmt(doc.get("admin_id")),
        teacher_ids=teacher_ids,
        created_by_tutor_id=_fmt(doc.get("created_by_tutor_id")),
        paper_version_id=_fmt(doc.get("paper_version_id")),
        paper_content_hash=_fmt(doc.get("paper_content_hash")),
        capture_mode=doc.get("capture_mode", "pen"),
        student_self_submission_enabled=bool(doc.get("student_self_submission_enabled", False)),
        student_submission_max_pages=int(doc.get("student_submission_max_pages") or 20),
        session_request_id=_fmt(doc.get("session_request_id")),
    )


def _processing_job_to_response(doc: Dict[str, Any]) -> ProcessingJobResponse:
    def _fmt(value: Any) -> Optional[str]:
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value) if value is not None else None

    return ProcessingJobResponse(
        job_id=str(doc.get("job_id") or ""),
        submission_id=str(doc.get("submission_id") or ""),
        student_id=_fmt(doc.get("student_id")),
        status=str(doc.get("status") or "queued"),
        attempts=int(doc.get("attempts") or 0),
        last_error=_fmt(doc.get("last_error")),
        created_at=_fmt(doc.get("created_at")),
        updated_at=_fmt(doc.get("updated_at")),
        finished_at=_fmt(doc.get("finished_at")),
        reprocess_count=int(doc.get("reprocess_count") or 0),
        reprocess_requested_at=_fmt(doc.get("reprocess_requested_at")),
        reprocess_requested_by=_fmt(doc.get("reprocess_requested_by")),
        segmentation=dict(doc.get("segmentation") or {}),
        evaluation=dict(doc.get("evaluation") or {}),
    )


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

_indexed_collections: set[str] = set()


async def _ensure_indexes(collection) -> None:
    # Each tenant owns a separate Mongo database.  A single process-wide
    # boolean used to skip index creation after the first tenant, leaving
    # later tenants without the uniqueness guarantees this API relies on.
    collection_key = str(getattr(collection, "full_name", "")) or repr(collection)
    if collection_key in _indexed_collections:
        return
    await collection.create_index("exam_id", unique=True)
    await collection.create_index("lifecycle_state")
    await collection.create_index("prepared_document_id")
    await collection.create_index("created_by")
    await collection.create_index("created_by_tutor_id")
    await collection.create_index("admin_id")
    await collection.create_index(
        "session_request_key", unique=True, sparse=True, name="uniq_session_request"
    )
    _indexed_collections.add(collection_key)


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


def _safe_string_dict(value: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: Dict[str, str] = {}
    for key, item in value.items():
        if key is None or item is None:
            continue
        result[str(key)] = str(item)
    return result


def _normalize_roster(roster: Optional[List[str]]) -> List[str]:
    """Normalize roster ids without allowing duplicate student entries."""
    normalized: List[str] = []
    seen: set[str] = set()
    for raw_student_id in roster or []:
        student_id = str(raw_student_id or "").strip()
        if not student_id or student_id in seen:
            continue
        seen.add(student_id)
        normalized.append(student_id)
    return normalized


def _normalize_capture_mode(raw_capture_mode: Optional[str], exam_type: str) -> str:
    capture_mode = str(raw_capture_mode or "pen").strip().lower()
    if capture_mode not in CAPTURE_MODES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"capture_mode must be one of: {sorted(CAPTURE_MODES)}",
        )
    if exam_type == "dcr" and capture_mode != "pen":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="DCR sessions require pen capture; camera fallback is PCR-only",
        )
    return capture_mode


def _normalize_student_self_submission_config(
    *,
    enabled: Optional[bool],
    max_pages: Optional[int],
    exam_type: str,
    capture_mode: str,
) -> tuple[bool, int]:
    """Validate the opt-in student raster-copy upload channel.

    This is deliberately separate from the mobile invigilator camera path.  A
    student can only submit a raster copy for a PCR session that was explicitly
    configured for camera-capable capture.  Existing pen-only/DCR sessions keep
    their current behavior because the default is disabled.
    """
    normalized_enabled = bool(enabled)
    normalized_max_pages = int(max_pages or 20)
    if not 1 <= normalized_max_pages <= 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="student_submission_max_pages must be between 1 and 50",
        )
    if normalized_enabled and exam_type != "pcr":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Student self-submission is available for PCR sessions only",
        )
    if normalized_enabled and capture_mode not in {"camera", "hybrid"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Student self-submission requires camera or hybrid capture mode",
        )
    return normalized_enabled, normalized_max_pages


def _new_exam_id() -> str:
    """Generate a server-owned conducted-session id."""
    return f"exam-{uuid.uuid4().hex}"


def _session_request_key(current_user: Dict[str, Any], request_id: Optional[str]) -> Optional[str]:
    if not request_id:
        return None
    # Automatic PCR camera collection belongs to the finalized paper, not to
    # whichever admin happened to click Activate.  Keeping this key
    # actor-independent makes concurrent/retried activation idempotent.
    if str(request_id).startswith("auto-pcr-camera:"):
        return str(request_id).strip()
    actor = str(current_user.get("user_id") or "unknown")
    return f"{actor}:{str(request_id).strip()}"


async def _build_preflight(
    tenant_db: Any,
    exam_doc: Dict[str, Any],
) -> ExamPreflightResponse:
    """Return the server-authoritative arm/readiness checks for a session."""
    exam_id = str(exam_doc.get("exam_id") or "")
    exam_type = str(exam_doc.get("exam_type") or "")
    capture_mode = str(exam_doc.get("capture_mode") or "pen")
    roster = _normalize_roster(exam_doc.get("roster"))
    pen_bindings = _safe_string_dict(exam_doc.get("pen_bindings"))
    hub_assignments = list(exam_doc.get("hub_assignments") or [])
    checks: List[PreflightCheck] = []

    paper_version_id = str(exam_doc.get("paper_version_id") or "")
    snapshot_ready = exam_doc.get("session_snapshot_status") == "ready"
    checks.append(
        PreflightCheck(
            id="paper_snapshot",
            label="Immutable paper snapshot",
            ready=bool(paper_version_id and snapshot_ready),
            detail=(
                f"Paper version: {paper_version_id}"
                if paper_version_id and snapshot_ready
                else "Session question metadata has not been materialized"
            ),
        )
    )

    question_collection = "evalpen_questions" if exam_type == "pcr" else "exampen_answer_keys"
    question_count = await tenant_db[question_collection].count_documents({"exam_id": exam_id})
    checks.append(
        PreflightCheck(
            id="session_questions",
            label="Session question metadata",
            ready=question_count > 0,
            detail=(
                f"{question_count} immutable question record(s)"
                if question_count
                else "No session-scoped question metadata found"
            ),
        )
    )

    checks.append(
        PreflightCheck(
            id="roster",
            label="Student roster",
            ready=bool(roster),
            detail=(f"{len(roster)} student(s)" if roster else "Add at least one student before arming"),
        )
    )

    duration = exam_doc.get("duration_minutes")
    checks.append(
        PreflightCheck(
            id="duration",
            label="Exam duration",
            ready=isinstance(duration, int) and not isinstance(duration, bool) and duration > 0,
            detail=(f"{duration} minutes" if isinstance(duration, int) and duration > 0 else "Set a positive duration"),
        )
    )

    if capture_mode in {"pen", "hybrid"}:
        hub_ids = [str(item.get("hub_id") or "") for item in hub_assignments if item.get("hub_id")]
        checks.append(
            PreflightCheck(
                id="hub_assignment",
                label="Capture hub assignment",
                ready=bool(hub_ids),
                detail=(f"{len(hub_ids)} hub(s) assigned" if hub_ids else "Assign at least one hub"),
            )
        )

        unavailable_hubs: List[str] = []
        for hub_id in hub_ids:
            hub = await tenant_db["exampen_hubs"].find_one({"hub_id": hub_id})
            online = bool(
                hub
                and hub.get("assigned_exam_id") == exam_id
                and str(hub.get("status") or "").lower() not in {"offline", "error", "revoked"}
                and str(hub.get("health") or "ok").lower() not in {"error", "unavailable"}
                and hub.get("last_heartbeat_at") is not None
            )
            if not online:
                unavailable_hubs.append(hub_id)
        checks.append(
            PreflightCheck(
                id="hub_health",
                label="Hub health",
                ready=bool(hub_ids) and not unavailable_hubs,
                detail=(
                    "All assigned hubs are online"
                    if hub_ids and not unavailable_hubs
                    else f"Waiting for healthy heartbeat: {', '.join(unavailable_hubs or hub_ids)}"
                ),
            )
        )

    if capture_mode == "pen":
        bound_students = set(pen_bindings.values())
        missing_bindings = [student_id for student_id in roster if student_id not in bound_students]
        checks.append(
            PreflightCheck(
                id="pen_bindings",
                label="Pen-to-student bindings",
                ready=not missing_bindings and bool(roster),
                detail=(
                    "Every rostered student has a pen binding"
                    if roster and not missing_bindings
                    else f"Missing bindings for: {', '.join(missing_bindings[:10]) or 'all students'}"
                ),
            )
        )
    elif capture_mode == "camera":
        checks.append(
            PreflightCheck(
                id="camera_capture",
                label="Camera fallback",
                ready=exam_type == "pcr",
                detail="Camera capture is configured for PCR" if exam_type == "pcr" else "Camera is PCR-only",
            )
        )

    if bool(exam_doc.get("student_self_submission_enabled", False)):
        max_pages = int(exam_doc.get("student_submission_max_pages") or 20)
        checks.append(
            PreflightCheck(
                id="student_self_submission",
                label="Student web copy submission",
                ready=(
                    exam_type == "pcr"
                    and capture_mode in {"camera", "hybrid"}
                    and bool(roster)
                ),
                detail=(
                    f"Rostered students can upload up to {max_pages} photographed/scanned pages"
                    if exam_type == "pcr" and capture_mode in {"camera", "hybrid"} and roster
                    else "Requires a PCR camera/hybrid session with a roster"
                ),
            )
        )

    return ExamPreflightResponse(
        exam_id=exam_id,
        lifecycle_state=str(exam_doc.get("lifecycle_state") or "draft"),
        ready_to_arm=all(check.ready for check in checks),
        checks=checks,
    )


async def _ready_for_eval_issues(tenant_db: Any, exam_doc: Dict[str, Any]) -> List[str]:
    """Return upload/processing blockers before an exam enters review."""
    exam_id = str(exam_doc.get("exam_id") or "")
    absent = {str(student_id) for student_id in (exam_doc.get("absent_student_ids") or [])}
    expected_students = set(_normalize_roster(exam_doc.get("roster"))) - absent
    if not expected_students:
        return ["No expected students remain; add a roster or record attendance correctly"]

    submissions = await tenant_db["evalpen_submissions"].find({"exam_id": exam_id}).to_list(length=5000)
    by_student = {str(item.get("student_id")): item for item in submissions if item.get("student_id")}
    missing = sorted(expected_students - set(by_student))
    issues: List[str] = []
    if missing:
        issues.append(f"Missing canonical submissions for: {', '.join(missing[:20])}")

    # DCR does not use the PCR OCR/segmentation job queue.  Once every
    # expected copy is canonically ingested it is ready for its separate DCR
    # evaluation path; requiring a non-existent PCR job would strand it in
    # uploading forever.
    if str(exam_doc.get("exam_type") or "") != "pcr":
        return issues

    submission_ids = [
        str(by_student[student_id].get("submission_id"))
        for student_id in expected_students
        if student_id in by_student and by_student[student_id].get("submission_id")
    ]
    jobs = await tenant_db["exampen_processing_jobs"].find(
        {"submission_id": {"$in": submission_ids}}
    ).to_list(length=5000)
    jobs_by_submission = {str(job.get("submission_id")): job for job in jobs}
    pending_students: List[str] = []
    failed_students: List[str] = []
    for student_id in sorted(expected_students):
        submission = by_student.get(student_id)
        if not submission:
            continue
        job = jobs_by_submission.get(str(submission.get("submission_id")))
        job_status = str((job or {}).get("status") or "not_enqueued")
        if job_status == "failed":
            failed_students.append(student_id)
        elif job_status not in {"completed", "blocked_for_review"}:
            pending_students.append(student_id)
    if pending_students:
        issues.append(f"PCR processing is still pending for: {', '.join(pending_students[:20])}")
    if failed_students:
        issues.append(f"PCR processing failed for: {', '.join(failed_students[:20])}")
    return issues


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

    if not body.prepared_document_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A conducted exam session must be created from a finalized paper",
        )

    source_document = await _load_prepared_document(tenant_db, body.prepared_document_id)
    # The finalized document is canonical.  A caller may not change its type,
    # title, owner, or teacher visibility while creating a live session.
    derived_exam_type = source_document.get("exam_mode")
    derived_title = str(source_document.get("title") or "") or None
    derived_duration_minutes = _prepared_document_duration_minutes(source_document)
    derived_admin_id = (
        str(source_document.get("admin_id"))
        if source_document.get("admin_id") is not None
        else None
    )
    raw_teachers = source_document.get("teacher_ids") or []
    derived_teacher_ids = [str(t) for t in raw_teachers] if isinstance(raw_teachers, list) else []

    # Tutor safety: a tutor may only create an exam from a prepared document
    # whose teacher_ids is empty/open or explicitly lists the tutor. Admins
    # and b2c_admin bypass this check.
    created_by_tutor_id = _current_tutor_id(current_user)
    if (
        created_by_tutor_id is not None
        and derived_teacher_ids
        and created_by_tutor_id not in derived_teacher_ids
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Prepared document is not assigned to this tutor",
        )

    # Fallback admin_id when the prepared document predates owner metadata.
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

    capture_mode = _normalize_capture_mode(body.capture_mode, derived_exam_type)
    student_self_submission_enabled, student_submission_max_pages = (
        _normalize_student_self_submission_config(
            enabled=body.student_self_submission_enabled,
            max_pages=body.student_submission_max_pages,
            exam_type=derived_exam_type,
            capture_mode=capture_mode,
        )
    )
    roster = _normalize_roster(body.roster)
    collection = tenant_db["exampen_exams"]
    await _ensure_indexes(collection)

    request_id = str(body.request_id or "").strip() or None
    request_key = _session_request_key(current_user, request_id)
    if request_key:
        existing = await collection.find_one({"session_request_key": request_key})
        if existing is not None:
            return _doc_to_response(existing)

    normalized_pen_bindings = _normalize_pen_bindings(
        body.pen_bindings,
        roster,
    )

    from services.exampen_paper_service import (
        delete_session_snapshot,
        load_or_create_paper_snapshot,
        snapshot_paper_to_session,
    )

    paper_version, snapshot_questions = await load_or_create_paper_snapshot(
        tenant_db,
        source_document,
    )
    exam_id = _new_exam_id()

    doc = _build_exam_doc(
        exam_id=exam_id,
        exam_type=derived_exam_type,
        current_user=current_user,
        title=derived_title,
        prepared_document_id=body.prepared_document_id,
        roster=roster,
        pen_bindings=normalized_pen_bindings,
        duration_minutes=derived_duration_minutes,
        admin_id=derived_admin_id,
        teacher_ids=derived_teacher_ids,
        created_by_tutor_id=created_by_tutor_id,
        paper_version_id=paper_version["paper_version_id"],
        paper_content_hash=paper_version.get("content_hash"),
        capture_mode=capture_mode,
        student_self_submission_enabled=student_self_submission_enabled,
        student_submission_max_pages=student_submission_max_pages,
        session_request_id=request_id,
        legacy_requested_exam_id=str(body.exam_id or "").strip() or None,
    )
    if request_key:
        doc["session_request_key"] = request_key

    try:
        await collection.insert_one(doc)
    except Exception as exc:
        if hasattr(exc, "code") and exc.code == 11000:
            if request_key:
                existing = await collection.find_one({"session_request_key": request_key})
                if existing is not None:
                    return _doc_to_response(existing)
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Session request already exists")
        raise

    try:
        snapshot_summary = await snapshot_paper_to_session(
            tenant_db,
            exam_id=exam_id,
            paper_version=paper_version,
            snapshot_questions=snapshot_questions,
        )
        await collection.update_one(
            {"exam_id": exam_id},
            {
                "$set": {
                    "session_snapshot_status": "ready",
                    "session_snapshot_summary": snapshot_summary,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
    except Exception:
        # Do not leave a selectable session that points at incomplete or
        # mutable question metadata.  The immutable paper version itself is
        # retained for audit and a safe retry.
        await delete_session_snapshot(tenant_db, exam_id, derived_exam_type)
        await collection.delete_one({"exam_id": exam_id})
        logger.exception("Failed to materialize paper snapshot for session %s", exam_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not materialize the finalized paper for this exam session",
        )

    doc = await collection.find_one({"exam_id": exam_id})

    logger.info(
        "Exam session %s created from paper %s as %s by %s",
        exam_id,
        body.prepared_document_id,
        derived_exam_type,
        current_user.get("user_id"),
    )
    return _doc_to_response(doc)


async def _camera_roster_for_document(
    tenant_db: Any,
    document: Dict[str, Any],
) -> List[str]:
    """Return every active student owned by the paper's institution/class."""
    standard = str(document.get("standard") or "").strip()
    if not standard:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Set the paper class before activating camera answer-copy uploads",
        )

    query: Dict[str, Any] = {
        "grade": standard,
        "is_active": True,
    }
    if document.get("admin_id") is not None:
        query["admin_id"] = document["admin_id"]

    students = await tenant_db["students"].find(
        query,
        projection={"_id": 1, "student_id": 1},
    ).to_list(length=5000)
    roster = _normalize_roster(
        [
            str(student.get("student_id") or student.get("_id") or "")
            for student in students
        ]
    )
    if not roster:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No active students were found in Class {standard}",
        )
    return roster


def _automatic_camera_request_id(document: Dict[str, Any]) -> str:
    finalized_at = document.get("exam_finalized_at") or "finalized"
    if hasattr(finalized_at, "isoformat"):
        finalized_at = finalized_at.isoformat()
    return (
        f"auto-pcr-camera:{str(document.get('document_id') or '').strip()}:"
        f"{str(finalized_at)}"
    )[:128]


async def ensure_default_pcr_camera_collection(
    *,
    prepared_document_id: str,
    current_user: Dict[str, Any],
    db: DatabaseManager,
) -> ExamDetailResponse:
    """Ensure a finalized PCR paper has one live camera-only collection.

    This is the server-side activation contract.  The ``exampen_exams``
    record remains an internal lifecycle container, while student capture is
    always camera upload: no hub, pen binding, or hybrid setup is involved.
    Repeated calls are idempotent and also repair an active paper whose
    automatic collection record is missing.
    """
    tenant_db = await _get_tenant_db(db, current_user)
    document = await _load_prepared_document(tenant_db, prepared_document_id)
    if str(document.get("exam_mode") or "").strip().lower() != "pcr":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Automatic camera collection is available for finalized PCR papers only",
        )

    roster = await _camera_roster_for_document(tenant_db, document)
    collection = tenant_db["exampen_exams"]
    await _ensure_indexes(collection)

    paper_sessions = await collection.find(
        {"prepared_document_id": prepared_document_id}
    ).sort("created_at", -1).to_list(length=100)
    session = next(
        (
            item
            for item in paper_sessions
            if str(item.get("lifecycle_state") or "draft") in ACTIVE_COLLECTION_STATES
        ),
        None,
    )

    if session is None and paper_sessions:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This finalized paper already has a closed collection. "
                "Create a new finalized paper before opening another class submission."
            ),
        )

    if session is None:
        created = await create_exam(
            body=ExamCreateRequest(
                prepared_document_id=prepared_document_id,
                request_id=_automatic_camera_request_id(document),
                roster=roster,
                pen_bindings={},
                capture_mode="camera",
                student_self_submission_enabled=True,
                student_submission_max_pages=DEFAULT_PCR_CAMERA_MAX_PAGES,
            ),
            current_user=current_user,
            db=db,
        )
        session = await collection.find_one({"exam_id": created.exam_id})
    else:
        lifecycle = str(session.get("lifecycle_state") or "draft")
        capture_mode = str(session.get("capture_mode") or "pen")
        automatic = str(session.get("session_request_id") or "").startswith(
            "auto-pcr-camera:"
        )
        if lifecycle != "draft" and capture_mode != "camera":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This paper already has a non-camera collection in progress. "
                    "It cannot be converted while students may be submitting."
                ),
            )

        if lifecycle == "draft" or automatic:
            # For an automatic camera collection, refreshing activation safely
            # adds newly enrolled class students and upgrades the page limit.
            existing_roster = _normalize_roster(session.get("roster"))
            desired_roster = (
                roster
                if lifecycle in {"draft", "armed"}
                else _normalize_roster([*existing_roster, *roster])
            )
            await collection.update_one(
                {"exam_id": session["exam_id"]},
                {
                    "$set": {
                        "roster": desired_roster,
                        "pen_bindings": {},
                        "capture_mode": "camera",
                        "student_self_submission_enabled": True,
                        "student_submission_max_pages": DEFAULT_PCR_CAMERA_MAX_PAGES,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )
            session = await collection.find_one({"exam_id": session["exam_id"]})
        elif not bool(session.get("student_self_submission_enabled", False)):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The existing camera collection does not allow student uploads",
            )

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera answer-copy collection could not be created",
        )

    exam_id = str(session.get("exam_id") or "")
    lifecycle = str(session.get("lifecycle_state") or "draft")
    if lifecycle == "draft":
        preflight = await _build_preflight(tenant_db, session)
        if not preflight.ready_to_arm:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "Camera answer-copy collection is not ready",
                    "preflight": preflight.model_dump(),
                },
            )
        await collection.update_one(
            {"exam_id": exam_id, "lifecycle_state": "draft"},
            {
                "$set": {
                    "lifecycle_state": "armed",
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
        session = await collection.find_one({"exam_id": exam_id})
        lifecycle = str((session or {}).get("lifecycle_state") or "draft")

    if lifecycle == "armed":
        now = datetime.now(timezone.utc)
        await collection.update_one(
            {"exam_id": exam_id, "lifecycle_state": "armed"},
            {
                "$set": {
                    "lifecycle_state": "in_progress",
                    "started_at": now,
                    "updated_at": now,
                }
            },
        )
        session = await collection.find_one({"exam_id": exam_id})
        lifecycle = str((session or {}).get("lifecycle_state") or "armed")

    if lifecycle != "in_progress":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Camera answer-copy collection is not open",
        )
    return _doc_to_response(session)


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
    "/{exam_id}/setup",
    response_model=ExamDetailResponse,
    summary="Update mutable conducted-session setup while it is still draft",
    responses={
        400: {"description": "No setup fields supplied or session is no longer draft"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        409: {"description": "Session changed concurrently"},
    },
)
async def update_exam_setup(
    exam_id: str,
    body: ExamSetupUpdateRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> ExamDetailResponse:
    """Amend roster/capture configuration before arming without replacing a paper.

    A draft can be repaired after a hub discovery or pen-binding mistake, but
    the immutable paper snapshot and all lifecycle data remain untouched.
    """
    if (
        body.roster is None
        and body.pen_bindings is None
        and body.capture_mode is None
        and body.student_self_submission_enabled is None
        and body.student_submission_max_pages is None
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Provide roster, pen_bindings, capture_mode, or student self-submission "
                "settings to update draft setup"
            ),
        )

    tenant_db = await _get_tenant_db(db, current_user)
    collection = tenant_db["exampen_exams"]
    doc = await collection.find_one({"exam_id": exam_id})
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Exam {exam_id} not found")
    _require_tutor_visibility(doc, current_user)
    if doc.get("lifecycle_state") != "draft":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session setup can only be changed while the session is in draft",
        )

    roster = _normalize_roster(body.roster) if body.roster is not None else _normalize_roster(doc.get("roster"))
    capture_mode = _normalize_capture_mode(
        body.capture_mode if body.capture_mode is not None else doc.get("capture_mode"),
        str(doc.get("exam_type") or ""),
    )
    student_self_submission_enabled, student_submission_max_pages = (
        _normalize_student_self_submission_config(
            enabled=(
                body.student_self_submission_enabled
                if body.student_self_submission_enabled is not None
                else doc.get("student_self_submission_enabled", False)
            ),
            max_pages=(
                body.student_submission_max_pages
                if body.student_submission_max_pages is not None
                else doc.get("student_submission_max_pages", 20)
            ),
            exam_type=str(doc.get("exam_type") or ""),
            capture_mode=capture_mode,
        )
    )
    pen_bindings = _normalize_pen_bindings(
        body.pen_bindings if body.pen_bindings is not None else doc.get("pen_bindings"),
        roster,
    )

    now = datetime.now(timezone.utc)
    result = await collection.update_one(
        {"exam_id": exam_id, "lifecycle_state": "draft"},
        {
            "$set": {
                "roster": roster,
                "pen_bindings": pen_bindings,
                "capture_mode": capture_mode,
                "student_self_submission_enabled": student_self_submission_enabled,
                "student_submission_max_pages": student_submission_max_pages,
                "updated_at": now,
            }
        },
    )
    if result.matched_count != 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session setup changed concurrently; refresh and try again",
        )
    updated = await collection.find_one({"exam_id": exam_id})
    return _doc_to_response(updated)


@router.get(
    "/{exam_id}/preflight",
    response_model=ExamPreflightResponse,
    summary="Get server-authoritative session preflight checks",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_preflight(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> ExamPreflightResponse:
    """Expose the same checks used by the arm transition to every client."""
    tenant_db = await _get_tenant_db(db, current_user)
    doc = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Exam {exam_id} not found")
    _require_tutor_visibility(doc, current_user)
    return await _build_preflight(tenant_db, doc)


@router.post(
    "/{exam_id}/attendance/absent",
    summary="Record an absent rostered student before review",
    responses={
        400: {"description": "Student cannot be marked absent in the current state"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam or student not found"},
    },
)
async def mark_student_absent(
    exam_id: str,
    body: MarkAbsentRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> ExamDetailResponse:
    """Explicit absence prevents an expected student from blocking review."""
    tenant_db = await _get_tenant_db(db, current_user)
    collection = tenant_db["exampen_exams"]
    doc = await collection.find_one({"exam_id": exam_id})
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Exam {exam_id} not found")
    _require_tutor_visibility(doc, current_user)
    if doc.get("lifecycle_state") not in {"collection_closed", "uploading"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Students can be marked absent only after collection closes",
        )
    student_id = body.student_id.strip()
    if student_id not in _normalize_roster(doc.get("roster")):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student is not in this exam roster")
    await collection.update_one(
        {"exam_id": exam_id},
        {
            "$addToSet": {"absent_student_ids": student_id},
            "$push": {
                "attendance_audit": {
                    "student_id": student_id,
                    "status": "absent",
                    "note": body.note,
                    "recorded_by": current_user.get("user_id"),
                    "recorded_at": datetime.now(timezone.utc),
                }
            },
            "$set": {"updated_at": datetime.now(timezone.utc)},
        },
    )
    updated = await collection.find_one({"exam_id": exam_id})
    # If the final blocker was an absent student, do not require an admin to
    # manually retry the lifecycle transition after every completed PCR job.
    # The coordinator uses a compare-and-set update, so it is safe to invoke
    # from both attendance and worker completion paths.
    if updated and updated.get("lifecycle_state") == "uploading":
        from services.exampen_workflow import _maybe_mark_exam_ready_for_review

        await _maybe_mark_exam_ready_for_review(tenant_db, exam_id)
        updated = await collection.find_one({"exam_id": exam_id})
    return _doc_to_response(updated)


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

    if body.target_state == "armed":
        preflight = await _build_preflight(tenant_db, doc)
        if not preflight.ready_to_arm:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "Exam session cannot be armed until preflight passes",
                    "preflight": preflight.model_dump(),
                },
            )

    if body.target_state == "ready_for_eval":
        blockers = await _ready_for_eval_issues(tenant_db, doc)
        if blockers:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "Exam cannot enter review until uploads and processing are complete",
                    "errors": blockers,
                },
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

    result = await collection.update_one(
        {"exam_id": exam_id, "lifecycle_state": current_state},
        update,
    )
    if result.matched_count != 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Exam lifecycle changed concurrently; refresh and try again",
        )

    # A fast camera/pen worker can finish while the invigilator is moving the
    # session from collection_closed to uploading.  Re-run the durable
    # coordinator at that boundary so such an exam reaches ready_for_eval
    # automatically rather than being stranded in uploading.
    if body.target_state == "uploading":
        from services.exampen_workflow import _maybe_mark_exam_ready_for_review

        await _maybe_mark_exam_ready_for_review(tenant_db, exam_id)

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
    hub_collection = tenant_db["exampen_hubs"]

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

    hub_doc = await hub_collection.find_one({"hub_id": body.hub_id})
    if hub_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hub {body.hub_id} not found",
        )
    assigned_exam_id = hub_doc.get("assigned_exam_id")
    if assigned_exam_id and assigned_exam_id != exam_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Hub {body.hub_id} is already assigned to exam {assigned_exam_id}",
        )
    if not _is_tutor_admin_role(current_user):
        # Reuse the hub's existing selected-tutor policy.  Visibility of the
        # paper alone must not grant a tutor control over every school hub.
        from api.v1.hub_ops_async import _is_hub_selected_for_tutor

        if not _is_hub_selected_for_tutor(hub_doc, current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tutor is not authorised for this hub",
            )

    now = datetime.now(timezone.utc)
    new_assignment = {
        "hub_id": body.hub_id,
        "hub_name": body.hub_name,
        "assigned_at": now,
        "session_started_at": None,
        "session_ended_at": None,
        "status": "assigned",
    }

    # Both projections must agree before the hub begins polling for an exam.
    # Mongo deployments without a replica-set transaction use explicit
    # compensation if the second write fails.
    reserved = await hub_collection.update_one(
        {
            "hub_id": body.hub_id,
            "$or": [
                {"assigned_exam_id": None},
                {"assigned_exam_id": {"$exists": False}},
                {"assigned_exam_id": exam_id},
            ],
        },
        {"$set": {"assigned_exam_id": exam_id, "assigned_at": now}},
    )
    if reserved.matched_count != 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Hub {body.hub_id} could not be reserved for this exam",
        )

    try:
        updated = await collection.update_one(
            {
                "exam_id": exam_id,
                "lifecycle_state": "draft",
                "hub_assignments.hub_id": {"$ne": body.hub_id},
            },
            {
                "$push": {"hub_assignments": new_assignment},
                "$set": {"updated_at": now},
            },
        )
        if updated.matched_count != 1:
            raise RuntimeError("Exam changed while assigning hub")
    except Exception:
        await hub_collection.update_one(
            {"hub_id": body.hub_id, "assigned_exam_id": exam_id},
            {"$set": {"assigned_exam_id": None, "assigned_at": None}},
        )
        logger.exception("Rolled back hub %s reservation for exam %s", body.hub_id, exam_id)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Exam changed while assigning hub; refresh and try again",
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
    hub_collection = tenant_db["exampen_hubs"]

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
    updated = await collection.update_one(
        {"exam_id": exam_id},
        {
            "$pull": {"hub_assignments": {"hub_id": hub_id}},
            "$set": {"updated_at": now},
        },
    )
    if updated.matched_count != 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Exam changed while unassigning hub; refresh and try again",
        )

    # Clear only this exact assignment.  A concurrent reassignment must never
    # be removed by an old browser/mobile request.
    await hub_collection.update_one(
        {"hub_id": hub_id, "assigned_exam_id": exam_id},
        {"$set": {"assigned_exam_id": None, "assigned_at": None}},
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

    absent_student_ids = {str(student_id) for student_id in (doc.get("absent_student_ids") or [])}
    roster = set(_normalize_roster(doc.get("roster"))) - absent_student_ids
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
        absent_student_ids=sorted(absent_student_ids),
    )


@router.get(
    "/{exam_id}/processing",
    response_model=ProcessingJobListResponse,
    summary="List PCR processing jobs for a conducted exam",
)
async def list_processing_jobs(
    exam_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> ProcessingJobListResponse:
    tenant_db = await _get_tenant_db(db, current_user)
    exam = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if exam is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Exam {exam_id} not found")
    _require_tutor_visibility(exam, current_user)
    cursor = tenant_db["exampen_processing_jobs"].find({"exam_id": exam_id}).sort("created_at", -1)
    items = await cursor.to_list(length=5000)
    return ProcessingJobListResponse(
        exam_id=exam_id,
        items=[_processing_job_to_response(item) for item in items],
    )


@router.post(
    "/{exam_id}/processing/{job_id}/retry",
    response_model=ProcessingJobResponse,
    summary="Reprocess a PCR answer copy from its canonical uploaded pages",
)
async def retry_exam_processing_job(
    exam_id: str,
    job_id: str,
    body: Optional[ProcessingJobReprocessRequest] = None,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> ProcessingJobResponse:
    tenant_db = await _get_tenant_db(db, current_user)
    exam = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if exam is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Exam {exam_id} not found")
    _require_tutor_visibility(exam, current_user)
    job = await tenant_db["exampen_processing_jobs"].find_one({"job_id": job_id, "exam_id": exam_id})
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Processing job {job_id} not found")

    submission = await tenant_db["evalpen_submissions"].find_one(
        {"submission_id": job.get("submission_id"), "exam_id": exam_id},
        projection={"publication_status": 1},
    )
    if submission is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Canonical answer-copy submission not found",
        )
    if submission.get("publication_status") == "published":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Published results cannot be reprocessed. Unpublish or create a recheck workflow first.",
        )

    from services.exampen_review_lease import (
        SubmissionReviewBusyError,
        acquire_submission_review_lease,
        release_submission_review_lease,
    )

    actor_id = str(
        current_user.get("user_id")
        or current_user.get("tutor_id")
        or current_user.get("admin_id")
        or current_user.get("username")
        or "unknown"
    )
    try:
        review_lease_token = await acquire_submission_review_lease(
            tenant_db,
            str(job.get("submission_id") or ""),
            actor_id=actor_id,
            operation="reprocess",
        )
    except SubmissionReviewBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    try:
        from services.exampen_workflow import ProcessingJobBusyError, reprocess_processing_job

        # Recheck after acquiring the shared review fence. Publication may
        # have committed between the first read and the lease acquisition.
        current_submission = await tenant_db["evalpen_submissions"].find_one(
            {"submission_id": job.get("submission_id")},
            {"publication_status": 1},
        )
        if (current_submission or {}).get("publication_status") == "published":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Published results cannot be reprocessed. Create a recheck workflow first.",
            )
        retried = await reprocess_processing_job(
            tenant_db,
            db_name=str(current_user.get("db_name") or ""),
            job_id=job_id,
            requested_by=actor_id,
            reason=(body.reason if body else None),
        )
    except ProcessingJobBusyError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    finally:
        await release_submission_review_lease(
            tenant_db,
            str(job.get("submission_id") or ""),
            review_lease_token,
        )
    return _processing_job_to_response(retried)
