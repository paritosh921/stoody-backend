"""
ExamPen Stroke Ingest API — pen-originated chunked artifact upload from hubs.

Handles:
  - Chunked stroke upload per exam/pen
  - Deduplication via content hash
  - Finalization with checksum verification
  - Per-pen upload status query

Architecture:
    IMPLEMENTATION_PLAN.md §UP-004
    new-docs/api/stroke-ingest.openapi.yaml

Ownership Declaration:
    - Writes:  exampen_stroke_chunks (chunk tracking), evalpen_submissions + evalpen_answer_pages
               (via IngestService on finalization)
    - Reads from: exampen_exams (lifecycle validation), exampen_hubs (pen inventory validation)
    - Never writes to: exampen_exams

Hard constraints:
    - C1: MongoDB only
    - Exam must be in 'uploading' lifecycle state for chunk acceptance
    - Deduplication: hash(exam_id + pen_mac + chunk_index + payload_hash) → return existing if dup
    - Finalization triggers bridge to IngestService for canonical persistence
"""

from __future__ import annotations

import hashlib
import base64
import binascii
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator

from core.database import DatabaseManager
from core.upload_security.policies import get_upload_policy
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

router = APIRouter()

HUB_SCOPE_DATA_UPLOAD = "hub:data:upload"


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

def require_hub_or_admin(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Accept hub tokens (user_type=hub) or admin/tutor tokens."""
    allowed = {"hub", "admin", "tutor", "b2c_admin"}
    if current_user.get("user_type") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hub or admin access required for stroke ingest",
        )
    return current_user


def _require_hub_data_upload_scope(current_user: Dict[str, Any]) -> None:
    if current_user.get("user_type") != "hub":
        return
    raw_scopes = current_user.get("scopes") or []
    scopes = {raw_scopes} if isinstance(raw_scopes, str) else {str(scope) for scope in raw_scopes}
    if HUB_SCOPE_DATA_UPLOAD not in scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Hub token missing required scope: {HUB_SCOPE_DATA_UPLOAD}",
        )


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

class StrokeChunkUpload(BaseModel):
    exam_type: str = Field(..., description="dcr or pcr")
    student_id: str
    hub_id: Optional[str] = Field(None, description="Hub that uploaded this chunk")
    chunk_index: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=1)
    payload_base64: str = Field(..., description="Base64-encoded stroke chunk payload")
    payload_hash: Optional[str] = Field(
        None,
        description="Optional SHA-256 of the base64-encoded payload string; verified when supplied",
    )

    @field_validator("exam_type")
    @classmethod
    def validate_exam_type(cls, value: str) -> str:
        normalized = str(value or "").lower()
        if normalized not in {"dcr", "pcr"}:
            raise ValueError("exam_type must be dcr or pcr")
        return normalized

    @model_validator(mode="after")
    def validate_chunk_policy(self) -> "StrokeChunkUpload":
        policy = get_upload_policy("hub_stroke_chunk")
        if policy.max_total_chunks is not None and self.total_chunks > policy.max_total_chunks:
            raise ValueError(f"total_chunks exceeds {policy.max_total_chunks}")
        encoded = self.payload_base64 or ""
        if policy.max_payload_base64_bytes is not None and len(encoded.encode("ascii", errors="ignore")) > policy.max_payload_base64_bytes:
            raise ValueError(f"payload_base64 exceeds {policy.max_payload_base64_bytes} bytes")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("payload_base64 is not valid base64") from exc
        if policy.max_decoded_payload_bytes is not None and len(decoded) > policy.max_decoded_payload_bytes:
            raise ValueError(f"decoded payload exceeds {policy.max_decoded_payload_bytes} bytes")
        return self


class IngestAck(BaseModel):
    artifact_id: str
    chunk_index: int
    deduplicated: bool
    accepted_at: str


class FinalizeRequest(BaseModel):
    student_id: str
    expected_checksum: str = Field(..., description="SHA-256 of all base64-encoded payload strings concatenated in chunk_index order")
    total_chunks: int = Field(..., ge=1)
    hub_id: Optional[str] = Field(None, description="Hub that uploaded these chunks")
    pages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Page-level metadata extracted from pen data: [{page_number, raw_strokes}]",
    )

    @model_validator(mode="after")
    def validate_finalize_policy(self) -> "FinalizeRequest":
        policy = get_upload_policy("hub_stroke_finalize")
        if policy.max_total_chunks is not None and self.total_chunks > policy.max_total_chunks:
            raise ValueError(f"total_chunks exceeds {policy.max_total_chunks}")
        if len(self.expected_checksum or "") != 64:
            raise ValueError("expected_checksum must be a SHA-256 hex digest")
        try:
            int(self.expected_checksum, 16)
        except ValueError as exc:
            raise ValueError("expected_checksum must be hex") from exc
        if policy.max_pages is not None and len(self.pages) > policy.max_pages:
            raise ValueError(f"pages exceeds {policy.max_pages}")
        for page in self.pages:
            raw_strokes = page.get("raw_strokes") or []
            if policy.max_strokes_per_page is not None and isinstance(raw_strokes, list):
                if len(raw_strokes) > policy.max_strokes_per_page:
                    raise ValueError(f"raw_strokes exceeds {policy.max_strokes_per_page} per page")
            page_number = page.get("page_number")
            try:
                if page_number is not None and int(page_number) < 1:
                    raise ValueError("page_number must be positive")
            except (TypeError, ValueError) as exc:
                raise ValueError("page_number must be positive") from exc
        return self


class FinalizeResponse(BaseModel):
    exam_id: str
    pen_mac: str
    student_id: str
    submission_id: Optional[str] = None
    chunks_received: int
    checksum_match: bool
    ingested: bool
    message: str


class PenUploadStatus(BaseModel):
    exam_id: str
    pen_mac: str
    student_id: Optional[str] = None
    chunks_expected: int
    chunks_received: int
    finalized: bool
    checksum_verified: bool
    ingested: bool
    last_chunk_at: Optional[str] = None


class DedupCheckRequest(BaseModel):
    chunk_index: int
    payload_hash: str = Field(..., description="SHA-256 of the base64-encoded payload string for this chunk")
    hub_id: Optional[str] = Field(None, description="Hub probing for this chunk")


class DedupCheckResponse(BaseModel):
    exists: bool
    artifact_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

_indexes_ensured = False


async def _ensure_indexes(collection) -> None:
    global _indexes_ensured
    if _indexes_ensured:
        return
    await collection.create_index(
        [("exam_id", 1), ("pen_mac", 1), ("chunk_index", 1)],
        unique=True,
    )
    await collection.create_index("dedup_hash")
    _indexes_ensured = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_dedup_hash(exam_id: str, pen_mac: str, chunk_index: int, payload_hash: str) -> str:
    raw = f"{exam_id}:{pen_mac}:{chunk_index}:{payload_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _compute_payload_hash(payload_b64: str) -> str:
    return hashlib.sha256(payload_b64.encode()).hexdigest()


def _normalize_pen_mac(pen_mac: str) -> str:
    return str(pen_mac or "").upper()


def _expected_student_for_bound_pen(
    exam_doc: Dict[str, Any],
    pen_mac_upper: str,
) -> Optional[str]:
    bindings = exam_doc.get("pen_bindings") or {}
    if not isinstance(bindings, dict):
        return None
    for raw_mac, raw_student_id in bindings.items():
        if _normalize_pen_mac(str(raw_mac)) != pen_mac_upper:
            continue
        if raw_student_id is None:
            return None
        student_id = str(raw_student_id)
        return student_id or None
    return None


def _require_student_binding_match(
    exam_doc: Dict[str, Any],
    pen_mac_upper: str,
    student_id: str,
) -> None:
    expected_student_id = _expected_student_for_bound_pen(exam_doc, pen_mac_upper)
    if expected_student_id is None:
        return
    if str(student_id) != expected_student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Student {student_id} is not bound to pen {pen_mac_upper} "
                f"for exam {exam_doc.get('exam_id')}"
            ),
        )


def _fmt(v) -> Optional[str]:
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if v is not None:
        return str(v)
    return None


# ---------------------------------------------------------------------------
# Exam owner / caller authority resolution
# ---------------------------------------------------------------------------
#
# Task 2 of the shared collector plan: finalization must attribute ingest
# provenance to the exam OWNER (admin_id stored on exampen_exams), not to
# the caller's identity (hub id / admin user_id / tutor user_id).
#
# This helper is applied to ALL stroke ingest route handlers
# (upload_stroke_chunk, finalize_pen_upload, get_pen_upload_status,
# dedup_check) so hub-assignment authorization is consistent across the
# full surface, not just finalize.
#
# Visibility / hub-assignment rules mirror Task 1 in exam_orch_async:
#   - admin / b2c_admin callers always pass
#   - tutor callers pass iff (created_by_tutor_id == tutor id)
#                            OR (teacher_ids contains tutor id)
#                            OR (teacher_ids is empty / None / missing)
#   - hub tokens pass iff the hub is assigned to the exam (either via
#     exam_doc.hub_assignments OR exampen_hubs.assigned_exam_id) AND
#     the body.hub_id matches the token hub id when body.hub_id is supplied
#
# If exampen_exams.admin_id is missing/empty the exam is in an
# indeterminate ownership state — we refuse the request with 400 BEFORE
# IngestService is invoked so we never have to fall back to the caller's
# identity (which would silently reintroduce hub-id provenance).


async def _resolve_exam_context_for_ingest(
    tenant_db: Any,
    exam_id: str,
    current_user: Dict[str, Any],
    body_hub_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the exam, enforce caller authority, and return the canonical
    exam context used by every stroke-ingest route handler.

    Returns
    -------
    dict
        ``{"exam_doc": <exam_doc>, "admin_id": <canonical owner admin_id>,
           "exam_type": <"dcr"|"pcr">, "teacher_ids": [...]}``

    Raises
    ------
    HTTPException
        - 404 if the exam is missing
        - 403 if the caller is a hub token that does not match an assigned
          hub, or whose body.hub_id disagrees with the token hub id
        - 403 if the caller is a tutor that is not visible to the exam
          (Task 1 visibility rules)
        - 400 if the exam has no admin_id (data-integrity refusal; we
          will not fall back to the caller's identity)
    """
    exam_doc = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if exam_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {exam_id} not found",
        )

    user_type = (current_user.get("user_type") or "").lower()
    canonical_admin_id = exam_doc.get("admin_id")
    if canonical_admin_id is not None:
        canonical_admin_id = str(canonical_admin_id)

    raw_teacher_ids = exam_doc.get("teacher_ids") or []
    if not isinstance(raw_teacher_ids, list):
        raw_teacher_ids = []
    teacher_ids = [str(t) for t in raw_teacher_ids]

    if user_type == "hub":
        token_hub_id = (
            current_user.get("hub_id")
            or current_user.get("user_id")
        )
        # body.hub_id, when supplied, must match the token hub id.
        if body_hub_id is not None and token_hub_id is not None:
            if str(body_hub_id) != str(token_hub_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=(
                        "Hub token hub_id does not match body.hub_id "
                        f"(token={token_hub_id}, body={body_hub_id})"
                    ),
                )

        effective_hub_id = (
            str(body_hub_id) if body_hub_id is not None else (
                str(token_hub_id) if token_hub_id is not None else None
            )
        )

        # Determine assignment: either exam_doc.hub_assignments or
        # exampen_hubs.assigned_exam_id.
        assigned = False
        for ha in exam_doc.get("hub_assignments", []) or []:
            if str(ha.get("hub_id", "")) == effective_hub_id:
                assigned = True
                break

        if not assigned and effective_hub_id is not None:
            hub_doc = await tenant_db["exampen_hubs"].find_one(
                {"hub_id": effective_hub_id}
            )
            if hub_doc is not None and str(
                hub_doc.get("assigned_exam_id") or ""
            ) == str(exam_id):
                assigned = True

        if not assigned:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Hub {effective_hub_id} is not assigned to exam {exam_id}"
                ),
            )

    elif user_type in ("admin", "b2c_admin"):
        # Admin / b2c_admin bypass visibility; canonical admin_id is the
        # exam owner (not the caller).
        pass

    elif user_type == "tutor":
        raw_tutor_id = current_user.get("tutor_id") or current_user.get("user_id")
        tutor_id = str(raw_tutor_id) if raw_tutor_id is not None else None
        visible = False
        if tutor_id is not None:
            if exam_doc.get("created_by_tutor_id") == tutor_id:
                visible = True
            elif not teacher_ids:
                visible = True
            elif tutor_id in teacher_ids:
                visible = True
        if not visible:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Exam is not visible to this tutor",
            )

    # Other user types are already rejected by require_hub_or_admin.

    # Data-integrity guard: refuse to operate on an exam whose owner
    # admin_id is missing/empty. This prevents any caller path from
    # silently falling back to current_user.user_id (which would
    # reintroduce hub-id / tutor-id provenance).
    if not canonical_admin_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Exam {exam_id} has no admin_id (owner) set; cannot "
                "process stroke ingest until the exam owner is established"
            ),
        )

    return {
        "exam_doc": exam_doc,
        "admin_id": canonical_admin_id,
        "exam_type": exam_doc.get("exam_type"),
        "teacher_ids": teacher_ids,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/{exam_id}/{pen_mac}",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload one stroke chunk from a hub",
    responses={
        400: {"description": "Exam not in uploading state or invalid payload"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def upload_stroke_chunk(
    exam_id: str,
    pen_mac: str,
    body: StrokeChunkUpload,
    current_user: Dict[str, Any] = Depends(require_hub_or_admin),
    db: DatabaseManager = Depends(get_database),
) -> IngestAck:
    """Accept a single stroke chunk from a hub for a specific pen.

    Chunks are deduplicated by content hash. If the exact same chunk was
    already received, the existing artifact_id is returned.
    """
    _require_hub_data_upload_scope(current_user)
    tenant_db = await _get_tenant_db(db, current_user)

    # Resolve exam owner / caller authority BEFORE touching chunks so 404,
    # 403 (unassigned hub / invisible tutor) and 400 (missing admin_id) are
    # surfaced consistently across all stroke-ingest route handlers.
    exam_ctx = await _resolve_exam_context_for_ingest(
        tenant_db=tenant_db,
        exam_id=exam_id,
        current_user=current_user,
        body_hub_id=body.hub_id,
    )
    exam_doc = exam_ctx["exam_doc"]
    canonical_exam_type = exam_ctx["exam_type"]

    lifecycle = exam_doc.get("lifecycle_state", "draft")
    if lifecycle not in ("in_progress", "collection_closed", "uploading"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Exam {exam_id} is in state '{lifecycle}' — must be 'in_progress', 'collection_closed', or 'uploading' to accept chunks",
        )

    if canonical_exam_type not in ("dcr", "pcr"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Exam {exam_id} has no valid exam_type set",
        )
    if body.exam_type != canonical_exam_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Chunk exam_type {body.exam_type!r} does not match "
                f"canonical exam_type {canonical_exam_type!r}"
            ),
        )

    pen_mac_upper = _normalize_pen_mac(pen_mac)
    _require_student_binding_match(exam_doc, pen_mac_upper, body.student_id)
    chunk_col = tenant_db["exampen_stroke_chunks"]
    await _ensure_indexes(chunk_col)

    # Deduplication check
    payload_hash = _compute_payload_hash(body.payload_base64)
    if body.payload_hash is not None and body.payload_hash != payload_hash:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "payload_hash mismatch: supplied hash does not match "
                "payload_base64"
            ),
        )
    dedup_hash = _compute_dedup_hash(exam_id, pen_mac_upper, body.chunk_index, payload_hash)

    existing = await chunk_col.find_one({"dedup_hash": dedup_hash})
    if existing:
        logger.debug(
            "Duplicate chunk: exam=%s pen=%s chunk=%d",
            exam_id, pen_mac_upper, body.chunk_index,
        )
        return IngestAck(
            artifact_id=existing.get("artifact_id", str(existing["_id"])),
            chunk_index=body.chunk_index,
            deduplicated=True,
            accepted_at=_fmt(existing.get("received_at")) or datetime.now(timezone.utc).isoformat(),
        )

    # Store chunk
    now = datetime.now(timezone.utc)
    artifact_id = f"{exam_id}:{pen_mac_upper}:{body.chunk_index}"

    doc = {
        "artifact_id": artifact_id,
        "exam_id": exam_id,
        "exam_type": canonical_exam_type,
        "hub_id": body.hub_id,
        "pen_mac": pen_mac_upper,
        "student_id": body.student_id,
        "chunk_index": body.chunk_index,
        "total_chunks": body.total_chunks,
        "payload_base64": body.payload_base64,
        "payload_hash": payload_hash,
        "dedup_hash": dedup_hash,
        "received_at": now,
        "finalized": False,
    }

    try:
        await chunk_col.insert_one(doc)
    except Exception as exc:
        # Duplicate key — race condition, treat as dedup
        if hasattr(exc, "code") and exc.code == 11000:
            return IngestAck(
                artifact_id=artifact_id,
                chunk_index=body.chunk_index,
                deduplicated=True,
                accepted_at=now.isoformat(),
            )
        raise

    logger.info(
        "Chunk accepted: exam=%s pen=%s chunk=%d/%d student=%s",
        exam_id, pen_mac_upper, body.chunk_index, body.total_chunks, body.student_id,
    )

    return IngestAck(
        artifact_id=artifact_id,
        chunk_index=body.chunk_index,
        deduplicated=False,
        accepted_at=now.isoformat(),
    )


@router.post(
    "/{exam_id}/{pen_mac}/complete",
    summary="Finalize pen upload — verify checksum and trigger canonical ingest",
    responses={
        400: {"description": "Checksum mismatch or missing chunks"},
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def finalize_pen_upload(
    exam_id: str,
    pen_mac: str,
    body: FinalizeRequest,
    current_user: Dict[str, Any] = Depends(require_hub_or_admin),
    db: DatabaseManager = Depends(get_database),
) -> FinalizeResponse:
    """Finalize a pen's upload for an exam.

    Verifies all chunks are received, validates checksum, then bridges
    the assembled data into the canonical ingest substrate via IngestService.

    The canonical admin_id sent to IngestService is the exam OWNER's
    admin_id (read from exampen_exams.admin_id), not the caller's user_id.
    See Task 2 of the shared collector plan.
    """
    _require_hub_data_upload_scope(current_user)
    tenant_db = await _get_tenant_db(db, current_user)
    pen_mac_upper = _normalize_pen_mac(pen_mac)
    chunk_col = tenant_db["exampen_stroke_chunks"]

    # Resolve exam owner / caller authority BEFORE touching chunks so 404
    # and 403 are surfaced for missing exams and unauthorized hubs/tutors.
    exam_ctx = await _resolve_exam_context_for_ingest(
        tenant_db=tenant_db,
        exam_id=exam_id,
        current_user=current_user,
        body_hub_id=body.hub_id,
    )
    canonical_admin_id = exam_ctx["admin_id"]
    _require_student_binding_match(
        exam_ctx["exam_doc"],
        pen_mac_upper,
        body.student_id,
    )

    # Count received chunks
    received = await chunk_col.count_documents({
        "exam_id": exam_id,
        "pen_mac": pen_mac_upper,
    })

    if received < body.total_chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected {body.total_chunks} chunks but only {received} received",
        )

    # Fetch chunks in order and compute composite checksum
    cursor = chunk_col.find(
        {"exam_id": exam_id, "pen_mac": pen_mac_upper}
    ).sort("chunk_index", 1)
    chunks = await cursor.to_list(length=body.total_chunks + 10)

    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(chunk.get("payload_base64", "").encode())
    computed_checksum = hasher.hexdigest()

    checksum_match = computed_checksum == body.expected_checksum
    if not checksum_match:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Checksum mismatch: expected={body.expected_checksum}, computed={computed_checksum}",
        )

    # Mark chunks as finalized
    await chunk_col.update_many(
        {"exam_id": exam_id, "pen_mac": pen_mac_upper},
        {"$set": {"finalized": True}},
    )

    # Bridge to canonical ingest — reconstruct pages from chunks if not provided
    submission_id = None
    ingested = False

    pages_for_ingest = body.pages
    if not pages_for_ingest:
        # Reconstruct a single-page submission from the stored chunks.
        # Each chunk's payload is base64-encoded stroke data. We create one
        # page with all chunk payloads as the raw_strokes reference.
        import base64 as _b64

        page_payloads = []
        for chunk in chunks:
            try:
                raw = _b64.b64decode(chunk.get("payload_base64", ""))
                page_payloads.append(raw.decode("utf-8", errors="replace"))
            except Exception:
                page_payloads.append(chunk.get("payload_base64", ""))

        pages_for_ingest = [{
            "page_number": 1,
            "raw_strokes": [{"chunk_index": i, "data": p} for i, p in enumerate(page_payloads)],
        }]

    try:
        from api.v1._exampen_imports import load_exampen
        ingest_mod = load_exampen("ingest.service")
        IngestService = ingest_mod.IngestService

        service = IngestService(tenant_db)
        await service.initialize()

        result = await service.ingest_submission(
            exam_id=exam_id,
            student_id=body.student_id,
            admin_id=canonical_admin_id,
            source="ble_pen",
            pen_mac=pen_mac_upper,
            hub_id=body.hub_id,
            pages=pages_for_ingest,
        )
        submission_id = result.submission_id
        ingested = True

        logger.info(
            "Finalized and ingested: exam=%s pen=%s student=%s submission=%s",
            exam_id, pen_mac_upper, body.student_id, submission_id,
        )
    except (ImportError, AttributeError):
        logger.warning("IngestService not available — chunks finalized but not ingested")
    except Exception:
        logger.exception("Ingest failed during finalization for exam=%s pen=%s", exam_id, pen_mac_upper)

    return FinalizeResponse(
        exam_id=exam_id,
        pen_mac=pen_mac_upper,
        student_id=body.student_id,
        submission_id=submission_id,
        chunks_received=received,
        checksum_match=True,
        ingested=ingested,
        message="Upload finalized and ingested" if ingested else "Upload finalized, pending ingest",
    )


@router.get(
    "/{exam_id}/{pen_mac}/status",
    summary="Get upload status for a specific pen",
    responses={
        403: {"description": "Insufficient permissions"},
        404: {"description": "Exam not found"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def get_pen_upload_status(
    exam_id: str,
    pen_mac: str,
    current_user: Dict[str, Any] = Depends(require_hub_or_admin),
    db: DatabaseManager = Depends(get_database),
) -> PenUploadStatus:
    """Get upload acknowledgment status for a specific pen in an exam."""
    tenant_db = await _get_tenant_db(db, current_user)

    # Enforce same hub-assignment / tutor-visibility / admin_id rules as
    # the rest of the stroke-ingest surface.
    await _resolve_exam_context_for_ingest(
        tenant_db=tenant_db,
        exam_id=exam_id,
        current_user=current_user,
        body_hub_id=None,  # GET has no body
    )

    pen_mac_upper = pen_mac.upper()
    chunk_col = tenant_db["exampen_stroke_chunks"]

    # Get all chunks for this pen
    cursor = chunk_col.find(
        {"exam_id": exam_id, "pen_mac": pen_mac_upper}
    ).sort("chunk_index", 1)
    chunks = await cursor.to_list(length=1000)

    if not chunks:
        return PenUploadStatus(
            exam_id=exam_id,
            pen_mac=pen_mac_upper,
            chunks_expected=0,
            chunks_received=0,
            finalized=False,
            checksum_verified=False,
            ingested=False,
        )

    total_expected = chunks[0].get("total_chunks", 0)
    chunks_received = len(chunks)
    finalized = all(c.get("finalized", False) for c in chunks)
    student_id = chunks[0].get("student_id")
    last_chunk_at = max((c.get("received_at") for c in chunks), default=None)

    # Check if ingested in submissions
    ingested = False
    if finalized:
        sub = await tenant_db["evalpen_submissions"].find_one({
            "exam_id": exam_id,
            "pen_mac": pen_mac_upper,
        })
        ingested = sub is not None

    return PenUploadStatus(
        exam_id=exam_id,
        pen_mac=pen_mac_upper,
        student_id=student_id,
        chunks_expected=total_expected,
        chunks_received=chunks_received,
        finalized=finalized,
        checksum_verified=finalized,
        ingested=ingested,
        last_chunk_at=_fmt(last_chunk_at),
    )


@router.post(
    "/{exam_id}/{pen_mac}/dedup",
    summary="Check if a chunk already exists before uploading",
    responses={
        403: {"description": "Insufficient permissions"},
        503: {"description": "Tenant database unavailable"},
    },
)
async def dedup_check(
    exam_id: str,
    pen_mac: str,
    body: DedupCheckRequest,
    current_user: Dict[str, Any] = Depends(require_hub_or_admin),
    db: DatabaseManager = Depends(get_database),
) -> DedupCheckResponse:
    """Pre-upload deduplication check so hub can skip already-received chunks."""
    tenant_db = await _get_tenant_db(db, current_user)

    # Enforce same hub-assignment / tutor-visibility / admin_id rules as
    # the rest of the stroke-ingest surface before touching chunk storage.
    await _resolve_exam_context_for_ingest(
        tenant_db=tenant_db,
        exam_id=exam_id,
        current_user=current_user,
        body_hub_id=body.hub_id,
    )

    pen_mac_upper = pen_mac.upper()
    chunk_col = tenant_db["exampen_stroke_chunks"]

    dedup_hash = _compute_dedup_hash(exam_id, pen_mac_upper, body.chunk_index, body.payload_hash)
    existing = await chunk_col.find_one({"dedup_hash": dedup_hash})

    if existing:
        return DedupCheckResponse(
            exists=True,
            artifact_id=existing.get("artifact_id", str(existing["_id"])),
        )

    return DedupCheckResponse(exists=False)
