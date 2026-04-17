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
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from core.database import DatabaseManager
from api.v1.auth_async import get_current_user, get_database

logger = logging.getLogger(__name__)

router = APIRouter()


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
    chunk_index: int = Field(..., ge=0)
    total_chunks: int = Field(..., ge=1)
    payload_base64: str = Field(..., description="Base64-encoded stroke chunk payload")


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


def _fmt(v) -> Optional[str]:
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if v is not None:
        return str(v)
    return None


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
    tenant_db = await _get_tenant_db(db, current_user)

    # Validate exam exists and is in uploading state
    exam_doc = await tenant_db["exampen_exams"].find_one({"exam_id": exam_id})
    if exam_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Exam {exam_id} not found",
        )

    lifecycle = exam_doc.get("lifecycle_state", "draft")
    if lifecycle not in ("in_progress", "collection_closed", "uploading"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Exam {exam_id} is in state '{lifecycle}' — must be 'in_progress', 'collection_closed', or 'uploading' to accept chunks",
        )

    if body.exam_type not in ("dcr", "pcr"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="exam_type must be 'dcr' or 'pcr'",
        )

    pen_mac_upper = pen_mac.upper()
    chunk_col = tenant_db["exampen_stroke_chunks"]
    await _ensure_indexes(chunk_col)

    # Deduplication check
    payload_hash = _compute_payload_hash(body.payload_base64)
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
        "exam_type": body.exam_type,
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
    """
    tenant_db = await _get_tenant_db(db, current_user)
    pen_mac_upper = pen_mac.upper()
    chunk_col = tenant_db["exampen_stroke_chunks"]

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
            admin_id=current_user.get("user_id", "unknown"),
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
