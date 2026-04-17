"""
Ingest Domain Models
====================

Pydantic models for canonical conducted-exam artifacts persisted by the
shared ingest substrate.

Collections
-----------
- ``evalpen_submissions``  -- one document per conducted-exam submission
- ``evalpen_answer_pages`` -- one document per page of raw artifact data

Integrity rules (TAMPER_PROOF_SPEC Layer 1):
- Every document carries ``content_hash`` (SHA-256 hex digest).
- Every document carries ``_immutable: true`` after initial write.
- Repositories MUST refuse updates on documents marked immutable.

Provenance rules (DUAL_MODE_ARCHITECTURE Section 3.2, STATE_OWNERSHIP_MAP):
- ``admin_id``, ``student_id``, ``pen_mac``, timestamps, and exam refs
  are persisted together to prevent mis-attribution (ING-02).

References
----------
- Failure modes: ING-01 (artifact loss), ING-02 (mis-attribution), ING-03 (duplicates)
- Test IDs: U-ING-01 (provenance fields), U-ING-02 (content hash)
- API schema: new-docs/api/eval-submissions.openapi.yaml
- Event schema: new-docs/contracts/events/eval.submission.received.schema.json
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ArtifactSource(str, enum.Enum):
    """Origin of the conducted-exam artifact.

    Maps to ``source`` in eval-submissions.openapi.yaml and the
    eval.submission.received event schema.
    """

    BLE_PEN = "ble_pen"
    CAMERA = "camera"


class SubmissionStatus(str, enum.Enum):
    """Processing status of a submission after ingest."""

    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class PageRef(BaseModel):
    """A reference to a single answer page within a submission.

    Matches ``page_refs[*]`` in ``CreateSubmissionRequest``.
    """

    page_num: int = Field(..., ge=1, description="1-based page number")
    raw_asset_ref: Optional[str] = Field(
        default=None,
        description="Opaque reference to raw asset in object storage (if applicable)",
    )


# ---------------------------------------------------------------------------
# Canonical document: evalpen_submissions
# ---------------------------------------------------------------------------

class ConductedExamSubmission(BaseModel):
    """Canonical conducted-exam submission record.

    Stored in ``evalpen_submissions`` within the tenant DB (``skb_<tenant>``).

    Provenance fields satisfy U-ING-01.  ``content_hash`` satisfies U-ING-02.
    ``_immutable`` enforces write-once semantics (TAMPER_PROOF_SPEC Layer 1).
    """

    # --- identity ---
    submission_id: str = Field(
        ..., description="Globally unique submission identifier"
    )
    exam_id: str = Field(..., description="Conducted exam identifier")
    student_id: str = Field(..., description="Student identity")
    admin_id: str = Field(..., description="Tenant admin who owns this exam context")

    # --- provenance ---
    source: ArtifactSource = Field(
        ..., description="Artifact origin (ble_pen or camera)"
    )
    pen_mac: Optional[str] = Field(
        default=None,
        description="BLE pen MAC address (required for ble_pen source)",
    )
    hub_id: Optional[str] = Field(
        default=None,
        description="Hub identity that uploaded this submission (for per-hub rollup)",
    )
    page_count: int = Field(
        default=0, ge=0, description="Total number of answer pages"
    )
    page_refs: List[PageRef] = Field(
        default_factory=list,
        description="Ordered references to per-page answer artifacts",
    )

    # --- integrity ---
    content_hash: str = Field(
        ...,
        description="SHA-256 hex digest over canonical page content",
    )
    immutable: bool = Field(
        default=True,
        alias="_immutable",
        description="Write-once flag; repository refuses updates when True",
    )

    # --- status ---
    upload_status: str = Field(
        default="received",
        description="Upload acknowledgment status: received | acknowledged. "
                    "Set to 'received' at ingest, 'acknowledged' after canonical write confirmed.",
    )
    segmentation_status: SubmissionStatus = Field(
        default=SubmissionStatus.PENDING,
        description="Downstream processing status (set by engines, not ingest)",
    )

    # --- timestamps ---
    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of ingest",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of document creation",
    )

    class Config:
        populate_by_name = True
        use_enum_values = True

    def to_mongo_doc(self) -> Dict[str, Any]:
        """Serialize to a MongoDB-ready dictionary.

        Uses ``model_dump`` with ``by_alias=True`` so the ``_immutable``
        field is stored under its correct key in MongoDB.
        """
        doc = self.model_dump(by_alias=True, mode="json")
        # Ensure datetime fields are proper datetime objects for MongoDB
        doc["submitted_at"] = self.submitted_at
        doc["created_at"] = self.created_at
        return doc


# ---------------------------------------------------------------------------
# Canonical document: evalpen_answer_pages
# ---------------------------------------------------------------------------

class AnswerPage(BaseModel):
    """Canonical answer-page artifact record.

    Stored in ``evalpen_answer_pages`` within the tenant DB.  Each document
    holds the raw stroke or image data for one page of a conducted exam.

    The ``content_hash`` is computed over the raw payload so that any
    alteration is detectable (TAMPER_PROOF_SPEC Layer 1, TAMP-02).
    """

    # --- identity ---
    page_id: str = Field(..., description="Unique page document identifier")
    submission_id: str = Field(
        ..., description="Parent submission this page belongs to"
    )
    exam_id: str = Field(..., description="Conducted exam identifier")
    student_id: str = Field(..., description="Student identity")
    admin_id: str = Field(..., description="Tenant admin identity")

    # --- page data ---
    page_number: int = Field(..., ge=1, description="1-based page number")
    source: ArtifactSource = Field(..., description="Artifact origin")
    pen_mac: Optional[str] = Field(
        default=None,
        description="BLE pen MAC address (when source is ble_pen)",
    )
    raw_strokes: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Canonical stroke vectors for pen-originated pages",
    )
    raw_image_ref: Optional[str] = Field(
        default=None,
        description="Reference to raw camera/scan image asset",
    )

    # --- integrity ---
    content_hash: str = Field(
        ...,
        description="SHA-256 hex digest over the raw page payload",
    )
    immutable: bool = Field(
        default=True,
        alias="_immutable",
        description="Write-once flag; repository refuses updates when True",
    )

    # --- timestamps ---
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of document creation",
    )

    class Config:
        populate_by_name = True
        use_enum_values = True

    def to_mongo_doc(self) -> Dict[str, Any]:
        """Serialize to a MongoDB-ready dictionary."""
        doc = self.model_dump(by_alias=True, mode="json")
        doc["created_at"] = self.created_at
        return doc


# ---------------------------------------------------------------------------
# Service result envelope
# ---------------------------------------------------------------------------

class IngestResult(BaseModel):
    """Return value from the ingest service layer.

    Carries the persisted ``submission_id`` and status metadata for the
    caller (typically an API route handler in SWM-010/SWM-012).
    """

    submission_id: str
    content_hash: str
    page_count: int
    segmentation_status: SubmissionStatus = SubmissionStatus.PENDING
    already_existed: bool = Field(
        default=False,
        description="True when a duplicate ingest was detected (ING-03 idempotency)",
    )
