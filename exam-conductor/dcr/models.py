"""
DCR Pydantic models.

Defines the data shapes for DCR submissions, recognition outputs, match results,
and stored result documents. Schema matches DUAL_MODE_ARCHITECTURE.md §4.5–§4.6.

Collections (read by DCR):
  - evalpen_submissions    (read-only from DCR; owned by ingest substrate)
  - evalpen_answer_pages   (read-only from DCR; owned by ingest substrate)

Collections (owned by DCR):
  - exampen_dcr_results    (writable owner: DCR engine)

Test IDs: U-DCR-01, U-DCR-02
Failure modes: DCR-01 (low confidence), DCR-02 (numeric mismatch), DCR-03 (scope creep)
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MatchType(str, enum.Enum):
    """
    Match classification between recognized text and the answer key.
    Spec: DUAL_MODE_ARCHITECTURE.md §4.5
    Test: U-DCR-02
    """

    EXACT_MATCH = "exact_match"
    PARTIAL_MATCH = "partial_match"
    NUMERIC_MATCH = "numeric_match"
    NO_MATCH = "no_match"


class AuditAction(str, enum.Enum):
    """Actions that produce audit trail entries."""

    ENGINE_SCORED = "engine_scored"
    GATE_FALLBACK = "gate_fallback"
    MANUAL_OVERRIDE = "manual_override"
    RESCORE = "rescore"


# ---------------------------------------------------------------------------
# Ingest substrate models (read-only from DCR perspective — C5)
# ---------------------------------------------------------------------------

class DCRSubmissionPage(BaseModel):
    """
    Read-only projection of an ``evalpen_answer_pages`` document.

    Each document holds raw stroke or image data for one page of a conducted
    exam.  The DCR engine reads these; it never mutates them (C5).
    """

    page_id: str = Field(..., description="Unique page document identifier")
    submission_id: str = Field(..., description="Parent submission ID")
    exam_id: str = Field(..., description="Conducted exam identifier")
    student_id: str = Field(..., description="Student identity")
    admin_id: str = Field(..., description="Tenant admin identity")
    page_number: int = Field(..., ge=1, description="1-based page number")
    source: Optional[str] = Field(
        default=None,
        description="Artifact origin (ble_pen or camera)",
    )
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
    content_hash: str = Field(
        ...,
        description="SHA-256 hex digest over the raw page payload",
    )
    immutable: bool = Field(True, alias="_immutable")

    class Config:
        populate_by_name = True


class PageRef(BaseModel):
    """Reference to a single answer page within a submission."""

    page_num: int = Field(..., ge=1, description="1-based page number")
    raw_asset_ref: Optional[str] = Field(
        default=None,
        description="Opaque reference to raw asset in object storage (if applicable)",
    )


class DCRSubmission(BaseModel):
    """
    Read-only projection of an ``evalpen_submissions`` document.

    The DCR engine reads these records; it never mutates them (C5).
    Pages are stored separately in ``evalpen_answer_pages`` and fetched
    via ``DCRRepository.get_submission_pages()``.
    """

    submission_id: str
    exam_id: str
    student_id: str
    admin_id: str
    source: Optional[str] = Field(
        default=None,
        description="Artifact origin (ble_pen or camera)",
    )
    pen_mac: Optional[str] = None
    page_count: int = Field(default=0, ge=0)
    page_refs: List[PageRef] = Field(default_factory=list)
    content_hash: Optional[str] = None
    immutable: bool = Field(True, alias="_immutable")
    segmentation_status: Optional[str] = Field(
        default=None,
        description="Downstream processing status (pending, complete, failed)",
    )
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        populate_by_name = True


# ---------------------------------------------------------------------------
# Answer key (read from exam metadata — not owned by DCR)
# ---------------------------------------------------------------------------

class AnswerKey(BaseModel):
    """
    Expected answer for a single question, consumed by the matcher.

    The DCR engine reads answer keys from exam metadata; it does not own them.
    """

    question_id: str
    expected_text: str = Field(
        ...,
        description="Canonical expected answer text (e.g. 'Paris', '42.0', 'A').",
    )
    max_score: float = Field(
        ..., ge=0,
        description="Maximum marks for this question.",
    )
    match_mode: Optional[str] = Field(
        default=None,
        description=(
            "Optional hint to the matcher. Values: 'exact', 'numeric', 'case_insensitive'. "
            "When absent the matcher applies all heuristics."
        ),
    )
    numeric_tolerance: Optional[float] = Field(
        default=None,
        ge=0,
        description="Absolute tolerance for numeric matching (DCR-02 mitigation).",
    )
    page_number: Optional[int] = Field(
        default=None,
        description="Expected page number where the answer appears.",
    )


# ---------------------------------------------------------------------------
# Recognition output (produced by recognizer, consumed by matcher)
# ---------------------------------------------------------------------------

class RecognitionOutput(BaseModel):
    """
    Output of the HWR recognizer for a single question region.

    Test: U-DCR-01
    Failure mode: DCR-01 (low confidence → route to fallback)
    """

    question_id: str
    recognized_text: str
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Recognition confidence in [0, 1].",
    )
    page_number: Optional[int] = None
    raw_logits: Optional[List[float]] = Field(
        default=None,
        description="Optional raw model logits retained for diagnostics.",
    )


# ---------------------------------------------------------------------------
# Match output (produced by matcher, consumed by service for result storage)
# ---------------------------------------------------------------------------

class MatchOutput(BaseModel):
    """
    Output of template matching for a single question.

    Test: U-DCR-02
    Failure mode: DCR-02 (numeric tolerance), DCR-03 (scope creep guard)
    """

    question_id: str
    match_type: MatchType
    score: float = Field(..., ge=0)
    max_score: float = Field(..., ge=0)
    recognized_text: str
    expected_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Audit trail entry
# ---------------------------------------------------------------------------

class DCRAuditEntry(BaseModel):
    """
    Append-only audit trail entry for a DCR result.
    Ref: DUAL_MODE_ARCHITECTURE.md §4.5 — audit_trail[]
    """

    action: AuditAction
    actor: Optional[str] = Field(
        default=None,
        description="Identifier of the actor. 'engine' for automated, user_id for manual.",
    )
    previous_score: Optional[float] = None
    new_score: Optional[float] = None
    previous_match_type: Optional[MatchType] = None
    new_match_type: Optional[MatchType] = None
    gate_call_ref: Optional[str] = Field(
        default=None,
        description="Reference to the LLM gate call log entry, if a fallback was used.",
    )
    note: Optional[str] = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# DCR result document (writable owner: DCR engine)
# ---------------------------------------------------------------------------

class DCRResult(BaseModel):
    """
    Stored result for one question in ``exampen_dcr_results``.

    Schema: DUAL_MODE_ARCHITECTURE.md §4.5–§4.6
    Unique index: { exam_id, student_id, question_id }
    """

    exam_id: str
    student_id: str
    question_id: str
    recognized_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    match_type: MatchType
    score: float = Field(..., ge=0)
    max_score: float = Field(..., ge=0)
    audit_trail: List[DCRAuditEntry] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Service-level request / response envelopes
# ---------------------------------------------------------------------------

class DCREvaluateRequest(BaseModel):
    """
    Request envelope for DCR evaluation of a conducted exam.

    Minimum request shape per DUAL_MODE_ARCHITECTURE.md §4.2.
    """

    submission_id: str
    exam_id: str
    student_id: str
    question_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional subset of questions to evaluate. None = all questions.",
    )


class DCRQuestionResult(BaseModel):
    """Per-question result within the evaluation response."""

    question_id: str
    recognized_text: str
    confidence: float
    match_type: MatchType
    score: float
    max_score: float
    used_gate_fallback: bool = False


class DCREvaluateResponse(BaseModel):
    """
    Response envelope for a DCR evaluation batch.

    Returned by DCRService.evaluate().
    """

    submission_id: str
    exam_id: str
    student_id: str
    results: List[DCRQuestionResult] = Field(default_factory=list)
    total_score: float = 0.0
    total_max_score: float = 0.0
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-question errors that did not halt the entire batch.",
    )
