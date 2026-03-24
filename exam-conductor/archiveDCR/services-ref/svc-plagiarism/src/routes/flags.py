"""HTTP routes for plagiarism flag CRUD and teacher verdicts.

Matches the OpenAPI spec at new-docs/api/plagiarism.openapi.yaml.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..domain.verdict_logic import validate_verdict
from ..storage.flag_repo import FlagRepo, FlagRow

router = APIRouter(prefix="/api/v1/plagiarism", tags=["plagiarism"])


# ---- request / response models -------------------------------------------- #


class VerdictRequest(BaseModel):
    teacher_id: str
    verdict: str
    reason: str = Field(..., min_length=5)


class MatchingSegment(BaseModel):
    student_a_text: str
    student_b_text: str


class Evidence(BaseModel):
    matching_segments: list[MatchingSegment]
    temporal_correlation_score: float | None = None
    seating_proximity_score: float | None = None


class FlagSummaryResponse(BaseModel):
    flag_id: str
    exam_id: str
    student_a_id: str
    student_b_id: str
    question_id: str
    composite_score: float
    severity: str
    teacher_verdict: str


class FlagDetailResponse(FlagSummaryResponse):
    evidence: Evidence
    verdict_reason: str | None = None
    verdict_by: str | None = None
    verdict_at: str | None = None


class FlagListResponse(BaseModel):
    items: list[FlagSummaryResponse]


# ---- helpers -------------------------------------------------------------- #


def _to_summary(row: FlagRow) -> FlagSummaryResponse:
    return FlagSummaryResponse(
        flag_id=row.flag_id,
        exam_id=row.exam_id,
        student_a_id=row.student_a_id,
        student_b_id=row.student_b_id,
        question_id=row.question_id,
        composite_score=row.composite_score,
        severity=row.severity,
        teacher_verdict=row.teacher_verdict,
    )


def _to_detail(row: FlagRow) -> FlagDetailResponse:
    evidence = Evidence(
        matching_segments=[
            MatchingSegment(
                student_a_text=row.student_a_text,
                student_b_text=row.student_b_text,
            ),
        ],
        temporal_correlation_score=row.temporal_corr,
        seating_proximity_score=row.proximity_score,
    )
    return FlagDetailResponse(
        flag_id=row.flag_id,
        exam_id=row.exam_id,
        student_a_id=row.student_a_id,
        student_b_id=row.student_b_id,
        question_id=row.question_id,
        composite_score=row.composite_score,
        severity=row.severity,
        teacher_verdict=row.teacher_verdict,
        evidence=evidence,
        verdict_reason=row.verdict_reason,
        verdict_by=row.verdict_by,
        verdict_at=(
            row.verdict_at.isoformat() if row.verdict_at else None
        ),
    )


def _get_repo(request: Request) -> FlagRepo:
    return request.app.state.flag_repo  # type: ignore[no-any-return]


# ---- endpoints ------------------------------------------------------------ #


@router.get("/exams/{exam_id}/flags", response_model=FlagListResponse)
async def list_flags(exam_id: str, request: Request) -> FlagListResponse:
    """List all plagiarism flags for an exam."""
    repo = _get_repo(request)
    rows = await repo.list_by_exam(exam_id)
    return FlagListResponse(items=[_to_summary(r) for r in rows])


@router.get("/flags/{flag_id}", response_model=FlagDetailResponse)
async def get_flag(flag_id: str, request: Request) -> FlagDetailResponse:
    """Get detailed plagiarism evidence for one flag."""
    repo = _get_repo(request)
    row = await repo.get_by_id(flag_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Flag not found")
    return _to_detail(row)


@router.patch("/flags/{flag_id}/verdict", response_model=FlagDetailResponse)
async def record_verdict(
    flag_id: str,
    body: VerdictRequest,
    request: Request,
) -> FlagDetailResponse:
    """Record a teacher verdict on a plagiarism flag.

    Mandatory reason for all verdicts. NEVER auto-penalizes -- teacher
    review is always required.
    """
    validation = validate_verdict(body.verdict, body.reason)
    if not validation.valid:
        raise HTTPException(
            status_code=422,
            detail={"errors": validation.errors},
        )

    repo = _get_repo(request)

    # Verify flag exists
    existing = await repo.get_by_id(flag_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Flag not found")

    updated = await repo.update_verdict(
        flag_id=flag_id,
        teacher_id=body.teacher_id,
        verdict=body.verdict,
        reason=body.reason,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Flag not found")

    return _to_detail(updated)
