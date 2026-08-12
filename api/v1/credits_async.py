"""Student credits API.

Provides policy management and credit visibility endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt, StrictStr
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager
from services import student_credits
from utils.tutor_scoping import get_tutor_scoped_students

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


def _as_dict(obj: BaseModel) -> Dict[str, Any]:
    """Compatibility helper for pydantic v1/v2 output conversion."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_unset=True)
    return obj.dict(exclude_unset=True)


class CreditTierItem(BaseModel):
    id: StrictStr
    name: StrictStr
    min_credits: StrictInt = Field(ge=0)
    accent: StrictStr
    icon: StrictStr

    class Config:
        extra = "forbid"


class CreditPolicyUpdate(BaseModel):
    enabled: Optional[StrictBool] = None
    semantic_judge_enabled: Optional[StrictBool] = None
    stroke_acceptance_threshold: Optional[StrictFloat] = Field(default=None, ge=0.0, le=1.0)
    image_acceptance_threshold: Optional[StrictFloat] = Field(default=None, ge=0.0, le=1.0)
    max_randomness_score: Optional[StrictFloat] = Field(default=None, ge=0.0, le=1.0)
    min_strokes: Optional[StrictInt] = Field(default=None, ge=0)
    min_points: Optional[StrictInt] = Field(default=None, ge=0)
    min_path_length_mm: Optional[StrictFloat] = Field(default=None, ge=0.0)
    min_page_coverage: Optional[StrictFloat] = Field(default=None, ge=0.0, le=1.0)
    min_image_width: Optional[StrictInt] = Field(default=None, ge=1)
    min_image_height: Optional[StrictInt] = Field(default=None, ge=1)
    min_written_coverage: Optional[StrictFloat] = Field(default=None, ge=0.0, le=1.0)
    max_written_coverage: Optional[StrictFloat] = Field(default=None, ge=0.0, le=1.0)
    min_blur_variance: Optional[StrictFloat] = Field(default=None, ge=0.0)
    min_ink_density: Optional[StrictFloat] = Field(default=None, ge=0.0, le=1.0)
    max_ink_density: Optional[StrictFloat] = Field(default=None, ge=0.0, le=1.0)
    max_skew_angle: Optional[StrictFloat] = Field(default=None, ge=0.0, le=180.0)
    max_perspective_distortion: Optional[StrictFloat] = Field(default=None, ge=0.0)
    max_glare_ratio: Optional[StrictFloat] = Field(default=None, ge=0.0, le=1.0)
    max_overexposure_ratio: Optional[StrictFloat] = Field(default=None, ge=0.0, le=1.0)
    max_edge_clipping_ratio: Optional[StrictFloat] = Field(default=None, ge=0.0, le=1.0)
    stroke_mm_per_credit_unit: Optional[StrictFloat] = Field(default=None, gt=0.0)
    stroke_credits_per_unit: Optional[StrictInt] = Field(default=None, gt=0)
    image_credits_per_page: Optional[StrictInt] = Field(default=None, ge=0)
    max_stroke_credits_per_page: Optional[StrictInt] = Field(default=None, ge=0)
    max_image_credits_per_submission: Optional[StrictInt] = Field(default=None, ge=0)
    daily_credit_cap: Optional[StrictInt] = Field(default=None, ge=0)
    max_attempts: Optional[StrictInt] = Field(default=None, gt=0)
    lease_seconds: Optional[StrictInt] = Field(default=None, gt=0)
    tiers: Optional[List[CreditTierItem]] = None

    class Config:
        extra = "forbid"


class CreditPolicyResponse(BaseModel):
    enabled: StrictBool
    semantic_judge_enabled: StrictBool
    stroke_acceptance_threshold: StrictFloat
    image_acceptance_threshold: StrictFloat
    max_randomness_score: StrictFloat
    min_strokes: StrictInt
    min_points: StrictInt
    min_path_length_mm: StrictFloat
    min_page_coverage: StrictFloat
    min_image_width: StrictInt
    min_image_height: StrictInt
    min_written_coverage: StrictFloat
    max_written_coverage: StrictFloat
    min_blur_variance: StrictFloat
    min_ink_density: StrictFloat
    max_ink_density: StrictFloat
    max_skew_angle: StrictFloat
    max_perspective_distortion: StrictFloat
    max_glare_ratio: StrictFloat
    max_overexposure_ratio: StrictFloat
    max_edge_clipping_ratio: StrictFloat
    stroke_mm_per_credit_unit: StrictFloat
    stroke_credits_per_unit: StrictInt
    image_credits_per_page: StrictInt
    max_stroke_credits_per_page: StrictInt
    max_image_credits_per_submission: StrictInt
    daily_credit_cap: StrictInt
    max_attempts: StrictInt
    lease_seconds: StrictInt
    tiers: List[CreditTierItem]
    version: StrictInt
    earning_started_at: Optional[Any] = None
    updated_at: Optional[Any] = None

    class Config:
        extra = "forbid"


class CreditPolicyEnvelope(BaseModel):
    success: StrictBool = True
    data: CreditPolicyResponse

    class Config:
        extra = "forbid"


class CreditSummaryResponse(BaseModel):
    success: StrictBool = True
    data: Dict[str, Any]

    class Config:
        extra = "forbid"


class CreditLeaderboardItem(BaseModel):
    rank: StrictInt = Field(gt=0)
    student_record_id: Optional[StrictStr] = None
    student_id: Optional[StrictStr] = None
    display_name: StrictStr
    is_self: StrictBool
    total_credits: StrictInt
    tier: Dict[str, Any]

    class Config:
        extra = "forbid"


class CreditLeaderboardResponse(BaseModel):
    success: StrictBool = True
    data: List[CreditLeaderboardItem]

    class Config:
        extra = "forbid"


def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    if current_user.get("user_type") not in {"admin", "b2c_admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


def require_student_tutor_or_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    if current_user.get("user_type") not in {"student", "tutor", "admin", "b2c_admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student, tutor, or admin access required",
        )
    return current_user


def _coerce_student_record_id(raw: Any) -> Optional[str]:
    if not raw:
        return None
    candidate = str(raw).strip()
    return candidate or None


async def _tenant_db(db: DatabaseManager, current_user: Dict[str, Any]):
    db_name = str(current_user.get("db_name") or "").strip()
    if not db_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context is missing from token",
        )
    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant database is unavailable",
        )
    return tenant_db


async def _find_student_by_identifier(tenant_db: Any, identifier: str) -> Optional[Dict[str, Any]]:
    normalized = identifier.strip()
    if not normalized:
        return None

    ors: List[Dict[str, Any]] = []
    lowered = normalized.lower()
    if ObjectId.is_valid(normalized):
        ors.append({"_id": ObjectId(normalized)})
    ors.extend(
        [
            {"student_id": normalized},
            {"student_id": lowered},
            {"username": normalized},
            {"username_lower": lowered},
        ]
    )
    return await tenant_db["students"].find_one({"$or": ors})


async def _resolve_subject_student(
    current_user: Dict[str, Any],
    tenant_db: Any,
    db: DatabaseManager,
    *,
    student_record_id: Optional[str],
    student_id: Optional[str],
) -> Dict[str, Any]:
    user_type = current_user.get("user_type") or ""
    requested_identifier: Optional[str] = student_record_id or student_id

    if user_type == "student":
        student = await student_credits.resolve_student_record(tenant_db, current_user)
        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student profile not found",
            )
        if requested_identifier:
            requested = await _find_student_by_identifier(tenant_db, requested_identifier)
            if not requested or str(requested.get("_id")) != str(student.get("_id")):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Students may only access their own credit summary",
                )
        return student

    if user_type == "tutor":
        if not requested_identifier:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="student_record_id or student_id is required for tutor callers",
            )
        target = await _find_student_by_identifier(tenant_db, requested_identifier)
        if not target:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found",
            )
        tutor_id = str(current_user.get("tutor_id") or current_user.get("teacher_id") or "").strip()
        if not tutor_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tutor identity not available",
            )

        admin_oid = None
        for value in (
            current_user.get("admin_id"),
            current_user.get("user_id"),
            current_user.get("tenant_id"),
        ):
            if not value:
                continue
            try:
                admin_oid = ObjectId(str(value))
                break
            except Exception:
                continue

        visible_students = await get_tutor_scoped_students(
            tutor_id=tutor_id,
            admin_oid=admin_oid,
            db=db,
            projection={"_id": 1},
        )
        visible_ids = {str(student.get("_id")) for student in visible_students}
        if str(target.get("_id")) not in visible_ids:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tutor is not assigned to this student",
            )
        return target

    # admin / b2c_admin
    if not requested_identifier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="student_record_id or student_id is required for admin callers",
        )
    target = await _find_student_by_identifier(tenant_db, requested_identifier)
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Student not found",
        )
    return target


async def _policy_dict_to_response(policy: Dict[str, Any]) -> Dict[str, Any]:
    response_payload = {
        "enabled": bool(policy.get("enabled", True)),
        "semantic_judge_enabled": bool(policy.get("semantic_judge_enabled", True)),
        "stroke_acceptance_threshold": float(policy.get("stroke_acceptance_threshold", 0.0)),
        "image_acceptance_threshold": float(policy.get("image_acceptance_threshold", 0.0)),
        "max_randomness_score": float(policy.get("max_randomness_score", 0.0)),
        "min_strokes": int(policy.get("min_strokes", 0)),
        "min_points": int(policy.get("min_points", 0)),
        "min_path_length_mm": float(policy.get("min_path_length_mm", 0.0)),
        "min_page_coverage": float(policy.get("min_page_coverage", 0.0)),
        "min_image_width": int(policy.get("min_image_width", 0)),
        "min_image_height": int(policy.get("min_image_height", 0)),
        "min_written_coverage": float(policy.get("min_written_coverage", 0.0)),
        "max_written_coverage": float(policy.get("max_written_coverage", 0.0)),
        "min_blur_variance": float(policy.get("min_blur_variance", 0.0)),
        "min_ink_density": float(policy.get("min_ink_density", 0.0)),
        "max_ink_density": float(policy.get("max_ink_density", 0.0)),
        "max_skew_angle": float(policy.get("max_skew_angle", 0.0)),
        "max_perspective_distortion": float(policy.get("max_perspective_distortion", 0.0)),
        "max_glare_ratio": float(policy.get("max_glare_ratio", 0.0)),
        "max_overexposure_ratio": float(policy.get("max_overexposure_ratio", 0.0)),
        "max_edge_clipping_ratio": float(policy.get("max_edge_clipping_ratio", 0.0)),
        "stroke_mm_per_credit_unit": float(policy.get("stroke_mm_per_credit_unit", 0.0)),
        "stroke_credits_per_unit": int(policy.get("stroke_credits_per_unit", 0)),
        "image_credits_per_page": int(policy.get("image_credits_per_page", 0)),
        "max_stroke_credits_per_page": int(policy.get("max_stroke_credits_per_page", 0)),
        "max_image_credits_per_submission": int(policy.get("max_image_credits_per_submission", 0)),
        "daily_credit_cap": int(policy.get("daily_credit_cap", 0)),
        "max_attempts": int(policy.get("max_attempts", 0)),
        "lease_seconds": int(policy.get("lease_seconds", 0)),
        "tiers": list(policy.get("tiers") or []),
        "version": int(policy.get("version") or 1),
        "earning_started_at": policy.get("earning_started_at"),
        "updated_at": policy.get("updated_at"),
    }
    return CreditPolicyResponse(**response_payload).dict()


@router.get("/policy", response_model=CreditPolicyEnvelope)
@limiter.limit("60/minute")
async def get_credit_policy(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
):
    tenant_db = await _tenant_db(db, current_user)
    policy = await student_credits.get_credit_policy(
        tenant_db,
        admin_id=str(current_user.get("user_id", "")),
    )
    return CreditPolicyEnvelope(success=True, data=CreditPolicyResponse(**await _policy_dict_to_response(policy)))


@router.put("/policy", response_model=CreditPolicyEnvelope)
@limiter.limit("20/minute")
async def update_credit_policy(
    request: Request,
    payload: CreditPolicyUpdate,
    current_user: Dict[str, Any] = Depends(require_admin),
    db: DatabaseManager = Depends(get_database),
):
    tenant_db = await _tenant_db(db, current_user)
    changes = _as_dict(payload)
    policy = await student_credits.update_credit_policy(
        tenant_db,
        changes,
        admin_id=str(current_user.get("user_id", "")),
    )
    return CreditPolicyEnvelope(
        success=True,
        data=CreditPolicyResponse(**await _policy_dict_to_response(policy)),
    )


@router.get("/summary", response_model=CreditSummaryResponse)
@limiter.limit("60/minute")
async def get_credit_summary(
    request: Request,
    student_record_id: Optional[StrictStr] = Query(default=None),
    student_id: Optional[StrictStr] = Query(default=None),
    current_user: Dict[str, Any] = Depends(require_student_tutor_or_admin),
    db: DatabaseManager = Depends(get_database),
):
    tenant_db = await _tenant_db(db, current_user)
    target = await _resolve_subject_student(
        current_user,
        tenant_db,
        db,
        student_record_id=_coerce_student_record_id(student_record_id),
        student_id=_coerce_student_record_id(student_id),
    )
    summary = await student_credits.get_student_credit_summary(tenant_db, target)
    return CreditSummaryResponse(
        success=True,
        data={
            "student_record_id": str(target.get("_id") or ""),
            "student_id": target.get("student_id"),
            "summary": summary,
        },
    )


@router.get("/leaderboard", response_model=CreditLeaderboardResponse)
@limiter.limit("60/minute")
async def get_credit_leaderboard(
    request: Request,
    limit: int = Query(default=50, ge=1, le=100),
    current_user: Dict[str, Any] = Depends(require_student_tutor_or_admin),
    db: DatabaseManager = Depends(get_database),
):
    tenant_db = await _tenant_db(db, current_user)
    user_type = current_user.get("user_type")

    allowed_ids: Optional[Iterable[str]] = None
    viewer_student_record_id = ""
    private_peer_labels = False

    if user_type == "student":
        student = await student_credits.resolve_student_record(tenant_db, current_user)
        if not student:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student profile not found",
            )
        viewer_student_record_id = str(student.get("_id") or "")
        private_peer_labels = True

    elif user_type == "tutor":
        tutor_id = str(
            current_user.get("tutor_id")
            or current_user.get("teacher_id")
            or ""
        ).strip()
        if not tutor_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tutor identity not available",
            )
        admin_oid = None
        for value in (
            current_user.get("admin_id"),
            current_user.get("user_id"),
            current_user.get("tenant_id"),
        ):
            if not value:
                continue
            try:
                admin_oid = ObjectId(str(value))
                break
            except Exception:
                continue

        students = await get_tutor_scoped_students(
            tutor_id=tutor_id,
            admin_oid=admin_oid,
            db=db,
            projection={"_id": 1},
        )
        allowed_ids = {str(student.get("_id")) for student in students if student.get("_id")}

    rows = await student_credits.get_credit_leaderboard(
        tenant_db,
        allowed_student_record_ids=allowed_ids,
        viewer_student_record_id=viewer_student_record_id,
        private_peer_labels=private_peer_labels,
        limit=limit,
    )
    return CreditLeaderboardResponse(
        success=True,
        data=[CreditLeaderboardItem(**row) for row in rows],
    )
