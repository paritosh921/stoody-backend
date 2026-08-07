"""Audited ExamPen recheck requests for published student results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from api.v1.auth_async import get_current_user, get_database
from api.v1.evalpen_review_async import (
    ScoreOverrideRequest,
    _check_student_in_scope,
    _get_tenant_db,
    _get_tutor_scoped_student_ids,
    override_evaluation_score,
    require_admin_or_tutor,
)
from api.v1.evalpen_student_bff_async import (
    _get_student_identity_ids,
    require_student,
)
from core.database import DatabaseManager

router = APIRouter()

OPEN_STATUSES = {"open", "under_review"}
RESOLVED_STATUSES = {
    "resolved_no_change",
    "resolved_score_updated",
    "resolved_explained",
}
_indexed_collections: set[str] = set()


class RecheckCreateRequest(BaseModel):
    exam_id: str = Field(..., min_length=1, max_length=160)
    question_id: str = Field(..., min_length=1, max_length=160)
    reason: str = Field(..., min_length=5, max_length=2000)


class RecheckResolveRequest(BaseModel):
    status: str
    teacher_response: str = Field(..., min_length=3, max_length=4000)
    updated_score: Optional[float] = Field(default=None, ge=0)
    updated_max_score: Optional[float] = Field(default=None, gt=0)


class RecheckItem(BaseModel):
    request_id: str
    exam_id: str
    student_id: str
    question_id: str
    question_number: Optional[int] = None
    submission_id: str
    status: str
    reason: str
    teacher_response: Optional[str] = None
    original_score: float
    original_max_score: float
    updated_score: Optional[float] = None
    updated_max_score: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    review_started_at: Optional[datetime] = None


class RecheckListResponse(BaseModel):
    items: List[RecheckItem] = Field(default_factory=list)


def _public_item(document: Dict[str, Any]) -> RecheckItem:
    return RecheckItem(
        request_id=str(document.get("request_id") or ""),
        exam_id=str(document.get("exam_id") or ""),
        student_id=str(document.get("student_id") or ""),
        question_id=str(document.get("question_id") or ""),
        question_number=document.get("question_number"),
        submission_id=str(document.get("submission_id") or ""),
        status=str(document.get("status") or "open"),
        reason=str(document.get("reason") or ""),
        teacher_response=document.get("teacher_response"),
        original_score=float(document.get("original_score") or 0.0),
        original_max_score=float(document.get("original_max_score") or 0.0),
        updated_score=document.get("updated_score"),
        updated_max_score=document.get("updated_max_score"),
        created_at=document.get("created_at") or datetime.now(timezone.utc),
        updated_at=document.get("updated_at"),
        resolved_at=document.get("resolved_at"),
        assigned_to=document.get("assigned_to"),
        review_started_at=document.get("review_started_at"),
    )


async def _ensure_indexes(collection: Any) -> None:
    collection_key = str(getattr(collection, "full_name", "")) or repr(collection)
    if collection_key in _indexed_collections:
        return
    await collection.create_index("request_id", unique=True)
    await collection.create_index([("exam_id", 1), ("created_at", -1)])
    await collection.create_index([("student_id", 1), ("created_at", -1)])
    # active_key is removed on resolution. It prevents concurrent duplicate
    # requests while allowing a student to request a later recheck.
    await collection.create_index("active_key", unique=True, sparse=True)
    _indexed_collections.add(collection_key)


def _published_score_row(
    submission: Dict[str, Any],
    question_id: str,
) -> Optional[Dict[str, Any]]:
    snapshot = submission.get("publication_snapshot")
    if not isinstance(snapshot, dict):
        return None
    for row in snapshot.get("score_rows") or []:
        if isinstance(row, dict) and str(row.get("question_id") or "") == question_id:
            return row
    return None


async def _evaluation_id_for_question(
    tenant_db: Any,
    submission_id: str,
    question_id: str,
    score_row: Dict[str, Any],
) -> Optional[str]:
    snapshot_id = str(score_row.get("evaluation_id") or "").strip()
    if snapshot_id:
        return snapshot_id
    responses = await tenant_db["evalpen_detected_responses"].find(
        {
            "submission_id": submission_id,
            "question_id": question_id,
            "eval_status": {"$ne": "superseded"},
        },
        {"_id": 0, "response_id": 1},
    ).to_list(length=100)
    response_ids = [
        str(item.get("response_id") or "")
        for item in responses
        if str(item.get("response_id") or "")
    ]
    if not response_ids:
        return None
    evaluation = await tenant_db["evalpen_evaluations"].find_one(
        {"response_id": {"$in": response_ids}},
        {"_id": 0, "evaluation_id": 1},
        sort=[("evaluated_at", -1), ("created_at", -1)],
    )
    return str((evaluation or {}).get("evaluation_id") or "").strip() or None


@router.post(
    "/requests",
    response_model=RecheckItem,
    status_code=status.HTTP_201_CREATED,
)
async def create_recheck_request(
    body: RecheckCreateRequest,
    current_user: Dict[str, Any] = Depends(require_student),
    db: DatabaseManager = Depends(get_database),
) -> RecheckItem:
    tenant_db = await _get_tenant_db(db, current_user)
    student_ids = await _get_student_identity_ids(tenant_db, current_user)
    submission = await tenant_db["evalpen_submissions"].find_one(
        {
            "exam_id": body.exam_id,
            "student_id": {"$in": student_ids},
            "publication_status": "published",
        }
    )
    if submission is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No published result was found for this exam",
        )

    question_id = body.question_id.strip()
    score_row = _published_score_row(submission, question_id)
    if score_row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This question is not part of the published result",
        )

    submission_id = str(submission.get("submission_id") or "")
    student_id = str(submission.get("student_id") or student_ids[0])
    evaluation_id = await _evaluation_id_for_question(
        tenant_db,
        submission_id,
        question_id,
        score_row,
    )
    now = datetime.now(timezone.utc)
    document = {
        "request_id": f"recheck-{uuid4().hex}",
        "exam_id": body.exam_id.strip(),
        "student_id": student_id,
        "submission_id": submission_id,
        "question_id": question_id,
        "question_number": score_row.get("question_number"),
        "evaluation_id": evaluation_id,
        "status": "open",
        "reason": body.reason.strip(),
        "teacher_response": None,
        "original_score": float(score_row.get("score") or 0.0),
        "original_max_score": float(score_row.get("max_score") or 0.0),
        "updated_score": None,
        "updated_max_score": None,
        "created_at": now,
        "updated_at": now,
        "resolved_at": None,
        "created_by": str(current_user.get("user_id") or student_id),
        "active_key": f"{submission_id}:{question_id}",
    }
    collection = tenant_db["evalpen_recheck_requests"]
    await _ensure_indexes(collection)
    try:
        await collection.insert_one(document)
    except DuplicateKeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A recheck for this question is already open",
        ) from exc
    return _public_item(document)


@router.get("/requests", response_model=RecheckListResponse)
async def list_recheck_requests(
    exam_id: Optional[str] = Query(default=None),
    student_id: Optional[str] = Query(default=None),
    request_status: Optional[str] = Query(default=None, alias="status"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
) -> RecheckListResponse:
    tenant_db = await _get_tenant_db(db, current_user)
    query: Dict[str, Any] = {}
    if exam_id:
        query["exam_id"] = exam_id
    if request_status:
        if request_status not in OPEN_STATUSES | RESOLVED_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Unsupported recheck status",
            )
        query["status"] = request_status

    if current_user.get("user_type") in {"student", "b2c_user"}:
        student_ids = await _get_student_identity_ids(tenant_db, current_user)
        query["student_id"] = {"$in": student_ids}
    elif current_user.get("user_type") in {"admin", "b2c_admin", "tutor"}:
        scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)
        if student_id:
            _check_student_in_scope(student_id, scoped_ids)
            query["student_id"] = student_id
        elif scoped_ids is not None:
            query["student_id"] = {"$in": list(scoped_ids)}
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Recheck access is not available for this account",
        )

    documents = await tenant_db["evalpen_recheck_requests"].find(
        query,
        {"_id": 0, "active_key": 0, "resolution_lock": 0, "evaluation_id": 0},
    ).sort("created_at", -1).to_list(length=5000)
    return RecheckListResponse(items=[_public_item(item) for item in documents])


@router.post("/requests/{request_id}/claim", response_model=RecheckItem)
async def claim_recheck_request(
    request_id: str,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> RecheckItem:
    """Claim a request so two teachers cannot independently resolve it."""
    tenant_db = await _get_tenant_db(db, current_user)
    collection = tenant_db["evalpen_recheck_requests"]
    await _ensure_indexes(collection)
    request_document = await collection.find_one({"request_id": request_id})
    if request_document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recheck request not found",
        )

    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)
    _check_student_in_scope(str(request_document.get("student_id") or ""), scoped_ids)
    request_status = str(request_document.get("status") or "open")
    if request_status in RESOLVED_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This recheck request is already resolved",
        )

    actor_id = str(current_user.get("user_id") or "unknown")
    assigned_to = str(request_document.get("assigned_to") or "")
    if request_status == "under_review" and assigned_to:
        if assigned_to != actor_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Another teacher is reviewing this request",
            )
        return _public_item(request_document)

    now = datetime.now(timezone.utc)
    claimed = await collection.find_one_and_update(
        {
            "request_id": request_id,
            "status": {"$in": list(OPEN_STATUSES)},
            "$or": [
                {"assigned_to": {"$exists": False}},
                {"assigned_to": None},
                {"assigned_to": actor_id},
            ],
        },
        {
            "$set": {
                "status": "under_review",
                "assigned_to": actor_id,
                "review_started_at": now,
                "updated_at": now,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if claimed is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another teacher claimed this request",
        )
    return _public_item(claimed)


@router.post("/requests/{request_id}/resolve", response_model=RecheckItem)
async def resolve_recheck_request(
    request_id: str,
    body: RecheckResolveRequest,
    current_user: Dict[str, Any] = Depends(require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
) -> RecheckItem:
    if body.status not in RESOLVED_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A resolved recheck status is required",
        )
    if body.status == "resolved_score_updated" and body.updated_score is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="An updated score is required",
        )

    tenant_db = await _get_tenant_db(db, current_user)
    collection = tenant_db["evalpen_recheck_requests"]
    request_document = await collection.find_one({"request_id": request_id})
    if request_document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recheck request not found",
        )
    scoped_ids = await _get_tutor_scoped_student_ids(current_user, db)
    _check_student_in_scope(str(request_document.get("student_id") or ""), scoped_ids)
    if str(request_document.get("status") or "") in RESOLVED_STATUSES:
        return _public_item(request_document)

    actor_id = str(current_user.get("user_id") or "unknown")
    assigned_to = str(request_document.get("assigned_to") or "")
    if (
        str(request_document.get("status") or "") == "under_review"
        and assigned_to
        and assigned_to != actor_id
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another teacher is reviewing this request",
        )

    original_max = float(request_document.get("original_max_score") or 0.0)
    if body.updated_max_score is not None and abs(body.updated_max_score - original_max) > 1e-9:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The published question maximum is immutable",
        )
    if body.updated_score is not None and body.updated_score > original_max:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Updated score cannot exceed the published question maximum",
        )

    lock_token = uuid4().hex
    previous_status = str(request_document.get("status") or "open")
    claimed = await collection.find_one_and_update(
        {
            "request_id": request_id,
            "status": {"$in": list(OPEN_STATUSES)},
            "resolution_lock": {"$exists": False},
            "$or": [
                {"status": "open"},
                {"assigned_to": actor_id},
                {"assigned_to": {"$exists": False}},
                {"assigned_to": None},
            ],
        },
        {
            "$set": {
                "status": "under_review",
                "assigned_to": actor_id,
                "review_started_at": (
                    request_document.get("review_started_at")
                    or datetime.now(timezone.utc)
                ),
                "resolution_lock": lock_token,
                "updated_at": datetime.now(timezone.utc),
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if claimed is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This recheck is already being resolved",
        )

    try:
        if body.status == "resolved_score_updated":
            evaluation_id = str(claimed.get("evaluation_id") or "")
            if not evaluation_id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="This question does not support an audited score amendment",
                )
            await override_evaluation_score(
                evaluation_id=evaluation_id,
                body=ScoreOverrideRequest(
                    new_score=float(body.updated_score),
                    reason=f"Recheck {request_id}: {body.teacher_response.strip()}",
                    amend_published=True,
                    recheck_request_id=request_id,
                ),
                current_user=current_user,
                db=db,
            )

        now = datetime.now(timezone.utc)
        resolved = await collection.find_one_and_update(
            {"request_id": request_id, "resolution_lock": lock_token},
            {
                "$set": {
                    "status": body.status,
                    "teacher_response": body.teacher_response.strip(),
                    "updated_score": (
                        float(body.updated_score)
                        if body.status == "resolved_score_updated"
                        else None
                    ),
                    "updated_max_score": (
                        original_max
                        if body.status == "resolved_score_updated"
                        else None
                    ),
                    "updated_at": now,
                    "resolved_at": now,
                    "resolved_by": str(current_user.get("user_id") or "unknown"),
                },
                "$unset": {"active_key": "", "resolution_lock": ""},
            },
            return_document=ReturnDocument.AFTER,
        )
        if resolved is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The recheck changed while it was being resolved",
            )
        return _public_item(resolved)
    except Exception:
        await collection.update_one(
            {"request_id": request_id, "resolution_lock": lock_token},
            {
                "$set": {
                    "status": previous_status,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$unset": {"resolution_lock": ""},
            },
        )
        raise
