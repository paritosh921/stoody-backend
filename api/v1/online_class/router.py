import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_current_user, get_database
from api.v1.meeting_async import resolve_business_student_id
from api.v1.online_class.models import (
    CreateLockRequest,
    LockResponse,
    CreateSubmissionRequest,
    SubmissionResponse,
    SubmissionResultItem,
)
from api.v1.online_class.locks import (
    create_lock,
    get_current_lock,
    get_lock_by_id,
    end_lock,
)
from api.v1.online_class.submissions import (
    create_or_update_submission,
    get_submissions_for_lock,
)
from core.database import DatabaseManager
from services.online_class import jitsi_provider_service

logger = logging.getLogger(__name__)

router = APIRouter()
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)

CANVAS_SHARE_REQUESTS_COLLECTION = "online_class_canvas_share_requests"
CANVAS_SHARE_REQUEST_TTL_SECONDS = 45


class CanvasProviderDetails(BaseModel):
    provider: Optional[str] = None
    domain: Optional[str] = None
    room_name: Optional[str] = None
    url: Optional[str] = None
    token_required: bool = False
    token: Optional[str] = None
    configured: bool = False


class CanvasShareSessionResponse(BaseModel):
    teacher_room: CanvasProviderDetails


class StudentCanvasRequestBody(BaseModel):
    student_ids: Optional[List[str]] = None


class StudentCanvasRoom(BaseModel):
    student_id: str
    room: CanvasProviderDetails


class StudentCanvasRequestResponse(BaseModel):
    active: bool
    requested_student_ids: List[str]
    monitor_rooms: List[StudentCanvasRoom]


class StudentCanvasPublishResponse(BaseModel):
    requested: bool
    student_id: str
    publish_room: Optional[CanvasProviderDetails] = None


def _require_tutor(current_user: Dict[str, Any]):
    if current_user.get("user_type") != "tutor":
        raise HTTPException(status_code=403, detail="Tutor access required")
    return current_user


def _require_student(current_user: Dict[str, Any]):
    if current_user.get("user_type") != "student":
        raise HTTPException(status_code=403, detail="Student access required")
    return current_user


def _current_user_identity(current_user: Dict[str, Any]) -> tuple[str, str, str]:
    user_id = (
        current_user.get("user_id")
        or current_user.get("tutor_id")
        or current_user.get("student_id")
        or ""
    )
    user_name = current_user.get("name") or current_user.get("username") or "Participant"
    user_email = current_user.get("email") or ""
    return str(user_id), str(user_name), str(user_email)


def _canvas_provider_details(
    room_name: str,
    current_user: Dict[str, Any],
    moderator: bool,
) -> CanvasProviderDetails:
    if not jitsi_provider_service.jwt_available:
        raise HTTPException(
            status_code=503,
            detail="Online class canvas sharing requires Jitsi JWT enforcement",
        )
    user_id, user_name, user_email = _current_user_identity(current_user)
    details = jitsi_provider_service.get_provider_details_for_room(
        room_name=room_name,
        user_id=user_id,
        user_name=user_name,
        user_email=user_email,
        moderator=moderator,
    )
    provider = CanvasProviderDetails(**details)
    if not provider.configured:
        raise HTTPException(
            status_code=503,
            detail="Online class canvas video provider is not configured",
        )
    return provider


def _teacher_canvas_room_name(meeting_id: str) -> str:
    return jitsi_provider_service.generate_canvas_room_name(meeting_id, "teacher")


def _student_canvas_room_name(meeting_id: str, student_id: str) -> str:
    return jitsi_provider_service.generate_canvas_room_name(
        meeting_id,
        "student",
        student_id=student_id,
    )


def _validate_requested_student_ids(
    meeting: Dict[str, Any],
    requested_student_ids: Optional[List[str]],
) -> List[str]:
    invited = [str(sid) for sid in meeting.get("invited_student_ids", []) if sid]
    if not invited:
        raise HTTPException(status_code=400, detail="No invited students for this meeting")

    requested = [str(sid) for sid in (requested_student_ids or invited) if sid]
    if not requested:
        requested = invited

    invited_set = set(invited)
    invalid = sorted(set(requested) - invited_set)
    if invalid:
        raise HTTPException(
            status_code=403,
            detail=f"Student(s) not invited to this meeting: {', '.join(invalid)}",
        )

    return sorted(set(requested), key=requested.index)


def _canvas_request_is_expired(doc: Dict[str, Any], now: datetime) -> bool:
    updated_at = doc.get("updated_at")
    if not isinstance(updated_at, datetime):
        return True
    return now - updated_at > timedelta(seconds=CANVAS_SHARE_REQUEST_TTL_SECONDS)


async def _get_active_canvas_request(
    db: DatabaseManager,
    meeting_id: str,
) -> Optional[Dict[str, Any]]:
    active = await db.mongo_find_one(
        CANVAS_SHARE_REQUESTS_COLLECTION,
        {"meeting_id": meeting_id, "status": "active"},
    )
    if not active:
        return None

    now = datetime.utcnow()
    if not _canvas_request_is_expired(active, now):
        return active

    await db.mongo_update_one(
        CANVAS_SHARE_REQUESTS_COLLECTION,
        {"meeting_id": meeting_id, "status": "active"},
        {"$set": {"status": "expired", "ended_at": now}},
    )
    return None


async def _verify_meeting_active(db: DatabaseManager, meeting_id: str) -> Dict[str, Any]:
    meeting = await db.mongo_find_one("meetings", {"meeting_id": meeting_id})
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if meeting.get("status") != "active":
        raise HTTPException(status_code=400, detail="Meeting is not active")
    return meeting


async def _verify_tutor_owns_meeting(db: DatabaseManager, meeting_id: str, tutor_id: str) -> Dict[str, Any]:
    meeting = await db.mongo_find_one("meetings", {"meeting_id": meeting_id})
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if meeting.get("tutor_id") != tutor_id:
        raise HTTPException(status_code=403, detail="Not authorized for this meeting")
    return meeting


async def _verify_student_invited(db: DatabaseManager, meeting_id: str, student_id: str) -> Dict[str, Any]:
    meeting = await db.mongo_find_one("meetings", {"meeting_id": meeting_id})
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if student_id not in meeting.get("invited_student_ids", []):
        raise HTTPException(status_code=403, detail="Student not invited to this meeting")
    return meeting


@router.get("/meetings/{meeting_id}/canvas-share/session", response_model=CanvasShareSessionResponse)
@limiter.limit("30/minute")
async def api_get_canvas_share_session(
    request: Request,
    meeting_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    user_type = current_user.get("user_type")
    moderator = False
    if user_type == "tutor":
        await _verify_tutor_owns_meeting(db, meeting_id, current_user.get("tutor_id"))
        moderator = True
    elif user_type == "student":
        student_id = await resolve_business_student_id(current_user, db)
        if not student_id:
            raise HTTPException(status_code=403, detail="Could not resolve student identity")
        await _verify_student_invited(db, meeting_id, student_id)
    else:
        raise HTTPException(status_code=403, detail="Access denied")

    await _verify_meeting_active(db, meeting_id)
    room_name = _teacher_canvas_room_name(meeting_id)
    return CanvasShareSessionResponse(
        teacher_room=_canvas_provider_details(room_name, current_user, moderator=moderator)
    )


@router.post(
    "/meetings/{meeting_id}/canvas-share/student-requests",
    response_model=StudentCanvasRequestResponse,
)
@limiter.limit("10/minute")
async def api_request_student_canvas_streams(
    request: Request,
    meeting_id: str,
    body: StudentCanvasRequestBody,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    _require_tutor(current_user)
    tutor_id = current_user.get("tutor_id")
    meeting = await _verify_tutor_owns_meeting(db, meeting_id, tutor_id)
    await _verify_meeting_active(db, meeting_id)
    requested_student_ids = _validate_requested_student_ids(meeting, body.student_ids)

    now = datetime.utcnow()
    doc = {
        "meeting_id": meeting_id,
        "tutor_id": tutor_id,
        "requested_student_ids": requested_student_ids,
        "status": "active",
        "updated_at": now,
    }
    existing = await _get_active_canvas_request(db, meeting_id)
    if existing:
        await db.mongo_update_one(
            CANVAS_SHARE_REQUESTS_COLLECTION,
            {"meeting_id": meeting_id, "status": "active"},
            {"$set": doc},
        )
    else:
        await db.mongo_insert_one(
            CANVAS_SHARE_REQUESTS_COLLECTION,
            {**doc, "created_at": now},
        )

    monitor_rooms = [
        StudentCanvasRoom(
            student_id=student_id,
            room=_canvas_provider_details(
                _student_canvas_room_name(meeting_id, student_id),
                current_user,
                moderator=True,
            ),
        )
        for student_id in requested_student_ids
    ]
    return StudentCanvasRequestResponse(
        active=True,
        requested_student_ids=requested_student_ids,
        monitor_rooms=monitor_rooms,
    )


@router.delete(
    "/meetings/{meeting_id}/canvas-share/student-requests",
    response_model=StudentCanvasRequestResponse,
)
@limiter.limit("10/minute")
async def api_stop_student_canvas_streams(
    request: Request,
    meeting_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    _require_tutor(current_user)
    tutor_id = current_user.get("tutor_id")
    await _verify_tutor_owns_meeting(db, meeting_id, tutor_id)
    await db.mongo_update_one(
        CANVAS_SHARE_REQUESTS_COLLECTION,
        {"meeting_id": meeting_id, "status": "active"},
        {"$set": {"status": "ended", "ended_at": datetime.utcnow()}},
    )
    return StudentCanvasRequestResponse(
        active=False,
        requested_student_ids=[],
        monitor_rooms=[],
    )


@router.get(
    "/meetings/{meeting_id}/canvas-share/student-publish",
    response_model=StudentCanvasPublishResponse,
)
@limiter.limit("30/minute")
async def api_get_student_canvas_publish_session(
    request: Request,
    meeting_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    _require_student(current_user)
    student_id = await resolve_business_student_id(current_user, db)
    if not student_id:
        raise HTTPException(status_code=403, detail="Could not resolve student identity")

    await _verify_student_invited(db, meeting_id, student_id)
    await _verify_meeting_active(db, meeting_id)

    active_request = await _get_active_canvas_request(db, meeting_id)
    requested = bool(
        active_request
        and student_id in active_request.get("requested_student_ids", [])
    )
    if not requested:
        return StudentCanvasPublishResponse(requested=False, student_id=student_id)

    room_name = _student_canvas_room_name(meeting_id, student_id)
    return StudentCanvasPublishResponse(
        requested=True,
        student_id=student_id,
        publish_room=_canvas_provider_details(room_name, current_user, moderator=False),
    )


@router.post("/meetings/{meeting_id}/locks", response_model=LockResponse)
@limiter.limit("10/minute")
async def api_create_lock(
    request: Request,
    meeting_id: str,
    body: CreateLockRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    _require_tutor(current_user)
    tutor_id = current_user.get("tutor_id")
    await _verify_tutor_owns_meeting(db, meeting_id, tutor_id)
    await _verify_meeting_active(db, meeting_id)

    try:
        lock = await create_lock(
            db=db,
            meeting_id=meeting_id,
            tutor_id=tutor_id,
            question_text=body.question_text,
            question_image_id=body.question_image_id,
            question_bbox=body.question_bbox,
            duration_seconds=body.duration_seconds,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return LockResponse(**lock)


@router.get("/meetings/{meeting_id}/locks/current")
@limiter.limit("30/minute")
async def api_get_current_lock(
    request: Request,
    meeting_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    user_type = current_user.get("user_type")
    if user_type == "tutor":
        await _verify_tutor_owns_meeting(db, meeting_id, current_user.get("tutor_id"))
    elif user_type == "student":
        student_id = await resolve_business_student_id(current_user, db)
        await _verify_student_invited(db, meeting_id, student_id)
    else:
        raise HTTPException(status_code=403, detail="Access denied")

    lock = await get_current_lock(db, meeting_id)
    if not lock:
        return {"lock": None}
    return {"lock": LockResponse(**lock)}


@router.post("/meetings/{meeting_id}/locks/{lock_id}/end", response_model=LockResponse)
@limiter.limit("10/minute")
async def api_end_lock(
    request: Request,
    meeting_id: str,
    lock_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    _require_tutor(current_user)
    tutor_id = current_user.get("tutor_id")
    await _verify_tutor_owns_meeting(db, meeting_id, tutor_id)

    try:
        lock = await end_lock(db, meeting_id, lock_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return LockResponse(**lock)


@router.get("/meetings/{meeting_id}/locks/{lock_id}/results")
@limiter.limit("30/minute")
async def api_get_lock_results(
    request: Request,
    meeting_id: str,
    lock_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    _require_tutor(current_user)
    tutor_id = current_user.get("tutor_id")
    await _verify_tutor_owns_meeting(db, meeting_id, tutor_id)

    lock = await get_lock_by_id(db, meeting_id, lock_id)
    if not lock:
        raise HTTPException(status_code=404, detail="Lock not found")

    raw_submissions = await get_submissions_for_lock(db, meeting_id, lock_id)
    results: List[SubmissionResultItem] = []
    for sub in raw_submissions:
        student_doc = await db.mongo_find_one("students", {"student_id": sub.get("student_id")})
        student_name = None
        if student_doc:
            student_name = student_doc.get("name") or student_doc.get("username")
        results.append(
            SubmissionResultItem(
                submission_id=sub["submission_id"],
                student_id=sub["student_id"],
                student_name=student_name,
                canvas_pages=sub.get("canvas_pages", []),
                answer_text=sub.get("answer_text"),
                time_spent=sub.get("time_spent"),
                analysis_status=sub.get("analysis_status", "pending"),
                score=sub.get("score"),
                is_correct=sub.get("is_correct"),
                student_answer=sub.get("student_answer"),
                work_shown=sub.get("work_shown"),
                what_went_wrong=sub.get("what_went_wrong"),
                correct_solution=sub.get("correct_solution"),
                analysis_error=sub.get("analysis_error"),
                analysis_completed_at=sub.get("analysis_completed_at"),
                analysis_failed_at=sub.get("analysis_failed_at"),
                created_at=sub.get("created_at"),
            )
        )
    return {"lock": LockResponse(**lock), "submissions": results}


@router.post("/meetings/{meeting_id}/locks/{lock_id}/submissions", response_model=SubmissionResponse)
@limiter.limit("10/minute")
async def api_create_submission(
    request: Request,
    meeting_id: str,
    lock_id: str,
    body: CreateSubmissionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    _require_student(current_user)
    student_id = await resolve_business_student_id(current_user, db)
    if not student_id:
        raise HTTPException(status_code=403, detail="Could not resolve student identity")
    await _verify_student_invited(db, meeting_id, student_id)
    await _verify_meeting_active(db, meeting_id)

    lock = await get_lock_by_id(db, meeting_id, lock_id)
    if not lock:
        raise HTTPException(status_code=404, detail="Lock not found")

    sub = await create_or_update_submission(
        db=db,
        meeting_id=meeting_id,
        lock_id=lock_id,
        student_id=student_id,
        canvas_pages=body.canvas_pages,
        question_page_refs=body.question_page_refs,
        answer_text=body.answer_text,
        time_spent=body.time_spent,
        client_submitted_at=body.client_submitted_at,
    )
    from services.online_class.analysis_service import run_submission_analysis
    task = asyncio.create_task(run_submission_analysis(db, current_user, lock, sub.copy()))
    task.add_done_callback(_log_analysis_task_error)
    return SubmissionResponse(**sub)


def _log_analysis_task_error(task: asyncio.Task) -> None:
    try:
        task.result()
    except Exception:
        logger.exception("Online-class submission analysis task crashed")
