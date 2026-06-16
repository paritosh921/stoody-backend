import asyncio
import base64
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_current_user, get_database
from api.v1.meeting_async import resolve_business_student_id
from api.v1.online_class.models import (
    CreateLockRequest,
    LockResponse,
    CreateSubmissionRequest,
    ReanalyzeSubmissionRequest,
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
from api.v1.strokes_async import (
    CanvasPageBatchRequest,
    CanvasPageUpsert,
    _build_merged_page_doc,
    _build_metadata_refresh,
    _page_doc,
)
from core.database import DatabaseManager
from core.user_identity import canonical_canvas_user_id, canvas_user_id_variants
from services.online_class import jitsi_provider_service

logger = logging.getLogger(__name__)

router = APIRouter()
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)

CANVAS_SHARE_REQUESTS_COLLECTION = "online_class_canvas_share_requests"
CANVAS_SHARE_REQUEST_TTL_SECONDS = 45
TEACHER_CANVAS_MODE_FIELD = "teacher_canvas_mode"
TEACHER_CANVAS_MODE_UPDATED_AT_FIELD = "teacher_canvas_mode_updated_at"
TEACHER_CANVAS_MODE_UPDATED_BY_FIELD = "teacher_canvas_mode_updated_by"


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


class MonitoringStudentItem(BaseModel):
    student_id: str
    student_name: Optional[str] = None
    username: Optional[str] = None
    pen_mac: Optional[str] = None
    pen_connected: bool = False
    pen_last_frame_ts: Optional[float] = None
    pen_battery: Optional[int] = None
    pen_page_no: Optional[int] = None
    pen_book_type: Optional[str] = None
    joined: bool = False


class MonitoringStudentsResponse(BaseModel):
    students: List[MonitoringStudentItem]


class MonitoringPageMeta(BaseModel):
    page_key: str
    copy_id: Optional[str] = None
    book_type: str
    page_number: int
    stroke_count: int = 0
    first_activity: Optional[float] = None
    last_activity: Optional[float] = None
    last_modified: Optional[str] = None
    client_last_modified: Optional[float] = None
    version: int = 1


class MonitoringPageListResponse(BaseModel):
    count: int
    pages: List[MonitoringPageMeta]
    server_time: str


class MonitoringPageResponse(BaseModel):
    page: Dict[str, Any]


class TeacherCanvasModeRequest(BaseModel):
    mode: Literal["live", "stream"]


class TeacherCanvasModeResponse(BaseModel):
    mode: Literal["live", "stream"]
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None


class TeacherLiveCanvasEventsResponse(BaseModel):
    success: bool
    upserted: int = 0
    modified: int = 0
    count: int = 0


class OnlineClassNoteClassItem(BaseModel):
    meeting_id: str
    copy_id: str
    topic: Optional[str] = None
    subject: Optional[str] = None
    standard: Optional[str] = None
    section: Optional[str] = None
    tutor_name: Optional[str] = None
    status: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    page_count: int = 0
    stroke_count: int = 0
    first_activity: Optional[float] = None
    latest_activity: Optional[float] = None


class OnlineClassNotesResponse(BaseModel):
    classes: List[OnlineClassNoteClassItem]


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


def _online_class_copy_id(meeting_id: str) -> str:
    return f"online-{meeting_id}"


def _validate_requested_student_ids(
    meeting: Dict[str, Any],
    requested_student_ids: Optional[List[str]],
) -> List[str]:
    invited = [str(sid) for sid in meeting.get("invited_student_ids", []) if sid]
    if not invited:
        raise HTTPException(status_code=400, detail="No invited students for this meeting")

    if requested_student_ids is None:
        requested = [str(sid) for sid in meeting.get("joined_student_ids", []) if sid]
    else:
        requested = [str(sid) for sid in requested_student_ids if sid]

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


def _dedupe_identity_values(values: List[Any]) -> List[Any]:
    seen: set[str] = set()
    deduped: List[Any] = []
    for value in values:
        if value is None or value == "":
            continue
        marker = f"{type(value).__name__}:{value}"
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(value)
    return deduped


def _datetime_to_epoch_ms(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp() * 1000
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric * 1000 if numeric < 100_000_000_000 else numeric
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.timestamp() * 1000
        except ValueError:
            try:
                numeric = float(value)
                return numeric * 1000 if numeric < 100_000_000_000 else numeric
            except ValueError:
                return None
    return None


def _page_activity_max_ms(page: Dict[str, Any]) -> Optional[float]:
    candidates: List[float] = []
    for key in ("last_activity", "first_activity", "client_last_modified", "last_modified"):
        ts = _datetime_to_epoch_ms(page.get(key))
        if ts is not None:
            candidates.append(ts)

    for stroke in page.get("strokes") or []:
        if not isinstance(stroke, dict):
            continue
        for key in ("endedAt", "startedAt", "timestamp"):
            ts = _datetime_to_epoch_ms(stroke.get(key))
            if ts is not None:
                candidates.append(ts)

    return max(candidates) if candidates else None


def _filter_canvas_pages_since_session_start(
    pages: List[Dict[str, Any]],
    started_at: Any,
) -> List[Dict[str, Any]]:
    started_ms = _datetime_to_epoch_ms(started_at)
    if started_ms is None:
        return pages
    return [
        page
        for page in pages
        if (_page_activity_max_ms(page) is not None and _page_activity_max_ms(page) >= started_ms)
    ]


def _encode_monitoring_page_key(copy_id: Optional[str], book_type: str, page_number: int) -> str:
    payload = {
        "copy_id": copy_id,
        "book_type": str(book_type).upper(),
        "page_number": int(page_number),
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_monitoring_page_key(page_key: str) -> Dict[str, Any]:
    try:
        padded = page_key + ("=" * (-len(page_key) % 4))
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
        book_type = str(payload["book_type"]).upper()
        page_number = int(payload["page_number"])
        copy_id = payload.get("copy_id")
        return {
            "copy_id": copy_id,
            "book_type": book_type,
            "page_number": page_number,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid monitoring page key") from exc


def _serialize_datetime(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return None


def _build_monitoring_page_meta(page: Dict[str, Any]) -> Dict[str, Any]:
    book_type = str(page.get("book_type") or "MS").upper()
    page_number = int(page.get("page_number") or 0)
    copy_id = page.get("copy_id")
    return {
        "page_key": _encode_monitoring_page_key(copy_id, book_type, page_number),
        "copy_id": copy_id,
        "book_type": book_type,
        "page_number": page_number,
        "stroke_count": int(page.get("stroke_count") or len(page.get("strokes") or []) or 0),
        "first_activity": page.get("first_activity"),
        "last_activity": page.get("last_activity"),
        "last_modified": _serialize_datetime(page.get("last_modified")),
        "client_last_modified": page.get("client_last_modified"),
        "version": int(page.get("version") or 1),
    }


def _normalize_monitoring_page_doc(page: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(page)
    normalized.pop("_id", None)
    for key in ("last_modified", "created_at", "updated_at"):
        if isinstance(normalized.get(key), datetime):
            normalized[key] = normalized[key].isoformat()
    normalized["book_type"] = str(normalized.get("book_type") or "MS").upper()
    normalized["page_number"] = int(normalized.get("page_number") or 0)
    normalized["page_key"] = _encode_monitoring_page_key(
        normalized.get("copy_id"),
        normalized["book_type"],
        normalized["page_number"],
    )
    normalized["stroke_count"] = int(normalized.get("stroke_count") or len(normalized.get("strokes") or []) or 0)
    return normalized


async def _resolve_student_canvas_user_ids(
    db: DatabaseManager,
    student_id: str,
) -> List[Any]:
    student_doc = await db.mongo_find_one("students", {"student_id": student_id})
    values: List[Any] = [student_id]
    if student_doc:
        values.extend([
            student_doc.get("username"),
            student_doc.get("email"),
            student_doc.get("_id"),
            str(student_doc.get("_id")) if student_doc.get("_id") is not None else None,
            student_doc.get("user_id"),
        ])
    return _dedupe_identity_values(values)


async def _resolve_student_pen_mac(db: DatabaseManager, student_id: str) -> Optional[str]:
    student_doc = await db.mongo_find_one("students", {"student_id": student_id})
    candidates = []

    binding = await db.mongo_find_one(
        "student_pen_bindings",
        {"user_id": student_id, "status": "active"},
    )
    if not binding:
        binding = await db.mongo_find_one(
            "student_pen_bindings",
            {"student_id": student_id, "status": "active"},
        )
    if not binding and student_doc:
        binding = await db.mongo_find_one(
            "student_pen_bindings",
            {"user_id": str(student_doc.get("_id")), "status": "active"},
        )
    if binding:
        candidates.append(binding.get("pen_mac"))
    candidates.extend([
        student_doc.get("pen_mac") if student_doc else None,
        student_doc.get("assigned_pen_mac") if student_doc else None,
        student_doc.get("bluetooth_address") if student_doc else None,
    ])

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().upper()
    return None


def _normalize_monitoring_pen_mac(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized or None


def _monitoring_pen_status_from_registry(
    pen_states: List[Dict[str, Any]],
    pen_mac: Optional[str],
) -> Dict[str, Any]:
    target_mac = _normalize_monitoring_pen_mac(pen_mac)
    if not target_mac:
        return {}

    for pen in pen_states:
        if _normalize_monitoring_pen_mac(pen.get("pen_mac")) != target_mac:
            continue
        return {
            "pen_connected": bool(pen.get("connected")),
            "pen_last_frame_ts": pen.get("last_frame_ts"),
            "pen_battery": pen.get("battery"),
            "pen_page_no": pen.get("page_no"),
            "pen_book_type": pen.get("book_type"),
        }
    return {}


async def _list_monitoring_pen_states() -> List[Dict[str, Any]]:
    try:
        from main_async import app
        registry = getattr(app.state, "dashboard_registry", None)
        if not registry:
            return []
        return await registry.list_pen_states()
    except Exception as exc:
        logger.debug("Failed to read dashboard pen registry for monitoring: %s", exc)
        return []


async def _resolve_tutor_canvas_user_ids(
    db: DatabaseManager,
    tutor_id: str,
    current_user: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    values: List[Any] = [tutor_id]
    if current_user:
        values.extend([
            current_user.get("username"),
            current_user.get("email"),
            current_user.get("user_id"),
            current_user.get("teacher_id"),
            current_user.get("tutor_id"),
        ])
    tutor_doc = await db.mongo_find_one("tutors", {"tutor_id": tutor_id})
    if tutor_doc:
        values.extend([
            tutor_doc.get("username"),
            tutor_doc.get("email"),
            tutor_doc.get("_id"),
            str(tutor_doc.get("_id")) if tutor_doc.get("_id") is not None else None,
            tutor_doc.get("user_id"),
        ])
    return _dedupe_identity_values(values)


async def _get_teacher_canvas_mode(
    db: DatabaseManager,
    meeting_id: str,
) -> Dict[str, Optional[str]]:
    meeting = await db.mongo_find_one("meetings", {"meeting_id": meeting_id})
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    mode = meeting.get(TEACHER_CANVAS_MODE_FIELD) or "live"
    if mode not in {"live", "stream"}:
        mode = "live"
    updated_at = meeting.get(TEACHER_CANVAS_MODE_UPDATED_AT_FIELD)
    return {
        "mode": mode,
        "updated_at": _serialize_datetime(updated_at),
        "updated_by": meeting.get(TEACHER_CANVAS_MODE_UPDATED_BY_FIELD),
    }


async def _set_teacher_canvas_mode(
    db: DatabaseManager,
    meeting_id: str,
    mode: Literal["live", "stream"],
    updated_by: str,
) -> Dict[str, Optional[str]]:
    now = datetime.utcnow()
    await db.mongo_update_one(
        "meetings",
        {"meeting_id": meeting_id},
        {
            "$set": {
                TEACHER_CANVAS_MODE_FIELD: mode,
                TEACHER_CANVAS_MODE_UPDATED_AT_FIELD: now,
                TEACHER_CANVAS_MODE_UPDATED_BY_FIELD: updated_by,
            }
        },
    )
    return {
        "mode": mode,
        "updated_at": now.isoformat(),
        "updated_by": updated_by,
    }


async def _find_canvas_pages_for_user_ids(
    db: DatabaseManager,
    user_ids: List[Any],
    *,
    projection: Optional[Dict[str, Any]] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    if not user_ids:
        return []
    return await db.mongo_find(
        "canvas_pages",
        {"user_id": {"$in": user_ids}},
        projection=projection,
        sort=[("last_modified", -1)],
        limit=limit,
    )


async def _find_teacher_online_class_pages(
    db: DatabaseManager,
    meeting: Dict[str, Any],
    *,
    tutor_user: Optional[Dict[str, Any]] = None,
    projection: Optional[Dict[str, Any]] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    tutor_id = str(meeting.get("tutor_id") or "")
    user_ids = await _resolve_tutor_canvas_user_ids(db, tutor_id, tutor_user)
    if not user_ids:
        return []
    copy_id = _online_class_copy_id(str(meeting.get("meeting_id") or ""))
    return await db.mongo_find(
        "canvas_pages",
        {
            "user_id": {"$in": user_ids},
            "copy_id": copy_id,
            "$or": [
                {"stroke_count": {"$gt": 0}},
                {"strokes.0": {"$exists": True}},
            ],
        },
        projection=projection,
        sort=[("last_modified", -1)],
        limit=limit,
    )


def _page_activity_min_ms(page: Dict[str, Any]) -> Optional[float]:
    candidates: List[float] = []
    for key in ("first_activity", "last_activity", "client_last_modified", "last_modified"):
        ts = _datetime_to_epoch_ms(page.get(key))
        if ts is not None:
            candidates.append(ts)

    for stroke in page.get("strokes") or []:
        if not isinstance(stroke, dict):
            continue
        for key in ("startedAt", "endedAt", "timestamp"):
            ts = _datetime_to_epoch_ms(stroke.get(key))
            if ts is not None:
                candidates.append(ts)

    return min(candidates) if candidates else None


def _build_online_class_note_item(
    meeting: Dict[str, Any],
    pages: List[Dict[str, Any]],
) -> OnlineClassNoteClassItem:
    activity_max_values = [
        value for value in (_page_activity_max_ms(page) for page in pages) if value is not None
    ]
    activity_min_values = [
        value for value in (_page_activity_min_ms(page) for page in pages) if value is not None
    ]
    return OnlineClassNoteClassItem(
        meeting_id=str(meeting.get("meeting_id") or ""),
        copy_id=_online_class_copy_id(str(meeting.get("meeting_id") or "")),
        topic=meeting.get("topic"),
        subject=meeting.get("subject"),
        standard=meeting.get("standard"),
        section=meeting.get("section"),
        tutor_name=meeting.get("tutor_name"),
        status=meeting.get("status"),
        scheduled_at=meeting.get("scheduled_at"),
        started_at=meeting.get("started_at"),
        ended_at=meeting.get("ended_at"),
        page_count=len(pages),
        stroke_count=sum(int(page.get("stroke_count") or len(page.get("strokes") or []) or 0) for page in pages),
        first_activity=min(activity_min_values) if activity_min_values else None,
        latest_activity=max(activity_max_values) if activity_max_values else None,
    )


async def _verify_user_can_read_online_class_notes(
    db: DatabaseManager,
    meeting_id: str,
    current_user: Dict[str, Any],
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    user_type = current_user.get("user_type")
    if user_type == "tutor":
        meeting = await _verify_tutor_owns_meeting(
            db,
            meeting_id,
            current_user.get("tutor_id"),
            current_user=current_user,
        )
        return meeting, current_user
    if user_type == "student":
        student_id = await resolve_business_student_id(current_user, db)
        if not student_id:
            raise HTTPException(status_code=403, detail="Could not resolve student identity")
        meeting = await _verify_student_invited(db, meeting_id, student_id, current_user=current_user)
        return meeting, None
    raise HTTPException(status_code=403, detail="Access denied")


async def _get_teacher_online_class_page(
    db: DatabaseManager,
    meeting: Dict[str, Any],
    page_key: str,
    *,
    tutor_user: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    decoded = _decode_monitoring_page_key(page_key)
    meeting_id = str(meeting.get("meeting_id") or "")
    expected_copy_id = _online_class_copy_id(meeting_id)
    if decoded.get("copy_id") and decoded.get("copy_id") != expected_copy_id:
        raise HTTPException(status_code=404, detail="Canvas page not found for this class")

    user_ids = await _resolve_tutor_canvas_user_ids(
        db,
        str(meeting.get("tutor_id") or ""),
        tutor_user,
    )
    query: Dict[str, Any] = {
        "user_id": {"$in": user_ids},
        "copy_id": expected_copy_id,
        "book_type": decoded["book_type"],
        "page_number": decoded["page_number"],
    }
    page = await db.mongo_find_one("canvas_pages", query)
    if not page:
        raise HTTPException(status_code=404, detail="Canvas page not found for this class")
    return _normalize_monitoring_page_doc(page)


async def _get_monitoring_page(
    db: DatabaseManager,
    user_ids: List[Any],
    page_key: str,
) -> Dict[str, Any]:
    decoded = _decode_monitoring_page_key(page_key)
    query: Dict[str, Any] = {
        "user_id": {"$in": user_ids},
        "book_type": decoded["book_type"],
        "page_number": decoded["page_number"],
    }
    if decoded.get("copy_id"):
        query["copy_id"] = decoded["copy_id"]
    page = await db.mongo_find_one("canvas_pages", query)
    if not page:
        raise HTTPException(status_code=404, detail="Canvas page not found")
    return _normalize_monitoring_page_doc(page)


def _canvas_page_identity_from_existing(existing: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "user_id": existing.get("user_id"),
        "copy_id": existing.get("copy_id"),
        "book_type": existing.get("book_type"),
        "page_number": existing.get("page_number"),
    }


async def _upsert_teacher_live_canvas_events(
    db: DatabaseManager,
    meeting: Dict[str, Any],
    current_user: Dict[str, Any],
    pages: List[CanvasPageUpsert],
) -> Dict[str, Any]:
    meeting_id = str(meeting.get("meeting_id") or "")
    user_id = canonical_canvas_user_id(current_user)
    user_ids = canvas_user_id_variants(current_user)
    if user_id not in user_ids:
        user_ids.append(user_id)
    admin_id = current_user.get("admin_id") or meeting.get("admin_id")
    default_copy_id = f"online-{meeting_id}"

    upserted = 0
    modified = 0
    for raw_page in pages:
        now = datetime.utcnow()
        page = raw_page.model_copy(update={
            "copy_id": raw_page.copy_id or default_copy_id,
            "source": "online_class_teacher_live",
            "session_id": raw_page.session_id or meeting_id,
            "stroke_count": raw_page.stroke_count if raw_page.stroke_count is not None else len(raw_page.strokes or []),
        })
        page_filter: Dict[str, Any] = {
            "user_id": {"$in": user_ids},
            "copy_id": page.copy_id,
            "book_type": page.book_type.upper(),
            "page_number": page.page_number,
        }
        existing = await db.mongo_find_one("canvas_pages", page_filter)

        if existing is None:
            doc = _page_doc(user_id, admin_id, page, now, copy_id=page.copy_id)
            doc["meeting_id"] = meeting_id
            ok = await db.mongo_update_one(
                "canvas_pages",
                {
                    "user_id": user_id,
                    "copy_id": page.copy_id,
                    "book_type": page.book_type.upper(),
                    "page_number": page.page_number,
                },
                {"$set": doc},
                upsert=True,
            )
            if ok:
                upserted += 1
            continue

        doc, added_count = _build_merged_page_doc(
            existing_doc=existing,
            user_id=user_id,
            admin_id=admin_id,
            page=page,
            now=now,
            copy_id=page.copy_id,
        )
        doc["meeting_id"] = meeting_id
        update_fields = doc if added_count > 0 else {
            **_build_metadata_refresh(existing, page, now),
            "meeting_id": meeting_id,
            "source": "online_class_teacher_live",
        }
        if update_fields:
            ok = await db.mongo_update_one(
                "canvas_pages",
                _canvas_page_identity_from_existing(existing),
                {"$set": update_fields},
            )
            if ok:
                modified += 1

    return {
        "success": True,
        "upserted": upserted,
        "modified": modified,
        "count": len(pages),
    }


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


def _normalize_scope_id(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)


def _verify_meeting_admin_boundary(
    meeting: Dict[str, Any],
    current_user: Optional[Dict[str, Any]] = None,
) -> None:
    if not current_user:
        return
    meeting_admin_id = _normalize_scope_id(meeting.get("admin_id"))
    user_admin_id = _normalize_scope_id(
        current_user.get("admin_id") or current_user.get("created_by")
    )
    if not meeting_admin_id or not user_admin_id:
        return
    if meeting_admin_id != user_admin_id:
        raise HTTPException(status_code=403, detail="Tenant boundary mismatch for this meeting")


async def _verify_tutor_owns_meeting(
    db: DatabaseManager,
    meeting_id: str,
    tutor_id: str,
    current_user: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meeting = await db.mongo_find_one("meetings", {"meeting_id": meeting_id})
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if meeting.get("tutor_id") != tutor_id:
        raise HTTPException(status_code=403, detail="Not authorized for this meeting")
    _verify_meeting_admin_boundary(meeting, current_user)
    return meeting


async def _verify_student_invited(
    db: DatabaseManager,
    meeting_id: str,
    student_id: str,
    current_user: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meeting = await db.mongo_find_one("meetings", {"meeting_id": meeting_id})
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if student_id not in meeting.get("invited_student_ids", []):
        raise HTTPException(status_code=403, detail="Student not invited to this meeting")
    _verify_meeting_admin_boundary(meeting, current_user)
    return meeting


def _build_submission_result_item(
    *,
    submission: Dict[str, Any],
    lock: Dict[str, Any],
    meeting_id: str,
    lock_id: str,
    student_name: Optional[str] = None,
) -> SubmissionResultItem:
    canvas_pages = submission.get("canvas_pages") or []
    return SubmissionResultItem(
        submission_id=submission["submission_id"],
        meeting_id=str(submission.get("meeting_id") or meeting_id),
        lock_id=str(submission.get("lock_id") or lock_id),
        student_id=submission["student_id"],
        student_name=student_name,
        question_text=lock.get("question_text"),
        question_page_refs=submission.get("question_page_refs"),
        canvas_pages=canvas_pages,
        canvas_image_count=len(canvas_pages),
        answer_text=submission.get("answer_text"),
        time_spent=submission.get("time_spent"),
        analysis_status=submission.get("analysis_status", "pending"),
        score=submission.get("score"),
        is_correct=submission.get("is_correct"),
        student_answer=submission.get("student_answer"),
        work_shown=submission.get("work_shown"),
        what_went_wrong=submission.get("what_went_wrong"),
        correct_solution=submission.get("correct_solution"),
        analysis_error=submission.get("analysis_error"),
        analysis_completed_at=submission.get("analysis_completed_at"),
        analysis_failed_at=submission.get("analysis_failed_at"),
        created_at=submission.get("created_at"),
        updated_at=submission.get("updated_at"),
    )


@router.get("/notes", response_model=OnlineClassNotesResponse)
@limiter.limit("30/minute")
async def api_list_online_class_notes(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    user_type = current_user.get("user_type")
    if user_type == "student":
        student_id = await resolve_business_student_id(current_user, db)
        if not student_id:
            raise HTTPException(status_code=403, detail="Could not resolve student identity")
        meeting_query: Dict[str, Any] = {
            "invited_student_ids": student_id,
            "status": {"$in": ["active", "ended"]},
        }
        tutor_user: Optional[Dict[str, Any]] = None
    elif user_type == "tutor":
        meeting_query = {
            "tutor_id": current_user.get("tutor_id"),
            "status": {"$in": ["active", "ended"]},
        }
        tutor_user = current_user
    else:
        raise HTTPException(status_code=403, detail="Access denied")

    meetings = await db.mongo_find(
        "meetings",
        meeting_query,
        sort=[("started_at", -1), ("scheduled_at", -1)],
        limit=100,
    )

    classes: List[OnlineClassNoteClassItem] = []
    for meeting in meetings:
        _verify_meeting_admin_boundary(meeting, current_user)
        pages = await _find_teacher_online_class_pages(
            db,
            meeting,
            tutor_user=tutor_user,
            projection={"strokes": 0},
            limit=500,
        )
        classes.append(_build_online_class_note_item(meeting, pages))

    return OnlineClassNotesResponse(classes=classes)


@router.get(
    "/notes/{meeting_id}/pages",
    response_model=MonitoringPageListResponse,
)
@limiter.limit("30/minute")
async def api_list_online_class_note_pages(
    request: Request,
    meeting_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    meeting, tutor_user = await _verify_user_can_read_online_class_notes(db, meeting_id, current_user)
    pages = await _find_teacher_online_class_pages(
        db,
        meeting,
        tutor_user=tutor_user,
        projection={"strokes": 0},
        limit=1000,
    )
    metas = [_build_monitoring_page_meta(page) for page in pages]
    return MonitoringPageListResponse(
        count=len(metas),
        pages=metas,
        server_time=datetime.utcnow().isoformat(),
    )


@router.get(
    "/notes/{meeting_id}/pages/{page_key}",
    response_model=MonitoringPageResponse,
)
@limiter.limit("60/minute")
async def api_get_online_class_note_page(
    request: Request,
    meeting_id: str,
    page_key: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    meeting, tutor_user = await _verify_user_can_read_online_class_notes(db, meeting_id, current_user)
    page = await _get_teacher_online_class_page(db, meeting, page_key, tutor_user=tutor_user)
    return MonitoringPageResponse(page=page)


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
        await _verify_tutor_owns_meeting(db, meeting_id, current_user.get("tutor_id"), current_user=current_user)
        moderator = True
    elif user_type == "student":
        student_id = await resolve_business_student_id(current_user, db)
        if not student_id:
            raise HTTPException(status_code=403, detail="Could not resolve student identity")
        await _verify_student_invited(db, meeting_id, student_id, current_user=current_user)
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
    meeting = await _verify_tutor_owns_meeting(db, meeting_id, tutor_id, current_user=current_user)
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
    await _verify_tutor_owns_meeting(db, meeting_id, tutor_id, current_user=current_user)
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

    await _verify_student_invited(db, meeting_id, student_id, current_user=current_user)
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


@router.get(
    "/meetings/{meeting_id}/monitoring/students",
    response_model=MonitoringStudentsResponse,
)
@limiter.limit("30/minute")
async def api_get_monitoring_students(
    request: Request,
    meeting_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    _require_tutor(current_user)
    meeting = await _verify_tutor_owns_meeting(db, meeting_id, current_user.get("tutor_id"), current_user=current_user)
    await _verify_meeting_active(db, meeting_id)

    invited_ids = [str(student_id) for student_id in meeting.get("invited_student_ids", []) if student_id]
    joined_ids = {str(student_id) for student_id in meeting.get("joined_student_ids", []) if student_id}
    if not invited_ids:
        return MonitoringStudentsResponse(students=[])

    student_docs = await db.mongo_find("students", {"student_id": {"$in": invited_ids}})
    by_id = {str(doc.get("student_id")): doc for doc in student_docs}
    pen_states = await _list_monitoring_pen_states()
    students: List[MonitoringStudentItem] = []
    for student_id in invited_ids:
        student_doc = by_id.get(student_id, {})
        pen_mac = await _resolve_student_pen_mac(db, student_id)
        pen_status = _monitoring_pen_status_from_registry(pen_states, pen_mac)
        students.append(
            MonitoringStudentItem(
                student_id=student_id,
                student_name=(
                    student_doc.get("name")
                    or student_doc.get("full_name")
                    or student_doc.get("username")
                ),
                username=student_doc.get("username"),
                pen_mac=pen_mac,
                **pen_status,
                joined=student_id in joined_ids,
            )
        )
    return MonitoringStudentsResponse(students=students)


@router.get(
    "/meetings/{meeting_id}/monitoring/students/{student_id}/pages",
    response_model=MonitoringPageListResponse,
)
@limiter.limit("30/minute")
async def api_get_monitoring_student_pages(
    request: Request,
    meeting_id: str,
    student_id: str,
    since_session_start: bool = Query(True),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    _require_tutor(current_user)
    meeting = await _verify_tutor_owns_meeting(db, meeting_id, current_user.get("tutor_id"), current_user=current_user)
    await _verify_meeting_active(db, meeting_id)
    if student_id not in meeting.get("invited_student_ids", []):
        raise HTTPException(status_code=403, detail="Student not invited to this meeting")

    user_ids = await _resolve_student_canvas_user_ids(db, student_id)
    pages = await _find_canvas_pages_for_user_ids(
        db,
        user_ids,
        projection={"strokes": 0},
    )
    if since_session_start:
        pages = _filter_canvas_pages_since_session_start(pages, meeting.get("started_at"))

    metas = [_build_monitoring_page_meta(page) for page in pages]
    return MonitoringPageListResponse(
        count=len(metas),
        pages=metas,
        server_time=datetime.utcnow().isoformat(),
    )


@router.get(
    "/meetings/{meeting_id}/monitoring/students/{student_id}/pages/{page_key}",
    response_model=MonitoringPageResponse,
)
@limiter.limit("60/minute")
async def api_get_monitoring_student_page(
    request: Request,
    meeting_id: str,
    student_id: str,
    page_key: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    _require_tutor(current_user)
    meeting = await _verify_tutor_owns_meeting(db, meeting_id, current_user.get("tutor_id"), current_user=current_user)
    await _verify_meeting_active(db, meeting_id)
    if student_id not in meeting.get("invited_student_ids", []):
        raise HTTPException(status_code=403, detail="Student not invited to this meeting")

    user_ids = await _resolve_student_canvas_user_ids(db, student_id)
    page = await _get_monitoring_page(db, user_ids, page_key)
    filtered = _filter_canvas_pages_since_session_start([page], meeting.get("started_at"))
    if not filtered:
        raise HTTPException(status_code=404, detail="Canvas page not active in this class session")
    return MonitoringPageResponse(page=page)


@router.get(
    "/meetings/{meeting_id}/monitoring/strokes/stream",
    response_model=MonitoringPageListResponse,
)
@limiter.limit("60/minute")
async def api_get_monitoring_stroke_stream_snapshot(
    request: Request,
    meeting_id: str,
    student_id: str = Query(...),
    after: Optional[float] = Query(None, description="Epoch ms cursor for changed pages"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    _require_tutor(current_user)
    meeting = await _verify_tutor_owns_meeting(db, meeting_id, current_user.get("tutor_id"), current_user=current_user)
    await _verify_meeting_active(db, meeting_id)
    if student_id not in meeting.get("invited_student_ids", []):
        raise HTTPException(status_code=403, detail="Student not invited to this meeting")

    user_ids = await _resolve_student_canvas_user_ids(db, student_id)
    pages = await _find_canvas_pages_for_user_ids(
        db,
        user_ids,
        projection={"strokes": 0},
    )
    pages = _filter_canvas_pages_since_session_start(pages, meeting.get("started_at"))
    if after is not None:
        pages = [
            page
            for page in pages
            if (_page_activity_max_ms(page) is not None and _page_activity_max_ms(page) > after)
        ]
    metas = [_build_monitoring_page_meta(page) for page in pages]
    return MonitoringPageListResponse(
        count=len(metas),
        pages=metas,
        server_time=datetime.utcnow().isoformat(),
    )


@router.get(
    "/meetings/{meeting_id}/teacher-canvas/mode",
    response_model=TeacherCanvasModeResponse,
)
@limiter.limit("60/minute")
async def api_get_teacher_canvas_mode(
    request: Request,
    meeting_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    user_type = current_user.get("user_type")
    if user_type == "tutor":
        await _verify_tutor_owns_meeting(db, meeting_id, current_user.get("tutor_id"), current_user=current_user)
    elif user_type == "student":
        student_id = await resolve_business_student_id(current_user, db)
        if not student_id:
            raise HTTPException(status_code=403, detail="Could not resolve student identity")
        await _verify_student_invited(db, meeting_id, student_id, current_user=current_user)
    else:
        raise HTTPException(status_code=403, detail="Access denied")
    await _verify_meeting_active(db, meeting_id)
    return TeacherCanvasModeResponse(**await _get_teacher_canvas_mode(db, meeting_id))


@router.post(
    "/meetings/{meeting_id}/teacher-canvas/mode",
    response_model=TeacherCanvasModeResponse,
)
@limiter.limit("20/minute")
async def api_set_teacher_canvas_mode(
    request: Request,
    meeting_id: str,
    body: TeacherCanvasModeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    _require_tutor(current_user)
    tutor_id = current_user.get("tutor_id")
    await _verify_tutor_owns_meeting(db, meeting_id, tutor_id, current_user=current_user)
    await _verify_meeting_active(db, meeting_id)
    if body.mode == "live":
        await db.mongo_update_one(
            CANVAS_SHARE_REQUESTS_COLLECTION,
            {"meeting_id": meeting_id, "status": "active"},
            {"$set": {"status": "ended", "ended_at": datetime.utcnow()}},
        )
    return TeacherCanvasModeResponse(**await _set_teacher_canvas_mode(db, meeting_id, body.mode, tutor_id))


@router.get(
    "/meetings/{meeting_id}/teacher-canvas/live/stream",
    response_model=MonitoringPageListResponse,
)
@limiter.limit("60/minute")
async def api_get_teacher_live_canvas_stream_snapshot(
    request: Request,
    meeting_id: str,
    after: Optional[float] = Query(None, description="Epoch ms cursor for changed pages"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    user_type = current_user.get("user_type")
    if user_type == "tutor":
        meeting = await _verify_tutor_owns_meeting(db, meeting_id, current_user.get("tutor_id"), current_user=current_user)
        tutor_user = current_user
    elif user_type == "student":
        student_id = await resolve_business_student_id(current_user, db)
        if not student_id:
            raise HTTPException(status_code=403, detail="Could not resolve student identity")
        meeting = await _verify_student_invited(db, meeting_id, student_id, current_user=current_user)
        tutor_user = None
    else:
        raise HTTPException(status_code=403, detail="Access denied")
    await _verify_meeting_active(db, meeting_id)

    tutor_id = str(meeting.get("tutor_id") or "")
    user_ids = await _resolve_tutor_canvas_user_ids(db, tutor_id, tutor_user)
    pages = await _find_canvas_pages_for_user_ids(
        db,
        user_ids,
        projection={"strokes": 0},
    )
    pages = _filter_canvas_pages_since_session_start(pages, meeting.get("started_at"))
    if after is not None:
        pages = [
            page
            for page in pages
            if (_page_activity_max_ms(page) is not None and _page_activity_max_ms(page) > after)
        ]
    metas = [_build_monitoring_page_meta(page) for page in pages]
    return MonitoringPageListResponse(
        count=len(metas),
        pages=metas,
        server_time=datetime.utcnow().isoformat(),
    )


@router.get(
    "/meetings/{meeting_id}/teacher-canvas/live/pages/{page_key}",
    response_model=MonitoringPageResponse,
)
@limiter.limit("60/minute")
async def api_get_teacher_live_canvas_page(
    request: Request,
    meeting_id: str,
    page_key: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    user_type = current_user.get("user_type")
    if user_type == "tutor":
        meeting = await _verify_tutor_owns_meeting(db, meeting_id, current_user.get("tutor_id"), current_user=current_user)
        tutor_user = current_user
    elif user_type == "student":
        student_id = await resolve_business_student_id(current_user, db)
        if not student_id:
            raise HTTPException(status_code=403, detail="Could not resolve student identity")
        meeting = await _verify_student_invited(db, meeting_id, student_id, current_user=current_user)
        tutor_user = None
    else:
        raise HTTPException(status_code=403, detail="Access denied")
    await _verify_meeting_active(db, meeting_id)

    tutor_id = str(meeting.get("tutor_id") or "")
    user_ids = await _resolve_tutor_canvas_user_ids(db, tutor_id, tutor_user)
    page = await _get_monitoring_page(db, user_ids, page_key)
    filtered = _filter_canvas_pages_since_session_start([page], meeting.get("started_at"))
    if not filtered:
        raise HTTPException(status_code=404, detail="Canvas page not active in this class session")
    return MonitoringPageResponse(page=page)


@router.post(
    "/meetings/{meeting_id}/teacher-canvas/live/events",
    response_model=TeacherLiveCanvasEventsResponse,
)
@limiter.limit("60/minute")
async def api_post_teacher_live_canvas_events(
    request: Request,
    meeting_id: str,
    body: CanvasPageBatchRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    _require_tutor(current_user)
    tutor_id = current_user.get("tutor_id")
    meeting = await _verify_tutor_owns_meeting(db, meeting_id, tutor_id, current_user=current_user)
    await _verify_meeting_active(db, meeting_id)
    mode = await _get_teacher_canvas_mode(db, meeting_id)
    if mode["mode"] != "live":
        raise HTTPException(status_code=409, detail="Teacher canvas is not in Live Canvas mode")
    return TeacherLiveCanvasEventsResponse(
        **await _upsert_teacher_live_canvas_events(db, meeting, current_user, body.pages)
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
    await _verify_tutor_owns_meeting(db, meeting_id, tutor_id, current_user=current_user)
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
        await _verify_tutor_owns_meeting(db, meeting_id, current_user.get("tutor_id"), current_user=current_user)
    elif user_type == "student":
        student_id = await resolve_business_student_id(current_user, db)
        await _verify_student_invited(db, meeting_id, student_id, current_user=current_user)
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
    await _verify_tutor_owns_meeting(db, meeting_id, tutor_id, current_user=current_user)

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
    await _verify_tutor_owns_meeting(db, meeting_id, tutor_id, current_user=current_user)

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
            _build_submission_result_item(
                submission=sub,
                lock=lock,
                meeting_id=meeting_id,
                lock_id=lock_id,
                student_name=student_name,
            )
        )
    return {"lock": LockResponse(**lock), "submissions": results}


@router.post("/meetings/{meeting_id}/locks/{lock_id}/submissions/{submission_id}/reanalyze", response_model=SubmissionResponse)
@limiter.limit("10/minute")
async def api_reanalyze_submission(
    request: Request,
    meeting_id: str,
    lock_id: str,
    submission_id: str,
    body: ReanalyzeSubmissionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database),
):
    _require_tutor(current_user)
    tutor_id = current_user.get("tutor_id")
    await _verify_tutor_owns_meeting(db, meeting_id, tutor_id, current_user=current_user)

    lock = await get_lock_by_id(db, meeting_id, lock_id)
    if not lock:
        raise HTTPException(status_code=404, detail="Lock not found")

    submission = await db.mongo_find_one(
        "online_class_submissions",
        {"submission_id": submission_id, "meeting_id": meeting_id, "lock_id": lock_id},
    )
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    now = datetime.utcnow()
    reset_fields = {
        "analysis_status": "pending",
        "analysis_error": None,
        "analysis_failed_at": None,
        "analysis_completed_at": None,
        "reanalyze_requested_at": now,
        "reanalyze_requested_by": tutor_id,
        "reanalyze_comments": body.tutor_comments.strip()[:2000] if body.tutor_comments else None,
        "updated_at": now,
    }
    await db.mongo_update_one(
        "online_class_submissions",
        {"submission_id": submission_id},
        {"$set": reset_fields},
    )
    submission.update(reset_fields)

    from services.online_class.analysis_service import run_submission_analysis
    task = asyncio.create_task(
        run_submission_analysis(
            db,
            current_user,
            lock,
            submission.copy(),
            tutor_comments=body.tutor_comments,
        )
    )
    task.add_done_callback(_log_analysis_task_error)
    return SubmissionResponse(**submission)


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
    await _verify_student_invited(db, meeting_id, student_id, current_user=current_user)
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
