import asyncio
from datetime import datetime, timedelta

import pytest

from api.v1.online_class.locks import create_lock, end_lock
from api.v1.online_class.submissions import create_or_update_submission
from api.v1.meeting_async import resolve_business_student_id, resolve_notification_recipient_ids
from services.online_class.jitsi_provider import JitsiProviderService


class FakeDb:
    def __init__(self):
        self.collections = {
            "online_class_locks": [],
            "online_class_submissions": [],
            "canvas_pages": [],
            "meetings": [],
            "students": [],
        }

    def _value_at_path(self, doc, key):
        current = doc
        for part in str(key).split("."):
            if isinstance(current, list):
                if not part.isdigit():
                    return None
                idx = int(part)
                if idx >= len(current):
                    return None
                current = current[idx]
                continue
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    def _matches(self, doc, query):
        for key, value in query.items():
            if key == "$or":
                if not any(self._matches(doc, clause) for clause in value):
                    return False
                continue
            current = self._value_at_path(doc, key)
            if isinstance(value, dict) and "$in" in value:
                if current not in value["$in"]:
                    return False
                continue
            if isinstance(value, dict) and "$gt" in value:
                if current is None or current <= value["$gt"]:
                    return False
                continue
            if isinstance(value, dict) and "$gte" in value:
                if current is None or current < value["$gte"]:
                    return False
                continue
            if isinstance(value, dict) and "$exists" in value:
                exists = current is not None
                if exists != value["$exists"]:
                    return False
                continue
            if isinstance(current, list):
                if value not in current:
                    return False
                continue
            if current != value:
                return False
        return True

    async def mongo_find_one(self, collection, query):
        for doc in self.collections.get(collection, []):
            if self._matches(doc, query):
                return doc.copy()
        return None

    async def mongo_find(self, collection, query, projection=None, sort=None, limit=None, skip=None):
        docs = [doc.copy() for doc in self.collections.get(collection, []) if self._matches(doc, query)]
        if projection:
            excluded = {key for key, value in projection.items() if value == 0}
            docs = [{key: value for key, value in doc.items() if key not in excluded} for doc in docs]
        if sort:
            for key, direction in reversed(sort):
                docs.sort(key=lambda item: item.get(key) or 0, reverse=direction < 0)
        if skip:
            docs = docs[skip:]
        if limit is not None:
            docs = docs[:limit]
        return docs

    async def mongo_insert_one(self, collection, doc):
        self.collections.setdefault(collection, []).append(doc.copy())
        return doc.get("lock_id") or doc.get("submission_id")

    async def mongo_update_one(self, collection, query, update, upsert=False):
        updates = update.get("$set", {})
        for doc in self.collections.get(collection, []):
            if all(doc.get(key) == value for key, value in query.items()):
                doc.update(updates)
                return True
        if upsert:
            new_doc = query.copy()
            new_doc.update(updates)
            self.collections.setdefault(collection, []).append(new_doc)
            return True
        return False


def test_create_lock_rejects_second_active_lock():
    asyncio.run(_test_create_lock_rejects_second_active_lock())


async def _test_create_lock_rejects_second_active_lock():
    db = FakeDb()

    first = await create_lock(
        db=db,
        meeting_id="MTG123",
        tutor_id="tutor-1",
        question_text="Solve x + 1 = 2",
        question_image_id=None,
        question_bbox=None,
        duration_seconds=120,
    )

    assert first["status"] == "active"
    assert first["meeting_id"] == "MTG123"

    with pytest.raises(ValueError, match="active lock already exists"):
        await create_lock(
            db=db,
            meeting_id="MTG123",
            tutor_id="tutor-1",
            question_text="Second",
            question_image_id=None,
            question_bbox=None,
            duration_seconds=120,
        )


def test_end_lock_is_idempotent():
    asyncio.run(_test_end_lock_is_idempotent())


async def _test_end_lock_is_idempotent():
    db = FakeDb()
    lock = await create_lock(
        db=db,
        meeting_id="MTG123",
        tutor_id="tutor-1",
        question_text=None,
        question_image_id=None,
        question_bbox=None,
        duration_seconds=60,
    )

    ended = await end_lock(db, "MTG123", lock["lock_id"])
    ended_again = await end_lock(db, "MTG123", lock["lock_id"])

    assert ended["status"] == "ended"
    assert ended_again["status"] == "ended"
    assert ended_again["lock_id"] == lock["lock_id"]


def test_verify_tutor_owns_meeting_allows_only_owner():
    asyncio.run(_test_verify_tutor_owns_meeting_allows_only_owner())


async def _test_verify_tutor_owns_meeting_allows_only_owner():
    from fastapi import HTTPException
    from api.v1.online_class.router import _verify_tutor_owns_meeting

    db = FakeDb()
    await db.mongo_insert_one(
        "meetings",
        {
            "meeting_id": "MTG123",
            "tutor_id": "tutor-owner",
            "status": "active",
        },
    )

    meeting = await _verify_tutor_owns_meeting(db, "MTG123", "tutor-owner")
    assert meeting["meeting_id"] == "MTG123"

    with pytest.raises(HTTPException) as exc:
        await _verify_tutor_owns_meeting(db, "MTG123", "other-tutor")

    assert exc.value.status_code == 403
    assert "Not authorized" in exc.value.detail


def test_verify_tutor_owns_meeting_enforces_admin_boundary():
    asyncio.run(_test_verify_tutor_owns_meeting_enforces_admin_boundary())


async def _test_verify_tutor_owns_meeting_enforces_admin_boundary():
    from fastapi import HTTPException
    from api.v1.online_class.router import _verify_tutor_owns_meeting

    db = FakeDb()
    await db.mongo_insert_one(
        "meetings",
        {
            "meeting_id": "MTG123",
            "tutor_id": "tutor-owner",
            "admin_id": "admin-a",
            "status": "active",
        },
    )

    meeting = await _verify_tutor_owns_meeting(
        db,
        "MTG123",
        "tutor-owner",
        current_user={"admin_id": "admin-a"},
    )
    assert meeting["meeting_id"] == "MTG123"

    with pytest.raises(HTTPException) as exc:
        await _verify_tutor_owns_meeting(
            db,
            "MTG123",
            "tutor-owner",
            current_user={"admin_id": "admin-b"},
        )

    assert exc.value.status_code == 403
    assert "tenant" in exc.value.detail.lower()


def test_verify_student_invited_blocks_other_students():
    asyncio.run(_test_verify_student_invited_blocks_other_students())


async def _test_verify_student_invited_blocks_other_students():
    from fastapi import HTTPException
    from api.v1.online_class.router import _verify_student_invited

    db = FakeDb()
    await db.mongo_insert_one(
        "meetings",
        {
            "meeting_id": "MTG123",
            "status": "active",
            "invited_student_ids": ["STU_INVITED_1"],
        },
    )

    meeting = await _verify_student_invited(db, "MTG123", "STU_INVITED_1")
    assert meeting["meeting_id"] == "MTG123"

    with pytest.raises(HTTPException) as exc:
        await _verify_student_invited(db, "MTG123", "STU_OTHER_2")

    assert exc.value.status_code == 403
    assert "not invited" in exc.value.detail


def test_verify_student_invited_enforces_admin_boundary():
    asyncio.run(_test_verify_student_invited_enforces_admin_boundary())


async def _test_verify_student_invited_enforces_admin_boundary():
    from fastapi import HTTPException
    from api.v1.online_class.router import _verify_student_invited

    db = FakeDb()
    await db.mongo_insert_one(
        "meetings",
        {
            "meeting_id": "MTG123",
            "admin_id": "admin-a",
            "status": "active",
            "invited_student_ids": ["STU_INVITED_1"],
        },
    )

    meeting = await _verify_student_invited(
        db,
        "MTG123",
        "STU_INVITED_1",
        current_user={"admin_id": "admin-a"},
    )
    assert meeting["meeting_id"] == "MTG123"

    with pytest.raises(HTTPException) as exc:
        await _verify_student_invited(
            db,
            "MTG123",
            "STU_INVITED_1",
            current_user={"admin_id": "admin-b"},
        )

    assert exc.value.status_code == 403
    assert "tenant" in exc.value.detail.lower()


def test_verify_meeting_active_rejects_inactive_meetings():
    asyncio.run(_test_verify_meeting_active_rejects_inactive_meetings())


async def _test_verify_meeting_active_rejects_inactive_meetings():
    from fastapi import HTTPException
    from api.v1.online_class.router import _verify_meeting_active

    db = FakeDb()
    await db.mongo_insert_one(
        "meetings",
        {
            "meeting_id": "MTG123",
            "status": "scheduled",
        },
    )

    with pytest.raises(HTTPException) as exc:
        await _verify_meeting_active(db, "MTG123")

    assert exc.value.status_code == 400
    assert "not active" in exc.value.detail


def test_duplicate_submission_updates_existing_record():
    asyncio.run(_test_duplicate_submission_updates_existing_record())


def test_submission_result_item_includes_detail_contract_fields():
    from api.v1.online_class.router import _build_submission_result_item

    created_at = datetime.utcnow()
    updated_at = created_at + timedelta(seconds=5)
    lock = {
        "question_text": "Solve x + 1 = 2",
    }
    submission = {
        "submission_id": "sub-1",
        "meeting_id": "MTG123",
        "lock_id": "lck-1",
        "student_id": "STU_1",
        "canvas_pages": ["img-a", "img-b"],
        "question_page_refs": {"0": {"copyId": "online-MTG123"}},
        "answer_text": "x = 1",
        "time_spent": 20,
        "analysis_status": "completed",
        "score": 0.75,
        "is_correct": False,
        "student_answer": "x = 1",
        "work_shown": "Moved 1 to the other side",
        "what_went_wrong": "Arithmetic error",
        "correct_solution": "x = 1",
        "created_at": created_at,
        "updated_at": updated_at,
    }

    result = _build_submission_result_item(
        submission=submission,
        lock=lock,
        meeting_id="MTG123",
        lock_id="lck-1",
        student_name="Asha",
    )

    assert result.meeting_id == "MTG123"
    assert result.lock_id == "lck-1"
    assert result.question_text == "Solve x + 1 = 2"
    assert result.question_page_refs == {"0": {"copyId": "online-MTG123"}}
    assert result.canvas_image_count == 2
    assert result.updated_at == updated_at


async def _test_duplicate_submission_updates_existing_record():
    db = FakeDb()

    first = await create_or_update_submission(
        db=db,
        meeting_id="MTG123",
        lock_id="lck-1",
        student_id="student-1",
        canvas_pages=["img-a"],
        question_page_refs={"pages": [1]},
        answer_text=None,
        time_spent=20,
        client_submitted_at=datetime.utcnow(),
    )
    second = await create_or_update_submission(
        db=db,
        meeting_id="MTG123",
        lock_id="lck-1",
        student_id="student-1",
        canvas_pages=["img-b"],
        question_page_refs={"pages": [2]},
        answer_text="updated",
        time_spent=30,
        client_submitted_at=datetime.utcnow(),
    )

    assert second["submission_id"] == first["submission_id"]
    assert second["canvas_pages"] == ["img-b"]
    assert second["answer_text"] == "updated"
    assert len(db.collections["online_class_submissions"]) == 1


def test_jitsi_provider_uses_base_url_host_when_domain_missing(monkeypatch):
    monkeypatch.delenv("ONLINE_CLASS_JITSI_DOMAIN", raising=False)
    monkeypatch.delenv("ONLINE_CLASS_JITSI_JWT_SECRET", raising=False)
    monkeypatch.setenv("ONLINE_CLASS_JITSI_BASE_URL", "https://class.stoody.in")
    monkeypatch.setenv("ONLINE_CLASS_JITSI_JWT_ENABLED", "true")

    provider = JitsiProviderService()
    details = provider.get_provider_details("MTG 123")

    assert details["configured"] is False
    assert details["domain"] == "class.stoody.in"
    assert details["room_name"] == "stoody-MTG-123"
    assert details["url"] == ""
    assert details["token_required"] is True
    assert details["token"] is None


class FakeStudentDb:
    def __init__(self, students=None):
        self.students = students or {}

    async def mongo_find_one(self, collection, query):
        if collection == "students":
            if "_id" in query:
                from bson import ObjectId
                oid = query["_id"]
                for sid, doc in self.students.items():
                    if doc.get("_id") == oid:
                        return doc.copy()
            if "student_id" in query:
                for sid, doc in self.students.items():
                    if doc.get("student_id") == query["student_id"]:
                        return doc.copy()
        return None

    async def mongo_find(self, collection, query):
        if collection == "students":
            results = []
            in_filter = None
            if isinstance(query, dict) and "student_id" in query:
                sf = query["student_id"]
                if isinstance(sf, dict) and "$in" in sf:
                    in_filter = set(sf["$in"])
            for sid, doc in self.students.items():
                if in_filter is not None:
                    if doc.get("student_id") in in_filter:
                        results.append(doc.copy())
                else:
                    results.append(doc.copy())
            return results
        return []


def test_resolve_business_student_id_with_objectid():
    asyncio.run(_test_resolve_with_objectid())


async def _test_resolve_with_objectid():
    from bson import ObjectId
    oid = ObjectId()
    db = FakeStudentDb(students={
        "stu1": {"_id": oid, "student_id": "STU_Lavyansh_536995"},
    })
    current_user = {"user_type": "student", "user_id": str(oid)}
    result = await resolve_business_student_id(current_user, db)
    assert result == "STU_Lavyansh_536995"


def test_resolve_business_student_id_already_business():
    asyncio.run(_test_resolve_already_business())


async def _test_resolve_already_business():
    db = FakeStudentDb()
    current_user = {"user_type": "student", "user_id": "STU_Already_123"}
    result = await resolve_business_student_id(current_user, db)
    assert result == "STU_Already_123"


def test_resolve_business_student_id_non_student_returns_none():
    asyncio.run(_test_resolve_non_student())


async def _test_resolve_non_student():
    db = FakeStudentDb()
    current_user = {"user_type": "tutor", "user_id": "tutor-1"}
    result = await resolve_business_student_id(current_user, db)
    assert result is None


def test_resolve_business_student_id_fallback_to_raw():
    asyncio.run(_test_resolve_fallback())


async def _test_resolve_fallback():
    db = FakeStudentDb(students={})
    current_user = {"user_type": "student", "user_id": "some-legacy-id"}
    result = await resolve_business_student_id(current_user, db)
    assert result == "some-legacy-id"


class FakeMeetingDb:
    def __init__(self, meetings=None, students=None):
        self.meetings = meetings or {}
        self.students = students or {}
        self.updates = []

    async def mongo_find_one(self, collection, query):
        if collection == "meetings":
            if "meeting_id" in query:
                return self.meetings.get(query["meeting_id"], {}).copy()
        if collection == "students":
            if "_id" in query:
                from bson import ObjectId
                oid = query["_id"]
                for doc in self.students.values():
                    if doc.get("_id") == oid:
                        return doc.copy()
        return None

    async def mongo_update_one(self, collection, query, update):
        self.updates.append((collection, query, update))
        return True


def test_online_class_router_student_uses_resolved_id():
    asyncio.run(_test_online_class_router_student())


async def _test_online_class_router_student():
    from bson import ObjectId
    oid = ObjectId()
    db = FakeMeetingDb(
        meetings={
            "MTG1": {
                "meeting_id": "MTG1",
                "status": "active",
                "tutor_id": "tutor-1",
                "invited_student_ids": ["STU_Resolved_999"],
            },
        },
        students={"stu1": {"_id": oid, "student_id": "STU_Resolved_999"}},
    )
    current_user = {"user_type": "student", "user_id": str(oid)}
    student_id = await resolve_business_student_id(current_user, db)
    assert student_id == "STU_Resolved_999"

    meeting = await db.mongo_find_one("meetings", {"meeting_id": "MTG1"})
    assert student_id in meeting.get("invited_student_ids", [])


def test_resolve_business_student_id_with_objectid_instance():
    asyncio.run(_test_resolve_with_objectid_instance())


async def _test_resolve_with_objectid_instance():
    from bson import ObjectId
    oid = ObjectId()
    db = FakeStudentDb(students={
        "stu1": {"_id": oid, "student_id": "STU_RawOid_111"},
    })
    current_user = {"user_type": "student", "user_id": oid}
    result = await resolve_business_student_id(current_user, db)
    assert result == "STU_RawOid_111"


def test_notification_recipient_ids_are_objectid_strings():
    asyncio.run(_test_notification_recipient_ids_are_objectid_strings())


async def _test_notification_recipient_ids_are_objectid_strings():
    from bson import ObjectId
    oid1 = ObjectId()
    oid2 = ObjectId()
    db = FakeStudentDb(students={
        "s1": {"_id": oid1, "student_id": "STU_Alice_001"},
        "s2": {"_id": oid2, "student_id": "STU_Bob_002"},
    })
    result = await resolve_notification_recipient_ids(
        db, ["STU_Alice_001", "STU_Bob_002"]
    )
    assert len(result) == 2
    assert str(oid1) in result
    assert str(oid2) in result
    assert "STU_Alice_001" not in result
    assert "STU_Bob_002" not in result


def test_notification_recipient_ids_fallback_for_missing_docs():
    asyncio.run(_test_notification_recipient_ids_fallback_for_missing_docs())


async def _test_notification_recipient_ids_fallback_for_missing_docs():
    from bson import ObjectId
    oid1 = ObjectId()
    db = FakeStudentDb(students={
        "s1": {"_id": oid1, "student_id": "STU_Present_001"},
    })
    result = await resolve_notification_recipient_ids(
        db, ["STU_Present_001", "STU_Missing_999"]
    )
    assert len(result) == 2
    assert str(oid1) in result
    assert "STU_Missing_999" in result


def test_notification_recipient_ids_empty_input():
    asyncio.run(_test_notification_recipient_ids_empty_input())


async def _test_notification_recipient_ids_empty_input():
    db = FakeStudentDb()
    result = await resolve_notification_recipient_ids(db, [])
    assert result == []


def test_canvas_request_defaults_to_joined_students():
    from api.v1.online_class.router import _validate_requested_student_ids

    meeting = {
        "invited_student_ids": ["STU_2", "STU_1", "STU_3"],
        "joined_student_ids": ["STU_1"],
    }
    assert _validate_requested_student_ids(meeting, None) == ["STU_1"]


def test_canvas_request_defaults_to_empty_when_no_students_joined():
    from api.v1.online_class.router import _validate_requested_student_ids

    meeting = {"invited_student_ids": ["STU_2", "STU_1"], "joined_student_ids": []}
    assert _validate_requested_student_ids(meeting, None) == []


def test_canvas_request_rejects_uninvited_student():
    from fastapi import HTTPException
    from api.v1.online_class.router import _validate_requested_student_ids

    meeting = {"invited_student_ids": ["STU_1"]}
    with pytest.raises(HTTPException) as exc:
        _validate_requested_student_ids(meeting, ["STU_1", "STU_2"])
    assert exc.value.status_code == 403
    assert "STU_2" in exc.value.detail


def test_expired_canvas_request_is_ended_and_ignored():
    asyncio.run(_test_expired_canvas_request_is_ended_and_ignored())


async def _test_expired_canvas_request_is_ended_and_ignored():
    from api.v1.online_class.router import (
        CANVAS_SHARE_REQUESTS_COLLECTION,
        _get_active_canvas_request,
    )

    db = FakeDb()
    await db.mongo_insert_one(
        CANVAS_SHARE_REQUESTS_COLLECTION,
        {
            "meeting_id": "MTG1",
            "status": "active",
            "requested_student_ids": ["STU_1"],
            "updated_at": datetime.utcnow() - timedelta(minutes=10),
        },
    )

    result = await _get_active_canvas_request(db, "MTG1")

    assert result is None
    stored = db.collections[CANVAS_SHARE_REQUESTS_COLLECTION][0]
    assert stored["status"] == "expired"
    assert stored["ended_at"] is not None


def test_canvas_provider_details_requires_jwt_for_private_canvas_rooms(monkeypatch):
    import importlib
    from fastapi import HTTPException
    router_module = importlib.import_module("api.v1.online_class.router")

    monkeypatch.setenv("ONLINE_CLASS_JITSI_DOMAIN", "class.stoody.in")
    monkeypatch.setenv("ONLINE_CLASS_JITSI_JWT_ENABLED", "false")
    monkeypatch.delenv("ONLINE_CLASS_JITSI_BASE_URL", raising=False)
    monkeypatch.delenv("ONLINE_CLASS_JITSI_JWT_SECRET", raising=False)
    provider = JitsiProviderService()
    monkeypatch.setattr(router_module, "jitsi_provider_service", provider)

    with pytest.raises(HTTPException) as exc:
        router_module._canvas_provider_details(
            "stoody-MTG1-canvas-student-STU-1",
            {"user_type": "tutor", "tutor_id": "tutor-1", "name": "Tutor"},
            moderator=True,
        )

    assert exc.value.status_code == 503
    assert "JWT" in exc.value.detail


def test_monitoring_page_filter_uses_meeting_start_and_page_key_round_trips():
    from api.v1.online_class.router import (
        _build_monitoring_page_meta,
        _decode_monitoring_page_key,
        _filter_canvas_pages_since_session_start,
    )

    started_at = datetime(2026, 6, 12, 9, 0, 0)
    before_start = started_at - timedelta(minutes=5)
    after_start = started_at + timedelta(minutes=2)
    pages = [
        {
            "copy_id": "online-MTG1",
            "book_type": "MS",
            "page_number": 1,
            "stroke_count": 2,
            "first_activity": before_start.timestamp() * 1000,
            "last_activity": before_start.timestamp() * 1000,
        },
        {
            "copy_id": "online-MTG1",
            "book_type": "MS",
            "page_number": 2,
            "stroke_count": 3,
            "first_activity": before_start.timestamp() * 1000,
            "last_activity": after_start.timestamp() * 1000,
        },
        {
            "copy_id": "online-MTG1",
            "book_type": "LN",
            "page_number": 3,
            "stroke_count": 1,
            "last_modified": after_start,
        },
    ]

    filtered = _filter_canvas_pages_since_session_start(pages, started_at)

    assert [page["page_number"] for page in filtered] == [2, 3]
    meta = _build_monitoring_page_meta(filtered[0])
    assert meta["page_key"]
    assert meta["copy_id"] == "online-MTG1"
    assert meta["book_type"] == "MS"
    assert meta["page_number"] == 2
    assert _decode_monitoring_page_key(meta["page_key"]) == {
        "copy_id": "online-MTG1",
        "book_type": "MS",
        "page_number": 2,
    }


def test_online_class_notes_list_returns_invited_classes_with_page_counts():
    asyncio.run(_test_online_class_notes_list_returns_invited_classes_with_page_counts())


def test_online_class_copy_id_is_valid_virtual_canvas_scope():
    asyncio.run(_test_online_class_copy_id_is_valid_virtual_canvas_scope())


async def _test_online_class_copy_id_is_valid_virtual_canvas_scope():
    from api.v1.copy_sets_async import resolve_copy_id

    db = FakeDb()
    resolved = await resolve_copy_id(
        "online-MTG3RFZNZYD",
        {"user_type": "tutor", "user_id": "tutor-1", "tutor_id": "tutor-1"},
        db,
    )

    assert resolved == "online-MTG3RFZNZYD"


async def _test_online_class_notes_list_returns_invited_classes_with_page_counts():
    from starlette.requests import Request
    from api.v1.online_class.router import api_list_online_class_notes

    db = FakeDb()
    db.collections["meetings"].append({
        "meeting_id": "MTG1",
        "status": "ended",
        "tutor_id": "tutor-1",
        "tutor_name": "Teacher",
        "topic": "Algebra class",
        "subject": "Math",
        "standard": "10",
        "section": "A",
        "admin_id": "admin-1",
        "invited_student_ids": ["STU_1"],
        "scheduled_at": datetime(2026, 6, 12, 9, 0, 0),
        "started_at": datetime(2026, 6, 12, 9, 5, 0),
        "ended_at": datetime(2026, 6, 12, 10, 0, 0),
    })
    db.collections["meetings"].append({
        "meeting_id": "MTG3",
        "status": "active",
        "tutor_id": "tutor-1",
        "tutor_name": "Teacher",
        "topic": "Geometry class",
        "subject": "Math",
        "standard": "10",
        "section": "A",
        "admin_id": "admin-1",
        "invited_student_ids": ["STU_1"],
        "scheduled_at": datetime(2026, 6, 12, 8, 0, 0),
        "started_at": datetime(2026, 6, 12, 8, 5, 0),
    })
    db.collections["canvas_pages"].extend([
        {
            "user_id": "tutor-1",
            "copy_id": "online-MTG1",
            "book_type": "MS",
            "page_number": 0,
            "stroke_count": 2,
            "first_activity": 1710000000000,
            "last_activity": 1710000001000,
            "strokes": [{"id": "s1"}, {"id": "s2"}],
        },
        {
            "user_id": "tutor-1",
            "copy_id": "online-MTG2",
            "book_type": "MS",
            "page_number": 0,
            "stroke_count": 5,
            "strokes": [{"id": "other"}],
        },
        {
            "user_id": "tutor-1",
            "copy_id": "online-MTG1",
            "book_type": "MS",
            "page_number": 1,
            "stroke_count": 0,
            "strokes": [],
        },
    ])
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/api/v1/online-class/notes",
        "headers": [(b"host", b"testserver")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "scheme": "http",
    })

    response = await api_list_online_class_notes(
        request=request,
        current_user={"user_type": "student", "student_id": "STU_1", "admin_id": "admin-1"},
        db=db,
    )

    assert len(response.classes) == 2
    classes_by_id = {item.meeting_id: item for item in response.classes}
    item = classes_by_id["MTG1"]
    assert item.meeting_id == "MTG1"
    assert item.copy_id == "online-MTG1"
    assert item.topic == "Algebra class"
    assert item.page_count == 1
    assert item.stroke_count == 2

    empty_item = classes_by_id["MTG3"]
    assert empty_item.copy_id == "online-MTG3"
    assert empty_item.topic == "Geometry class"
    assert empty_item.page_count == 0
    assert empty_item.stroke_count == 0


def test_online_class_note_page_fetch_scopes_to_teacher_meeting_copy():
    asyncio.run(_test_online_class_note_page_fetch_scopes_to_teacher_meeting_copy())


async def _test_online_class_note_page_fetch_scopes_to_teacher_meeting_copy():
    from fastapi import HTTPException
    from api.v1.online_class.router import _build_monitoring_page_meta, _get_teacher_online_class_page

    db = FakeDb()
    meeting = {"meeting_id": "MTG1", "status": "ended", "tutor_id": "tutor-1"}
    page = {
        "user_id": "tutor-1",
        "copy_id": "online-MTG1",
        "book_type": "MS",
        "page_number": 0,
        "stroke_count": 1,
        "strokes": [{"id": "s1", "points": [[1, 2, 0.5]]}],
    }
    db.collections["canvas_pages"].append(page)

    meta = _build_monitoring_page_meta(page)
    result = await _get_teacher_online_class_page(db, meeting, meta["page_key"])

    assert result["copy_id"] == "online-MTG1"
    assert result["page_key"] == meta["page_key"]
    assert result["strokes"][0]["id"] == "s1"

    wrong_page_key = _build_monitoring_page_meta({
        "copy_id": "online-MTG2",
        "book_type": "MS",
        "page_number": 0,
    })["page_key"]
    with pytest.raises(HTTPException) as exc:
        await _get_teacher_online_class_page(db, meeting, wrong_page_key)
    assert exc.value.status_code == 404


def test_resolve_student_canvas_user_ids_includes_backend_identity_variants():
    asyncio.run(_test_resolve_student_canvas_user_ids())


async def _test_resolve_student_canvas_user_ids():
    from bson import ObjectId
    from api.v1.online_class.router import _resolve_student_canvas_user_ids

    oid = ObjectId()
    db = FakeDb()
    db.collections["students"].append({
        "_id": oid,
        "student_id": "STU_Lavyansh_536995",
        "username": "lavyansh",
    })

    ids = await _resolve_student_canvas_user_ids(db, "STU_Lavyansh_536995")

    assert "STU_Lavyansh_536995" in ids
    assert "lavyansh" in ids
    assert str(oid) in [str(value) for value in ids]
    assert oid in ids


def test_teacher_canvas_mode_defaults_live_and_persists_stream():
    asyncio.run(_test_teacher_canvas_mode_defaults_live_and_persists_stream())


async def _test_teacher_canvas_mode_defaults_live_and_persists_stream():
    from api.v1.online_class.router import (
        _get_teacher_canvas_mode,
        _set_teacher_canvas_mode,
    )

    db = FakeDb()
    db.collections["meetings"].append({
        "meeting_id": "MTG1",
        "status": "active",
        "tutor_id": "tutor-1",
    })

    default_mode = await _get_teacher_canvas_mode(db, "MTG1")
    assert default_mode["mode"] == "live"

    updated = await _set_teacher_canvas_mode(db, "MTG1", "stream", "tutor-1")
    assert updated["mode"] == "stream"
    assert updated["updated_by"] == "tutor-1"

    stored = await _get_teacher_canvas_mode(db, "MTG1")
    assert stored["mode"] == "stream"


def test_teacher_live_canvas_events_upsert_through_online_class_facade():
    asyncio.run(_test_teacher_live_canvas_events_upsert())


async def _test_teacher_live_canvas_events_upsert():
    from api.v1.online_class.router import _upsert_teacher_live_canvas_events
    from api.v1.strokes_async import CanvasPageUpsert

    db = FakeDb()
    meeting = {
        "meeting_id": "MTG1",
        "status": "active",
        "tutor_id": "tutor-1",
        "admin_id": "admin-1",
    }
    current_user = {
        "user_type": "tutor",
        "tutor_id": "tutor-1",
        "user_id": "oid-tutor-1",
        "username": "tutor_user",
        "admin_id": "admin-1",
    }
    page = CanvasPageUpsert(
        book_type="ms",
        page_number=2,
        copy_id="online-MTG1",
        strokes=[{
            "id": "stroke-1",
            "points": [[1, 2, 0.5], [3, 4, 0.6]],
            "strokeWidth": 1.5,
            "color": "#111111",
            "tool": "pen",
            "timestamp": 1710000000000,
        }],
        client_last_modified=1710000000000,
        first_activity=1710000000000,
        last_activity=1710000000100,
    )

    result = await _upsert_teacher_live_canvas_events(db, meeting, current_user, [page])

    assert result == {
        "success": True,
        "upserted": 1,
        "modified": 0,
        "count": 1,
    }
    assert len(db.collections["canvas_pages"]) == 1
    stored = db.collections["canvas_pages"][0]
    assert stored["user_id"] == "tutor_user"
    assert stored["admin_id"] == "admin-1"
    assert stored["meeting_id"] == "MTG1"
    assert stored["copy_id"] == "online-MTG1"
    assert stored["book_type"] == "MS"
    assert stored["page_number"] == 2
    assert stored["source"] == "online_class_teacher_live"
    assert stored["stroke_count"] == 1


def test_teacher_live_canvas_events_merge_duplicate_strokes():
    asyncio.run(_test_teacher_live_canvas_events_merge_duplicate_strokes())


async def _test_teacher_live_canvas_events_merge_duplicate_strokes():
    from api.v1.online_class.router import _upsert_teacher_live_canvas_events
    from api.v1.strokes_async import CanvasPageUpsert

    db = FakeDb()
    meeting = {"meeting_id": "MTG1", "status": "active", "tutor_id": "tutor-1"}
    current_user = {
        "user_type": "tutor",
        "tutor_id": "tutor-1",
        "user_id": "oid-tutor-1",
        "username": "tutor_user",
    }
    first_page = CanvasPageUpsert(
        book_type="MS",
        page_number=1,
        copy_id="online-MTG1",
        strokes=[{"id": "same", "points": [[1, 1, 0.5]]}],
    )
    second_page = CanvasPageUpsert(
        book_type="MS",
        page_number=1,
        copy_id="online-MTG1",
        strokes=[
            {"id": "same", "points": [[1, 1, 0.5]]},
            {"id": "new", "points": [[2, 2, 0.5]]},
        ],
    )

    await _upsert_teacher_live_canvas_events(db, meeting, current_user, [first_page])
    result = await _upsert_teacher_live_canvas_events(db, meeting, current_user, [second_page])

    assert result["modified"] == 1
    stored = db.collections["canvas_pages"][0]
    assert [stroke["id"] for stroke in stored["strokes"]] == ["same", "new"]
    assert stored["stroke_count"] == 2


class AnalysisFakeDb(FakeDb):
    def __init__(self):
        super().__init__()

    async def mongo_find(self, collection, query):
        return []


def _mock_text_call_success(db, current_user, prompt, **kwargs):
    return {
        "success": True,
        "response": '{"score": 0.8, "is_correct": false, "student_answer": "x=1", "work_shown": "solved", "what_went_wrong": "sign error", "correct_solution": "x=3"}',
    }


def _mock_vision_call_success(db, current_user, images, prompt, **kwargs):
    return {
        "success": True,
        "response": '{"score": 1.0, "is_correct": true, "student_answer": "42", "work_shown": "calculated", "what_went_wrong": null, "correct_solution": "42"}',
    }


def _mock_text_call_failure(db, current_user, prompt, **kwargs):
    raise RuntimeError("LLM service unavailable")


def test_analysis_success_sets_fields_on_submission():
    asyncio.run(_test_analysis_success())


async def _test_analysis_success():
    from unittest.mock import patch

    db = AnalysisFakeDb()
    current_user = {"db_name": "test_db"}
    lock = {"question_text": "Solve x + 1 = 2"}
    submission = {
        "submission_id": "sub-test123",
        "meeting_id": "MTG1",
        "lock_id": "lck-1",
        "student_id": "stu-1",
        "canvas_pages": [],
        "answer_text": "x = 1",
        "analysis_status": "pending",
    }

    with patch("api.v1.practice_async._gate_text_call", side_effect=_mock_text_call_success), \
         patch("api.v1.practice_async._gate_vision_call", side_effect=_mock_vision_call_success):
        from services.online_class.analysis_service import run_submission_analysis
        result = await run_submission_analysis(db, current_user, lock, submission)

    assert result["analysis_status"] == "completed"
    assert result["score"] == 0.8
    assert result["is_correct"] is False
    assert result["student_answer"] == "x=1"
    assert result["work_shown"] == "solved"
    assert result["what_went_wrong"] == "sign error"
    assert result["correct_solution"] == "x=3"
    assert result["analysis_completed_at"] is not None
    assert result["analysis_error"] is None


def test_analysis_failure_preserves_canvas_and_marks_failed():
    asyncio.run(_test_analysis_failure())


async def _test_analysis_failure():
    from unittest.mock import patch

    db = AnalysisFakeDb()
    current_user = {"db_name": "test_db"}
    lock = {"question_text": "Solve for x"}
    submission = {
        "submission_id": "sub-fail001",
        "meeting_id": "MTG1",
        "lock_id": "lck-1",
        "student_id": "stu-1",
        "canvas_pages": ["data:image/png;base64,ABC123"],
        "question_page_refs": {"pages": [3]},
        "answer_text": "my answer",
        "analysis_status": "pending",
    }

    with patch("api.v1.practice_async._gate_vision_call", side_effect=_mock_text_call_failure), \
         patch("api.v1.practice_async._gate_text_call", side_effect=_mock_text_call_failure):
        from services.online_class.analysis_service import run_submission_analysis
        result = await run_submission_analysis(db, current_user, lock, submission)

    assert result["analysis_status"] == "failed"
    assert result["analysis_error"] is not None
    assert result["analysis_failed_at"] is not None
    assert result["canvas_pages"] == ["data:image/png;base64,ABC123"]
    assert result["question_page_refs"] == {"pages": [3]}
    assert result["answer_text"] == "my answer"
    assert result["score"] is None
    assert result["is_correct"] is None


def test_duplicate_submission_resets_stale_analysis_fields():
    asyncio.run(_test_duplicate_resets_stale_analysis_fields())


async def _test_duplicate_resets_stale_analysis_fields():
    db = AnalysisFakeDb()

    first = await create_or_update_submission(
        db=db,
        meeting_id="MTG1",
        lock_id="lck-1",
        student_id="stu-1",
        canvas_pages=[],
        question_page_refs={"pages": [1]},
        answer_text="4",
        time_spent=10,
        client_submitted_at=datetime.utcnow(),
    )

    stale_fields = {
        "analysis_status": "completed",
        "score": 1.0,
        "is_correct": True,
        "student_answer": "4",
        "work_shown": "old work",
        "what_went_wrong": None,
        "correct_solution": "2 + 2 = 4",
        "analysis_error": None,
        "analysis_completed_at": datetime.utcnow(),
        "analysis_failed_at": None,
    }
    await db.mongo_update_one(
        "online_class_submissions",
        {"submission_id": first["submission_id"]},
        {"$set": stale_fields},
    )

    second = await create_or_update_submission(
        db=db,
        meeting_id="MTG1",
        lock_id="lck-1",
        student_id="stu-1",
        canvas_pages=["img-b"],
        question_page_refs={"pages": [2]},
        answer_text="5",
        time_spent=20,
        client_submitted_at=datetime.utcnow(),
    )

    assert second["submission_id"] == first["submission_id"]
    assert second["canvas_pages"] == ["img-b"]
    assert second["answer_text"] == "5"
    assert second["analysis_status"] == "pending"
    assert second["score"] is None
    assert second["is_correct"] is None
    assert second["student_answer"] is None
    assert second["work_shown"] is None
    assert second["correct_solution"] is None
    assert len(db.collections["online_class_submissions"]) == 1
