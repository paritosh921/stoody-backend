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
        }

    async def mongo_find_one(self, collection, query):
        for doc in self.collections.get(collection, []):
            if all(doc.get(key) == value for key, value in query.items()):
                return doc.copy()
        return None

    async def mongo_insert_one(self, collection, doc):
        self.collections.setdefault(collection, []).append(doc.copy())
        return doc.get("lock_id") or doc.get("submission_id")

    async def mongo_update_one(self, collection, query, update):
        updates = update.get("$set", {})
        for doc in self.collections.get(collection, []):
            if all(doc.get(key) == value for key, value in query.items()):
                doc.update(updates)
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


def test_duplicate_submission_updates_existing_record():
    asyncio.run(_test_duplicate_submission_updates_existing_record())


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


def test_canvas_request_defaults_to_invited_students():
    from api.v1.online_class.router import _validate_requested_student_ids

    meeting = {"invited_student_ids": ["STU_2", "STU_1"]}
    assert _validate_requested_student_ids(meeting, None) == ["STU_2", "STU_1"]


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
