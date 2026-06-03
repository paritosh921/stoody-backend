import asyncio
from datetime import datetime

import pytest

from api.v1.online_class.locks import create_lock, end_lock
from api.v1.online_class.submissions import create_or_update_submission
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
    monkeypatch.setenv("ONLINE_CLASS_JITSI_BASE_URL", "https://class.stoody.in")
    monkeypatch.setenv("ONLINE_CLASS_JITSI_JWT_ENABLED", "true")

    provider = JitsiProviderService()
    details = provider.get_provider_details("MTG 123")

    assert details["configured"] is True
    assert details["domain"] == "class.stoody.in"
    assert details["room_name"] == "stoody-MTG-123"
    assert details["url"] == "https://class.stoody.in/stoody-MTG-123"
    assert details["token_required"] is True
    assert details["token"] is None
