from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest
from fastapi import HTTPException


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_test"]


def _admin_user() -> Dict[str, Any]:
    return {"user_id": "admin-1", "user_type": "admin", "db_name": "skb_test"}


def _tutor_user(tutor_id: str = "tut-1") -> Dict[str, Any]:
    return {
        "user_id": f"user-{tutor_id}",
        "user_type": "tutor",
        "tutor_id": tutor_id,
        "admin_id": "admin-1",
        "db_name": "skb_test",
    }


async def _seed_document(db, **overrides) -> None:
    doc = {
        "document_id": "doc-1",
        "title": "PCR Paper",
        "exam_mode": "pcr",
        "exam_finalized": True,
        "total_minutes": 45,
        "admin_id": "admin-1",
        "teacher_ids": ["tut-1"],
        "is_active": True,
    }
    doc.update(overrides)
    await db["documents"].insert_one(doc)
    await db["questions"].insert_one(
        {
            "id": f"{doc['document_id']}-q1",
            "document_id": doc["document_id"],
            "text": "Explain the approved test question.",
            "subject": "Science",
            "question_type": "mcq" if doc.get("exam_mode") == "dcr" else "subjective",
            "points": 4,
            "correct_answer": "A",
            "rubric": "Award marks for the approved key point.",
        }
    )


async def _create_exam(db, current_user, **body_overrides):
    from api.v1.exam_orch_async import ExamCreateRequest, create_exam

    body = ExamCreateRequest(
        exam_id=body_overrides.pop("exam_id", "exam-1"),
        exam_type=body_overrides.pop("exam_type", ""),
        prepared_document_id=body_overrides.pop("prepared_document_id", "doc-1"),
        **body_overrides,
    )
    with patch("api.v1.exam_orch_async._get_tenant_db", return_value=db):
        return await create_exam(body=body, current_user=current_user, db=None)


async def _list_exams(db, current_user):
    from api.v1.exam_orch_async import list_exams

    with patch("api.v1.exam_orch_async._get_tenant_db", return_value=db):
        return await list_exams(current_user=current_user, db=None)


async def _get_exam(db, exam_id: str, current_user):
    from api.v1.exam_orch_async import get_exam

    with patch("api.v1.exam_orch_async._get_tenant_db", return_value=db):
        return await get_exam(exam_id=exam_id, current_user=current_user, db=None)


async def _get_preflight(db, exam_id: str, current_user):
    from api.v1.exam_orch_async import get_preflight

    with patch("api.v1.exam_orch_async._get_tenant_db", return_value=db):
        return await get_preflight(exam_id=exam_id, current_user=current_user, db=None)


async def _update_exam_setup(db, exam_id: str, current_user, **fields):
    from api.v1.exam_orch_async import ExamSetupUpdateRequest, update_exam_setup

    with patch("api.v1.exam_orch_async._get_tenant_db", return_value=db):
        return await update_exam_setup(
            exam_id=exam_id,
            body=ExamSetupUpdateRequest(**fields),
            current_user=current_user,
            db=None,
        )


async def _ensure_camera_collection(db, current_user, document_id: str = "doc-1"):
    from api.v1.exam_orch_async import ensure_default_pcr_camera_collection

    with patch("api.v1.exam_orch_async._get_tenant_db", return_value=db):
        return await ensure_default_pcr_camera_collection(
            prepared_document_id=document_id,
            current_user=current_user,
            db=None,
        )


@pytest.mark.asyncio
async def test_finalized_pcr_activation_opens_one_camera_only_class_collection():
    db = _fresh_db()
    await _seed_document(
        db,
        standard="11",
        admin_id="admin-1",
        is_active=False,
    )
    await db["students"].insert_many(
        [
            {
                "student_id": "STU-11-A",
                "grade": "11",
                "admin_id": "admin-1",
                "is_active": True,
            },
            {
                "student_id": "STU-11-B",
                "grade": "11",
                "admin_id": "admin-1",
                "is_active": True,
            },
            {
                "student_id": "STU-INACTIVE",
                "grade": "11",
                "admin_id": "admin-1",
                "is_active": False,
            },
            {
                "student_id": "STU-OTHER-CLASS",
                "grade": "12",
                "admin_id": "admin-1",
                "is_active": True,
            },
            {
                "student_id": "STU-OTHER-INSTITUTE",
                "grade": "11",
                "admin_id": "admin-2",
                "is_active": True,
            },
        ]
    )

    first = await _ensure_camera_collection(db, _admin_user())
    second = await _ensure_camera_collection(db, _admin_user())

    assert first.exam_id == second.exam_id
    assert first.lifecycle_state == "in_progress"
    assert first.capture_mode == "camera"
    assert first.student_self_submission_enabled is True
    assert first.student_submission_max_pages == 40
    assert first.pen_bindings == {}
    assert set(first.roster) == {"STU-11-A", "STU-11-B"}
    assert await db["exampen_exams"].count_documents(
        {"prepared_document_id": "doc-1"}
    ) == 1


@pytest.mark.asyncio
async def test_create_exam_maps_prepared_document_to_tutor_owner():
    db = _fresh_db()
    await _seed_document(
        db,
        document_id="doc-1",
        title="DCR Final Paper",
        exam_mode="dcr",
        admin_id="admin-doc",
        teacher_ids=["tut-1", "tut-2"],
        total_minutes=90,
    )

    result = await _create_exam(
        db,
        _tutor_user("tut-1"),
        exam_type="pcr",
        prepared_document_id="doc-1",
        duration_minutes=15,
    )

    assert result.title == "DCR Final Paper"
    assert result.exam_type == "dcr"
    assert result.duration_minutes == 90
    assert result.admin_id == "admin-doc"
    assert result.teacher_ids == ["tut-1", "tut-2"]
    assert result.created_by_tutor_id == "tut-1"


@pytest.mark.asyncio
async def test_create_exam_persists_pen_bindings_for_hub_assignment():
    db = _fresh_db()
    await _seed_document(db)

    result = await _create_exam(
        db,
        _tutor_user("tut-1"),
        roster=["student-1"],
        pen_bindings={"aa:bb:cc:dd:ee:ff": "student-1"},
    )

    stored = await db["exampen_exams"].find_one({"exam_id": result.exam_id})
    fetched = await _get_exam(db, result.exam_id, _tutor_user("tut-1"))

    assert result.pen_bindings == {"AA:BB:CC:DD:EE:FF": "student-1"}
    assert stored["pen_bindings"] == {"AA:BB:CC:DD:EE:FF": "student-1"}
    assert fetched.pen_bindings == {"AA:BB:CC:DD:EE:FF": "student-1"}


@pytest.mark.asyncio
async def test_pcr_session_uses_server_id_and_session_scoped_immutable_metadata():
    db = _fresh_db()
    await _seed_document(db)

    result = await _create_exam(
        db,
        _tutor_user("tut-1"),
        exam_id="client-chosen-id",
        roster=["student-1"],
        pen_bindings={"aa:bb:cc:dd:ee:ff": "student-1"},
    )

    assert result.exam_id.startswith("exam-")
    assert result.exam_id != "client-chosen-id"
    assert result.paper_version_id

    session_question = await db["evalpen_questions"].find_one(
        {"exam_id": result.exam_id}
    )
    assert session_question["question_id"] == f"{result.exam_id}::doc-1-q1"
    assert session_question["source_question_id"] == "doc-1-q1"
    assert session_question["question_number"] == 1
    assert session_question["immutable"] is True
    assert session_question["reference_solution"] == "A"

    solution = await db["evalpen_solutions"].find_one(
        {"question_id": session_question["question_id"]}
    )
    assert solution["solution_source"] == "teacher"
    assert solution["reference_solution"] == "A"


@pytest.mark.asyncio
async def test_preflight_blocks_arm_until_hub_and_pen_bindings_are_ready():
    db = _fresh_db()
    await _seed_document(db)
    result = await _create_exam(
        db,
        _tutor_user("tut-1"),
        roster=["student-1"],
    )

    preflight = await _get_preflight(db, result.exam_id, _tutor_user("tut-1"))
    checks = {check.id: check for check in preflight.checks}

    assert not preflight.ready_to_arm
    assert checks["paper_snapshot"].ready
    assert checks["hub_assignment"].ready is False
    assert checks["pen_bindings"].ready is False


@pytest.mark.asyncio
async def test_draft_setup_can_be_repaired_without_replacing_immutable_paper():
    db = _fresh_db()
    await _seed_document(db)
    created = await _create_exam(
        db,
        _tutor_user("tut-1"),
        roster=["student-1"],
        pen_bindings={"AA:BB:CC:DD:EE:FF": "student-1"},
    )

    updated = await _update_exam_setup(
        db,
        created.exam_id,
        _tutor_user("tut-1"),
        roster=["student-2"],
        pen_bindings={"11:22:33:44:55:66": "student-2"},
        capture_mode="pen",
    )

    assert updated.exam_id == created.exam_id
    assert updated.paper_version_id == created.paper_version_id
    assert updated.roster == ["student-2"]
    assert updated.pen_bindings == {"11:22:33:44:55:66": "student-2"}


@pytest.mark.asyncio
async def test_dcr_uploading_session_does_not_require_pcr_processing_job():
    from api.v1.exam_orch_async import _ready_for_eval_issues
    from services.exampen_workflow import _maybe_mark_exam_ready_for_review

    db = _fresh_db()
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "dcr-session",
            "exam_type": "dcr",
            "lifecycle_state": "uploading",
            "roster": ["student-1"],
            "absent_student_ids": [],
        }
    )
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": "dcr-submission-1",
            "exam_id": "dcr-session",
            "student_id": "student-1",
        }
    )
    exam = await db["exampen_exams"].find_one({"exam_id": "dcr-session"})

    assert await _ready_for_eval_issues(db, exam) == []
    await _maybe_mark_exam_ready_for_review(db, "dcr-session")

    updated = await db["exampen_exams"].find_one({"exam_id": "dcr-session"})
    assert updated["lifecycle_state"] == "ready_for_eval"


@pytest.mark.asyncio
async def test_legacy_pcr_snapshot_can_freeze_an_accepted_answer_mapping():
    from services.exampen_paper_service import load_or_create_paper_snapshot

    db = _fresh_db()
    document = {
        "document_id": "legacy-pcr",
        "title": "Legacy PCR",
        "exam_mode": "pcr",
        "exam_finalized": True,
        "total_minutes": 45,
    }
    await db["documents"].insert_one(document)
    await db["questions"].insert_one(
        {
            "id": "legacy-q1",
            "document_id": "legacy-pcr",
            "text": "Explain the accepted solution.",
            "points": 5,
        }
    )
    await db["answer_question_mappings"].insert_one(
        {
            "document_id": "legacy-pcr",
            "mapping_id": "mapping-1",
            "question_id": "legacy-q1",
            "answer_text": "Teacher-reviewed answer.",
            "review_status": "accepted",
            "manual_review_required": False,
            "source": "answer_sheet",
        }
    )

    version, questions = await load_or_create_paper_snapshot(db, document)

    assert version["paper_version_id"]
    assert questions[0]["question"]["reference_solution"] == "Teacher-reviewed answer."
    stored_document = await db["documents"].find_one({"document_id": "legacy-pcr"})
    assert stored_document["exam_paper_version_id"] == version["paper_version_id"]
    assert stored_document["exam_snapshot_marking_plan"]["questions_using_approved_mapping"] == 1


@pytest.mark.asyncio
async def test_create_exam_rejects_invalid_pen_bindings():
    db = _fresh_db()
    await _seed_document(db)

    with pytest.raises(HTTPException) as exc:
        await _create_exam(
            db,
            _tutor_user("tut-1"),
            roster=["student-1"],
            pen_bindings={"not-a-mac": "student-1"},
        )
    assert exc.value.status_code == 400

    with pytest.raises(HTTPException) as exc:
        await _create_exam(
            db,
            _tutor_user("tut-1"),
            roster=["student-1"],
            pen_bindings={"AA:BB:CC:DD:EE:FF": "student-2"},
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_prepared_document_duration_rejects_bool_and_uses_total_minutes_fallback():
    db = _fresh_db()
    await _seed_document(db, duration_minutes=True, total_minutes=75)

    result = await _create_exam(
        db,
        _tutor_user("tut-1"),
        prepared_document_id="doc-1",
        duration_minutes=15,
    )

    assert result.duration_minutes == 75


@pytest.mark.asyncio
async def test_tutor_cannot_create_exam_from_other_tutor_paper():
    db = _fresh_db()
    await _seed_document(db, teacher_ids=["tut-other"])

    with pytest.raises(HTTPException) as exc:
        await _create_exam(db, _tutor_user("tut-1"))

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_unfinalized_or_untyped_document_is_not_exam_ready():
    db = _fresh_db()
    await _seed_document(db, document_id="draft", exam_finalized=False)
    await _seed_document(db, document_id="untyped", exam_mode=None)

    for document_id in ("draft", "untyped"):
        with pytest.raises(HTTPException) as exc:
            await _create_exam(db, _admin_user(), prepared_document_id=document_id)
        assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_tutor_list_and_get_are_scoped_to_visible_exams():
    db = _fresh_db()
    await db["exampen_exams"].insert_many(
        [
            {"exam_id": "created", "created_by_tutor_id": "tut-1", "teacher_ids": []},
            {"exam_id": "assigned", "teacher_ids": ["tut-1"]},
            {"exam_id": "open", "teacher_ids": []},
            {"exam_id": "hidden", "teacher_ids": ["tut-2"]},
        ]
    )

    visible = await _list_exams(db, _tutor_user("tut-1"))

    assert [exam.exam_id for exam in visible.items] == ["created", "assigned", "open"]
    assert (await _get_exam(db, "assigned", _tutor_user("tut-1"))).exam_id == "assigned"
    with pytest.raises(HTTPException) as exc:
        await _get_exam(db, "hidden", _tutor_user("tut-1"))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_admin_can_see_exam_hidden_from_tutor():
    db = _fresh_db()
    await db["exampen_exams"].insert_one(
        {"exam_id": "hidden", "teacher_ids": ["tut-2"], "admin_id": "admin-1"}
    )

    result = await _get_exam(db, "hidden", _admin_user())

    assert result.exam_id == "hidden"
