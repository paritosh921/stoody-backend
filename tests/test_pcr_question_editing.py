from __future__ import annotations

from mongomock_motor import AsyncMongoMockClient
import pytest
from fastapi import HTTPException


class _ScopedDatabase:
    def __init__(self, tenant_db):
        self.tenant_db = tenant_db

    async def mongo_find_one(self, collection_name, query, projection=None):
        return await self.tenant_db[collection_name].find_one(query, projection)

    async def mongo_find(self, collection_name, query, projection=None):
        return await self.tenant_db[collection_name].find(
            query,
            projection,
        ).to_list(length=1000)

    async def mongo_update_one(self, collection_name, query, update):
        result = await self.tenant_db[collection_name].update_one(query, update)
        return result.modified_count > 0


async def _seed_pcr_question(tenant_db, *, finalized: bool) -> None:
    await tenant_db["documents"].insert_one(
        {
            "document_id": "pcr-paper",
            "document_type": "Test Series",
            "exam_mode": "pcr",
            "exam_finalized": finalized,
        }
    )
    await tenant_db["questions"].insert_one(
        {
            "id": "pcr-question-1",
            "document_id": "pcr-paper",
            "text": "Original question",
            "question_type": "subjective",
            "points": 4,
            "marking_criteria": [
                {
                    "criterion_id": "criterion-1",
                    "description": "Original criterion",
                    "max_marks": 4,
                }
            ],
        }
    )


def _admin_user():
    return {
        "user_id": "admin-1",
        "user_type": "admin",
        "db_name": "skb_test",
    }


@pytest.mark.asyncio
async def test_unfinalized_pcr_question_and_current_marking_plan_are_saved():
    from api.v1.pdf_async import update_question

    tenant_db = AsyncMongoMockClient()["skb_test"]
    await _seed_pcr_question(tenant_db, finalized=False)
    db = _ScopedDatabase(tenant_db)

    await update_question.__wrapped__(
        request=None,
        question_id="pcr-question-1",
        question_data={
            "text": "Updated question",
            "question_type": "subjective",
            "points": 4,
            "marking_criteria": [
                {
                    "criterion_id": "criterion-1",
                    "description": "Updated teacher criterion",
                    "max_marks": 4,
                    "acceptable_evidence": "Equivalent reasoning is accepted",
                }
            ],
        },
        current_user=_admin_user(),
        db=db,
    )

    stored = await tenant_db["questions"].find_one({"id": "pcr-question-1"})
    assert stored["text"] == "Updated question"
    assert stored["question_type"] == "subjective"
    assert stored["marking_criteria"] == [
        {
            "criterion_id": "criterion-1",
            "description": "Updated teacher criterion",
            "max_marks": 4.0,
            "acceptable_evidence": "Equivalent reasoning is accepted",
        }
    ]


@pytest.mark.asyncio
async def test_finalized_pcr_question_catalog_cannot_be_mutated_in_place():
    from api.v1.pdf_async import update_question

    tenant_db = AsyncMongoMockClient()["skb_test"]
    await _seed_pcr_question(tenant_db, finalized=True)
    db = _ScopedDatabase(tenant_db)

    with pytest.raises(HTTPException) as exc:
        await update_question.__wrapped__(
            request=None,
            question_id="pcr-question-1",
            question_data={"text": "Unsafe in-place change"},
            current_user=_admin_user(),
            db=db,
        )

    assert exc.value.status_code == 409
    stored = await tenant_db["questions"].find_one({"id": "pcr-question-1"})
    assert stored["text"] == "Original question"


@pytest.mark.asyncio
async def test_question_mark_edit_preserves_authoritative_paper_total_and_records_mismatch():
    from api.v1.pdf_async import update_question

    tenant_db = AsyncMongoMockClient()["skb_test"]
    await _seed_pcr_question(tenant_db, finalized=False)
    await tenant_db["documents"].update_one(
        {"document_id": "pcr-paper"},
        {
            "$set": {
                "total_points": 4,
                "total_points_source": "visual_question_marks",
                "marks_extraction_summary": {
                    "expected_total": 4,
                    "calculated_total": 4,
                    "reconciled": True,
                },
            }
        },
    )
    db = _ScopedDatabase(tenant_db)

    await update_question.__wrapped__(
        request=None,
        question_id="pcr-question-1",
        question_data={"points": 3},
        current_user=_admin_user(),
        db=db,
    )

    stored_question = await tenant_db["questions"].find_one({"id": "pcr-question-1"})
    stored_document = await tenant_db["documents"].find_one({"document_id": "pcr-paper"})
    assert stored_question["points"] == 3
    assert stored_document["total_points"] == 4
    assert stored_document["total_points_source"] == "visual_question_marks"
    assert stored_document["marks_extraction_summary"]["calculated_total"] == 3
    assert stored_document["marks_extraction_summary"]["expected_total"] == 4
    assert stored_document["marks_extraction_summary"]["reconciled"] is False
    assert stored_document["marks_review_required"] is True


@pytest.mark.asyncio
async def test_subjective_to_objective_transition_replaces_the_grading_authority():
    from api.v1.pdf_async import update_question

    tenant_db = AsyncMongoMockClient()["skb_test"]
    await _seed_pcr_question(tenant_db, finalized=False)
    await tenant_db["questions"].update_one(
        {"id": "pcr-question-1"},
        {
            "$set": {
                "reference_solution": "An old worked answer",
                "rubric": "An old Subjective rubric",
                "assessment_units": [
                    {
                        "unit_id": "unit-1",
                        "label": "Whole question",
                        "prompt": "Original question",
                        "max_marks": 4,
                        "reference_solution": "An old worked answer",
                        "marking_criteria": [],
                    }
                ],
                "correct_answer": "A",
            }
        },
    )
    db = _ScopedDatabase(tenant_db)

    await update_question.__wrapped__(
        request=None,
        question_id="pcr-question-1",
        question_data={
            "question_type": "mcq",
            "options": ["One", "Two", "Three"],
            "enhanced_options": [
                {"label": "A", "type": "text", "content": "One"},
                {"label": "B", "type": "text", "content": "Two"},
                {"label": "C", "type": "text", "content": "Three"},
            ],
            "correct_answer": "B",
            "penalty": 1,
        },
        current_user=_admin_user(),
        db=db,
    )

    stored = await tenant_db["questions"].find_one({"id": "pcr-question-1"})
    assert stored["question_type"] == "mcq"
    assert stored["correct_answer"] == "B"
    assert stored["penalty"] == 1
    assert stored["reference_solution"] is None
    assert stored["rubric"] is None
    assert stored["marking_criteria"] == []
    assert stored["assessment_units"] == []
    assert stored["response_selection"] is None
    assert stored["marking_plan_generation_error"] is None


@pytest.mark.asyncio
async def test_objective_to_subjective_transition_clears_key_and_penalty_but_keeps_options():
    from api.v1.pdf_async import update_question

    tenant_db = AsyncMongoMockClient()["skb_test"]
    await _seed_pcr_question(tenant_db, finalized=False)
    await tenant_db["questions"].update_one(
        {"id": "pcr-question-1"},
        {
            "$set": {
                "question_type": "mcq",
                "options": ["One", "Two"],
                "correct_answer": "B",
                "penalty": 1,
            }
        },
    )
    db = _ScopedDatabase(tenant_db)

    await update_question.__wrapped__(
        request=None,
        question_id="pcr-question-1",
        question_data={
            "question_type": "subjective",
            # A direct/old client cannot retain an Objective key accidentally.
            "correct_answer": "B",
            "penalty": 1,
        },
        current_user=_admin_user(),
        db=db,
    )

    stored = await tenant_db["questions"].find_one({"id": "pcr-question-1"})
    assert stored["question_type"] == "subjective"
    assert stored["correct_answer"] == ""
    assert stored["penalty"] == 0
    assert stored["options"] == ["One", "Two"]
    assert stored["marking_criteria"] == []
    assert stored["marking_plan_generation_status"] == "not_generated"
    assert "prepare and review" in stored["marking_plan_generation_error"]


@pytest.mark.asyncio
async def test_objective_answer_must_reference_a_saved_option_when_options_exist():
    from api.v1.pdf_async import update_question

    tenant_db = AsyncMongoMockClient()["skb_test"]
    await _seed_pcr_question(tenant_db, finalized=False)
    db = _ScopedDatabase(tenant_db)

    with pytest.raises(HTTPException) as exc:
        await update_question.__wrapped__(
            request=None,
            question_id="pcr-question-1",
            question_data={
                "question_type": "mcq",
                "options": ["One", "Two"],
                "correct_answer": "C",
            },
            current_user=_admin_user(),
            db=db,
        )

    assert exc.value.status_code == 422
    assert "not one of the saved options" in str(exc.value.detail)
    stored = await tenant_db["questions"].find_one({"id": "pcr-question-1"})
    assert stored["question_type"] == "subjective"


@pytest.mark.asyncio
async def test_objective_alias_is_not_treated_as_a_grading_contract_change():
    from api.v1.pdf_async import update_question

    tenant_db = AsyncMongoMockClient()["skb_test"]
    await _seed_pcr_question(tenant_db, finalized=False)
    await tenant_db["questions"].update_one(
        {"id": "pcr-question-1"},
        {
            "$set": {
                "question_type": "objective",
                "options": ["One", "Two"],
                "correct_answer": "B",
            }
        },
    )
    db = _ScopedDatabase(tenant_db)

    await update_question.__wrapped__(
        request=None,
        question_id="pcr-question-1",
        question_data={"question_type": "mcq"},
        current_user=_admin_user(),
        db=db,
    )

    stored = await tenant_db["questions"].find_one({"id": "pcr-question-1"})
    assert stored["question_type"] == "mcq"
    assert stored["correct_answer"] == "B"


@pytest.mark.asyncio
async def test_inline_option_image_is_persisted_and_invalidates_subjective_plan(monkeypatch):
    import api.v1.pdf_async as pdf_async

    tenant_db = AsyncMongoMockClient()["skb_test"]
    await _seed_pcr_question(tenant_db, finalized=False)
    await tenant_db["questions"].update_one(
        {"id": "pcr-question-1"},
        {
            "$set": {
                "assessment_units": [
                    {
                        "unit_id": "unit-1",
                        "label": "Whole question",
                        "prompt": "Original question",
                        "max_marks": 4,
                        "reference_solution": "Original answer",
                        "marking_criteria": [],
                    }
                ],
                "marking_plan_generation_status": "completed",
            }
        },
    )

    async def save_option_image(**_kwargs):
        return [
            {
                "id": "stored-option-image",
                "filename": "stored-option-image.png",
                "path": "uploads/pdf_images/pcr-paper/stored-option-image.png",
                "url": "/api/v1/images/stored-option-image",
                "size": 68,
            }
        ]

    monkeypatch.setattr(pdf_async, "save_image_to_disk", save_option_image)
    tiny_png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
        "ASsJTYQAAAAASUVORK5CYII="
    )

    await pdf_async.update_question.__wrapped__(
        request=None,
        question_id="pcr-question-1",
        question_data={
            "enhanced_options": [
                {
                    "id": "opt_temp",
                    "label": "A",
                    "type": "image",
                    "content": f"data:image/png;base64,{tiny_png}",
                }
            ]
        },
        current_user=_admin_user(),
        db=_ScopedDatabase(tenant_db),
    )

    stored = await tenant_db["questions"].find_one({"id": "pcr-question-1"})
    stored_option = stored["enhanced_options"][0]
    assert stored_option["image_id"] == "stored-option-image"
    assert stored_option["content"] == "/api/v1/images/stored-option-image"
    assert "base64Data" not in stored_option
    assert stored["assessment_units"] == []
    assert stored["marking_plan_generation_status"] == "not_generated"
    assert "structure changed" in stored["marking_plan_generation_error"].lower()


@pytest.mark.asyncio
async def test_new_question_image_storage_failure_is_not_silently_accepted(monkeypatch):
    import api.v1.pdf_async as pdf_async

    tenant_db = AsyncMongoMockClient()["skb_test"]
    await _seed_pcr_question(tenant_db, finalized=False)

    async def fail_image_storage(**_kwargs):
        return []

    monkeypatch.setattr(pdf_async, "save_image_to_disk", fail_image_storage)
    tiny_png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
        "ASsJTYQAAAAASUVORK5CYII="
    )

    with pytest.raises(HTTPException) as exc:
        await pdf_async.update_question.__wrapped__(
            request=None,
            question_id="pcr-question-1",
            question_data={
                "question_figures": [
                    {
                        "id": "question_image_temp",
                        "base64Data": f"data:image/png;base64,{tiny_png}",
                        "type": "diagram",
                    }
                ]
            },
            current_user=_admin_user(),
            db=_ScopedDatabase(tenant_db),
        )

    assert exc.value.status_code == 422
    assert "could not be stored" in str(exc.value.detail)
    stored = await tenant_db["questions"].find_one({"id": "pcr-question-1"})
    assert stored.get("question_figures") is None
