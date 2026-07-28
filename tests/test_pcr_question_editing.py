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
