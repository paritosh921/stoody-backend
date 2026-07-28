from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from mongomock_motor import AsyncMongoMockClient


class _ScopedDatabase:
    """Small adapter matching the tenant-scoped methods used by pdf_async."""

    def __init__(self, tenant_db):
        self.tenant_db = tenant_db

    async def mongo_find_one(self, collection_name, query, projection=None):
        return await self.tenant_db[collection_name].find_one(query, projection)

    async def mongo_find(
        self,
        collection_name,
        query,
        projection=None,
        sort=None,
        limit=1000,
    ):
        cursor = self.tenant_db[collection_name].find(query, projection)
        if sort:
            cursor = cursor.sort(sort)
        return await cursor.to_list(length=limit)

    async def mongo_update_one(self, collection_name, query, update):
        result = await self.tenant_db[collection_name].update_one(query, update)
        return result.modified_count > 0


async def _seed_document(tenant_db, *, active: bool) -> None:
    await tenant_db["documents"].insert_one(
        {
            "document_id": "camera-paper",
            "title": "Camera Paper",
            "document_type": "Test Series",
            "question_type": "subjective",
            "exam_mode": "pcr",
            "exam_finalized": True,
            "is_active": active,
            "total_minutes": 60,
            "standard": "11",
        }
    )
    await tenant_db["questions"].insert_one(
        {
            "id": "camera-paper-q1",
            "document_id": "camera-paper",
            "question_type": "subjective",
            "text": "Explain the result.",
            "points": 4,
        }
    )


@pytest.mark.asyncio
async def test_active_pcr_metadata_reconciliation_opens_camera_collection():
    from api.v1.pdf_async import update_document_metadata

    tenant_db = AsyncMongoMockClient()["skb_test"]
    await _seed_document(tenant_db, active=True)
    scoped_db = _ScopedDatabase(tenant_db)
    ensure_collection = AsyncMock()

    with (
        patch(
            "api.v1.pdf_async._build_test_series_activation_errors",
            return_value=[],
        ),
        patch(
            "api.v1.exam_orch_async.ensure_default_pcr_camera_collection",
            ensure_collection,
        ),
    ):
        await update_document_metadata.__wrapped__(
            request=None,
            document_id="camera-paper",
            metadata={"is_active": True},
            current_user={
                "user_id": "admin-1",
                "user_type": "admin",
                "db_name": "skb_test",
            },
            db=scoped_db,
        )

    ensure_collection.assert_awaited_once_with(
        prepared_document_id="camera-paper",
        current_user={
            "user_id": "admin-1",
            "user_type": "admin",
            "db_name": "skb_test",
        },
        db=scoped_db,
    )


@pytest.mark.asyncio
async def test_new_pcr_activation_rolls_back_when_camera_collection_fails():
    from api.v1.pdf_async import update_document_metadata

    tenant_db = AsyncMongoMockClient()["skb_test"]
    await _seed_document(tenant_db, active=False)
    scoped_db = _ScopedDatabase(tenant_db)

    with (
        patch(
            "api.v1.pdf_async._build_test_series_activation_errors",
            return_value=[],
        ),
        patch(
            "api.v1.exam_orch_async.ensure_default_pcr_camera_collection",
            AsyncMock(
                side_effect=HTTPException(
                    status_code=409,
                    detail="Camera collection could not be opened",
                )
            ),
        ),
        pytest.raises(HTTPException) as exc,
    ):
        await update_document_metadata.__wrapped__(
            request=None,
            document_id="camera-paper",
            metadata={"is_active": True},
            current_user={
                "user_id": "admin-1",
                "user_type": "admin",
                "db_name": "skb_test",
            },
            db=scoped_db,
        )

    assert exc.value.status_code == 409
    stored = await tenant_db["documents"].find_one({"document_id": "camera-paper"})
    assert stored["is_active"] is False
