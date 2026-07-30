from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_note_page_index_is_copy_scoped_without_legacy_recreation():
    """Two physical copies may legitimately contain the same notebook page."""

    from core.database import DatabaseManager
    from mongomock_motor import AsyncMongoMockClient

    db = AsyncMongoMockClient()["skb_index_test"]
    await db["note_classifications"].insert_many(
        [
            {
                "user_id": "student-1",
                "copy_id": "copy-a",
                "pen_mac": "AA:BB:CC:DD:EE:FF",
                "book_type": "LS",
                "page_number": 144,
            },
            {
                "user_id": "student-1",
                "copy_id": "copy-b",
                "pen_mac": "AA:BB:CC:DD:EE:FF",
                "book_type": "LS",
                "page_number": 144,
            },
        ]
    )
    manager = DatabaseManager()

    await manager.ensure_indexes_for_db(db)

    indexes = await db["note_classifications"].index_information()
    assert "uniq_note_page" not in indexes
    assert indexes["uniq_note_page_v2"]["unique"] is True
    assert indexes["uniq_note_page_v2"]["key"] == [
        ("user_id", 1),
        ("copy_id", 1),
        ("book_type", 1),
        ("page_number", 1),
    ]
    assert db.name in manager._indexed_dbs
