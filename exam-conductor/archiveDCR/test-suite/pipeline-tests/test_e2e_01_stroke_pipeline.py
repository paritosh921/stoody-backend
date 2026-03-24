"""
E2E-01: Stroke ingestion -> processing -> storage.

Services involved: svc-stroke-ingest, svc-stroke-proc, TimescaleDB.

What it proves:
    Stroke data flows through the full ingestion pipeline. A raw chunk
    published on ``stroke.raw`` is picked up by svc-stroke-proc, deduplicated,
    normalized, committed to TimescaleDB, and a ``stroke.processed`` event is
    emitted.  Duplicate chunks produce only a single DB row (idempotency).

Test-ID: E2E-01  (TEST_SUITE_SPEC.md section 2.3)
Level: L5 (multi-service pipeline)
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


class TestStrokeIngestionPipeline:
    """E2E-01 — stroke.raw -> stroke.processed -> TimescaleDB row."""

    async def test_raw_event_produces_processed_event(
        self,
        publish_event,
        event_waiter,
        stroke_factory,
    ):
        """Publish stroke.raw and verify stroke.processed is emitted."""
        exam_id = str(uuid.uuid4())
        pen_mac = "AA:BB:CC:DD:EE:01"

        raw_event = stroke_factory.create_raw_event(
            exam_id=exam_id,
            pen_mac=pen_mac,
            chunk_index=0,
            total_chunks=1,
        )

        # Subscribe BEFORE publishing so we don't miss the event.
        waiter = event_waiter.wait_for_event(
            "stroke.processed",
            filter_fn=lambda e: (
                e.get("exam_id") == exam_id
                and e.get("pen_mac") == pen_mac
            ),
        )

        await publish_event("stroke.raw", raw_event)

        processed = await waiter
        assert processed["event_type"] == "stroke.processed"
        assert processed["exam_id"] == exam_id
        assert processed["pen_mac"] == pen_mac
        assert isinstance(processed["page_assignments"], list)
        assert len(processed["page_assignments"]) > 0

    async def test_processed_stroke_persisted_in_timescaledb(
        self,
        publish_event,
        event_waiter,
        stroke_factory,
        pg_pool,
    ):
        """After stroke.processed, a corresponding row exists in TimescaleDB."""
        exam_id = str(uuid.uuid4())
        pen_mac = "AA:BB:CC:DD:EE:02"

        raw_event = stroke_factory.create_raw_event(
            exam_id=exam_id,
            pen_mac=pen_mac,
        )

        waiter = event_waiter.wait_for_event(
            "stroke.processed",
            filter_fn=lambda e: e.get("exam_id") == exam_id,
        )

        await publish_event("stroke.raw", raw_event)
        await waiter

        # Query TimescaleDB for the committed stroke.
        async with pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT exam_id, pen_mac, chunk_index
                FROM stroke_data
                WHERE exam_id = $1 AND pen_mac = $2
                LIMIT 1
                """,
                exam_id,
                pen_mac,
            )

        assert row is not None, "Stroke row not found in TimescaleDB"
        assert str(row["exam_id"]) == exam_id
        assert row["pen_mac"] == pen_mac

    async def test_duplicate_chunk_produces_single_row(
        self,
        publish_event,
        event_waiter,
        stroke_factory,
        pg_pool,
    ):
        """Same chunk sent twice should result in exactly one DB row (dedup)."""
        exam_id = str(uuid.uuid4())
        pen_mac = "AA:BB:CC:DD:EE:03"

        raw_event = stroke_factory.create_raw_event(
            exam_id=exam_id,
            pen_mac=pen_mac,
            chunk_index=0,
            total_chunks=1,
        )

        waiter = event_waiter.wait_for_event(
            "stroke.processed",
            filter_fn=lambda e: e.get("exam_id") == exam_id,
        )

        # Publish twice (duplicate).
        await publish_event("stroke.raw", raw_event)
        await publish_event("stroke.raw", raw_event)

        await waiter
        # Brief grace period for the duplicate to be processed (or rejected).
        await asyncio.sleep(2)

        async with pg_pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM stroke_data
                WHERE exam_id = $1 AND pen_mac = $2 AND chunk_index = 0
                """,
                exam_id,
                pen_mac,
            )

        assert count == 1, f"Expected 1 row, got {count} (dedup failed)"

    async def test_multi_chunk_upload_reassembled(
        self,
        publish_event,
        event_waiter,
        stroke_factory,
    ):
        """Multiple chunks for the same pen produce a single processed event."""
        exam_id = str(uuid.uuid4())
        pen_mac = "AA:BB:CC:DD:EE:04"
        total_chunks = 3

        waiter = event_waiter.wait_for_event(
            "stroke.processed",
            filter_fn=lambda e: e.get("exam_id") == exam_id,
        )

        for idx in range(total_chunks):
            raw = stroke_factory.create_raw_event(
                exam_id=exam_id,
                pen_mac=pen_mac,
                chunk_index=idx,
                total_chunks=total_chunks,
            )
            await publish_event("stroke.raw", raw)

        processed = await waiter
        assert processed["exam_id"] == exam_id
        assert processed["pen_mac"] == pen_mac

    async def test_stroke_raw_schema_compliance(
        self,
        stroke_factory,
    ):
        """Factory-produced stroke.raw events conform to contract schema."""
        event = stroke_factory.create_raw_event(
            exam_id=str(uuid.uuid4()),
            pen_mac="AA:BB:CC:DD:EE:05",
        )

        required_fields = [
            "event_id",
            "event_type",
            "event_version",
            "occurred_at",
            "exam_id",
            "pen_mac",
            "chunk_index",
            "total_chunks",
            "payload_base64",
            "checksum_crc32",
            "upload_path",
        ]
        for f in required_fields:
            assert f in event, f"Missing required field: {f}"
        assert event["event_type"] == "stroke.raw"
        assert event["event_version"] == "1.0.0"
        assert event["upload_path"] in ("wifi", "mobile")
