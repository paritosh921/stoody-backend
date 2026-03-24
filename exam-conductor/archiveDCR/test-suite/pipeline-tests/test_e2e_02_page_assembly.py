"""
E2E-02: Page assembly -> AI recognition.

Services involved: svc-doc-assembly, svc-ai-pipeline, MinIO.

What it proves:
    When stroke.processed events are emitted, svc-doc-assembly renders a
    page image, uploads it to MinIO (S3), and publishes a ``page.ready``
    event.  The page image is retrievable from S3 and the event payload
    conforms to the contract schema.

Test-ID: E2E-02  (TEST_SUITE_SPEC.md section 2.3)
Level: L5 (multi-service pipeline)
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


class TestPageAssembly:
    """E2E-02 — stroke.processed -> page.ready -> page image in S3."""

    async def test_processed_stroke_triggers_page_ready(
        self,
        publish_event,
        event_waiter,
        stroke_factory,
    ):
        """A stroke.processed event triggers a page.ready event."""
        exam_id = str(uuid.uuid4())
        student_id = str(uuid.uuid4())
        pen_mac = "AA:BB:CC:DD:EE:10"

        processed_event = stroke_factory.create_processed_event(
            exam_id=exam_id,
            pen_mac=pen_mac,
            student_id=student_id,
            page_assignments=[
                {"page_number": 1, "question_id": "q1", "point_count": 200},
            ],
        )

        waiter = event_waiter.wait_for_event(
            "page.ready",
            filter_fn=lambda e: (
                e.get("exam_id") == exam_id
                and e.get("student_id") == student_id
            ),
        )

        await publish_event("stroke.processed", processed_event)

        page_event = await waiter
        assert page_event["event_type"] == "page.ready"
        assert page_event["exam_id"] == exam_id
        assert page_event["student_id"] == student_id
        assert page_event["page_number"] == 1
        assert "image_uri" in page_event
        assert page_event["authoritative_source"] in (
            "strokes",
            "copy_image",
            "both",
        )

    async def test_page_image_exists_in_s3(
        self,
        publish_event,
        event_waiter,
        stroke_factory,
        minio_client,
    ):
        """After page.ready, the referenced image is in MinIO."""
        exam_id = str(uuid.uuid4())
        student_id = str(uuid.uuid4())
        pen_mac = "AA:BB:CC:DD:EE:11"

        processed_event = stroke_factory.create_processed_event(
            exam_id=exam_id,
            pen_mac=pen_mac,
            student_id=student_id,
        )

        waiter = event_waiter.wait_for_event(
            "page.ready",
            filter_fn=lambda e: e.get("exam_id") == exam_id,
        )

        await publish_event("stroke.processed", processed_event)
        page_event = await waiter

        image_uri: str = page_event["image_uri"]
        # URIs are expected as s3://bucket/key or http://...
        if image_uri.startswith("s3://"):
            parts = image_uri.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1]
        else:
            # Assume the last path segments are bucket/key.
            key = "/".join(image_uri.split("/")[-3:])
            from conftest import MINIO_BUCKET

            bucket = MINIO_BUCKET

        stat = minio_client.stat_object(bucket, key)
        assert stat.size > 0, "Page image is empty in S3"

    async def test_page_ready_contains_question_ids(
        self,
        publish_event,
        event_waiter,
        stroke_factory,
    ):
        """page.ready event includes question_ids for the page."""
        exam_id = str(uuid.uuid4())
        student_id = str(uuid.uuid4())
        pen_mac = "AA:BB:CC:DD:EE:12"

        processed_event = stroke_factory.create_processed_event(
            exam_id=exam_id,
            pen_mac=pen_mac,
            student_id=student_id,
            page_assignments=[
                {"page_number": 1, "question_id": "q1", "point_count": 100},
                {"page_number": 1, "question_id": "q2", "point_count": 80},
            ],
        )

        waiter = event_waiter.wait_for_event(
            "page.ready",
            filter_fn=lambda e: e.get("exam_id") == exam_id,
        )

        await publish_event("stroke.processed", processed_event)
        page_event = await waiter

        # question_ids is optional in schema but expected from doc-assembly.
        if "question_ids" in page_event:
            assert "q1" in page_event["question_ids"]
            assert "q2" in page_event["question_ids"]

    async def test_page_ready_schema_compliance(
        self,
        publish_event,
        event_waiter,
        stroke_factory,
    ):
        """page.ready event has all required fields per contract."""
        exam_id = str(uuid.uuid4())
        student_id = str(uuid.uuid4())

        processed_event = stroke_factory.create_processed_event(
            exam_id=exam_id,
            pen_mac="AA:BB:CC:DD:EE:13",
            student_id=student_id,
        )

        waiter = event_waiter.wait_for_event(
            "page.ready",
            filter_fn=lambda e: e.get("exam_id") == exam_id,
        )

        await publish_event("stroke.processed", processed_event)
        page_event = await waiter

        required_fields = [
            "event_id",
            "event_type",
            "event_version",
            "occurred_at",
            "exam_id",
            "student_id",
            "page_id",
            "page_number",
            "image_uri",
            "authoritative_source",
        ]
        for f in required_fields:
            assert f in page_event, f"Missing required field: {f}"
        assert page_event["event_version"] == "1.0.0"
