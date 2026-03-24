"""Integration tests for ProcessedStrokeConsumer.

Test IDs: I-DA-C01 through I-DA-C05
Markers: integration (mocked NATS, S3, PG)

Validates:
- Bug 2 fix: student_id in buffer key prevents cross-student collision
- Bug 3 fix: real stroke data parsed from event, not placeholders
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.models import CanonicalPoint, Stroke
from src.events.processed_consumer import (
    ProcessedStrokeConsumer,
    _parse_canonical_points,
    _build_question_regions,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

@dataclass
class FakeUploadResult:
    bucket: str = "test-bucket"
    key: str = "test-key"
    etag: str = "test-etag"

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


@dataclass
class FakeS3:
    uploads: list[dict[str, Any]] = field(default_factory=list)

    async def upload_page_svg(
        self,
        exam_id: str,
        student_id: str,
        page_number: int,
        svg_content: str,
    ) -> FakeUploadResult:
        self.uploads.append({
            "exam_id": exam_id,
            "student_id": student_id,
            "page_number": page_number,
            "svg_content": svg_content,
        })
        return FakeUploadResult(
            key=f"{exam_id}/{student_id}/page_{page_number}.svg",
        )


@dataclass
class FakePageRepo:
    pages: list[dict[str, Any]] = field(default_factory=list)

    async def save_page(self, doc: Any, s3_uri: str, page_id: str) -> None:
        self.pages.append({
            "exam_id": doc.exam_id,
            "student_id": doc.student_id,
            "page_number": doc.page_number,
            "s3_uri": s3_uri,
            "page_id": page_id,
            "question_results": doc.question_results,
            "svg_content": doc.svg_content,
        })


def _make_consumer() -> tuple[
    ProcessedStrokeConsumer, FakeS3, FakePageRepo, AsyncMock
]:
    js_mock = AsyncMock()
    # Stub publish to return an ack-like object with .seq
    ack = MagicMock()
    ack.seq = 1
    js_mock.publish.return_value = ack
    s3 = FakeS3()
    repo = FakePageRepo()
    consumer = ProcessedStrokeConsumer(js=js_mock, s3=s3, page_repo=repo)
    return consumer, s3, repo, js_mock


_EXAM_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


# ---------------------------------------------------------------------------
# I-DA-C01: Two students on the same exam+page do not collide
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_student_buffer_isolation():
    """Strokes from student_a and student_b on the same (exam, page)
    must be assembled independently and not lose data.
    """
    consumer, s3, repo, _ = _make_consumer()

    event_a = {
        "exam_id": _EXAM_ID,
        "student_id": "student-A",
        "pen_mac": "AA:AA:AA:AA:AA:AA",
        "normalized_stroke_uri": f"timescaledb://strokes/{_EXAM_ID}/AA:AA:AA:AA:AA:AA",
        "page_assignments": [
            {
                "page_number": 1,
                "question_id": "q1",
                "point_count": 2,
                "normalized_points": [
                    {"x_mm": 10.0, "y_mm": 20.0, "pressure": 0.8},
                    {"x_mm": 11.0, "y_mm": 21.0, "pressure": 0.7},
                ],
            },
        ],
    }

    event_b = {
        "exam_id": _EXAM_ID,
        "student_id": "student-B",
        "pen_mac": "BB:BB:BB:BB:BB:BB",
        "normalized_stroke_uri": f"timescaledb://strokes/{_EXAM_ID}/BB:BB:BB:BB:BB:BB",
        "page_assignments": [
            {
                "page_number": 1,
                "question_id": "q1",
                "point_count": 3,
                "normalized_points": [
                    {"x_mm": 50.0, "y_mm": 60.0, "pressure": 0.9},
                    {"x_mm": 51.0, "y_mm": 61.0, "pressure": 0.6},
                    {"x_mm": 52.0, "y_mm": 62.0, "pressure": 0.5},
                ],
            },
        ],
    }

    await consumer._process_event(event_a)
    await consumer._process_event(event_b)

    # Two separate pages assembled — one per student
    assert len(repo.pages) == 2

    page_a = next(p for p in repo.pages if p["student_id"] == "student-A")
    page_b = next(p for p in repo.pages if p["student_id"] == "student-B")

    assert page_a["exam_id"] == _EXAM_ID
    assert page_b["exam_id"] == _EXAM_ID
    assert page_a["page_number"] == 1
    assert page_b["page_number"] == 1

    # Each page must have its own SVG (not share content)
    assert page_a["s3_uri"] != page_b["s3_uri"]


# ---------------------------------------------------------------------------
# I-DA-C02: Real stroke points parsed from event, not empty placeholders
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_real_stroke_data_parsed():
    """Strokes must contain actual CanonicalPoints from the event
    payload, not empty placeholders.
    """
    consumer, s3, repo, _ = _make_consumer()

    event = {
        "exam_id": _EXAM_ID,
        "student_id": "student-1",
        "pen_mac": "CC:CC:CC:CC:CC:CC",
        "normalized_stroke_uri": f"timescaledb://strokes/{_EXAM_ID}/CC:CC:CC:CC:CC:CC",
        "page_assignments": [
            {
                "page_number": 1,
                "question_id": "q1",
                "point_count": 2,
                "normalized_points": [
                    {"x_mm": 15.0, "y_mm": 25.0, "pressure": 0.8, "tilt_x": 0.1, "tilt_y": 0.2, "timestamp_ms": 100},
                    {"x_mm": 16.0, "y_mm": 26.0, "pressure": 0.7, "tilt_x": 0.0, "tilt_y": 0.0, "timestamp_ms": 101},
                ],
            },
        ],
    }

    await consumer._process_event(event)

    assert len(repo.pages) == 1
    page = repo.pages[0]

    # The SVG must contain path data (not just an empty page)
    assert "<path" in page["svg_content"], (
        "SVG should contain stroke paths from real data"
    )


# ---------------------------------------------------------------------------
# I-DA-C03: _parse_canonical_points converts raw dicts correctly
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_parse_canonical_points():
    raw = [
        {"x_mm": 10.0, "y_mm": 20.0, "pressure": 0.5, "tilt_x": 0.1, "tilt_y": 0.2, "timestamp_ms": 42},
        {"x_mm": 11.0, "y_mm": 21.0, "pressure": 0.6},
    ]

    points = _parse_canonical_points(raw)

    assert len(points) == 2
    assert isinstance(points[0], CanonicalPoint)
    assert points[0].x == 10.0
    assert points[0].y == 20.0
    assert points[0].pressure == 0.5
    assert points[0].tilt_x == 0.1
    assert points[0].tilt_y == 0.2
    assert points[0].timestamp_ms == 42

    # Second point: defaults for missing optional fields
    assert points[1].tilt_x == 0.0
    assert points[1].tilt_y == 0.0
    assert points[1].timestamp_ms == 0


# ---------------------------------------------------------------------------
# I-DA-C04: _build_question_regions distributes regions across page
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_build_question_regions():
    regions = _build_question_regions(["q1", "q2"], page_height=200.0)

    assert len(regions) == 2
    assert regions[0].question_id == "q1"
    assert regions[0].y_min == 0.0
    assert regions[0].y_max == 100.0
    assert regions[1].question_id == "q2"
    assert regions[1].y_min == 100.0
    assert regions[1].y_max == 200.0


# ---------------------------------------------------------------------------
# I-DA-C05: Empty question_ids produce empty regions list
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_build_question_regions_empty():
    regions = _build_question_regions([])
    assert regions == []
