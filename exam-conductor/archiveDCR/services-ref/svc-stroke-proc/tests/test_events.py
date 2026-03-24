"""Integration tests for the stroke processing pipeline.

Test IDs: I-SPROC-01 through I-SPROC-06
Markers: integration (mocked NATS + DB)

Tests the full flow: raw event -> dedup -> normalize -> commit -> publish.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.dedup import make_idempotency_key
from src.domain.normalizer import normalize_coordinates, compute_bbox_mm
from src.domain.page_assigner import build_page_assignments
from src.events.processed_publisher import (
    EVENT_TYPE,
    EVENT_VERSION,
    ProcessedStrokePublisher,
)
from src.events.raw_consumer import (
    RawStrokeConsumer,
    _decode_stroke_payload,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXAM_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_PEN_MAC = "AA:BB:CC:DD:EE:FF"


def _make_stroke_payload(
    strokes: list[dict[str, Any]] | None = None,
) -> str:
    """Build base64-encoded JSON stroke payload."""
    if strokes is None:
        strokes = [
            {
                "stroke_id": "s1",
                "page_number": 1,
                "book_type": "LS",
                "points": [
                    {"x": 1050, "y": 1485, "pressure": 0.7, "timestamp": 100},
                    {"x": 1100, "y": 1500, "pressure": 0.6, "timestamp": 101},
                ],
            }
        ]
    raw = json.dumps(strokes).encode()
    return base64.b64encode(raw).decode()


def _make_raw_event(**overrides: Any) -> dict[str, Any]:
    event = {
        "event_id": "evt-1",
        "event_type": "stroke.raw",
        "event_version": "1.0.0",
        "occurred_at": "2026-03-19T10:00:00Z",
        "exam_id": _EXAM_ID,
        "pen_mac": _PEN_MAC,
        "chunk_index": 0,
        "total_chunks": 1,
        "payload_base64": _make_stroke_payload(),
        "checksum_crc32": "00000000",
        "upload_path": "wifi",
    }
    event.update(overrides)
    return event


@dataclass
class FakeStrokeRepo:
    """In-memory repo for testing."""

    committed: list[dict[str, Any]] = field(default_factory=list)
    existing_keys: set[str] = field(default_factory=set)

    async def chunk_exists(self, idempotency_key: str) -> bool:
        return idempotency_key in self.existing_keys

    async def commit_processed_strokes(
        self, exam_id: str, pen_mac: str, chunk_index: int,
        idempotency_key: str, strokes: list[dict[str, Any]],
    ) -> bool:
        if idempotency_key in self.existing_keys:
            return False
        self.existing_keys.add(idempotency_key)
        self.committed.append({
            "exam_id": exam_id,
            "pen_mac": pen_mac,
            "chunk_index": chunk_index,
            "idempotency_key": idempotency_key,
            "strokes": strokes,
        })
        return True

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass


@dataclass
class FakePublisher:
    """Captures published events for assertions."""

    published: list[dict[str, Any]] = field(default_factory=list)

    async def publish_stroke_processed(
        self, exam_id: str, pen_mac: str,
        student_id: str | None,
        page_assignments: list[dict[str, Any]],
    ) -> None:
        self.published.append({
            "exam_id": exam_id,
            "pen_mac": pen_mac,
            "student_id": student_id,
            "page_assignments": page_assignments,
        })


# ---------------------------------------------------------------------------
# I-SPROC-01: Full pipeline — new chunk processed and published
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_full_pipeline_new_chunk():
    repo = FakeStrokeRepo()
    publisher = FakePublisher()
    nats_mock = AsyncMock()

    consumer = RawStrokeConsumer(
        nats_client=nats_mock,
        stroke_repo=repo,
        publisher=publisher,
    )

    event = _make_raw_event()
    await consumer._process_event(event)

    # DB commit happened
    assert len(repo.committed) == 1
    commit = repo.committed[0]
    assert commit["exam_id"] == _EXAM_ID
    assert commit["pen_mac"] == _PEN_MAC
    assert len(commit["strokes"]) == 1
    assert "normalized_points" in commit["strokes"][0]

    # Event published
    assert len(publisher.published) == 1
    pub = publisher.published[0]
    assert pub["exam_id"] == _EXAM_ID
    assert len(pub["page_assignments"]) > 0


# ---------------------------------------------------------------------------
# I-SPROC-02: Duplicate chunk skipped — no commit, no publish
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_duplicate_chunk_skipped():
    idem_key = make_idempotency_key(_EXAM_ID, _PEN_MAC, 0)
    repo = FakeStrokeRepo(existing_keys={idem_key})
    publisher = FakePublisher()
    nats_mock = AsyncMock()

    consumer = RawStrokeConsumer(
        nats_client=nats_mock,
        stroke_repo=repo,
        publisher=publisher,
    )

    event = _make_raw_event()
    await consumer._process_event(event)

    assert len(repo.committed) == 0
    assert len(publisher.published) == 0


# ---------------------------------------------------------------------------
# I-SPROC-03: Strokes are normalized with correct coordinates
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_strokes_normalized():
    repo = FakeStrokeRepo()
    publisher = FakePublisher()
    nats_mock = AsyncMock()

    consumer = RawStrokeConsumer(
        nats_client=nats_mock,
        stroke_repo=repo,
        publisher=publisher,
    )

    event = _make_raw_event()
    await consumer._process_event(event)

    strokes = repo.committed[0]["strokes"]
    assert len(strokes) == 1
    pts = strokes[0]["normalized_points"]
    assert len(pts) == 2

    # x=1050 pen units / 10 = 105.0 mm
    assert pts[0]["x_mm"] == 105.0
    # y=1485 pen units -> 297.0 - 148.5 = 148.5 mm
    assert pts[0]["y_mm"] == 148.5


# ---------------------------------------------------------------------------
# I-SPROC-04: Decode JSON payload
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_decode_json_payload():
    strokes_data = [
        {
            "stroke_id": "s1",
            "page_number": 1,
            "book_type": "LS",
            "points": [{"x": 100, "y": 200, "pressure": 0.5, "timestamp": 1}],
        }
    ]
    raw = json.dumps(strokes_data).encode()
    event = _make_raw_event()
    result = _decode_stroke_payload(raw, event)
    assert len(result) == 1
    assert result[0]["stroke_id"] == "s1"


# ---------------------------------------------------------------------------
# I-SPROC-05: Decode binary fallback payload
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_decode_binary_fallback():
    # Build a single 14-byte frame
    frame = bytearray(14)
    frame[0] = 0x02  # bookType
    frame[1] = 0x01  # pageNo
    frame[2:4] = (500).to_bytes(2, "big")  # X
    frame[4:6] = (600).to_bytes(2, "big")  # Y
    frame[6:8] = (512).to_bytes(2, "big")  # pressure (512/1023 ~ 0.5)
    frame[8:12] = (1000).to_bytes(4, "big")  # timestamp
    frame[12:14] = (0).to_bytes(2, "big")  # padding

    event = _make_raw_event()
    result = _decode_stroke_payload(bytes(frame), event)
    assert len(result) == 1
    points = result[0]["points"]
    assert len(points) == 1
    assert points[0]["x"] == 500
    assert points[0]["y"] == 600


# ---------------------------------------------------------------------------
# I-SPROC-06: ProcessedStrokePublisher builds correct event shape
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_publisher_event_shape():
    mock_client = AsyncMock()
    publisher = ProcessedStrokePublisher(mock_client)

    await publisher.publish_stroke_processed(
        exam_id=_EXAM_ID,
        pen_mac=_PEN_MAC,
        student_id="student-42",
        page_assignments=[
            {"page_number": 1, "question_id": "q1", "point_count": 10}
        ],
    )

    mock_client.publish.assert_awaited_once()
    call_args = mock_client.publish.call_args
    subject = call_args[0][0]
    event = call_args[0][1]

    assert subject == "EXAMPEN.stroke.processed"
    assert event["event_type"] == EVENT_TYPE
    assert event["event_version"] == EVENT_VERSION
    assert event["exam_id"] == _EXAM_ID
    assert event["pen_mac"] == _PEN_MAC
    assert event["student_id"] == "student-42"
    assert event["normalized_stroke_uri"].startswith("timescaledb://")
    assert len(event["page_assignments"]) == 1


# ---------------------------------------------------------------------------
# I-SPROC-07: Concurrent duplicate inserts result in exactly one commit
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_concurrent_duplicate_inserts_one_commit():
    """Simulate two concurrent calls to _process_event with the same
    (exam_id, pen_mac, chunk_index).  The FakeStrokeRepo now rejects
    duplicates via the idempotency key, ensuring exactly one commit
    and one publish.
    """
    import asyncio

    repo = FakeStrokeRepo()
    publisher = FakePublisher()
    nats_mock = AsyncMock()

    consumer = RawStrokeConsumer(
        nats_client=nats_mock,
        stroke_repo=repo,
        publisher=publisher,
    )

    event = _make_raw_event()

    # Fire two concurrent processing attempts for the same chunk
    results = await asyncio.gather(
        consumer._process_event(event),
        consumer._process_event(event),
        return_exceptions=True,
    )

    # Neither should raise
    for r in results:
        assert not isinstance(r, Exception), f"Unexpected exception: {r}"

    # Exactly one commit should have succeeded
    assert len(repo.committed) == 1, (
        f"Expected 1 commit but got {len(repo.committed)}"
    )

    # Exactly one publish
    assert len(publisher.published) == 1, (
        f"Expected 1 publish but got {len(publisher.published)}"
    )
