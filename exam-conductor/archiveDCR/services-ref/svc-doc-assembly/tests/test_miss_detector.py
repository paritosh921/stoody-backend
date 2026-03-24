"""Unit tests for miss indicator detection. Domain-only, ZERO I/O.

Test IDs: U-DA-M01 through U-DA-M09.
"""

from __future__ import annotations

import pytest

from src.domain.models import (
    CanonicalPoint,
    MissAutoState,
    MissOverrideState,
    QuestionRegion,
    QuestionResult,
    Stroke,
    SyncMetadata,
)
from src.domain.miss_detector import detect_all_regions, detect_miss_state


# ── helpers ──────────────────────────────────────────────────────────

def _stroke(stroke_id: str, points: list[tuple[float, float]]) -> Stroke:
    """Build a Stroke with unit pressure at each point."""
    return Stroke(
        stroke_id=stroke_id,
        points=[CanonicalPoint(x=x, y=y, pressure=1.0) for x, y in points],
    )


def _region(qid: str, x0: float, y0: float, x1: float, y1: float) -> QuestionRegion:
    return QuestionRegion(question_id=qid, x_min=x0, y_min=y0, x_max=x1, y_max=y1)


def _sync(complete: bool = True, connected: bool = True) -> SyncMetadata:
    return SyncMetadata(
        pen_mac="AA:BB:CC:DD:EE:FF",
        sync_complete=complete,
        pen_connected=connected,
        strokes_expected=True,
    )


# ── U-DA-M01: answered — strokes present in region ──────────────────

@pytest.mark.unit
class TestAnswered:
    def test_strokes_inside_region(self):
        strokes = [_stroke("s1", [(15.0, 25.0), (16.0, 26.0)])]
        region = _region("q1", 10, 20, 50, 60)
        result = detect_miss_state(strokes, region, _sync())
        assert result == MissAutoState.ANSWERED

    def test_single_point_on_boundary(self):
        strokes = [_stroke("s1", [(10.0, 20.0)])]
        region = _region("q1", 10, 20, 50, 60)
        result = detect_miss_state(strokes, region, _sync())
        assert result == MissAutoState.ANSWERED


# ── U-DA-M02: miss_no_strokes — no strokes in region at all ────────

@pytest.mark.unit
class TestMissNoStrokes:
    def test_no_strokes_connected_pen(self):
        """No strokes + connected pen -> PEN_INACTIVE (not NO_STROKES)."""
        region = _region("q1", 10, 20, 50, 60)
        result = detect_miss_state([], region, _sync(connected=True))
        assert result == MissAutoState.MISS_PEN_INACTIVE

    def test_strokes_outside_region(self):
        strokes = [_stroke("s1", [(100.0, 100.0), (101.0, 101.0)])]
        region = _region("q1", 10, 20, 50, 60)
        result = detect_miss_state(strokes, region, _sync())
        assert result == MissAutoState.MISS_NO_STROKES

    def test_strokes_barely_outside_region(self):
        strokes = [_stroke("s1", [(50.1, 30.0)])]
        region = _region("q1", 10, 20, 50, 60)
        result = detect_miss_state(strokes, region, _sync())
        assert result == MissAutoState.MISS_NO_STROKES


# ── U-DA-M03: miss_sync_failure — sync incomplete ───────────────────

@pytest.mark.unit
class TestMissSyncFailure:
    def test_sync_not_complete(self):
        strokes = [_stroke("s1", [(15.0, 25.0)])]
        region = _region("q1", 10, 20, 50, 60)
        result = detect_miss_state(strokes, region, _sync(complete=False))
        assert result == MissAutoState.MISS_SYNC_FAILURE

    def test_sync_metadata_none(self):
        strokes = [_stroke("s1", [(15.0, 25.0)])]
        region = _region("q1", 10, 20, 50, 60)
        result = detect_miss_state(strokes, region, None)
        assert result == MissAutoState.MISS_SYNC_FAILURE

    def test_sync_failure_overrides_stroke_presence(self):
        """Even if strokes exist in region, sync failure takes priority."""
        strokes = [_stroke("s1", [(15.0, 25.0)])]
        region = _region("q1", 10, 20, 50, 60)
        result = detect_miss_state(strokes, region, _sync(complete=False))
        assert result == MissAutoState.MISS_SYNC_FAILURE


# ── U-DA-M04: miss_pen_inactive — connected but no writing ─────────

@pytest.mark.unit
class TestMissPenInactive:
    def test_pen_connected_no_strokes(self):
        region = _region("q1", 10, 20, 50, 60)
        result = detect_miss_state([], region, _sync(connected=True))
        assert result == MissAutoState.MISS_PEN_INACTIVE

    def test_pen_not_connected_no_strokes(self):
        """If pen wasn't connected, we just see no strokes."""
        region = _region("q1", 10, 20, 50, 60)
        result = detect_miss_state(
            [], region, _sync(connected=False)
        )
        assert result == MissAutoState.MISS_NO_STROKES


# ── U-DA-M05: detect_all_regions — multi-question page ──────────────

@pytest.mark.unit
class TestDetectAllRegions:
    def test_mixed_states(self):
        strokes = [_stroke("s1", [(15.0, 25.0)])]
        regions = [
            _region("q1", 10, 20, 50, 60),   # answered
            _region("q2", 70, 80, 120, 130),  # miss_no_strokes
        ]
        result = detect_all_regions(strokes, regions, _sync())
        assert result["q1"] == MissAutoState.ANSWERED
        assert result["q2"] == MissAutoState.MISS_NO_STROKES

    def test_all_answered(self):
        strokes = [
            _stroke("s1", [(15.0, 25.0)]),
            _stroke("s2", [(75.0, 85.0)]),
        ]
        regions = [
            _region("q1", 10, 20, 50, 60),
            _region("q2", 70, 80, 120, 130),
        ]
        result = detect_all_regions(strokes, regions, _sync())
        assert all(v == MissAutoState.ANSWERED for v in result.values())

    def test_empty_regions(self):
        strokes = [_stroke("s1", [(15.0, 25.0)])]
        result = detect_all_regions(strokes, [], _sync())
        assert result == {}


# ── U-DA-M06: priority ordering ─────────────────────────────────────

@pytest.mark.unit
class TestPriorityOrdering:
    def test_sync_failure_beats_pen_inactive(self):
        region = _region("q1", 10, 20, 50, 60)
        result = detect_miss_state(
            [], region, _sync(complete=False, connected=True)
        )
        assert result == MissAutoState.MISS_SYNC_FAILURE

    def test_sync_failure_beats_answered(self):
        strokes = [_stroke("s1", [(15.0, 25.0)])]
        region = _region("q1", 10, 20, 50, 60)
        result = detect_miss_state(
            strokes, region, _sync(complete=False)
        )
        assert result == MissAutoState.MISS_SYNC_FAILURE


# ── U-DA-M07: QuestionResult display logic ──────────────────────────

@pytest.mark.unit
class TestQuestionResultDisplay:
    def test_no_override_shows_auto(self):
        qr = QuestionResult(
            question_id="q1",
            auto_state=MissAutoState.MISS_NO_STROKES,
        )
        assert qr.display_state == "miss_no_strokes"

    def test_override_takes_precedence(self):
        qr = QuestionResult(
            question_id="q1",
            auto_state=MissAutoState.MISS_NO_STROKES,
            override_state=MissOverrideState.ANSWERED_CONFIRMED,
        )
        assert qr.display_state == "answered_confirmed"

    def test_not_attempted_override(self):
        qr = QuestionResult(
            question_id="q1",
            auto_state=MissAutoState.ANSWERED,
            override_state=MissOverrideState.NOT_ATTEMPTED_CONFIRMED,
        )
        assert qr.display_state == "not_attempted_confirmed"
