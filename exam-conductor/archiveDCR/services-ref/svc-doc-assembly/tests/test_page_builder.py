"""Unit tests for page builder. Domain-only, ZERO I/O.

Test IDs: U-DA-P01 through U-DA-P06.
"""

from __future__ import annotations

import pytest

from src.domain.models import (
    CanonicalPoint,
    MissAutoState,
    QuestionRegion,
    Stroke,
    SyncMetadata,
)
from src.domain.page_builder import build_page


# ── helpers ──────────────────────────────────────────────────────────

def _stroke(stroke_id: str, points: list[tuple[float, float]]) -> Stroke:
    return Stroke(
        stroke_id=stroke_id,
        points=[CanonicalPoint(x=x, y=y, pressure=0.8) for x, y in points],
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


# ── U-DA-P01: basic assembly ────────────────────────────────────────

@pytest.mark.unit
class TestBasicAssembly:
    def test_builds_page_with_strokes_and_regions(self):
        strokes = [_stroke("s1", [(15, 25), (16, 26)])]
        regions = [_region("q1", 10, 20, 50, 60)]

        doc = build_page(
            strokes=strokes,
            question_regions=regions,
            sync_metadata=_sync(),
            exam_id="exam-001",
            student_id="stu-001",
            page_number=1,
        )

        assert doc.exam_id == "exam-001"
        assert doc.student_id == "stu-001"
        assert doc.page_number == 1
        assert "<svg" in doc.svg_content
        assert len(doc.question_results) == 1
        assert doc.question_results[0].question_id == "q1"
        assert doc.question_results[0].auto_state == MissAutoState.ANSWERED

    def test_page_dimensions(self):
        doc = build_page(
            strokes=[],
            question_regions=[],
            sync_metadata=_sync(),
            exam_id="e",
            student_id="s",
            page_number=1,
            page_width=148.0,
            page_height=210.0,
        )
        assert doc.page_width_mm == 148.0
        assert doc.page_height_mm == 210.0
        assert "148.0mm" in doc.svg_content


# ── U-DA-P02: empty page (no strokes) ───────────────────────────────

@pytest.mark.unit
class TestEmptyPage:
    def test_no_strokes_pen_connected(self):
        regions = [_region("q1", 10, 20, 50, 60)]
        doc = build_page(
            strokes=[],
            question_regions=regions,
            sync_metadata=_sync(connected=True),
            exam_id="e",
            student_id="s",
            page_number=1,
        )
        assert doc.question_results[0].auto_state == MissAutoState.MISS_PEN_INACTIVE

    def test_no_strokes_pen_disconnected(self):
        regions = [_region("q1", 10, 20, 50, 60)]
        doc = build_page(
            strokes=[],
            question_regions=regions,
            sync_metadata=_sync(connected=False),
            exam_id="e",
            student_id="s",
            page_number=1,
        )
        assert doc.question_results[0].auto_state == MissAutoState.MISS_NO_STROKES


# ── U-DA-P03: multi-question page with mixed states ─────────────────

@pytest.mark.unit
class TestMultiQuestion:
    def test_three_questions_mixed(self):
        strokes = [
            _stroke("s1", [(15, 25)]),   # hits q1
            _stroke("s2", [(75, 85)]),   # hits q2
            # q3 has no strokes
        ]
        regions = [
            _region("q1", 10, 20, 50, 60),
            _region("q2", 70, 80, 120, 130),
            _region("q3", 150, 160, 200, 210),
        ]
        doc = build_page(
            strokes=strokes,
            question_regions=regions,
            sync_metadata=_sync(),
            exam_id="e",
            student_id="s",
            page_number=1,
        )

        by_q = {qr.question_id: qr for qr in doc.question_results}
        assert by_q["q1"].auto_state == MissAutoState.ANSWERED
        assert by_q["q2"].auto_state == MissAutoState.ANSWERED
        assert by_q["q3"].auto_state == MissAutoState.MISS_NO_STROKES


# ── U-DA-P04: sync failure propagates to all regions ────────────────

@pytest.mark.unit
class TestSyncFailure:
    def test_sync_failure_marks_all_regions(self):
        strokes = [_stroke("s1", [(15, 25)])]
        regions = [
            _region("q1", 10, 20, 50, 60),
            _region("q2", 70, 80, 120, 130),
        ]
        doc = build_page(
            strokes=strokes,
            question_regions=regions,
            sync_metadata=_sync(complete=False),
            exam_id="e",
            student_id="s",
            page_number=1,
        )
        for qr in doc.question_results:
            assert qr.auto_state == MissAutoState.MISS_SYNC_FAILURE


# ── U-DA-P05: no question regions ───────────────────────────────────

@pytest.mark.unit
class TestNoRegions:
    def test_no_regions_empty_results(self):
        strokes = [_stroke("s1", [(15, 25)])]
        doc = build_page(
            strokes=strokes,
            question_regions=[],
            sync_metadata=_sync(),
            exam_id="e",
            student_id="s",
            page_number=1,
        )
        assert doc.question_results == []
        assert "<svg" in doc.svg_content


# ── U-DA-P06: SVG content includes strokes ──────────────────────────

@pytest.mark.unit
class TestSvgContent:
    def test_svg_contains_paths(self):
        strokes = [
            _stroke("s1", [(5, 5), (10, 10)]),
            _stroke("s2", [(20, 20), (30, 30)]),
        ]
        doc = build_page(
            strokes=strokes,
            question_regions=[],
            sync_metadata=_sync(),
            exam_id="e",
            student_id="s",
            page_number=1,
        )
        assert doc.svg_content.count("<path") == 2

    def test_svg_valid_structure(self):
        doc = build_page(
            strokes=[],
            question_regions=[],
            sync_metadata=_sync(),
            exam_id="e",
            student_id="s",
            page_number=1,
        )
        assert doc.svg_content.startswith('<?xml version="1.0"')
        assert doc.svg_content.endswith("</svg>")
