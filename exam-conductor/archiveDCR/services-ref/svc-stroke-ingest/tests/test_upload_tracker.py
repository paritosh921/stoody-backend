"""Unit tests for domain/upload_tracker.py — progress and reconciliation.

Test IDs: U-SINGEST-20 through U-SINGEST-28
Markers: unit (ZERO I/O)
"""

from __future__ import annotations

import pytest

from src.domain.upload_tracker import ExamUploadTracker, PenProgress

_EXAM_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_PEN_A = "AA:BB:CC:DD:EE:01"
_PEN_B = "AA:BB:CC:DD:EE:02"


# ---------------------------------------------------------------------------
# U-SINGEST-20: PenProgress — complete when all chunks acked
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pen_progress_complete():
    p = PenProgress(pen_mac=_PEN_A, total_chunks=3, acked_chunks=frozenset({0, 1, 2}))
    assert p.complete is True
    assert p.missing_chunks == []
    assert p.next_expected_chunk == 3


# ---------------------------------------------------------------------------
# U-SINGEST-21: PenProgress — incomplete with missing chunks
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pen_progress_incomplete():
    p = PenProgress(pen_mac=_PEN_A, total_chunks=4, acked_chunks=frozenset({0, 2}))
    assert p.complete is False
    assert p.missing_chunks == [1, 3]
    assert p.next_expected_chunk == 1


# ---------------------------------------------------------------------------
# U-SINGEST-22: PenProgress — empty acked set
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pen_progress_empty():
    p = PenProgress(pen_mac=_PEN_A, total_chunks=3)
    assert p.complete is False
    assert p.missing_chunks == [0, 1, 2]
    assert p.next_expected_chunk == 0


# ---------------------------------------------------------------------------
# U-SINGEST-23: PenProgress — zero total_chunks is never complete
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_pen_progress_zero_total():
    p = PenProgress(pen_mac=_PEN_A, total_chunks=0)
    assert p.complete is False


# ---------------------------------------------------------------------------
# U-SINGEST-24: ExamUploadTracker — record ack builds progress
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_tracker_record_ack():
    t = ExamUploadTracker(exam_id=_EXAM_ID)
    pen = t.record_ack(_PEN_A, 0, 3)
    assert pen.pen_mac == _PEN_A
    assert pen.acked_chunks == frozenset({0})
    assert pen.total_chunks == 3


# ---------------------------------------------------------------------------
# U-SINGEST-25: ExamUploadTracker — multiple acks accumulate
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_tracker_multiple_acks():
    t = ExamUploadTracker(exam_id=_EXAM_ID)
    t.record_ack(_PEN_A, 0, 3)
    t.record_ack(_PEN_A, 1, 3)
    pen = t.record_ack(_PEN_A, 2, 3)
    assert pen.complete is True
    assert len(pen.acked_chunks) == 3


# ---------------------------------------------------------------------------
# U-SINGEST-26: ExamUploadTracker — multiple pens tracked independently
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_tracker_multiple_pens():
    t = ExamUploadTracker(exam_id=_EXAM_ID)
    t.record_ack(_PEN_A, 0, 2)
    t.record_ack(_PEN_B, 0, 3)
    assert len(t.all_pens()) == 2

    a = t.get_pen_progress(_PEN_A)
    b = t.get_pen_progress(_PEN_B)
    assert a is not None and a.total_chunks == 2
    assert b is not None and b.total_chunks == 3


# ---------------------------------------------------------------------------
# U-SINGEST-27: ExamUploadTracker — all_complete
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_tracker_all_complete():
    t = ExamUploadTracker(exam_id=_EXAM_ID)
    assert t.all_complete() is False  # no pens yet

    t.record_ack(_PEN_A, 0, 1)
    assert t.all_complete() is True

    t.record_ack(_PEN_B, 0, 2)
    assert t.all_complete() is False  # pen B missing chunk 1

    t.record_ack(_PEN_B, 1, 2)
    assert t.all_complete() is True


# ---------------------------------------------------------------------------
# U-SINGEST-28: reconciliation_summary matches API schema
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_reconciliation_summary():
    t = ExamUploadTracker(exam_id=_EXAM_ID)
    t.record_ack(_PEN_A, 0, 2)
    t.record_ack(_PEN_A, 1, 2)
    t.record_ack(_PEN_B, 0, 3)

    summary = t.reconciliation_summary()
    assert len(summary) == 2

    # Find pen A
    pen_a = next(s for s in summary if s["pen_mac"] == _PEN_A)
    assert pen_a["complete"] is True
    assert pen_a["acked_chunks"] == [0, 1]
    assert pen_a["total_chunks"] == 2

    pen_b = next(s for s in summary if s["pen_mac"] == _PEN_B)
    assert pen_b["complete"] is False
    assert pen_b["acked_chunks"] == [0]
    assert pen_b["total_chunks"] == 3
