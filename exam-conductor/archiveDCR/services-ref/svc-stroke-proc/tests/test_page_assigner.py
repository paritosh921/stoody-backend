"""Unit tests for domain/page_assigner.py — question assignment from bbox overlap.

Test IDs: U-SPROC-30 through U-SPROC-39
Markers: unit (ZERO I/O)
"""

from __future__ import annotations

import pytest

from src.domain.page_assigner import (
    QuestionRegion,
    assign_strokes_to_questions,
    assign_to_question,
    build_page_assignments,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_Q1 = QuestionRegion("q1", 0.0, 0.0, 100.0, 50.0)
_Q2 = QuestionRegion("q2", 0.0, 50.0, 100.0, 120.0)
_Q3 = QuestionRegion("q3", 0.0, 120.0, 100.0, 200.0)

_REGIONS = [_Q1, _Q2, _Q3]


# ---------------------------------------------------------------------------
# U-SPROC-30: Stroke fully inside Q1
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assign_fully_inside():
    bbox = {"min_x": 10.0, "min_y": 10.0, "max_x": 50.0, "max_y": 40.0}
    assert assign_to_question(bbox, _REGIONS) == "q1"


# ---------------------------------------------------------------------------
# U-SPROC-31: Stroke fully inside Q2
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assign_fully_inside_q2():
    bbox = {"min_x": 10.0, "min_y": 60.0, "max_x": 90.0, "max_y": 110.0}
    assert assign_to_question(bbox, _REGIONS) == "q2"


# ---------------------------------------------------------------------------
# U-SPROC-32: Stroke spanning Q1 and Q2 — majority in Q2
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assign_spanning_majority_q2():
    # Q1 ends at y=50, Q2 starts at y=50
    # bbox from y=40 to y=100 -> overlap with Q1 = 10px, Q2 = 50px
    bbox = {"min_x": 10.0, "min_y": 40.0, "max_x": 90.0, "max_y": 100.0}
    assert assign_to_question(bbox, _REGIONS) == "q2"


# ---------------------------------------------------------------------------
# U-SPROC-33: Stroke spanning Q1 and Q2 — majority in Q1
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assign_spanning_majority_q1():
    # bbox from y=10 to y=55 -> overlap with Q1 = 40, Q2 = 5
    bbox = {"min_x": 10.0, "min_y": 10.0, "max_x": 90.0, "max_y": 55.0}
    assert assign_to_question(bbox, _REGIONS) == "q1"


# ---------------------------------------------------------------------------
# U-SPROC-34: Stroke outside all regions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assign_outside_all():
    bbox = {"min_x": 110.0, "min_y": 210.0, "max_x": 150.0, "max_y": 250.0}
    assert assign_to_question(bbox, _REGIONS) is None


# ---------------------------------------------------------------------------
# U-SPROC-35: None bbox returns None
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assign_none_bbox():
    assert assign_to_question(None, _REGIONS) is None


# ---------------------------------------------------------------------------
# U-SPROC-36: Empty regions returns None
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assign_empty_regions():
    bbox = {"min_x": 10.0, "min_y": 10.0, "max_x": 50.0, "max_y": 40.0}
    assert assign_to_question(bbox, []) is None


# ---------------------------------------------------------------------------
# U-SPROC-37: assign_strokes_to_questions enriches strokes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assign_strokes_to_questions():
    strokes = [
        {"stroke_id": "s1", "bbox": {"min_x": 10, "min_y": 10, "max_x": 50, "max_y": 40}},
        {"stroke_id": "s2", "bbox": {"min_x": 10, "min_y": 60, "max_x": 50, "max_y": 100}},
        {"stroke_id": "s3", "bbox": None},
    ]
    result = assign_strokes_to_questions(strokes, _REGIONS)
    assert result[0]["question_id"] == "q1"
    assert result[1]["question_id"] == "q2"
    assert result[2]["question_id"] is None


# ---------------------------------------------------------------------------
# U-SPROC-38: build_page_assignments groups by page and question
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_page_assignments():
    strokes = [
        {"page_number": 1, "question_id": "q1", "normalized_points": [1, 2, 3]},
        {"page_number": 1, "question_id": "q1", "normalized_points": [4, 5]},
        {"page_number": 1, "question_id": "q2", "normalized_points": [6]},
        {"page_number": 2, "question_id": "q3", "normalized_points": [7, 8]},
    ]
    assignments = build_page_assignments(strokes)

    assert len(assignments) == 3
    assert assignments[0] == {"page_number": 1, "question_id": "q1", "point_count": 5}
    assert assignments[1] == {"page_number": 1, "question_id": "q2", "point_count": 1}
    assert assignments[2] == {"page_number": 2, "question_id": "q3", "point_count": 2}


# ---------------------------------------------------------------------------
# U-SPROC-39: build_page_assignments with unassigned strokes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_page_assignments_unassigned():
    strokes = [
        {"page_number": 1, "question_id": None, "normalized_points": [1, 2]},
    ]
    assignments = build_page_assignments(strokes)
    assert len(assignments) == 1
    assert assignments[0]["question_id"] == "unassigned"
    assert assignments[0]["point_count"] == 2
