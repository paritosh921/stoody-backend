import pytest
from pydantic import ValidationError

from api.v1.strokes_async import (
    CANONICAL_STROKE_PROCESSING_VERSION,
    CanvasPageStroke,
    CanvasPageUpsert,
    _incoming_stroke_docs,
    _merge_stroke_docs,
)


def test_canvas_stroke_preserves_canonical_geometry_and_metadata():
    canonical_points = [
        [1.25, 2.5, 0, 120, 0.4706, 0],
        [1.75, 3.5, 24, 128, 0.502, 0],
    ]
    stroke = CanvasPageStroke(
        id="canonical-1",
        points=canonical_points,
        processingVersion=CANONICAL_STROKE_PROCESSING_VERSION,
        qualityFlags=["offline_replay_gap"],
        sourceMode="offlineReplay",
    )
    page = CanvasPageUpsert(book_type="MS", page_number=4, strokes=[stroke])

    stored = _incoming_stroke_docs(page)[0]

    assert stored["points"] == canonical_points
    assert stored["processingVersion"] == CANONICAL_STROKE_PROCESSING_VERSION
    assert stored["qualityFlags"] == ["offline_replay_gap"]


def test_canonical_stroke_rejects_non_canonical_point_shape():
    with pytest.raises(ValidationError):
        CanvasPageStroke(
            id="bad-canonical-1",
            points=[[1, 2, 0.5]],
            processingVersion=CANONICAL_STROKE_PROCESSING_VERSION,
        )


def test_stroke_merge_dedupes_by_stable_id_without_repairing_geometry():
    existing = [
        {
            "id": "canonical-1",
            "points": [[1.25, 2.5, 0, 120, 0.4706, 0]],
            "processingVersion": CANONICAL_STROKE_PROCESSING_VERSION,
            "qualityFlags": [],
        }
    ]
    incoming = [
        {
            "id": "canonical-1",
            "points": [[999, 999, 0, 1, 0.05, 0]],
            "processingVersion": CANONICAL_STROKE_PROCESSING_VERSION,
            "qualityFlags": ["duplicate_replay"],
        },
        {
            "id": "canonical-2",
            "points": [[2.25, 3.5, 0, 140, 0.549, 0]],
            "processingVersion": CANONICAL_STROKE_PROCESSING_VERSION,
            "qualityFlags": [],
        },
    ]

    merged, added = _merge_stroke_docs(existing, incoming)

    assert added == 1
    assert merged[0] == existing[0]
    assert merged[1] == incoming[1]
