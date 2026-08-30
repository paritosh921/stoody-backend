import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from api.v1.strokes_async import (
    CANONICAL_STROKE_PROCESSING_VERSION,
    CanvasPageStroke,
    CanvasPageUpsert,
    _incoming_stroke_docs,
    _merge_stroke_docs,
)


CORPUS_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "ble-stroke-replay-corpus"
    / "cases.json"
)


def load_corpus_case(case_id):
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    for item in corpus["cases"]:
        if item["case_id"] == case_id:
            return item
    raise KeyError(f"Corpus case not found: {case_id}")


def canonical_points_from_case(case_id):
    test_case = load_corpus_case(case_id)
    points = []
    for sample in test_case["input"]:
        pressure = int(sample["pressure"])
        if pressure <= 0:
            continue
        points.append(
            [
                float(sample["x"]),
                float(sample["y"]),
                float(sample["pen_ts"] if sample.get("pen_ts") is not None else sample.get("ingress_rx_ts", 0)),
                pressure,
                round(pressure / 255, 4),
                0,
            ]
        )
    return points


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
        pageNumber=4,
        bookType="MS",
        startedAt=0,
        endedAt=24,
    )
    page = CanvasPageUpsert(book_type="MS", page_number=4, strokes=[stroke])

    stored = _incoming_stroke_docs(page)[0]

    assert stored["points"] == canonical_points
    assert stored["processingVersion"] == CANONICAL_STROKE_PROCESSING_VERSION
    assert stored["qualityFlags"] == ["offline_replay_gap"]


def test_canvas_stroke_normalizes_legacy_null_quality_flags():
    stroke = CanvasPageStroke(
        id="legacy-null-flags",
        points=[[1, 2, 0, 120, 0.47, 0]],
        processingVersion=CANONICAL_STROKE_PROCESSING_VERSION,
        qualityFlags=None,
        sourceMode="live",
        pageNumber=4,
        bookType="MS",
        startedAt=10.0,
        endedAt=20.0,
    )

    assert stroke.qualityFlags == []


def test_canvas_stroke_preserves_replay_corpus_canonical_geometry():
    canonical_points = canonical_points_from_case("normal_stroke")
    stroke = CanvasPageStroke(
        id="corpus-normal-stroke",
        points=canonical_points,
        processingVersion=CANONICAL_STROKE_PROCESSING_VERSION,
        qualityFlags=[],
        sourceMode="live",
        pageNumber=7,
        bookType="MS",
        startedAt=canonical_points[0][2],
        endedAt=canonical_points[-1][2],
    )
    page = CanvasPageUpsert(book_type="MS", page_number=7, strokes=[stroke])

    stored = _incoming_stroke_docs(page)[0]

    assert stored["points"] == canonical_points
    assert stored["processingVersion"] == CANONICAL_STROKE_PROCESSING_VERSION


def test_canonical_stroke_rejects_non_canonical_point_shape():
    with pytest.raises(ValidationError):
        CanvasPageStroke(
            id="bad-canonical-1",
            points=[[1, 2, 0.5]],
            processingVersion=CANONICAL_STROKE_PROCESSING_VERSION,
        )


def test_canonical_stroke_requires_transport_metadata():
    with pytest.raises(ValidationError):
        CanvasPageStroke(
            id="missing-canonical-metadata",
            points=[[1, 2, 0, 120, 0.47, 0]],
            processingVersion=CANONICAL_STROKE_PROCESSING_VERSION,
            qualityFlags=[],
            sourceMode="live",
            pageNumber=4,
            bookType="MS",
            startedAt=10.0,
        )

    with pytest.raises(ValidationError):
        CanvasPageStroke(
            id="bad-source-mode",
            points=[[1, 2, 0, 120, 0.47, 0]],
            processingVersion=CANONICAL_STROKE_PROCESSING_VERSION,
            qualityFlags=[],
            sourceMode="renderer",
            pageNumber=4,
            bookType="MS",
            startedAt=10.0,
            endedAt=20.0,
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
