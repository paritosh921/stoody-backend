from __future__ import annotations

import pytest


def _questions():
    return [
        {"id": "q-a", "text": "First", "marks": 1, "rubric": "One mark"},
        {"id": "q-b", "text": "Second", "marks": 2, "rubric": "Two marks"},
    ]


def _regions():
    return {
        "regions": [
            {
                "id": "q-b",
                "pageNumber": 2,
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 40,
                "order": 2,
                "label": "Q2",
                "ocrStatus": "completed",
            },
            {
                "id": "q-a",
                "pageNumber": 1,
                "x": 0,
                "y": 10,
                "width": 100,
                "height": 40,
                "order": 1,
                "label": "Q1",
                "ocrStatus": "completed",
            },
        ],
        "excluded_pages": [],
    }


def test_question_layout_is_complete_and_uses_printed_order():
    from services.exampen_paper_service import build_question_layout

    layout, errors = build_question_layout(
        {"document_id": "DOC-1", "pages_count": 2},
        _questions(),
        _regions(),
    )

    assert errors == []
    assert [item["source_question_id"] for item in layout] == ["q-a", "q-b"]
    assert layout[1]["page_number"] == 2
    assert layout[0]["bbox_percent"]["y"] == 10.0


def test_question_layout_rejects_missing_extra_and_overlapping_regions():
    from services.exampen_paper_service import build_question_layout

    regions = _regions()
    regions["regions"][0]["id"] = "unknown"
    regions["regions"][0]["pageNumber"] = 1
    regions["regions"][0]["y"] = 20

    _layout, errors = build_question_layout(
        {"document_id": "DOC-1", "pages_count": 2},
        _questions(),
        regions,
    )

    assert any("Question q-b" in error and "no saved" in error for error in errors)
    assert any("Region unknown" in error and "no reviewed" in error for error in errors)


def test_question_layout_uses_complete_reviewed_ocr_anchors_when_manual_boxes_are_not_required():
    from services.exampen_paper_service import build_question_layout

    document = {
        "document_id": "DOC-OCR",
        "pages_count": 2,
        "ocr_manual_segmentation_recommended": False,
        "ocr_layout_summary": {
            "pages": [
                {
                    "page": 1,
                    "page_height": 800,
                    "question_anchors": [
                        {"number": "1", "x": 50, "y": 100},
                        {"number": "2", "x": 50, "y": 500},
                    ],
                },
                {
                    "page": 2,
                    "page_height": 800,
                    "question_anchors": [{"number": "3", "x": 50, "y": 100}],
                },
            ]
        },
    }
    questions = [
        {"id": "q-a", "text": "First", "marks": 1, "question_number": 1, "page_number": 1},
        {"id": "q-b", "text": "Second", "marks": 1, "question_number": 2, "page_number": 1},
        {"id": "q-c", "text": "Third", "marks": 1, "question_number": 3, "page_number": 2},
    ]

    layout, errors = build_question_layout(document, questions, None)

    assert errors == []
    assert [item["source_question_id"] for item in layout] == ["q-a", "q-b", "q-c"]
    assert [item["page_number"] for item in layout] == [1, 1, 2]
    assert all(item["layout_source"] == "reviewed_ocr_anchor" for item in layout)
    assert layout[0]["bbox_percent"]["y"] == 0.0
    assert layout[0]["bbox_percent"]["height"] == 37.5
    assert layout[1]["bbox_percent"]["y"] == 37.5
    assert layout[2]["bbox_percent"]["height"] == 100.0


def test_question_layout_does_not_bypass_incomplete_or_untrusted_ocr_anchors():
    from services.exampen_paper_service import build_question_layout

    document = {
        "document_id": "DOC-OCR",
        "pages_count": 1,
        "ocr_manual_segmentation_recommended": False,
        "ocr_layout_summary": {
            "pages": [{"page": 1, "question_anchors": [{"number": "1", "y": 100}]}]
        },
    }
    questions = [
        {"id": "q-a", "text": "First", "marks": 1, "question_number": 1, "page_number": 1},
        {"id": "q-b", "text": "Second", "marks": 1, "question_number": 2, "page_number": 1},
    ]

    layout, errors = build_question_layout(document, questions, None)
    assert layout == []
    assert errors == ["Question 2: no reviewed OCR page anchor"]

    document["ocr_manual_segmentation_recommended"] = True
    layout, errors = build_question_layout(document, questions[:1], None)
    assert layout == []
    assert "Segment every printed question" in errors[0]


def test_full_document_visual_contract_makes_ocr_layout_advisory(monkeypatch):
    from services.exampen_paper_service import resolve_question_layout_for_finalization

    monkeypatch.setenv("PCR_FULL_DOCUMENT_GRADING_ENABLED", "true")
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("PCR_FULL_DOCUMENT_GRADING_MODEL", "gpt-5.1")
    document = {
        "document_id": "DOC-VISUAL",
        "file_path": "s3://private/papers/doc-visual.pdf",
        "sha256": "paper-hash",
        "answer_sheet_path": "s3://private/solutions/doc-visual.pdf",
        "answer_sheet_sha256": "solution-hash",
        "pages_count": 1,
        "ocr_manual_segmentation_recommended": True,
        "ocr_layout_summary": {
            "pages": [
                {
                    "page": 1,
                    "question_anchors": [
                        {"number": number, "y": index * 40}
                        for index, number in enumerate(
                            ("1", "2", "1", "2", "3", "1", "2", "4", "1", "2", "3", "4", "5"),
                            start=1,
                        )
                    ],
                }
            ]
        },
    }
    questions = [
        {
            "id": f"q-{number}",
            "text": f"Question {number}",
            "marks": 5,
            "rubric": "Five marks",
            "question_number": number,
            "page_number": 1,
        }
        for number in range(1, 6)
    ]

    resolved = resolve_question_layout_for_finalization(document, questions, None)

    assert resolved["ready"] is True
    assert resolved["strategy"] == "full_document_visual"
    assert resolved["question_layout"] == []
    assert resolved["errors"] == []
    assert resolved["warnings"]
    assert (
        resolved["paper_context"]["version"]
        == "canonical-full-document-visual-v2"
    )
    assert resolved["paper_context"]["question_paper_sha256"] == "paper-hash"
    assert resolved["paper_context"]["teacher_solution_sha256"] == "solution-hash"
    assert resolved["paper_context"]["requires_question_regions"] is False


def test_layout_remains_required_when_full_document_visual_grading_is_disabled(monkeypatch):
    from services.exampen_paper_service import resolve_question_layout_for_finalization

    monkeypatch.setenv("PCR_FULL_DOCUMENT_GRADING_ENABLED", "false")
    document = {
        "document_id": "DOC-LEGACY",
        "file_path": "s3://private/papers/doc-legacy.pdf",
        "pages_count": 1,
        "ocr_manual_segmentation_recommended": True,
    }

    resolved = resolve_question_layout_for_finalization(
        document,
        [{"id": "q-1", "text": "First", "marks": 1, "rubric": "One mark"}],
        None,
    )

    assert resolved["ready"] is False
    assert resolved["strategy"] == "unavailable"
    assert any("Segment every printed question" in error for error in resolved["errors"])
    assert any("disabled" in error for error in resolved["errors"])


def test_public_readiness_uses_structured_preflight_categories():
    from api.v1.pdf_async import _public_pcr_finalization_readiness

    result = _public_pcr_finalization_readiness(
        {
            "document_id": "DOC-READY",
            "exam_mode": "pcr",
            "ocr_status": "completed",
        },
        {
            "ready": True,
            "questions": [{"id": "q-1"}],
            "errors": [],
            "marking_errors": [],
            "paper_context_errors": [],
            "warnings": ["OCR regions are advisory"],
            "marking_plan": {"questions_using_direct_solution": 1},
            "strategy": "full_document_visual",
            "paper_context": {
                "mode": "full_document_visual",
                "ready": True,
            },
        },
    )

    assert result["ready"] is True
    assert result["strategy"] == "full_document_visual"
    checks = {item["id"]: item for item in result["checks"]}
    assert checks["marking-plan"]["ready"] is True
    assert checks["paper-context"]["ready"] is True
    assert "OCR regions are advisory" in result["warnings"]


@pytest.mark.asyncio
async def test_paper_snapshot_hash_and_rows_include_layout():
    from mongomock_motor import AsyncMongoMockClient

    from services.exampen_paper_service import (
        build_question_layout,
        create_paper_snapshot,
    )

    db = AsyncMongoMockClient()["paper_layout_test"]
    document = {
        "document_id": "DOC-1",
        "exam_mode": "pcr",
        "title": "Layout paper",
        "pages_count": 2,
    }
    layout, errors = build_question_layout(document, _questions(), _regions())
    assert errors == []

    version = await create_paper_snapshot(
        db,
        document,
        _questions(),
        question_layout=layout,
    )

    assert version["layout_status"] == "verified"
    assert version["question_layout"] == layout
    rows = await db["exampen_paper_questions"].find(
        {"paper_version_id": version["paper_version_id"]}
    ).sort("position", 1).to_list(length=10)
    assert [row["layout"]["source_region_id"] for row in rows] == ["q-a", "q-b"]

    moved_layout = [dict(item) for item in layout]
    moved_layout[0] = {
        **moved_layout[0],
        "bbox_percent": {**moved_layout[0]["bbox_percent"], "y": 11.0},
    }
    moved = await create_paper_snapshot(
        db,
        document,
        _questions(),
        question_layout=moved_layout,
    )
    assert moved["content_hash"] != version["content_hash"]


@pytest.mark.asyncio
async def test_paper_snapshot_freezes_full_document_visual_context():
    from mongomock_motor import AsyncMongoMockClient

    from services.exampen_paper_service import create_paper_snapshot

    db = AsyncMongoMockClient()["paper_visual_context_test"]
    document = {
        "document_id": "DOC-VISUAL",
        "exam_mode": "pcr",
        "title": "Visual paper",
        "sha256": "paper-hash",
        "answer_sheet_sha256": "solution-hash",
    }
    context = {
        "version": "canonical-full-document-visual-v1",
        "mode": "full_document_visual",
        "ready": True,
        "question_paper_sha256": "paper-hash",
        "teacher_solution_sha256": "solution-hash",
        "requires_question_regions": False,
    }

    version = await create_paper_snapshot(
        db,
        document,
        _questions(),
        paper_context=context,
    )

    assert version["layout_status"] == "full_document_visual"
    assert version["paper_context"] == context
    assert version["snapshot_status"] == "ready"
