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


def test_paper_content_hash_includes_question_and_option_visuals():
    from services.exampen_paper_service import _content_hash

    document = {"document_id": "DOC-VISUAL-HASH", "exam_mode": "pcr"}
    base_question = {
        "id": "q-1",
        "text": "Use the diagram.",
        "marks": 1,
        "question_figures": [{"id": "figure-a", "path": "uploads/figure-a.png"}],
        "images": [{"id": "legacy-option-a", "path": "uploads/option-a.png"}],
        "enhanced_options": [
            {"label": "A", "type": "image", "image_id": "legacy-option-a"}
        ],
    }

    initial_hash = _content_hash(document, [base_question])
    changed_figure_hash = _content_hash(
        document,
        [{**base_question, "question_figures": [{"id": "figure-b", "path": "uploads/figure-b.png"}]}],
    )
    changed_legacy_option_hash = _content_hash(
        document,
        [{**base_question, "images": [{"id": "legacy-option-b", "path": "uploads/option-b.png"}]}],
    )

    assert changed_figure_hash != initial_hash
    assert changed_legacy_option_hash != initial_hash


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


@pytest.mark.asyncio
async def test_pcr_snapshot_materializes_and_reuses_hash_verified_private_assets(monkeypatch):
    """A finalized exam must survive deletion of its authoring upload path."""

    from mongomock_motor import AsyncMongoMockClient

    import services.exampen_paper_service as paper_service

    db = AsyncMongoMockClient()["paper_assets_test"]
    source_objects = {
        "s3://legacy/question.pdf": b"%PDF-1.4 immutable-question-paper",
        "s3://legacy/solution.pdf": b"%PDF-1.4 immutable-teacher-solution",
    }
    private_objects = {}

    async def fake_legacy_download(uri):
        return source_objects[uri]

    async def fake_private_upload(data, *, object_key, content_type, metadata=None):
        uri = f"s3://private-bucket/{object_key}"
        private_objects[uri] = bytes(data)
        return uri

    async def fake_private_download(uri, *, allowed_key_prefix=None, max_bytes=0):
        assert allowed_key_prefix == paper_service.CANONICAL_PAPER_ASSET_PREFIX
        assert max_bytes >= len(private_objects[uri])
        return private_objects[uri]

    monkeypatch.setattr(paper_service, "download_file", fake_legacy_download)
    monkeypatch.setattr(paper_service, "upload_private_object", fake_private_upload)
    monkeypatch.setattr(paper_service, "download_private_object", fake_private_download)

    import hashlib

    document = {
        "document_id": "DOC-ASSET",
        "file_path": "s3://legacy/question.pdf",
        "filename": "question.pdf",
        "sha256": hashlib.sha256(source_objects["s3://legacy/question.pdf"]).hexdigest(),
        "answer_sheet_path": "s3://legacy/solution.pdf",
        "answer_sheet_filename": "solution.pdf",
        "answer_sheet_sha256": hashlib.sha256(
            source_objects["s3://legacy/solution.pdf"]
        ).hexdigest(),
    }

    assets = await paper_service.materialize_paper_assets(db, document)
    assert set(assets) == {"question_paper", "teacher_solution"}
    assert assets["question_paper"]["storage_uri"].startswith(
        "s3://private-bucket/private/exampen/paper-assets/"
    )
    assert await paper_service.load_canonical_paper_asset(assets["question_paper"]) == (
        source_objects["s3://legacy/question.pdf"]
    )

    # Re-finalizing a paper with the same bytes checks the existing object,
    # rather than treating a mutable source path as an exam dependency.
    assets_again = await paper_service.materialize_paper_assets(db, document)
    assert assets_again == assets
    assert await db[paper_service.PAPER_ASSETS_COLLECTION].count_documents({}) == 2

    snapshot = await paper_service.create_paper_snapshot(
        db,
        {**document, "exam_mode": "pcr"},
        _questions(),
        paper_assets=assets,
    )
    assert snapshot["paper_assets"] == assets


@pytest.mark.asyncio
async def test_legacy_asset_backfill_refuses_a_document_that_drifted_after_finalization():
    from mongomock_motor import AsyncMongoMockClient

    from services.exampen_paper_service import (
        CanonicalPaperAssetError,
        migrate_legacy_paper_snapshot_assets,
    )

    db = AsyncMongoMockClient()["legacy_asset_safety_test"]
    await db["exampen_paper_versions"].insert_one(
        {
            "paper_version_id": "paper-legacy",
            "document_id": "DOC-LEGACY",
            "paper_context": {
                "question_paper_sha256": "frozen-question-sha",
                "teacher_solution_sha256": "frozen-solution-sha",
            },
        }
    )

    with pytest.raises(
        CanonicalPaperAssetError,
        match="cannot prove the question-paper SHA-256",
    ):
        await migrate_legacy_paper_snapshot_assets(
            db,
            {
                "document_id": "DOC-LEGACY",
                "sha256": "later-edited-question-sha",
                "answer_sheet_sha256": "frozen-solution-sha",
            },
            paper_version_id="paper-legacy",
        )
