"""Regression coverage for mixed handwritten PCR answer copies.

These tests cover the production failure mode where a student answers a few
questions across multiple pages without reliable Q markers.  The pipeline must
map the visible document regions first; it must never put the full copy under
Q1 and manufacture zeroes for every other question.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest


def _models():
    from api.v1._exampen_imports import load_exampen

    return load_exampen("pcr.domain.response_models")


def _ocr_service():
    from api.v1._exampen_imports import load_exampen

    return load_exampen("pcr.services.ocr_service")


def _mapping_service():
    from api.v1._exampen_imports import load_exampen

    return load_exampen("pcr.services.response_mapping_service")


def _submission_service():
    from api.v1._exampen_imports import load_exampen

    return load_exampen("pcr.services.submission_service")


def _page(page_number: int, blocks: List[tuple[str, float, float]]):
    models = _models()
    return models.PageOCR(
        page_number=page_number,
        page_width_mm=210.0,
        page_height_mm=297.0,
        text_blocks=[
            models.TextBlock(
                text=text,
                bbox=models.BoundingBox(
                    x_min=12.0,
                    y_min=y_start,
                    x_max=196.0,
                    y_max=y_end,
                ),
                confidence=0.96,
                source="camera",
            )
            for text, y_start, y_end in blocks
        ],
        source="camera",
        mean_ocr_confidence=0.96,
    )


def _response(
    response_id: str,
    question_number: int | None,
    text: str,
    page_number: int,
    y_start: float,
    y_end: float,
):
    models = _models()
    return models.DetectedResponse(
        response_id=response_id,
        question_number=question_number,
        sub_part=None,
        detected_text=text,
        source_pages=[
            models.SourcePageRef(
                page_number=page_number,
                y_start=y_start,
                y_end=y_end,
            )
        ],
        content_type=models.ContentType.TEXT_ONLY,
        text_coverage_ratio=1.0,
        segmentation_confidence=0.95,
        ocr_confidence=0.95,
        flags=[],
        word_count=len(text.split()),
        is_continuation=False,
    )


class _StaticVisionGate:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload
        self.calls: List[Dict[str, Any]] = []

    async def call(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(content=json.dumps(self.payload))


class _MemoryIngest:
    def __init__(self, pages: List[Dict[str, Any]]) -> None:
        self.pages = pages
        self.statuses: List[tuple[str, str]] = []

    async def get_submission(self, submission_id: str):
        assert submission_id == "SUB-MIXED"
        return {
            "submission_id": submission_id,
            "exam_id": "EXAM-MIXED",
            "student_id": "STU-1",
            "source": "camera",
        }

    async def get_answer_pages(self, submission_id: str):
        assert submission_id == "SUB-MIXED"
        return self.pages

    async def update_segmentation_status(self, submission_id: str, status: str):
        self.statuses.append((submission_id, status))
        return True


class _MemoryResponses:
    def __init__(self) -> None:
        self.docs: List[Dict[str, Any]] = []

    async def insert_responses_bulk(self, docs: List[Dict[str, Any]]):
        self.docs.extend(docs)
        return len(docs), 0

    async def update_eval_status(self, response_id: str, eval_status: str):
        for doc in self.docs:
            if doc["response_id"] == response_id:
                doc["eval_status"] = eval_status
                return True
        return False

    async def supersede_responses_for_submission(self, *_args, **_kwargs):
        return 0


class _StaticQuestions:
    async def get_questions_by_exam(self, exam_id: str):
        assert exam_id == "EXAM-MIXED"
        return [
            {
                "question_id": f"EXAM-MIXED::Q{number}",
                "question_number": number,
                "question_text": f"Question {number}",
                "max_marks": 4,
            }
            for number in range(1, 6)
        ]


class _MixedCopyOCR:
    def __init__(self, pages) -> None:
        self.pages = pages

    async def recognize_pages(self, _pages_data, *, source: str):
        assert source == "camera"
        return _ocr_service().OCRResult(
            pages=self.pages,
            source=source,
            metadata={"adapter": "mixed-copy-test"},
        )


class _MappedThreeAnswerCopy:
    def __init__(self) -> None:
        self.seen_pages = None

    async def map_submission(self, *, pages, answer_pages, numbered_questions, source):
        assert source == "camera"
        assert len(pages) == 2
        assert [number for number, _question in numbered_questions] == [1, 2, 3, 4, 5]
        assert all(str(page["raw_image_ref"]).startswith("s3://") for page in answer_pages)
        self.seen_pages = list(pages)
        mapping = _mapping_service()
        return mapping.DocumentAnswerMappingResult(
            responses=[
                _response("RESP-MAP-Q1", 1, "projectile working", 1, 15.0, 90.0),
                _response("RESP-MAP-Q3", 3, "work energy working", 1, 145.0, 230.0),
                _response("RESP-MAP-Q5", 5, "thermodynamics working", 2, 20.0, 140.0),
            ],
            assignment_details_by_response={
                response_id: {
                    "method": "document_vision_mapping",
                    "question_number": question_number,
                    "confidence": 0.96,
                    "manual_review_required": False,
                }
                for response_id, question_number in (
                    ("RESP-MAP-Q1", 1),
                    ("RESP-MAP-Q3", 3),
                    ("RESP-MAP-Q5", 5),
                )
            },
            coverage_is_reliable=True,
            manual_review_required=False,
        )


class _NoReliableDocumentMap:
    """Mapper double for proving that unreadable copies stay reviewable."""

    async def map_submission(self, **_kwargs):
        mapping = _mapping_service()
        return mapping.DocumentAnswerMappingResult(
            responses=[],
            coverage_is_reliable=False,
            manual_review_required=True,
            reason="The submitted handwriting could not be read safely",
        )


def test_document_mapping_catalog_uses_immutable_solution_and_rubric_aliases():
    """Imported/finalized papers retain the semantic anchors used for mapping."""
    mapping = _mapping_service()

    prompt = mapping._build_mapping_prompt(
        [_page(1, [])],
        [
            (
                3,
                {
                    "content": "Use work-energy theorem to find the speed.",
                    "solution": "W = delta K, then solve v.",
                    "metadata": {
                        "marking_criteria": [
                            {
                                "description": "Applies work-energy theorem",
                                "acceptable_evidence": "W = change in kinetic energy",
                            }
                        ]
                    },
                },
            )
        ],
    )

    marker = "Question catalog:\n"
    catalog = json.loads(prompt.split(marker, 1)[1].split("\nPage count:", 1)[0])
    assert catalog == [
        {
            "question_number": 3,
            "question": "Use work-energy theorem to find the speed.",
            "reference_solution": "W = delta K, then solve v.",
            "rubric": "",
            "marking_criteria": [
                {
                    "description": "Applies work-energy theorem",
                    "acceptable_evidence": "W = change in kinetic energy",
                }
            ],
        }
    ]


def test_multi_page_camera_copy_always_uses_document_mapping():
    """OCR finding a few labels cannot make an unlabelled third answer vanish."""
    mapping = _mapping_service()
    pages = [_page(1, [("Q1 working", 20.0, 80.0)]), _page(2, [("Q3 working", 30.0, 110.0)])]
    segmented = [
        _response("RESP-Q1", 1, "Q1 working", 1, 20.0, 80.0),
        _response("RESP-Q3", 3, "Q3 working", 2, 30.0, 110.0),
    ]
    questions = [(number, {"question_text": f"Question {number}"}) for number in range(1, 6)]

    assert mapping.needs_document_answer_mapping(
        pages=pages,
        segmented_responses=segmented,
        numbered_questions=questions,
        source="camera",
    ) is True
    assert mapping.needs_document_answer_mapping(
        pages=pages,
        segmented_responses=segmented,
        numbered_questions=questions,
        source="pen",
    ) is False


@pytest.mark.asyncio
async def test_document_mapper_maps_interleaved_regions_across_all_pages(monkeypatch):
    """A mixed copy is associated to Q1/Q3/Q5 before any marking occurs."""
    mapping = _mapping_service()
    pages = [
        _page(
            1,
            [
                ("Projectile: u = 20 m/s; range = 38.4 m.", 20.0, 90.0),
                ("Work energy: W = delta K.", 150.0, 225.0),
            ],
        ),
        _page(2, [("Thermodynamics: Q = delta U + W.", 30.0, 120.0)]),
    ]
    payload = {
        "document_coverage": {"complete": True, "confidence": 0.97},
        "answers": [
            {
                "question_number": 1,
                "confidence": 0.98,
                "mapping_basis": "layout_and_semantics",
                "regions": [{"page_number": 1, "y_start": 0, "y_end": 380}],
                "transcribed_text": "Projectile: u = 20 m/s; range = 38.4 m.",
            },
            {
                "question_number": 3,
                "confidence": 0.95,
                "mapping_basis": "layout_and_semantics",
                "regions": [{"page_number": 1, "y_start": 450, "y_end": 850}],
                "transcribed_text": "Work energy: W = delta K.",
            },
            {
                "question_number": 5,
                "confidence": 0.96,
                "mapping_basis": "continuation",
                "regions": [{"page_number": 2, "y_start": 0, "y_end": 550}],
                "transcribed_text": "Thermodynamics: Q = delta U + W.",
            },
        ],
        "unresolved_regions": [],
    }
    gate = _StaticVisionGate(payload)

    async def _image_loader(_reference: str):
        return "ZmFrZS1pbWFnZQ=="

    monkeypatch.setattr(mapping, "_resolve_image_base64", _image_loader)
    monkeypatch.setattr(mapping, "_get_ocr_vision_model", lambda: "test-vision")

    result = await mapping.DocumentAnswerMapper(gate).map_submission(
        pages=pages,
        answer_pages=[
            {"page_number": 1, "raw_image_ref": "s3://private/exampen/page-1.png"},
            {"page_number": 2, "raw_image_ref": "s3://private/exampen/page-2.png"},
        ],
        numbered_questions=[
            (number, {"question_text": f"Question {number}"})
            for number in range(1, 6)
        ],
        source="camera",
    )

    assert result.coverage_is_reliable is True
    assert result.manual_review_required is False
    assert [response.question_number for response in result.responses] == [1, 3, 5]
    assert "Projectile" in result.responses[0].detected_text
    assert "Work energy" in result.responses[1].detected_text
    assert "Thermodynamics" in result.responses[2].detected_text
    assert gate.calls[0]["kwargs"]["metadata"]["pcr_stage"] == "document_answer_mapping"
    content = gate.calls[0]["kwargs"]["messages"][0]["content"]
    assert sum(part["type"] == "image_url" for part in content) == 2


@pytest.mark.asyncio
async def test_document_mapper_merges_one_answer_continued_across_pages(monkeypatch):
    """One worked answer may span page breaks without becoming two attempts."""
    mapping = _mapping_service()
    pages = [
        _page(1, [("Q1 first working: ux = 16 m/s.", 20.0, 110.0)]),
        _page(2, [("Q1 continuation: range = 38.4 m.", 15.0, 105.0)]),
    ]
    gate = _StaticVisionGate(
        {
            "document_coverage": {"complete": True, "confidence": 0.96},
            "answers": [
                {
                    "question_number": 1,
                    "confidence": 0.97,
                    "mapping_basis": "layout_and_semantics",
                    "regions": [{"page_number": 1, "y_start": 0, "y_end": 430}],
                    "transcribed_text": "ux = 16 m/s",
                },
                {
                    "question_number": 1,
                    "confidence": 0.94,
                    "mapping_basis": "continuation",
                    "regions": [{"page_number": 2, "y_start": 0, "y_end": 430}],
                    "transcribed_text": "range = 38.4 m",
                },
            ],
            "unresolved_regions": [],
        }
    )

    async def _image_loader(_reference: str):
        return "ZmFrZS1pbWFnZQ=="

    monkeypatch.setattr(mapping, "_resolve_image_base64", _image_loader)
    monkeypatch.setattr(mapping, "_get_ocr_vision_model", lambda: "test-vision")

    result = await mapping.DocumentAnswerMapper(gate).map_submission(
        pages=pages,
        answer_pages=[
            {"page_number": 1, "raw_image_ref": "s3://private/exampen/page-1.png"},
            {"page_number": 2, "raw_image_ref": "s3://private/exampen/page-2.png"},
        ],
        numbered_questions=[(1, {"question_text": "Projectile motion"})],
        source="camera",
    )

    assert result.coverage_is_reliable is True
    assert len(result.responses) == 1
    response = result.responses[0]
    assert response.question_number == 1
    assert response.is_continuation is True
    assert [region.page_number for region in response.source_pages] == [1, 2]
    assert "first working" in response.detected_text
    assert "continuation" in response.detected_text
    assert result.assignment_details_by_response[str(response.response_id)][
        "continuation_segment_count"
    ] == 2


@pytest.mark.asyncio
async def test_document_mapper_blocks_coverage_when_visible_ocr_region_is_unmapped(monkeypatch):
    """An omitted answer region must never be converted into a zero-mark blank."""
    mapping = _mapping_service()
    pages = [
        _page(
            1,
            [
                ("Projectile answer: range = 38.4 m.", 20.0, 90.0),
                ("Work energy answer: W = delta K.", 150.0, 225.0),
            ],
        )
    ]
    gate = _StaticVisionGate(
        {
            "document_coverage": {"complete": True, "confidence": 0.99},
            "answers": [
                {
                    "question_number": 1,
                    "confidence": 0.98,
                    "mapping_basis": "layout_and_semantics",
                    "regions": [{"page_number": 1, "y_start": 0, "y_end": 380}],
                    "transcribed_text": "Projectile answer: range = 38.4 m.",
                }
            ],
            "unresolved_regions": [],
        }
    )

    async def _image_loader(_reference: str):
        return "ZmFrZS1pbWFnZQ=="

    monkeypatch.setattr(mapping, "_resolve_image_base64", _image_loader)
    monkeypatch.setattr(mapping, "_get_ocr_vision_model", lambda: "test-vision")

    result = await mapping.DocumentAnswerMapper(gate).map_submission(
        pages=pages,
        answer_pages=[{"page_number": 1, "raw_image_ref": "s3://private/exampen/page-1.png"}],
        numbered_questions=[
            (1, {"question_text": "Projectile motion"}),
            (3, {"question_text": "Work energy"}),
        ],
        source="camera",
    )

    assert result.coverage_is_reliable is False
    assert result.manual_review_required is True
    assert [response.question_number for response in result.responses] == [1, None]
    assert "Work energy" in result.responses[-1].detected_text


@pytest.mark.asyncio
async def test_document_mapper_routes_overlapping_regions_to_teacher_review(monkeypatch):
    """The mapper may not grade two questions from materially the same ink."""
    mapping = _mapping_service()
    pages = [_page(1, [("mixed handwritten working", 30.0, 260.0)])]
    gate = _StaticVisionGate(
        {
            "document_coverage": {"complete": True, "confidence": 0.99},
            "answers": [
                {
                    "question_number": 1,
                    "confidence": 0.98,
                    "mapping_basis": "layout_and_semantics",
                    "regions": [{"page_number": 1, "y_start": 0, "y_end": 650}],
                    "transcribed_text": "first proposed answer",
                },
                {
                    "question_number": 2,
                    "confidence": 0.98,
                    "mapping_basis": "layout_and_semantics",
                    "regions": [{"page_number": 1, "y_start": 350, "y_end": 900}],
                    "transcribed_text": "second proposed answer",
                },
            ],
            "unresolved_regions": [],
        }
    )

    async def _image_loader(_reference: str):
        return "ZmFrZS1pbWFnZQ=="

    monkeypatch.setattr(mapping, "_resolve_image_base64", _image_loader)
    monkeypatch.setattr(mapping, "_get_ocr_vision_model", lambda: "test-vision")

    result = await mapping.DocumentAnswerMapper(gate).map_submission(
        pages=pages,
        answer_pages=[{"page_number": 1, "raw_image_ref": "s3://private/exampen/page-1.png"}],
        numbered_questions=[(1, {"question_text": "Question 1"}), (2, {"question_text": "Question 2"})],
        source="camera",
    )

    assert result.coverage_is_reliable is False
    assert result.manual_review_required is True
    assert [response.question_number for response in result.responses] == [1, None]
    assert result.assignment_details_by_response[str(result.responses[-1].response_id)]["manual_review_required"] is True


@pytest.mark.asyncio
async def test_submission_service_does_not_collapse_mixed_copy_into_q1(monkeypatch):
    """Only confidently mapped answers get scoreable slots; proven blanks remain explicit."""
    submission = _submission_service()
    models = _models()
    pages = [
        _page(1, [("first and third answers interleaved", 15.0, 230.0)]),
        _page(2, [("fifth answer continues here", 20.0, 140.0)]),
    ]
    collapsed = models.SegmentationResult(
        responses=[
            _response(
                "RESP-SEGMENTED-WRONG-Q1",
                1,
                "all mixed OCR text was previously attached to Q1",
                1,
                0.0,
                297.0,
            )
        ],
        flags=[],
        page_count=2,
    )
    monkeypatch.setattr(submission, "segment_submission", lambda **_kwargs: collapsed)

    ingest = _MemoryIngest(
        [
            {"page_number": 1, "raw_image_ref": "s3://private/exampen/student-copy-1.png"},
            {"page_number": 2, "raw_image_ref": "s3://private/exampen/student-copy-2.png"},
        ]
    )
    responses = _MemoryResponses()
    service = submission.SubmissionService(
        ingest=ingest,
        response_repo=responses,
        question_repo=_StaticQuestions(),
        gate=object(),
        ocr_adapter=_MixedCopyOCR(pages),
        document_answer_mapper=_MappedThreeAnswerCopy(),
    )

    result = await service.process_submission("SUB-MIXED")

    assert result.blocked_count == 0
    assert result.response_count == 5
    slots = sorted(responses.docs, key=lambda doc: doc["question_number"])
    assert [slot["question_number"] for slot in slots] == [1, 2, 3, 4, 5]
    detected = [slot for slot in slots if not slot["is_missing_response"]]
    assert [slot["question_number"] for slot in detected] == [1, 3, 5]
    assert [slot["question_assignment"]["method"] for slot in detected] == [
        "document_vision_mapping",
        "document_vision_mapping",
        "document_vision_mapping",
    ]
    blanks = [slot for slot in slots if slot["is_missing_response"]]
    assert [slot["question_number"] for slot in blanks] == [2, 4]
    assert all(slot["eval_status"] == "ready" for slot in slots)
    assert ingest.statuses == [("SUB-MIXED", "complete")]


@pytest.mark.asyncio
async def test_submission_service_maps_private_images_when_text_ocr_is_empty(monkeypatch):
    """Handwriting OCR failure must not skip the private-image mapping stage."""
    submission = _submission_service()
    models = _models()
    pages = [_page(1, []), _page(2, [])]
    monkeypatch.setattr(
        submission,
        "segment_submission",
        lambda **_kwargs: models.SegmentationResult(
            responses=[], flags=[], page_count=2
        ),
    )

    ingest = _MemoryIngest(
        [
            {"page_number": 1, "raw_image_ref": "s3://private/exampen/student-copy-1.png"},
            {"page_number": 2, "raw_image_ref": "s3://private/exampen/student-copy-2.png"},
        ]
    )
    responses = _MemoryResponses()
    mapper = _MappedThreeAnswerCopy()
    service = submission.SubmissionService(
        ingest=ingest,
        response_repo=responses,
        question_repo=_StaticQuestions(),
        gate=object(),
        ocr_adapter=_MixedCopyOCR(pages),
        document_answer_mapper=mapper,
    )

    result = await service.process_submission("SUB-MIXED")

    assert result.error is None
    assert result.response_count == 5
    assert mapper.seen_pages is not None
    assert all(not page.text_blocks for page in mapper.seen_pages)
    assert [doc["question_number"] for doc in responses.docs if not doc["is_missing_response"]] == [1, 3, 5]
    assert ingest.statuses == [("SUB-MIXED", "complete")]


@pytest.mark.asyncio
async def test_submission_service_rebuilds_camera_page_evidence_when_ocr_returns_no_pages(monkeypatch):
    """An empty OCR result still has canonical S3 page images for vision mapping."""
    submission = _submission_service()
    models = _models()
    monkeypatch.setattr(
        submission,
        "segment_submission",
        lambda **_kwargs: models.SegmentationResult(
            responses=[], flags=[], page_count=2
        ),
    )

    ingest = _MemoryIngest(
        [
            {
                "page_number": 1,
                "raw_image_ref": "s3://private/exampen/student-copy-1.png",
                "image_width_px": 1500,
                "image_height_px": 2100,
            },
            {
                "page_number": 2,
                "raw_image_ref": "s3://private/exampen/student-copy-2.png",
            },
        ]
    )
    responses = _MemoryResponses()
    mapper = _MappedThreeAnswerCopy()
    service = submission.SubmissionService(
        ingest=ingest,
        response_repo=responses,
        question_repo=_StaticQuestions(),
        gate=object(),
        ocr_adapter=_MixedCopyOCR([]),
        document_answer_mapper=mapper,
    )

    result = await service.process_submission("SUB-MIXED")

    assert result.error is None
    assert result.page_count == 2
    assert mapper.seen_pages is not None
    assert [page.page_number for page in mapper.seen_pages] == [1, 2]
    assert mapper.seen_pages[0].image_width_px == 1500
    assert all(not page.text_blocks for page in mapper.seen_pages)


@pytest.mark.asyncio
async def test_unreadable_copy_becomes_one_blocked_review_row_not_fake_zeroes(monkeypatch):
    """If visual mapping fails, retain the copy for review without blank slots."""
    submission = _submission_service()
    models = _models()
    monkeypatch.setattr(
        submission,
        "segment_submission",
        lambda **_kwargs: models.SegmentationResult(
            responses=[], flags=[], page_count=2
        ),
    )

    ingest = _MemoryIngest(
        [
            {"page_number": 1, "raw_image_ref": "s3://private/exampen/student-copy-1.png"},
            {"page_number": 2, "raw_image_ref": "s3://private/exampen/student-copy-2.png"},
        ]
    )
    responses = _MemoryResponses()
    service = submission.SubmissionService(
        ingest=ingest,
        response_repo=responses,
        question_repo=_StaticQuestions(),
        gate=object(),
        ocr_adapter=_MixedCopyOCR([]),
        document_answer_mapper=_NoReliableDocumentMap(),
    )

    result = await service.process_submission("SUB-MIXED")

    assert result.error is None
    assert result.response_count == 1
    assert result.blocked_count == 1
    assert len(responses.docs) == 1
    review = responses.docs[0]
    assert review["question_id"] is None
    assert review["question_number"] is None
    assert review["is_missing_response"] is False
    assert review["manual_review_required"] is True
    assert review["eval_status"] == "blocked"
    assert [page["page_number"] for page in review["source_pages"]] == [1, 2]


def test_ocr_normalizes_document_layout_coordinates_from_vision():
    """Vision's 0..1000 boxes must become real page-space regions."""
    ocr = _ocr_service()
    blocks = ocr._parse_ocr_response_to_text_blocks(
        '[{"text":"Q3 work-energy answer","confidence":0.91,"bbox":'
        '{"x_min":100,"y_min":250,"x_max":900,"y_max":500}}]',
        page_width_mm=210.0,
        page_height_mm=297.0,
        source="camera",
    )

    assert len(blocks) == 1
    assert blocks[0].bbox.x_min == pytest.approx(21.0)
    assert blocks[0].bbox.y_min == pytest.approx(74.25)
    assert blocks[0].bbox.y_max == pytest.approx(148.5)
