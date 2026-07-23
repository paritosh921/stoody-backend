import asyncio

import pytest


def test_layout_preflight_flags_staggered_label_only_options():
    from services.layout_preflight_service import LayoutPreflightService

    text_items = [
        {"text": "a. B > C > A", "x": 20, "y": 10, "width": 80, "height": 10},
        {"text": "A > C > B", "x": 36, "y": 24, "width": 70, "height": 10},
        {"text": "b.", "x": 20, "y": 36, "width": 8, "height": 10},
        {"text": "c. A > B > C", "x": 20, "y": 50, "width": 80, "height": 10},
        {"text": "B > A > C", "x": 36, "y": 64, "width": 70, "height": 10},
        {"text": "d.", "x": 20, "y": 76, "width": 8, "height": 10},
    ]

    report = LayoutPreflightService().analyze(
        region_id="region-1",
        text_items=text_items,
        embedded_images=[],
    )

    assert report["has_text_layer"] is True
    assert "staggered_options" in report["layout_risks"]
    assert report["option_layout"]["label_only_lines"] == ["b", "d"]
    assert report["recommended_strategy"] == "ocr_with_layout_hints"


def test_layout_preflight_handles_scanned_region_without_text_layer():
    from services.layout_preflight_service import LayoutPreflightService

    report = LayoutPreflightService().analyze(
        region_id="scan-1",
        text_items=[],
        embedded_images=[],
    )

    assert report["has_text_layer"] is False
    assert report["layout_risks"] == []
    assert report["recommended_strategy"] == "ocr_then_validate"


def test_option_normalizer_attaches_label_only_lines_to_previous_text():
    from services.option_layout_normalizer import OptionLayoutNormalizer

    text_items = [
        {"text": "a. B > C > A", "x": 20, "y": 10, "width": 80, "height": 10},
        {"text": "A > C > B", "x": 36, "y": 24, "width": 70, "height": 10},
        {"text": "b.", "x": 20, "y": 36, "width": 8, "height": 10},
        {"text": "c. A > B > C", "x": 20, "y": 50, "width": 80, "height": 10},
        {"text": "B > A > C", "x": 36, "y": 64, "width": 70, "height": 10},
        {"text": "d.", "x": 20, "y": 76, "width": 8, "height": 10},
    ]

    result = OptionLayoutNormalizer().correct(text_items=text_items, layout_report={})

    assert result["options_by_label"] == {
        "a": "B > C > A",
        "b": "A > C > B",
        "c": "A > B > C",
        "d": "B > A > C",
    }
    assert result["manual_review_required"] is False
    assert [c["label"] for c in result["corrections"]] == ["b", "d"]


def test_extraction_validator_flags_invalid_mcq_options_for_review():
    from services.extraction_validator import ExtractionValidator

    result = ExtractionValidator().validate_question(
        question_text="Which order is correct?",
        options=["B > C > A", "", "B > C > A"],
        layout_report={"layout_risks": ["staggered_options"]},
        expected_option_count=4,
    )

    assert result["valid"] is False
    assert result["manual_review_required"] is True
    assert "option_count_mismatch" in result["reasons"]
    assert "empty_option" in result["reasons"]
    assert "duplicate_option_text" in result["reasons"]


def test_region_crop_service_returns_crop_pdf_preview_and_text_items():
    import fitz

    from services.region_crop_service import RegionCropService

    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((20, 40), "a. Alpha option", fontsize=12)
    page.insert_text((20, 60), "b. Beta option", fontsize=12)
    pdf_content = doc.tobytes()
    doc.close()

    crop = RegionCropService().crop(
        pdf_content=pdf_content,
        page_number=1,
        bbox={"x": 0, "y": 0, "width": 100, "height": 50},
        region_id="region-1",
        region_scope="question",
    )

    assert crop["region_id"] == "region-1"
    assert crop["region_pdf_bytes"].startswith(b"%PDF")
    assert crop["region_png_base64"]
    assert any(item["text"].startswith("a.") for item in crop["text_items"])
    assert crop["crop_metadata"]["page"] == 1


def test_document_layout_provider_pymupdf_detects_question_and_staggered_options(monkeypatch):
    import fitz

    from services.document_layout_provider import DocumentLayoutProvider

    monkeypatch.setenv("LITEPARSE_LAYOUT_ENABLED", "false")
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    page.insert_text((20, 30), "1. Which sequence is correct?", fontsize=10)
    page.insert_text((20, 50), "a. Alpha", fontsize=10)
    page.insert_text((35, 70), "Beta", fontsize=10)
    page.insert_text((20, 85), "b.", fontsize=10)
    page.insert_text((20, 105), "c. Gamma", fontsize=10)
    page.insert_text((35, 125), "Delta", fontsize=10)
    page.insert_text((20, 140), "d.", fontsize=10)
    page.insert_text((20, 160), "e. Epsilon", fontsize=10)
    pdf = doc.tobytes()
    doc.close()

    report = asyncio.run(
        DocumentLayoutProvider().analyze(
            pdf_bytes=pdf,
            document_id="doc-1",
            mode="question_paper",
        )
    )

    assert report["provider"] == "pymupdf"
    assert report["page_count"] == 1
    assert report["pages"][0]["question_anchors"][0]["number"] == "1"
    assert "staggered_options_possible" in report["document_layout_risks"]
    evidence = report["pages"][0]["question_option_evidence"][0]
    assert evidence["question_number"] == "1"
    assert evidence["option_labels_found"] == ["A", "B", "C", "D", "E"]
    assert evidence["expected_option_count"] == 5


def test_full_document_validator_marks_low_question_count_against_anchors():
    from api.v1.pdf_async import ExtractedQuestion
    from services.full_document_extraction_validator import FullDocumentExtractionValidator

    questions = [ExtractedQuestion(id="q1", text="Question 1", options=["a", "b", "c", "d"], metadata={"number": "1"})]
    layout_report = {
        "pages": [
            {"question_anchors": [{"number": "1"}, {"number": "2"}]},
        ]
    }

    summary = FullDocumentExtractionValidator().validate_questions(
        questions=questions,
        layout_report=layout_report,
    )

    assert summary["status"] == "manual_segmentation_recommended"
    assert "question_count_lower_than_layout_anchors" in summary["reasons"]


def test_full_document_validator_treats_repeated_numbered_subparts_as_advisory():
    from api.v1.pdf_async import ExtractedQuestion
    from services.full_document_extraction_validator import FullDocumentExtractionValidator

    questions = [
        ExtractedQuestion(
            id=f"q{number}",
            text=(
                f"Calculate the equilibrium result for reaction {number} "
                "and show every step in your method."
            ),
            metadata={"number": str(number)},
        )
        for number in range(1, 6)
    ]
    layout_report = {
        "pages": [
            {
                "question_anchors": [
                    {"number": number}
                    for number in ("1", "2", "1", "2", "3", "1", "2", "4", "1", "2", "3", "4", "5")
                ]
            }
        ]
    }

    summary = FullDocumentExtractionValidator().validate_questions(
        questions=questions,
        layout_report=layout_report,
        skip_option_extraction=True,
        objective_questions=False,
    )

    assert summary["status"] == "needs_review"
    assert summary["manual_segmentation_recommended"] is False
    assert "numbered_subparts_possible" in summary["reasons"]
    assert "question_count_lower_than_layout_anchors" not in summary["reasons"]


def test_full_document_validator_accepts_layout_supported_three_option_mcq():
    from api.v1.pdf_async import ExtractedQuestion
    from services.full_document_extraction_validator import FullDocumentExtractionValidator

    questions = [
        ExtractedQuestion(
            id="q1",
            text="Which statement is correct?",
            options=["alpha", "beta", "gamma"],
            metadata={"question_number": "1"},
        )
    ]
    layout_report = {
        "pages": [
            {
                "question_anchors": [{"number": "1"}],
                "question_option_evidence": [
                    {
                        "question_number": "1",
                        "option_labels_found": ["A", "B", "C"],
                        "expected_option_count": 3,
                        "evidence_confidence": 0.9,
                    }
                ],
            }
        ]
    }

    summary = FullDocumentExtractionValidator().validate_questions(
        questions=questions,
        layout_report=layout_report,
    )

    assert summary["status"] == "trusted_draft"
    assert summary["warnings"] == []


def test_full_document_validator_flags_missing_options_when_five_option_layout_has_two_extracted():
    from api.v1.pdf_async import ExtractedQuestion
    from services.full_document_extraction_validator import FullDocumentExtractionValidator

    questions = [
        ExtractedQuestion(
            id="q1",
            text="Choose the best alternative from the options.",
            options=["alpha", "beta"],
            metadata={"question_number": "1"},
        )
    ]
    layout_report = {
        "pages": [
            {
                "question_anchors": [{"number": "1"}],
                "question_option_evidence": [
                    {
                        "question_number": "1",
                        "option_labels_found": ["A", "B", "C", "D", "E"],
                        "expected_option_count": 5,
                        "evidence_confidence": 0.88,
                    }
                ],
            }
        ]
    }

    summary = FullDocumentExtractionValidator().validate_questions(
        questions=questions,
        layout_report=layout_report,
    )

    warning = summary["warnings"][0]
    assert "missing_options_detected" in warning["reasons"]
    assert warning["observed_option_count"] == 2
    assert warning["expected_option_count"] == 5
    assert warning["missing_option_labels"] == ["C", "D", "E"]


def test_full_document_validator_uses_ocr_markdown_option_evidence_for_scanned_pdf():
    from api.v1.pdf_async import ExtractedQuestion
    from services.full_document_extraction_validator import FullDocumentExtractionValidator

    questions = [
        ExtractedQuestion(
            id="q1",
            text="Choose the best answer from the following.",
            options=["alpha", "beta"],
            metadata={"question_number": "1"},
        )
    ]
    layout_report = {
        "has_text_layer": False,
        "pages": [
            {
                "question_anchors": [{"number": "1"}],
                "question_option_evidence": [],
            }
        ],
    }
    ocr_result = {
        "pages": [
            {
                "index": 0,
                "markdown": "1. Choose the best answer from the following.\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta",
            }
        ]
    }

    summary = FullDocumentExtractionValidator().validate_questions(
        questions=questions,
        layout_report=layout_report,
        ocr_result=ocr_result,
    )

    warning = summary["warnings"][0]
    assert "missing_options_detected" in warning["reasons"]
    assert warning["expected_option_count"] == 4
    assert warning["missing_option_labels"] == ["C", "D"]
    assert warning["option_evidence"]["source"] == "ocr_markdown"


def test_full_document_validator_accepts_layout_supported_two_option_objective_question():
    from api.v1.pdf_async import ExtractedQuestion
    from services.full_document_extraction_validator import FullDocumentExtractionValidator

    questions = [
        ExtractedQuestion(
            id="q1",
            text="The statement is true or false.",
            options=["True", "False"],
            metadata={"question_number": "1"},
        )
    ]
    layout_report = {
        "pages": [
            {
                "question_anchors": [{"number": "1"}],
                "question_option_evidence": [
                    {
                        "question_number": "1",
                        "option_labels_found": ["A", "B"],
                        "expected_option_count": 2,
                        "evidence_confidence": 0.86,
                    }
                ],
            }
        ]
    }

    summary = FullDocumentExtractionValidator().validate_questions(
        questions=questions,
        layout_report=layout_report,
    )

    assert "missing_options_detected" not in summary["reasons"]
    assert summary["status"] == "trusted_draft"


def test_full_document_validator_flags_incomplete_question_text():
    from api.v1.pdf_async import ExtractedQuestion
    from services.full_document_extraction_validator import FullDocumentExtractionValidator

    questions = [
        ExtractedQuestion(
            id="q1",
            text="Which of the",
            options=["alpha", "beta", "gamma"],
            metadata={"question_number": "1"},
        )
    ]

    summary = FullDocumentExtractionValidator().validate_questions(
        questions=questions,
        layout_report={"pages": [{"question_anchors": [{"number": "1"}]}]},
    )

    assert "incomplete_question_text" in summary["warnings"][0]["reasons"]
    assert summary["warnings"][0]["reason_severities"]["incomplete_question_text"] == "medium"


def test_document_layout_provider_does_not_infer_subjective_subparts_as_options():
    from services.document_layout_provider import DocumentLayoutProvider

    report = DocumentLayoutProvider()._text_report_for_page(
        text=(
            "1. Answer the following parts.\n"
            "(a) Prove that the sequence is convergent.\n"
            "(b) Explain why the limit exists.\n"
            "(c) Calculate the final value."
        ),
        page_number=1,
        mode="question",
        text_blocks=[],
        width=None,
        height=None,
    )

    evidence = report["question_option_evidence"][0]
    assert evidence["option_labels_found"] == ["A", "B", "C"]
    assert evidence["expected_option_count"] is None
    assert evidence["evidence_confidence"] < 0.7


def test_answer_sheet_block_normalizer_groups_anchor_numbered_answers():
    from services.answer_sheet_block_normalizer import AnswerSheetBlockNormalizer

    result = AnswerSheetBlockNormalizer().normalize(
        pages=[
            {"index": 1, "markdown": "exp 1. Start of answer\nsecond line\nexp 2. Next answer"}
        ],
        anchor_text="exp",
    )

    assert result["answer_count"] == 2
    assert result["answers"][0]["number"] == "1"
    assert "second line" in result["answers"][0]["text"]


def test_answer_sheet_block_normalizer_does_not_split_plain_numbered_steps():
    from services.answer_sheet_block_normalizer import AnswerSheetBlockNormalizer

    result = AnswerSheetBlockNormalizer().normalize(
        pages=[
            {
                "index": 1,
                "markdown": "exp 1. Worked answer starts\n1. square both sides\n2. simplify\ntherefore option A",
            }
        ],
        anchor_text="exp",
    )

    assert result["answer_count"] == 1
    assert "2. simplify" in result["answers"][0]["text"]


def test_document_layout_provider_answer_anchors_require_solution_cue(monkeypatch):
    import fitz

    from services.document_layout_provider import DocumentLayoutProvider

    monkeypatch.setenv("LITEPARSE_LAYOUT_ENABLED", "false")
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    page.insert_text((20, 30), "1. square both sides", fontsize=10)
    page.insert_text((20, 50), "2. simplify expression", fontsize=10)
    page.insert_text((20, 70), "exp 3. worked solution", fontsize=10)
    pdf = doc.tobytes()
    doc.close()

    report = asyncio.run(
        DocumentLayoutProvider().analyze(
            pdf_bytes=pdf,
            document_id="doc-1",
            mode="answer_sheet",
        )
    )

    anchors = report["pages"][0]["answer_anchors"]
    assert [anchor["number"] for anchor in anchors] == ["3"]


def test_answer_solution_coverage_not_expected_when_no_answer_mode():
    from services.answer_solution_coverage_service import AnswerSolutionCoverageService

    result = AnswerSolutionCoverageService().compute(
        document={"answer_solution_mode": "none"},
        questions=[{"id": "q1"}],
        mappings=[],
    )

    assert result["answer_solution_coverage_status"] == "not_expected"
    assert result["answer_solution_coverage_score"] == 0.0


def test_answer_solution_coverage_pending_when_answer_sheet_uploaded_not_processed():
    from services.answer_solution_coverage_service import AnswerSolutionCoverageService

    result = AnswerSolutionCoverageService().compute(
        document={
            "answer_solution_mode": "upload",
            "answer_sheet_path": "uploads/answer.pdf",
            "answer_sheet_ocr_status": "not_processed",
        },
        questions=[{"id": "q1"}, {"id": "q2"}],
        mappings=[],
    )

    assert result["answer_solution_coverage_status"] == "pending"
    assert result["answer_solution_coverage_summary"]["answer_source"] == "upload"
    assert "answer_sheet_ocr_pending" in result["answer_solution_coverage_summary"]["reasons"]


def test_answer_solution_coverage_not_ready_when_answer_sheet_maps_zero_answers():
    from services.answer_solution_coverage_service import AnswerSolutionCoverageService

    result = AnswerSolutionCoverageService().compute(
        document={
            "answer_solution_mode": "upload",
            "answer_sheet_path": "uploads/answer.pdf",
            "answer_sheet_ocr_status": "completed",
            "ocr_quality_score": 0.99,
        },
        questions=[{"id": f"q{i}"} for i in range(1, 6)],
        mappings=[],
    )

    assert result["answer_solution_coverage_status"] == "not_ready"
    assert result["answer_solution_coverage_score"] == 0.0
    assert result["answer_solution_coverage_summary"]["manual_segmentation_recommended"] is True
    assert "no_answers_mapped" in result["answer_solution_coverage_summary"]["reasons"]


def test_answer_solution_coverage_needs_review_with_partial_review_mappings():
    from services.answer_solution_coverage_service import AnswerSolutionCoverageService

    mappings = [
        {
            "question_id": f"q{i}",
            "answer_text": f"solution {i}",
            "manual_review_required": i in {1, 2},
            "source": "manual_answer_segmentation",
        }
        for i in range(1, 8)
    ]

    result = AnswerSolutionCoverageService().compute(
        document={
            "answer_solution_mode": "upload",
            "answer_sheet_path": "uploads/answer.pdf",
            "answer_sheet_processed_regions_count": 7,
        },
        questions=[{"id": f"q{i}"} for i in range(1, 11)],
        mappings=mappings,
    )

    assert result["answer_solution_coverage_status"] == "needs_review"
    assert result["answer_solution_coverage_score"] == 0.7
    assert result["answer_solution_coverage_summary"]["manual_review_count"] == 2


def test_answer_solution_coverage_ready_for_generated_solutions():
    from services.answer_solution_coverage_service import AnswerSolutionCoverageService

    result = AnswerSolutionCoverageService().compute(
        document={
            "answer_solution_mode": "auto",
            "generated_solutions_status": "completed",
            "generated_solutions_count": 3,
        },
        questions=[{"id": f"q{i}"} for i in range(1, 4)],
        mappings=[
            {
                "question_id": f"q{i}",
                "answer_text": f"solution {i}",
                "manual_review_required": False,
                "source": "ai_generated",
            }
            for i in range(1, 4)
        ],
    )

    assert result["answer_solution_coverage_status"] == "ready"
    assert result["answer_solution_coverage_score"] == 1.0
    assert result["answer_solution_coverage_summary"]["answer_source"] == "generated"


def test_answer_solution_coverage_upload_mode_ignores_generated_mappings():
    from services.answer_solution_coverage_service import AnswerSolutionCoverageService

    result = AnswerSolutionCoverageService().compute(
        document={
            "answer_solution_mode": "upload",
            "answer_sheet_path": "uploads/answer.pdf",
            "answer_sheet_ocr_status": "completed",
            "generated_solutions_count": 3,
        },
        questions=[{"id": f"q{i}"} for i in range(1, 4)],
        mappings=[
            {
                "question_id": f"q{i}",
                "answer_text": f"generated solution {i}",
                "manual_review_required": False,
                "source": "ai_generated",
                "mapping_strategy": "ai_generated_solution",
            }
            for i in range(1, 4)
        ],
    )

    assert result["answer_solution_coverage_status"] == "not_ready"
    assert result["answer_solution_coverage_score"] == 0.0
    assert result["answer_solution_coverage_summary"]["answer_source"] == "upload"
    assert "no_answers_mapped" in result["answer_solution_coverage_summary"]["reasons"]


def test_answer_solution_coverage_ignores_stale_question_mappings_and_clamps_score():
    from services.answer_solution_coverage_service import AnswerSolutionCoverageService

    result = AnswerSolutionCoverageService().compute(
        document={
            "answer_solution_mode": "upload",
            "answer_sheet_path": "uploads/answer.pdf",
            "answer_sheet_processed_regions_count": 2,
        },
        questions=[{"id": "q-current"}],
        mappings=[
            {
                "question_id": "q-old-1",
                "answer_text": "old solution 1",
                "manual_review_required": False,
                "source": "manual_answer_segmentation",
            },
            {
                "question_id": "q-old-2",
                "answer_text": "old solution 2",
                "manual_review_required": False,
                "source": "manual_answer_segmentation",
            },
        ],
    )

    assert result["answer_solution_coverage_status"] == "not_ready"
    assert result["answer_solution_coverage_score"] == 0.0
    assert result["answer_solution_coverage_summary"]["mapped_answer_count"] == 0
    assert result["answer_solution_coverage_summary"]["stale_mapping_count"] == 2


def test_answer_sheet_validator_flags_zero_mappings_when_questions_exist():
    from services.full_document_extraction_validator import FullDocumentExtractionValidator

    summary = FullDocumentExtractionValidator().validate_answer_sheet(
        extracted_text="1. Worked answer\n2. Worked answer",
        page_summaries=[{"index": 0, "markdown": "1. Worked answer\n2. Worked answer"}],
        layout_report={"pages": [{"answer_anchors": [{"number": "1"}, {"number": "2"}]}]},
        mapped_count=0,
        question_count=2,
    )

    assert "mapped_answer_count_lower_than_answer_anchors" in summary["reasons"]
    assert "no_answers_mapped" in summary["reasons"]


class _FakeGatewayDb:
    def __init__(self):
        self.events = []
        self.counters = {}
        self.fail_tenant_reservation = False

    async def insert_event(self, event):
        self.events.append(dict(event))
        return event["event_id"]

    async def update_event(self, event_id, updates):
        for event in self.events:
            if event["event_id"] == event_id:
                event.update(updates)
                return True
        return False

    async def reserve_usage(
        self,
        *,
        scope,
        subject_id,
        period="daily",
        period_key,
        tokens,
        limit=None,
        page_units=0,
        call_units=1,
        token_limit=None,
        page_limit=None,
        call_limit=None,
    ):
        if scope == "tenant" and self.fail_tenant_reservation:
            key = (scope, subject_id, period_key)
            return False, {"metric": "tokens", "limit": token_limit if token_limit is not None else limit, "used": self.counters.get(key, 0), "required": tokens}
        key = (scope, subject_id, period_key)
        used = self.counters.get(key, 0)
        effective_token_limit = token_limit if token_limit is not None else limit
        if effective_token_limit is not None and used + tokens > effective_token_limit:
            return False, {"metric": "tokens", "limit": effective_token_limit, "used": used, "required": tokens}
        page_key = (scope, subject_id, period_key, "pages")
        page_used = self.counters.get(page_key, 0)
        if page_limit is not None and page_used + page_units > page_limit:
            return False, {"metric": "page_units", "limit": page_limit, "used": page_used, "required": page_units}
        self.counters[key] = used + tokens
        self.counters[page_key] = page_used + page_units
        return True, {"metric": "tokens", "limit": effective_token_limit, "used": self.counters[key], "required": tokens}

    async def release_usage(self, *, scope, subject_id, period="daily", period_key, tokens, page_units=0, call_units=1):
        key = (scope, subject_id, period_key)
        self.counters[key] = max(0, self.counters.get(key, 0) - tokens)
        page_key = (scope, subject_id, period_key, "pages")
        self.counters[page_key] = max(0, self.counters.get(page_key, 0) - page_units)
        return True


def test_ai_gateway_blocks_before_provider_call(monkeypatch):
    from services.ai_gateway_service import AIGatewayService, AIUsageLimitExceeded

    monkeypatch.setenv("AI_GATEWAY_ENABLED", "true")
    monkeypatch.setenv("AI_BLOCK_ON_LIMIT", "true")
    monkeypatch.setenv("AI_DAILY_TOKEN_LIMIT_PER_USER", "10")
    db = _FakeGatewayDb()
    called = False

    async def provider_call():
        nonlocal called
        called = True
        return {"ok": True}

    async def run_case():
        with pytest.raises(AIUsageLimitExceeded):
            await AIGatewayService(db).call(
                user_id="user-1",
                tenant_id="tenant-1",
                document_id="doc-1",
                region_id="region-1",
                region_scope="question",
                stage="question_structuring",
                provider="groq",
                model="openai/gpt-oss-120b",
                input_kind="text",
                estimated_input_tokens=8,
                estimated_output_tokens=8,
                call_fn=provider_call,
            )

    asyncio.run(run_case())

    assert called is False
    assert db.events[-1]["status"] == "blocked"


def test_ai_gateway_records_success_event(monkeypatch):
    from services.ai_gateway_service import AIGatewayService

    monkeypatch.setenv("AI_GATEWAY_ENABLED", "true")
    monkeypatch.setenv("AI_DAILY_TOKEN_LIMIT_PER_USER", "100")
    db = _FakeGatewayDb()

    async def provider_call():
        return {"ok": True, "usage": {"prompt_tokens": 7, "completion_tokens": 3}}

    async def run_case():
        return await AIGatewayService(db).call(
            user_id="user-1",
            tenant_id="tenant-1",
            document_id="doc-1",
            region_id="region-1",
            region_scope="question",
            stage="question_structuring",
            provider="openai",
            model="gpt-5-mini",
            input_kind="text",
            estimated_input_tokens=8,
            estimated_output_tokens=8,
            call_fn=provider_call,
        )

    result = asyncio.run(run_case())

    assert result["ok"] is True
    assert db.events[-1]["status"] == "success"
    assert db.events[-1]["actual_input_tokens"] == 7
    assert db.events[-1]["actual_output_tokens"] == 3


def test_ai_gateway_records_success_when_soft_limit_allows_call(monkeypatch):
    from services.ai_gateway_service import AIGatewayService

    monkeypatch.setenv("AI_GATEWAY_ENABLED", "true")
    monkeypatch.setenv("AI_BLOCK_ON_LIMIT", "false")
    monkeypatch.setenv("AI_DAILY_TOKEN_LIMIT_PER_USER", "10")
    db = _FakeGatewayDb()
    called = False

    async def provider_call():
        nonlocal called
        called = True
        return {"ok": True, "usage": {"prompt_tokens": 5, "completion_tokens": 2}}

    async def run_case():
        return await AIGatewayService(db).call(
            user_id="user-1",
            tenant_id="tenant-1",
            document_id="doc-1",
            region_id="region-1",
            region_scope="question",
            stage="question_structuring",
            provider="openai",
            model="gpt-5-mini",
            input_kind="text",
            estimated_input_tokens=8,
            estimated_output_tokens=8,
            call_fn=provider_call,
        )

    result = asyncio.run(run_case())

    assert called is True
    assert result["ok"] is True
    assert [event["status"] for event in db.events] == ["blocked", "success"]
    assert db.events[-1]["actual_input_tokens"] == 5
    assert db.events[-1]["actual_output_tokens"] == 2


def test_ai_gateway_rolls_back_user_reservation_when_tenant_blocks(monkeypatch):
    from services.ai_gateway_service import AIGatewayService, AIUsageLimitExceeded

    monkeypatch.setenv("AI_GATEWAY_ENABLED", "true")
    monkeypatch.setenv("AI_BLOCK_ON_LIMIT", "true")
    monkeypatch.setenv("AI_DAILY_TOKEN_LIMIT_PER_USER", "100")
    db = _FakeGatewayDb()
    db.fail_tenant_reservation = True

    async def provider_call():
        return {"ok": True}

    async def run_case():
        with pytest.raises(AIUsageLimitExceeded):
            await AIGatewayService(db).call(
                user_id="user-1",
                tenant_id="tenant-1",
                document_id="doc-1",
                region_id="region-1",
                region_scope="question",
                stage="question_structuring",
                provider="openai",
                model="gpt-5-mini",
                input_kind="text",
                estimated_input_tokens=8,
                estimated_output_tokens=8,
                call_fn=provider_call,
            )

    asyncio.run(run_case())

    user_counters = {
        key: value
        for key, value in db.counters.items()
        if key[0] == "user"
    }
    assert all(value == 0 for value in user_counters.values())
    assert db.events[-1]["status"] == "blocked"
    assert db.events[-1]["error"] == "ai_token_limit_exceeded"


def test_ai_gateway_enforces_monthly_user_limit(monkeypatch):
    from services.ai_gateway_service import AIGatewayService, AIUsageLimitExceeded

    monkeypatch.setenv("AI_GATEWAY_ENABLED", "true")
    monkeypatch.setenv("AI_BLOCK_ON_LIMIT", "true")
    monkeypatch.delenv("AI_DAILY_TOKEN_LIMIT_PER_USER", raising=False)
    monkeypatch.setenv("AI_MONTHLY_TOKEN_LIMIT_PER_USER", "10")
    db = _FakeGatewayDb()

    async def provider_call():
        return {"ok": True}

    async def run_case():
        with pytest.raises(AIUsageLimitExceeded):
            await AIGatewayService(db).call(
                user_id="user-1",
                tenant_id="tenant-1",
                document_id="doc-1",
                region_id="region-1",
                region_scope="question",
                stage="question_structuring",
                provider="openai",
                model="gpt-5-mini",
                input_kind="text",
                estimated_input_tokens=8,
                estimated_output_tokens=8,
                call_fn=provider_call,
            )

    asyncio.run(run_case())

    assert db.events[-1]["status"] == "blocked"
    assert db.events[-1]["error"] == "ai_token_limit_exceeded"


def test_ai_gateway_enforces_page_units_before_provider_call(monkeypatch):
    from services.ai_gateway_service import AIGatewayService, AIUsageLimitExceeded

    monkeypatch.setenv("AI_GATEWAY_ENABLED", "true")
    monkeypatch.setenv("AI_BLOCK_ON_LIMIT", "true")
    monkeypatch.delenv("AI_DAILY_TOKEN_LIMIT_PER_USER", raising=False)
    monkeypatch.setenv("AI_DAILY_PAGE_LIMIT_PER_USER", "2")
    db = _FakeGatewayDb()
    called = False

    async def provider_call():
        nonlocal called
        called = True
        return {"ok": True}

    async def run_case():
        with pytest.raises(AIUsageLimitExceeded):
            await AIGatewayService(db).call(
                user_id="user-1",
                tenant_id="tenant-1",
                document_id="doc-1",
                region_id=None,
                region_scope="document",
                stage="ocr_primary",
                provider="mistral",
                model="mistral-ocr-latest",
                input_kind="pdf_region",
                estimated_input_tokens=1,
                input_units={"page_count": 3},
                call_fn=provider_call,
            )

    asyncio.run(run_case())

    assert called is False
    assert db.events[-1]["status"] == "blocked"
    assert db.events[-1]["metric"] == "page_units"


def test_ai_gateway_mongo_path_blocks_single_call_over_page_limit(monkeypatch):
    from services.ai_gateway_service import AIGatewayService, AIUsageLimitExceeded

    class FakeCollection:
        def __init__(self):
            self.docs = {}
            self.inserted_events = []

        async def insert_one(self, doc):
            if doc.get("event_id"):
                self.inserted_events.append(dict(doc))
                return type("Result", (), {"inserted_id": doc["event_id"]})()
            self.docs[doc["counter_id"]] = dict(doc)
            return type("Result", (), {"inserted_id": doc["counter_id"]})()

        async def find_one(self, query):
            if "counter_id" in query:
                return self.docs.get(query["counter_id"])
            return None

        async def find_one_and_update(self, query, update, upsert=False, return_document=None):
            counter_id = query.get("counter_id")
            doc = self.docs.get(counter_id)
            if doc is None and upsert:
                doc = {
                    "counter_id": counter_id,
                    "reserved_tokens": 0,
                    "reserved_page_units": 0,
                    "reserved_calls": 0,
                }
                self.docs[counter_id] = doc
            if doc is None:
                return None
            for condition in query.get("$and", []):
                matched_or = False
                for option in condition.get("$or", []):
                    field, requirement = next(iter(option.items()))
                    if "$exists" in requirement:
                        matched_or = matched_or or ((field in doc) is bool(requirement["$exists"]))
                    elif "$lte" in requirement:
                        matched_or = matched_or or (doc.get(field, 0) <= requirement["$lte"])
                if not matched_or:
                    return None
            for field, value in update.get("$inc", {}).items():
                doc[field] = doc.get(field, 0) + value
            for field, value in update.get("$set", {}).items():
                doc[field] = value
            return dict(doc)

        async def update_one(self, query, update):
            return type("Result", (), {"modified_count": 1})()

    class FakeContextDb(dict):
        def __init__(self):
            super().__init__()
            self["ai_usage_events"] = FakeCollection()
            self["ai_usage_counters"] = FakeCollection()
            self["ai_usage_limits"] = FakeCollection()

    class FakeMongoDb:
        def __init__(self):
            self.context = FakeContextDb()

        async def get_context_db(self):
            return self.context

    monkeypatch.setenv("AI_GATEWAY_ENABLED", "true")
    monkeypatch.setenv("AI_BLOCK_ON_LIMIT", "true")
    monkeypatch.delenv("AI_DAILY_TOKEN_LIMIT_PER_USER", raising=False)
    monkeypatch.setenv("AI_DAILY_PAGE_LIMIT_PER_USER", "2")
    db = FakeMongoDb()
    called = False

    async def provider_call():
        nonlocal called
        called = True
        return {"ok": True}

    async def run_case():
        with pytest.raises(AIUsageLimitExceeded):
            await AIGatewayService(db).call(
                user_id="user-1",
                tenant_id="tenant-1",
                document_id="doc-1",
                region_id=None,
                region_scope="document",
                stage="ocr_primary",
                provider="mistral",
                model="mistral-ocr-latest",
                input_kind="pdf_region",
                estimated_input_tokens=1,
                input_units={"page_count": 3},
                call_fn=provider_call,
            )

    asyncio.run(run_case())

    assert called is False
    assert all(
        doc.get("reserved_page_units", 0) == 0
        for doc in db.context["ai_usage_counters"].docs.values()
    )


def test_tutor_is_not_ai_usage_admin():
    from api.v1.ai_usage_async import _is_admin

    assert _is_admin({"user_type": "admin"}) is True
    assert _is_admin({"user_type": "b2c_admin"}) is True
    assert _is_admin({"user_type": "tutor"}) is False


def test_option_normalizer_groups_multiline_previous_option_text():
    from services.option_layout_normalizer import OptionLayoutNormalizer

    text_items = [
        {"text": "a. first option", "x": 20, "y": 10, "width": 80, "height": 10},
        {"text": "line one of beta", "x": 36, "y": 24, "width": 70, "height": 10},
        {"text": "line two of beta", "x": 36, "y": 36, "width": 70, "height": 10},
        {"text": "b.", "x": 20, "y": 48, "width": 8, "height": 10},
        {"text": "c. third option", "x": 20, "y": 62, "width": 80, "height": 10},
        {"text": "fourth option", "x": 36, "y": 76, "width": 70, "height": 10},
        {"text": "d.", "x": 20, "y": 88, "width": 8, "height": 10},
    ]

    result = OptionLayoutNormalizer().correct(text_items=text_items, layout_report={})

    assert result["options_by_label"]["b"] == "line one of beta\nline two of beta"
    assert result["options_by_label"]["d"] == "fourth option"


def test_answer_mapping_uses_question_number_cues_before_region_order():
    from services.answer_question_mapping_service import AnswerQuestionMappingService

    class FakeDb:
        def __init__(self):
            self.mappings = []

        async def mongo_update_one(self, collection_name, query, update, upsert=False):
            self.mappings.append(update["$set"])
            return True

        async def mongo_delete_many(self, collection_name, query):
            return 0

    service = AnswerQuestionMappingService()
    db = FakeDb()

    question_regions = [
        {"id": "q1", "label": "Q1", "pageNumber": 1, "x": 0, "y": 10},
        {"id": "q2", "label": "Q2", "pageNumber": 1, "x": 0, "y": 20},
    ]
    answer_regions = [
        {"id": "a-first", "extractedText": "2. worked solution for second", "pageNumber": 1, "x": 0, "y": 10},
        {"id": "a-second", "extractedText": "1. worked solution for first", "pageNumber": 1, "x": 0, "y": 20},
    ]

    mappings = asyncio.run(
        service.map_region_order(
            db=db,
            is_b2c=False,
            document_id="doc-1",
            question_regions=question_regions,
            answer_regions=answer_regions,
        )
    )

    by_answer = {mapping["answer_region_id"]: mapping for mapping in mappings}
    assert by_answer["a-first"]["question_id"] == "q2"
    assert by_answer["a-first"]["mapping_strategy"] == "question_number"
    assert by_answer["a-second"]["question_id"] == "q1"


def test_answer_context_uses_question_number_and_question_doc():
    from api.v1.pdf_async import _resolve_question_context_for_answer_region

    question_regions = [
        {"id": "q1", "label": "Q1", "extractedText": "1. first question", "pageNumber": 1, "x": 0, "y": 10},
        {"id": "q2", "label": "Q2", "extractedText": "2. second question", "pageNumber": 1, "x": 0, "y": 20},
    ]
    questions_by_id = {
        "q2": {
            "id": "q2",
            "text": "Which option is correct?",
            "options": ["alpha", "beta", "gamma", "delta"],
            "correct_answer": "C",
        }
    }

    context = _resolve_question_context_for_answer_region(
        answer_region={"id": "a1"},
        answer_text="2. exp: worked solution for second question",
        answer_region_order=0,
        question_regions=question_regions,
        questions_by_id=questions_by_id,
    )

    assert context["question_id"] == "q2"
    assert context["match_strategy"] == "question_number"
    assert context["question_text"] == "Which option is correct?"
    assert context["correct_answer"] == "C"


def test_generate_solutions_request_does_not_replace_existing_by_default():
    from api.v1.pdf_async import GenerateSolutionsRequest

    request = GenerateSolutionsRequest(confirmQuestionsReviewed=True)

    assert request.replaceExisting is False


def test_answer_mapping_preserves_answer_manual_review_flag():
    from services.answer_question_mapping_service import AnswerQuestionMappingService

    class FakeDb:
        def __init__(self):
            self.mappings = []

        async def mongo_update_one(self, collection_name, query, update, upsert=False):
            self.mappings.append(update["$set"])
            return True

        async def mongo_delete_many(self, collection_name, query):
            return 0

    mappings = asyncio.run(
        AnswerQuestionMappingService().map_region_order(
            db=FakeDb(),
            is_b2c=False,
            document_id="doc-1",
            question_regions=[{"id": "q1", "label": "Q1", "pageNumber": 1, "x": 0, "y": 10}],
            answer_regions=[
                {
                    "id": "a1",
                    "extractedText": "1. exp: worked solution",
                    "manualReviewRequired": True,
                    "pageNumber": 1,
                    "x": 0,
                    "y": 10,
                }
            ],
        )
    )

    assert mappings[0]["question_id"] == "q1"
    assert mappings[0]["manual_review_required"] is True


def test_full_answer_sheet_mapping_uses_answer_numbers():
    from services.answer_sheet_mapping_service import AnswerSheetMappingService

    class FakeDb:
        def __init__(self):
            self.mappings = []
            self.deleted_queries = []

        async def mongo_find(self, collection_name, query):
            return []

        async def mongo_delete_many(self, collection_name, query):
            self.deleted_queries.append(query)
            return 0

        async def mongo_update_one(self, collection_name, query, update, upsert=False):
            self.mappings.append(update["$set"])
            return True

    db = FakeDb()
    result = asyncio.run(
        AnswerSheetMappingService(vision_mapper=None).map_full_document_blocks(
            db=db,
            is_b2c=False,
            document_id="doc-1",
            question_docs=[
                {"id": "q1", "text": "1. First question text", "region_metadata": {"page": 1, "y": 10}},
                {"id": "q2", "text": "2. Second question text", "region_metadata": {"page": 1, "y": 20}},
            ],
            answer_blocks=[
                {"number": "2", "text": "Worked solution for question two with enough detail.", "confidence": 0.82},
                {"number": "1", "text": "Worked solution for question one with enough detail.", "confidence": 0.82},
            ],
        )
    )

    by_number = {mapping["answer_number"]: mapping for mapping in result["mappings"]}
    assert by_number["2"]["question_id"] == "q2"
    assert by_number["1"]["question_id"] == "q1"
    assert by_number["2"]["mapping_strategy"] == "answer_number"
    assert by_number["2"]["review_status"] == "accepted"
    assert result["mapped_count"] == 2
    assert len(db.mappings) == 2


def test_full_answer_sheet_mapping_uses_extraction_order_before_uuid_order():
    from services.answer_sheet_mapping_service import AnswerSheetMappingService

    class FakeDb:
        def __init__(self):
            self.mappings = []

        async def mongo_find(self, collection_name, query):
            return []

        async def mongo_delete_many(self, collection_name, query):
            return 0

        async def mongo_update_one(self, collection_name, query, update, upsert=False):
            self.mappings.append(update["$set"])
            return True

    db = FakeDb()
    result = asyncio.run(
        AnswerSheetMappingService(vision_mapper=None).map_full_document_blocks(
            db=db,
            is_b2c=False,
            document_id="doc-1",
            question_docs=[
                {"id": "zzz-q2", "text": "Second question text", "extraction_order": 2},
                {"id": "aaa-q1", "text": "First question text", "extraction_order": 1},
            ],
            answer_blocks=[
                {"number": "1", "text": "Worked solution for question one with enough detail.", "confidence": 0.9},
                {"number": "2", "text": "Worked solution for question two with enough detail.", "confidence": 0.9},
            ],
        )
    )

    by_number = {mapping["answer_number"]: mapping["question_id"] for mapping in result["mappings"]}
    assert by_number == {"1": "aaa-q1", "2": "zzz-q2"}
    assert result["mapped_count"] == 2


def test_full_answer_sheet_mapping_uses_vision_for_weak_cases():
    from services.answer_sheet_mapping_service import AnswerSheetMappingService

    class FakeVisionMapper:
        def __init__(self):
            self.called = False
            self.reasons = []

        async def map(self, **kwargs):
            self.called = True
            self.reasons = kwargs.get("reasons") or []
            return {
                "used": True,
                "provider": "openai",
                "model": "gpt-5.4-mini",
                "mappings": [
                    {
                        "question_id": "q1",
                        "answer_block_id": "answer_block_1",
                        "answer_number": "1",
                        "answer_text": "Vision confirmed worked solution for question one.",
                        "confidence": 0.91,
                        "manual_review_required": False,
                        "evidence": "Visible answer number 1 matches question 1.",
                    }
                ],
            }

    class FakeDb:
        async def mongo_find(self, collection_name, query):
            return []

        async def mongo_delete_many(self, collection_name, query):
            return 0

        async def mongo_update_one(self, collection_name, query, update, upsert=False):
            return True

    vision = FakeVisionMapper()
    result = asyncio.run(
        AnswerSheetMappingService(vision_mapper=vision).map_full_document_blocks(
            db=FakeDb(),
            is_b2c=False,
            document_id="doc-1",
            question_docs=[
                {"id": "q1", "text": "1. First question text"},
                {"id": "q2", "text": "2. Second question text"},
            ],
            answer_blocks=[
                {"text": "short", "confidence": 0.4},
            ],
            pdf_bytes=b"%PDF-test",
        )
    )

    assert vision.called is True
    assert "answer_question_count_mismatch" in vision.reasons
    assert result["mappings"][0]["mapping_strategy"] == "gpt_vision_mapper"
    assert result["mappings"][0]["review_status"] == "needs_review"
    assert result["mappings"][0]["manual_review_required"] is True
    assert result["mapped_count"] == 0
    assert result["summary"]["auto_acceptance_blocked"] is True
    assert result["summary"]["vision_used"] is True


def test_question_anchored_subjective_solution_does_not_require_an_option_label():
    from services.answer_sheet_vision_mapper import AnswerSheetVisionMapper

    result = AnswerSheetVisionMapper()._normalise_question_anchored_mapping(
        {
            "question_index": 1,
            "question_id": "q1",
            "answer_text": "The two integers are -12 and 3 because their sum is -9.",
            "final_answer_text": "-12 and 3",
            "correct_answer": "",
            "correct_answer_confidence": 0.0,
            "confidence": 0.98,
            "manual_review_required": False,
        },
        {
            1: {
                "id": "q1",
                "question_type": "subjective",
                "options": [],
            }
        },
    )

    assert result["requires_option_label"] is False
    assert result["correct_answer"] == ""
    assert result["manual_review_required"] is False


def test_answer_sheet_sideways_ruled_page_builds_bounded_orientation_candidates():
    from io import BytesIO

    from PIL import Image, ImageDraw

    from services.answer_sheet_vision_mapper import AnswerSheetVisionMapper

    image = Image.new("RGB", (700, 1000), "white")
    draw = ImageDraw.Draw(image)
    for x in range(70, 650, 55):
        draw.line((x, 40, x, 960), fill="black", width=3)
    output = BytesIO()
    image.save(output, format="PNG")

    mapper = AnswerSheetVisionMapper()
    renders = mapper._orientation_aware_renders(
        page_index=0,
        image_bytes=output.getvalue(),
        width=image.width,
        height=image.height,
    )

    assert {render["rotation_degrees"] for render in renders} == {90, 270}
    assert all(render["orientation_recovery_candidate"] is True for render in renders)
    assert all(render["width"] == image.height for render in renders)
    assert all(render["height"] == image.width for render in renders)


def test_answer_sheet_upright_ruled_page_does_not_duplicate_vision_images():
    from io import BytesIO

    from PIL import Image, ImageDraw

    from services.answer_sheet_vision_mapper import AnswerSheetVisionMapper

    image = Image.new("RGB", (700, 1000), "white")
    draw = ImageDraw.Draw(image)
    for y in range(80, 940, 55):
        draw.line((40, y, 660, y), fill="black", width=3)
    output = BytesIO()
    image.save(output, format="PNG")

    mapper = AnswerSheetVisionMapper()
    renders = mapper._orientation_aware_renders(
        page_index=0,
        image_bytes=output.getvalue(),
        width=image.width,
        height=image.height,
    )

    assert len(renders) == 1
    assert renders[0]["rotation_degrees"] == 0
    assert renders[0]["orientation_recovery_candidate"] is False


def test_successful_empty_question_anchored_result_does_not_make_duplicate_paid_call():
    from services.answer_sheet_mapping_service import AnswerSheetMappingService

    class FakeVisionMapper:
        def __init__(self):
            self.extract_calls = 0
            self.map_calls = 0

        async def extract_by_question(self, **kwargs):
            self.extract_calls += 1
            return {
                "used": True,
                "provider": "openai",
                "model": "gpt-5.4-mini",
                "mode": "question_anchored",
                "mappings": [],
                "orientation_recovery_used": True,
                "render_orientations": [
                    {
                        "physical_page": 1,
                        "rotation_degrees_clockwise": 90,
                        "orientation_recovery_candidate": True,
                    }
                ],
            }

        async def map(self, **kwargs):
            self.map_calls += 1
            raise AssertionError("legacy mapper must not repeat a successful full-page call")

    class FakeDb:
        async def mongo_find(self, collection_name, query):
            return []

        async def mongo_delete_many(self, collection_name, query):
            return 0

        async def mongo_update_one(self, collection_name, query, update, upsert=False):
            return True

    vision = FakeVisionMapper()
    result = asyncio.run(
        AnswerSheetMappingService(vision_mapper=vision).map_full_document_blocks(
            db=FakeDb(),
            is_b2c=False,
            document_id="doc-rotated",
            question_docs=[
                {"id": "q1", "text": "1. First question"},
                {"id": "q2", "text": "2. Second question"},
            ],
            answer_blocks=[{"text": "one weak OCR block", "confidence": 0.2}],
            pdf_bytes=b"%PDF-test",
        )
    )

    assert vision.extract_calls == 1
    assert vision.map_calls == 0
    assert result["summary"]["vision_orientation_recovery_used"] is True
    assert result["summary"]["vision_render_orientations"][0]["rotation_degrees_clockwise"] == 90


def test_question_anchored_objective_solution_without_option_label_requires_review():
    from services.answer_sheet_vision_mapper import AnswerSheetVisionMapper

    result = AnswerSheetVisionMapper()._normalise_question_anchored_mapping(
        {
            "question_index": 1,
            "question_id": "q1",
            "answer_text": "The visible working does not state the selected option.",
            "correct_answer": "",
            "correct_answer_confidence": 0.0,
            "confidence": 0.98,
            "manual_review_required": False,
        },
        {
            1: {
                "id": "q1",
                "question_type": "mcq",
                "options": ["First", "Second", "Third", "Fourth"],
            }
        },
    )

    assert result["requires_option_label"] is True
    assert result["manual_review_required"] is True


def test_full_answer_sheet_question_anchored_subjective_solutions_can_auto_accept():
    from services.answer_sheet_mapping_service import AnswerSheetMappingService

    class FakeVisionMapper:
        async def extract_by_question(self, **kwargs):
            return {
                "used": True,
                "provider": "openai",
                "model": "gpt-5.4-mini",
                "mode": "question_anchored",
                "mappings": [
                    {
                        "question_id": "q1",
                        "question_type": "subjective",
                        "requires_option_label": False,
                        "answer_block_id": "question_anchored_q_1",
                        "answer_item_id": "question_anchored_q_1",
                        "answer_number": "1",
                        "correct_answer": "",
                        "correct_answer_confidence": 0.0,
                        "answer_text": "The two integers are -12 and 3.",
                        "final_answer_text": "-12 and 3",
                        "mapping_strategy": "gpt_question_anchored",
                        "confidence": 0.98,
                        "manual_review_required": True,
                    },
                    {
                        "question_id": "q2",
                        "question_type": "subjective",
                        "requires_option_label": False,
                        "answer_block_id": "question_anchored_q_2",
                        "answer_item_id": "question_anchored_q_2",
                        "answer_number": "2",
                        "correct_answer": "",
                        "correct_answer_confidence": 0.0,
                        "answer_text": "Cube root 125 minus cube root 27 equals 5 minus 3.",
                        "final_answer_text": "2",
                        "mapping_strategy": "gpt_question_anchored",
                        "confidence": 0.96,
                        "manual_review_required": True,
                    },
                ],
            }

    class FakeDb:
        def __init__(self):
            self.mappings = []

        async def mongo_find(self, collection_name, query):
            return []

        async def mongo_delete_many(self, collection_name, query):
            return 0

        async def mongo_update_one(self, collection_name, query, update, upsert=False):
            self.mappings.append(update["$set"])
            return True

    db = FakeDb()
    result = asyncio.run(
        AnswerSheetMappingService(vision_mapper=FakeVisionMapper()).map_full_document_blocks(
            db=db,
            is_b2c=False,
            document_id="doc-subjective",
            question_docs=[
                {"id": "q1", "text": "1. Find the integers", "question_type": "subjective", "extraction_order": 1},
                {"id": "q2", "text": "2. Work out the expression", "question_type": "subjective", "extraction_order": 2},
            ],
            answer_blocks=[{"text": "OCR grouped the sheet badly.", "confidence": 0.2}],
            pdf_bytes=b"%PDF-test",
        )
    )

    assert result["mapped_count"] == 2
    assert result["manual_review_count"] == 0
    assert result["summary"]["auto_acceptance_blocked"] is False
    assert all(mapping["review_status"] == "accepted" for mapping in result["mappings"])
    assert all(mapping["correct_answer_candidate"] == "" for mapping in result["mappings"])
    assert all(mapping["requires_option_label"] is False for mapping in result["mappings"])


def test_full_answer_sheet_question_anchored_vision_can_auto_accept_complete_result():
    from services.answer_sheet_mapping_service import AnswerSheetMappingService

    class FakeVisionMapper:
        async def extract_by_question(self, **kwargs):
            return {
                "used": True,
                "provider": "openai",
                "model": "gpt-5.4-mini",
                "mode": "question_anchored",
                "mappings": [
                    {
                        "question_id": "q1",
                        "answer_block_id": "question_anchored_q_1",
                        "answer_item_id": "question_anchored_q_1",
                        "answer_number": "1",
                        "correct_answer": "B",
                        "correct_answer_confidence": 0.96,
                        "answer_text": "Teacher uploaded worked solution for question one.",
                        "mapping_strategy": "gpt_question_anchored",
                        "confidence": 0.95,
                        "manual_review_required": False,
                        "evidence": "Question index 1 visible as Ans 2.",
                    },
                    {
                        "question_id": "q2",
                        "answer_block_id": "question_anchored_q_2",
                        "answer_item_id": "question_anchored_q_2",
                        "answer_number": "2",
                        "correct_answer": "D",
                        "correct_answer_confidence": 0.97,
                        "answer_text": "Teacher uploaded worked solution for question two.",
                        "mapping_strategy": "gpt_question_anchored",
                        "confidence": 0.96,
                        "manual_review_required": False,
                        "evidence": "Question index 2 visible as Ans 4.",
                    },
                ],
            }

    class FakeDb:
        def __init__(self):
            self.mappings = []

        async def mongo_find(self, collection_name, query):
            return []

        async def mongo_delete_many(self, collection_name, query):
            return 0

        async def mongo_update_one(self, collection_name, query, update, upsert=False):
            self.mappings.append(update["$set"])
            return True

    db = FakeDb()
    result = asyncio.run(
        AnswerSheetMappingService(vision_mapper=FakeVisionMapper()).map_full_document_blocks(
            db=db,
            is_b2c=False,
            document_id="doc-1",
            question_docs=[
                {"id": "q1", "text": "1. First question text", "extraction_order": 1},
                {"id": "q2", "text": "2. Second question text", "extraction_order": 2},
            ],
            answer_blocks=[{"text": "OCR grouped the sheet badly.", "confidence": 0.2}],
            pdf_bytes=b"%PDF-test",
        )
    )

    assert result["mapped_count"] == 2
    assert result["manual_review_count"] == 0
    assert result["summary"]["auto_acceptance_blocked"] is False
    assert result["summary"]["manual_segmentation_recommended"] is False
    assert {m["correct_answer_candidate"] for m in result["mappings"]} == {"B", "D"}
    assert all(m["mapping_strategy"] == "gpt_question_anchored" for m in result["mappings"])


def test_full_answer_sheet_question_anchored_high_confidence_model_review_is_not_global_block():
    from services.answer_sheet_mapping_service import AnswerSheetMappingService

    class FakeVisionMapper:
        async def extract_by_question(self, **kwargs):
            return {
                "used": True,
                "provider": "openai",
                "model": "gpt-5.4-mini",
                "mode": "question_anchored",
                "mappings": [
                    {
                        "question_id": "q1",
                        "answer_block_id": "question_anchored_q_1",
                        "answer_item_id": "question_anchored_q_1",
                        "answer_number": "1",
                        "correct_answer": "B",
                        "correct_answer_confidence": 1.0,
                        "answer_text": "Teacher uploaded worked solution for question one.",
                        "mapping_strategy": "gpt_question_anchored",
                        "confidence": 1.0,
                        "manual_review_required": True,
                    },
                    {
                        "question_id": "q2",
                        "answer_block_id": "question_anchored_q_2",
                        "answer_item_id": "question_anchored_q_2",
                        "answer_number": "2",
                        "correct_answer": "D",
                        "correct_answer_confidence": 0.95,
                        "answer_text": "Teacher uploaded worked solution for question two.",
                        "mapping_strategy": "gpt_question_anchored",
                        "confidence": 0.95,
                        "manual_review_required": True,
                    },
                ],
            }

    class FakeDb:
        def __init__(self):
            self.mappings = []

        async def mongo_find(self, collection_name, query):
            return []

        async def mongo_delete_many(self, collection_name, query):
            return 0

        async def mongo_update_one(self, collection_name, query, update, upsert=False):
            self.mappings.append(update["$set"])
            return True

    db = FakeDb()
    result = asyncio.run(
        AnswerSheetMappingService(vision_mapper=FakeVisionMapper()).map_full_document_blocks(
            db=db,
            is_b2c=False,
            document_id="doc-1",
            question_docs=[
                {"id": "q1", "text": "1. First question text", "extraction_order": 1},
                {"id": "q2", "text": "2. Second question text", "extraction_order": 2},
            ],
            answer_blocks=[{"text": "OCR grouped the sheet badly.", "confidence": 0.2}],
            pdf_bytes=b"%PDF-test",
        )
    )

    assert result["mapped_count"] == 2
    assert result["manual_review_count"] == 0
    assert result["summary"]["auto_acceptance_blocked"] is False
    assert "answer_question_count_mismatch" in result["summary"]["vision_reasons"]
    assert all(mapping["review_status"] == "accepted" for mapping in result["mappings"])
    assert all(mapping["manual_review_required"] is False for mapping in result["mappings"])


def test_full_answer_sheet_vision_mappings_from_same_block_use_unique_answer_items():
    from services.answer_sheet_mapping_service import AnswerSheetMappingService

    class FakeVisionMapper:
        async def map(self, **kwargs):
            return {
                "used": True,
                "provider": "openai",
                "model": "gpt-5.4-mini",
                "mappings": [
                    {
                        "question_id": "q1",
                        "answer_block_id": "answer_block_1",
                        "answer_text": "Solution one from the same visual block.",
                        "confidence": 0.92,
                    },
                    {
                        "question_id": "q2",
                        "answer_block_id": "answer_block_1",
                        "answer_text": "Solution two from the same visual block.",
                        "confidence": 0.91,
                    },
                ],
            }

    class FakeDb:
        def __init__(self):
            self.answer_region_ids = []

        async def mongo_find(self, collection_name, query):
            return []

        async def mongo_delete_many(self, collection_name, query):
            return 0

        async def mongo_update_one(self, collection_name, query, update, upsert=False):
            answer_region_id = update["$set"]["answer_region_id"]
            assert answer_region_id not in self.answer_region_ids
            self.answer_region_ids.append(answer_region_id)
            return True

    db = FakeDb()
    result = asyncio.run(
        AnswerSheetMappingService(vision_mapper=FakeVisionMapper()).map_full_document_blocks(
            db=db,
            is_b2c=False,
            document_id="doc-1",
            question_docs=[
                {"id": "q1", "text": "1. First question text"},
                {"id": "q2", "text": "2. Second question text"},
            ],
            answer_blocks=[
                {"text": "A broad OCR block containing multiple solutions.", "confidence": 0.4},
            ],
            pdf_bytes=b"%PDF-test",
        )
    )

    assert result["mapped_count"] == 0
    assert result["manual_review_count"] == 2
    assert len(db.answer_region_ids) == 2
    assert all("answer_block_1" in answer_region_id for answer_region_id in db.answer_region_ids)
    assert db.answer_region_ids[0] != db.answer_region_ids[1]


def test_answer_key_reconciliation_auto_applies_complete_table():
    from services.answer_key_reconciliation_service import AnswerKeyReconciliationService

    class FakeDb:
        def __init__(self):
            self.updates = []

        async def mongo_update_one(self, collection_name, query, update, upsert=False):
            self.updates.append((collection_name, query, update))
            return True

    db = FakeDb()
    result = asyncio.run(
        AnswerKeyReconciliationService().reconcile(
            db=db,
            is_b2c=False,
            document_id="doc-1",
            question_docs=[
                {"id": "q1", "text": "1. First question", "correct_answer": ""},
                {"id": "q2", "text": "2. Second question", "correct_answer": None},
            ],
            page_summaries=[
                {"index": 0, "markdown": "| Q | Ans |\n| 1 | B |\n| 2 | D |"}
            ],
            mappings=[],
            mapping_summary={"auto_acceptance_blocked": True},
        )
    )

    assert result["summary"]["status"] == "ready"
    assert result["summary"]["extracted_count"] == 2
    assert result["summary"]["auto_applied_count"] == 2
    applied = [update["$set"]["correct_answer"] for _, _, update in db.updates]
    assert applied == ["B", "D"]


def test_answer_key_reconciliation_auto_applies_complete_high_confidence_mappings():
    from services.answer_key_reconciliation_service import AnswerKeyReconciliationService

    class FakeDb:
        def __init__(self):
            self.updates = []

        async def mongo_update_one(self, collection_name, query, update, upsert=False):
            self.updates.append((collection_name, query, update))
            return True

    db = FakeDb()
    result = asyncio.run(
        AnswerKeyReconciliationService().reconcile(
            db=db,
            is_b2c=False,
            document_id="doc-1",
            question_docs=[
                {"id": "q1", "text": "1. First question", "correct_answer": ""},
                {"id": "q2", "text": "2. Second question", "correct_answer": None},
            ],
            page_summaries=[],
            mappings=[
                {
                    "question_id": "q1",
                    "correct_answer_candidate": "B",
                    "correct_answer_confidence": 0.96,
                    "confidence": 0.95,
                    "review_status": "accepted",
                    "manual_review_required": False,
                    "mapping_strategy": "gpt_question_anchored",
                },
                {
                    "question_id": "q2",
                    "correct_answer_candidate": "D",
                    "correct_answer_confidence": 0.97,
                    "confidence": 0.96,
                    "review_status": "accepted",
                    "manual_review_required": False,
                    "mapping_strategy": "gpt_question_anchored",
                },
            ],
        )
    )

    assert result["summary"]["status"] == "ready"
    assert result["summary"]["extracted_count"] == 2
    assert result["summary"]["auto_applied_count"] == 2
    applied = [update["$set"]["correct_answer"] for _, _, update in db.updates]
    assert applied == ["B", "D"]


def test_answer_key_reconciliation_prefers_accepted_mappings_over_noisy_page_regex():
    from services.answer_key_reconciliation_service import AnswerKeyReconciliationService

    class FakeDb:
        def __init__(self):
            self.updates = []

        async def mongo_update_one(self, collection_name, query, update, upsert=False):
            self.updates.append((collection_name, query, update))
            return True

    db = FakeDb()
    result = asyncio.run(
        AnswerKeyReconciliationService().reconcile(
            db=db,
            is_b2c=False,
            document_id="doc-1",
            question_docs=[
                {"id": "q1", "text": "1. First question", "correct_answer": ""},
                {"id": "q2", "text": "2. Second question", "correct_answer": ""},
            ],
            page_summaries=[
                {
                    "index": 0,
                    "markdown": "Answer key: 1 B, 2 D\n1. A distractor-like worked-solution line\n2. A second noisy line",
                }
            ],
            mappings=[
                {
                    "question_id": "q1",
                    "correct_answer_candidate": "B",
                    "correct_answer_confidence": 1.0,
                    "confidence": 1.0,
                    "review_status": "accepted",
                    "manual_review_required": False,
                    "mapping_strategy": "gpt_question_anchored",
                },
                {
                    "question_id": "q2",
                    "correct_answer_candidate": "D",
                    "correct_answer_confidence": 1.0,
                    "confidence": 1.0,
                    "review_status": "accepted",
                    "manual_review_required": False,
                    "mapping_strategy": "gpt_question_anchored",
                },
            ],
        )
    )

    assert result["summary"]["status"] == "ready"
    assert result["summary"]["duplicate_count"] == 0
    assert result["summary"]["auto_applied_count"] == 2
    assert result["summary"]["review_required_count"] == 0
    applied = [update["$set"]["correct_answer"] for _, _, update in db.updates]
    assert applied == ["B", "D"]


def test_answer_key_reconciliation_does_not_auto_apply_partial_table():
    from services.answer_key_reconciliation_service import AnswerKeyReconciliationService

    class FakeDb:
        def __init__(self):
            self.updates = []

        async def mongo_update_one(self, collection_name, query, update, upsert=False):
            self.updates.append((collection_name, query, update))
            return True

    db = FakeDb()
    result = asyncio.run(
        AnswerKeyReconciliationService().reconcile(
            db=db,
            is_b2c=False,
            document_id="doc-1",
            question_docs=[
                {"id": "q1", "text": "1. First question", "correct_answer": ""},
                {"id": "q2", "text": "2. Second question", "correct_answer": ""},
            ],
            page_summaries=[{"index": 0, "markdown": "1. A"}],
            mappings=[],
        )
    )

    assert result["summary"]["status"] == "needs_review"
    assert result["summary"]["extracted_count"] == 1
    assert result["summary"]["auto_applied_count"] == 0
    assert result["summary"]["missing_count"] == 1
    assert db.updates == []


def test_full_answer_sheet_mapping_does_not_overwrite_manual_mapping():
    from services.answer_sheet_mapping_service import AnswerSheetMappingService

    class FakeDb:
        def __init__(self):
            self.mappings = []

        async def mongo_find(self, collection_name, query):
            return [
                {
                    "document_id": "doc-1",
                    "question_id": "q1",
                    "answer_text": "Manual answer",
                    "source": "manual_answer_segmentation",
                }
            ]

        async def mongo_delete_many(self, collection_name, query):
            return 0

        async def mongo_update_one(self, collection_name, query, update, upsert=False):
            self.mappings.append(update["$set"])
            return True

    db = FakeDb()
    result = asyncio.run(
        AnswerSheetMappingService(vision_mapper=None).map_full_document_blocks(
            db=db,
            is_b2c=False,
            document_id="doc-1",
            question_docs=[{"id": "q1", "text": "1. First question text"}],
            answer_blocks=[
                {"number": "1", "text": "Automatic answer should not replace manual mapping.", "confidence": 0.9}
            ],
        )
    )

    assert result["summary"]["protected_manual_mapping_count"] == 1
    assert result["mappings"] == []
    assert db.mappings == []


def test_coverage_counts_full_answer_sheet_ocr_mappings():
    from services.answer_solution_coverage_service import AnswerSolutionCoverageService

    result = AnswerSolutionCoverageService().compute(
        document={"answer_sheet_path": "answers.pdf", "answer_solution_mode": "upload", "answer_sheet_ocr_status": "completed"},
        questions=[{"id": "q1"}, {"id": "q2"}],
        mappings=[
            {
                "question_id": "q1",
                "answer_text": "Worked answer one",
                "source": "answer_sheet_full_ocr",
                "mapping_strategy": "answer_number",
                "confidence": 0.9,
                "manual_review_required": False,
                "review_status": "accepted",
            },
            {
                "question_id": "q2",
                "answer_text": "Worked answer two",
                "source": "answer_sheet_full_ocr",
                "mapping_strategy": "gpt_vision_mapper",
                "confidence": 0.91,
                "manual_review_required": False,
                "review_status": "accepted",
            },
        ],
    )

    assert result["answer_solution_coverage_status"] == "ready"
    assert result["answer_solution_coverage_summary"]["mapped_answer_count"] == 2


def test_test_series_activation_requires_correct_answers():
    from api.v1.pdf_async import _build_test_series_activation_errors

    errors = _build_test_series_activation_errors(
        document={"document_type": "Test Series", "total_minutes": 30},
        questions=[
            {"id": "q1", "correct_answer": "A"},
            {"id": "q2", "correct_answer": ""},
            {"id": "q3"},
        ],
    )

    assert errors == ["2, 3"]


def test_objective_practice_set_activation_requires_correct_answers():
    from api.v1.pdf_async import _build_test_series_activation_errors

    errors = _build_test_series_activation_errors(
        document={"document_type": "Practice Sets", "question_type": "mcq"},
        questions=[
            {"id": "q1", "correct_answer": "A"},
            {"id": "q2", "correct_answer": ""},
        ],
    )

    assert errors == ["2"]


def test_subjective_practice_set_activation_does_not_require_correct_answers():
    from api.v1.pdf_async import _build_test_series_activation_errors

    errors = _build_test_series_activation_errors(
        document={"document_type": "Practice Sets", "question_type": "subjective"},
        questions=[
            {"id": "q1", "question_type": "subjective", "correct_answer": ""},
        ],
    )

    assert errors == []


def test_test_series_activation_allows_saved_answer_candidate_mismatch():
    from api.v1.pdf_async import _build_test_series_activation_errors

    errors = _build_test_series_activation_errors(
        document={"document_type": "Test Series", "total_minutes": 30},
        questions=[
            {
                "id": "q1",
                "correct_answer": "A",
                "answer_key_candidate": {"correct_answer": "B"},
                "mapped_worked_answer": {"correct_answer_candidate": "B"},
            },
            {"id": "q2", "correct_answer": "C"},
        ],
    )

    assert errors == []


def test_test_series_activation_requires_uploaded_answer_sheet_full_mapping():
    from api.v1.pdf_async import _build_test_series_activation_errors

    errors = _build_test_series_activation_errors(
        document={
            "document_type": "Test Series",
            "total_minutes": 30,
            "answer_sheet_path": "uploads/answers.pdf",
        },
        questions=[
            {"id": "q1", "correct_answer": "A"},
            {"id": "q2", "correct_answer": "B"},
            {"id": "q3", "correct_answer": "C"},
        ],
        answer_coverage={
            "answer_solution_coverage_status": "not_ready",
            "answer_solution_coverage_summary": {
                "question_count": 3,
                "mapped_answer_count": 2,
                "manual_review_count": 0,
            },
        },
    )

    assert errors == [
        "Uploaded answer sheet is not fully mapped. 2/3 question(s) have mapped solutions."
    ]


def test_test_series_activation_allows_ready_test_series():
    from api.v1.pdf_async import _build_test_series_activation_errors

    errors = _build_test_series_activation_errors(
        document={
            "document_type": "Test Series",
            "total_minutes": 30,
            "answer_sheet_path": "uploads/answers.pdf",
        },
        questions=[
            {"id": "q1", "correct_answer": "A"},
            {"id": "q2", "correct_answer": "B"},
        ],
        answer_coverage={
            "answer_solution_coverage_status": "ready",
            "answer_solution_coverage_summary": {
                "question_count": 2,
                "mapped_answer_count": 2,
                "manual_review_count": 0,
            },
        },
    )

    assert errors == []
