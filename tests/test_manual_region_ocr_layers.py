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

    async def reserve_usage(self, *, scope, subject_id, period="daily", period_key, tokens, limit):
        if scope == "tenant" and self.fail_tenant_reservation:
            key = (scope, subject_id, period_key)
            return False, self.counters.get(key, 0)
        key = (scope, subject_id, period_key)
        used = self.counters.get(key, 0)
        if limit is not None and used + tokens > limit:
            return False, used
        self.counters[key] = used + tokens
        return True, self.counters[key]

    async def release_usage(self, *, scope, subject_id, period="daily", period_key, tokens):
        key = (scope, subject_id, period_key)
        self.counters[key] = max(0, self.counters.get(key, 0) - tokens)
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


def test_answer_mapping_preserves_answer_manual_review_flag():
    from services.answer_question_mapping_service import AnswerQuestionMappingService

    class FakeDb:
        def __init__(self):
            self.mappings = []

        async def mongo_update_one(self, collection_name, query, update, upsert=False):
            self.mappings.append(update["$set"])
            return True

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
