import asyncio
import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image


_tiny_png_buffer = io.BytesIO()
Image.new("RGB", (2, 2), "white").save(_tiny_png_buffer, format="PNG")
TINY_PNG = base64.b64encode(_tiny_png_buffer.getvalue()).decode("ascii")


COMPOUND_QUESTION_TEXT = """Answer the following questions: (5x2 = 10)
a) i. Compare aquatic and terrestrial animals.
   ii. Compare floating and submerged plants.
b) Describe the journey of a river.
c) List four uses of saline water.
d) Identify the pictured machine and state its use.
"""


def test_pcr_marking_plan_source_accepts_direct_saved_answer_without_mapping():
    from api.v1.pdf_async import _pcr_marking_plan_source

    solution, source = _pcr_marking_plan_source(
        {"reference_solution": "Teacher-approved direct answer"},
        None,
    )

    assert solution == "Teacher-approved direct answer"
    assert source == "saved_reference_solution"


def test_pcr_marking_plan_source_prefers_accepted_mapping():
    from api.v1.pdf_async import _pcr_marking_plan_source

    solution, source = _pcr_marking_plan_source(
        {"reference_solution": "Direct answer"},
        {"answer_text": "Accepted uploaded solution"},
    )

    assert solution == "Accepted uploaded solution"
    assert source == "accepted_mapping"


def test_compound_marks_formula_defines_leaf_assessment_unit_count():
    from api.v1.pdf_async import _expected_pcr_assessment_unit_count

    assert _expected_pcr_assessment_unit_count(COMPOUND_QUESTION_TEXT, 10) == 5
    assert _expected_pcr_assessment_unit_count("Explain photosynthesis. (5)", 5) is None


def test_compound_marking_plan_is_validated_and_flattened_per_leaf():
    import json

    from api.v1.pdf_async import _parse_pcr_marking_plan_draft

    units = []
    for index, label in enumerate(("4(a)(i)", "4(a)(ii)", "4(b)", "4(c)", "4(d)"), start=1):
        units.append(
            {
                "unit_id": f"unit_{index}",
                "label": label,
                "prompt": f"Leaf prompt {index}",
                "max_marks": 2,
                "scoring_model": "point_based",
                "reference_solution": f"Leaf answer {index}",
                "method_policy": {
                    "mode": "any_valid_method",
                    "required_method": None,
                    "allow_error_carried_forward": True,
                },
                "marking_criteria": [
                    {
                        "criterion_id": "fact_1",
                        "description": "First independently assessable fact",
                        "max_marks": 1,
                        "acceptable_evidence": "First valid fact",
                    },
                    {
                        "criterion_id": "fact_2",
                        "description": "Second independently assessable fact",
                        "max_marks": 1,
                        "acceptable_evidence": "Second valid fact",
                    },
                ],
                "figure_refs": ["Question figure 1"] if index == 5 else [],
            }
        )

    draft = _parse_pcr_marking_plan_draft(
        json.dumps({"assessment_units": units}),
        question_marks=10,
        question_text=COMPOUND_QUESTION_TEXT,
    )

    assert len(draft["assessment_units"]) == 5
    assert len(draft["marking_criteria"]) == 10
    assert sum(row["max_marks"] for row in draft["marking_criteria"]) == 10
    assert draft["marking_criteria"][0]["description"].startswith("[4(a)(i)]")
    assert draft["assessment_units"][-1]["figure_refs"] == ["Question figure 1"]


def test_marking_plan_rejects_visual_references_that_were_not_attached():
    import json

    from api.v1.pdf_async import _parse_pcr_marking_plan_draft

    payload = {
        "assessment_units": [
            {
                "unit_id": "unit_1",
                "label": "1",
                "prompt": "Identify the pictured machine.",
                "max_marks": 1,
                "scoring_model": "point_based",
                "reference_solution": "Hand pump",
                "figure_refs": ["Question figure 2"],
                "marking_criteria": [
                    {
                        "criterion_id": "name",
                        "description": "Names the machine",
                        "max_marks": 1,
                    }
                ],
            }
        ]
    }

    with pytest.raises(ValueError, match="does not match an attached visual label"):
        _parse_pcr_marking_plan_draft(
            json.dumps(payload),
            question_marks=1,
            question_text="Identify the pictured machine.",
            available_figure_labels=["Question figure 1"],
        )


def test_point_based_unit_rejects_a_multi_mark_criterion():
    from api.v1._exampen_imports import load_exampen

    policy = load_exampen("pcr.marking_policy")
    errors = policy.validate_assessment_units(
        [
            {
                "unit_id": "unit_1",
                "label": "4(a)",
                "prompt": "State two facts.",
                "max_marks": 2,
                "scoring_model": "point_based",
                "reference_solution": "Fact one and fact two",
                "marking_criteria": [
                    {
                        "criterion_id": "facts",
                        "description": "States both facts",
                        "max_marks": 2,
                    }
                ],
            }
        ],
        2,
        require_reference_solution=True,
    )

    assert any("at most 1 mark" in error for error in errors)


def test_holistic_unit_allows_a_multi_mark_quality_band():
    from api.v1._exampen_imports import load_exampen

    policy = load_exampen("pcr.marking_policy")
    errors = policy.validate_assessment_units(
        [
            {
                "unit_id": "essay",
                "label": "5",
                "prompt": "Write a coherent paragraph.",
                "max_marks": 3,
                "scoring_model": "holistic_banded",
                "reference_solution": "A relevant coherent paragraph.",
                "marking_criteria": [
                    {
                        "criterion_id": "quality",
                        "description": "Overall coherence and relevance",
                        "max_marks": 3,
                    }
                ],
            }
        ],
        3,
        require_reference_solution=True,
    )

    assert errors == []


def test_finalization_uses_same_assessment_unit_contract():
    from api.v1._exampen_imports import load_exampen
    from services.exampen_paper_service import validate_pcr_questions

    policy = load_exampen("pcr.marking_policy")
    units = policy.normalize_assessment_units(
        [
            {
                "unit_id": "part_a",
                "label": "4(a)",
                "prompt": "Name the machine and state its use.",
                "max_marks": 2,
                "scoring_model": "point_based",
                "reference_solution": "Hand pump; used to draw groundwater.",
                "figure_refs": ["Question figure 1"],
                "marking_criteria": [
                    {
                        "criterion_id": "name",
                        "description": "Names the machine",
                        "max_marks": 1,
                        "acceptable_evidence": "Hand pump",
                    },
                    {
                        "criterion_id": "use",
                        "description": "States its use",
                        "max_marks": 1,
                        "acceptable_evidence": "Draws groundwater",
                    },
                ],
            }
        ]
    )
    question = {
        "id": "Q4",
        "text": "Name the pictured machine and state its use.",
        "points": 2,
        "assessment_units": units,
        "marking_criteria": policy.flatten_assessment_unit_criteria(units),
        "reference_solution": policy.compose_assessment_unit_reference_solution(units),
    }

    assert validate_pcr_questions(
        [question],
        marking_policy={"mode": "criterion_rubric_v1"},
    ) == []

    question["marking_criteria"] = []
    errors = validate_pcr_questions(
        [question],
        marking_policy={"mode": "criterion_rubric_v1"},
    )
    assert any("projection is out of sync" in error for error in errors)


def test_single_question_save_refreshes_the_document_package_status(monkeypatch):
    import api.v1.pdf_async as pdf_async
    from api.v1._exampen_imports import load_exampen

    policy = load_exampen("pcr.marking_policy")
    units = policy.normalize_assessment_units(
        [
            {
                "unit_id": "unit_1",
                "label": "1",
                "prompt": "State one fact.",
                "max_marks": 1,
                "scoring_model": "point_based",
                "reference_solution": "A valid fact.",
                "marking_criteria": [
                    {
                        "criterion_id": "fact",
                        "description": "States one valid fact",
                        "max_marks": 1,
                    }
                ],
            }
        ]
    )
    question = {
        "id": "q1",
        "text": "State one fact.",
        "question_type": "subjective",
        "points": 1,
        "assessment_units": units,
        "marking_criteria": policy.flatten_assessment_unit_criteria(units),
        "reference_solution": policy.compose_assessment_unit_reference_solution(units),
    }

    class FakeDb:
        def __init__(self):
            self.document_update = None

        async def mongo_find(self, collection, query):
            assert collection == "questions"
            assert query == {"document_id": "doc1"}
            return [question]

        async def mongo_update_one(self, collection, query, update):
            assert collection == "documents"
            self.document_update = update["$set"]
            return True

    async def fake_refresh(**_kwargs):
        return {}

    db = FakeDb()
    monkeypatch.setattr(pdf_async, "refresh_answer_solution_coverage", fake_refresh)

    asyncio.run(
        pdf_async._refresh_pcr_authoring_package_status(
            db=db,
            document={
                "document_id": "doc1",
                "exam_mode": "pcr",
                "question_type": "subjective",
                "pcr_marking_policy": {"mode": "criterion_rubric_v1"},
            },
            is_b2c=False,
        )
    )

    assert db.document_update["generated_solutions_status"] == "completed"
    assert db.document_update["ai_grading_package_status"] == "ready_for_review"
    assert db.document_update["generated_marking_plan_failed_count"] == 0


def test_visual_question_detection_covers_figures_and_question_wording():
    from api.v1.pdf_async import _question_requires_visual_authoring

    assert _question_requires_visual_authoring(
        {"text": "Use the following diagram to calculate x."}
    )
    assert _question_requires_visual_authoring(
        {"text": "Calculate x.", "question_figures": [{"id": "fig-1"}]}
    )
    assert not _question_requires_visual_authoring(
        {"text": "Name the largest salt-producing state of India."}
    )


def test_visual_context_uses_inline_question_figure():
    from api.v1.pdf_async import _build_pcr_authoring_visual_context

    visuals = asyncio.run(
        _build_pcr_authoring_visual_context(
            document={},
            question={
                "text": "Use the diagram.",
                "question_figures": [
                    {"base64Data": f"data:image/png;base64,{TINY_PNG}"}
                ],
            },
            solution_images=[],
            gateway_context={"db": None, "is_b2c": False},
        )
    )

    assert visuals[0]["label"] == "Question figure 1"
    assert visuals[0]["data_uri"].startswith("data:image/png;base64,")


def test_visual_context_uses_inline_image_option():
    from api.v1.pdf_async import _build_pcr_authoring_visual_context

    visuals = asyncio.run(
        _build_pcr_authoring_visual_context(
            document={},
            question={
                "text": "Choose the matching graph.",
                "enhanced_options": [
                    {
                        "label": "A",
                        "type": "image",
                        "content": f"data:image/png;base64,{TINY_PNG}",
                    }
                ],
            },
            solution_images=[],
            gateway_context={"db": None, "is_b2c": False},
        )
    )

    assert visuals[0]["label"] == "Question option A"
    assert visuals[0]["data_uri"].startswith("data:image/png;base64,")


def test_visual_context_resolves_durable_image_option(monkeypatch):
    import api.v1.pdf_async as pdf_async

    class ImageDb:
        async def mongo_find_one(self, collection_name, query):
            assert collection_name == "images"
            assert query == {"_id": "stored-option-image"}
            return {
                "_id": "stored-option-image",
                "file_path": "uploads/options/stored-option-image.png",
                "content_type": "image/png",
            }

    async def read_stored(_path):
        return base64.b64decode(TINY_PNG)

    monkeypatch.setattr(pdf_async, "_read_authoring_storage_bytes", read_stored)
    visuals = asyncio.run(
        pdf_async._build_pcr_authoring_visual_context(
            document={},
            question={
                "text": "Choose the matching graph.",
                "enhanced_options": [
                    {
                        "id": "stored-option-image",
                        "image_id": "stored-option-image",
                        "label": "A",
                        "type": "image",
                        "content": "/api/v1/images/stored-option-image",
                    }
                ],
            },
            solution_images=[],
            gateway_context={"db": ImageDb(), "is_b2c": False},
        )
    )

    assert visuals[0]["label"] == "Question option A"
    assert visuals[0]["data_uri"].startswith("data:image/png;base64,")


def test_visual_context_fails_closed_when_required_image_is_unavailable(monkeypatch):
    import api.v1.pdf_async as pdf_async

    async def no_page(**_kwargs):
        return None

    monkeypatch.setattr(pdf_async, "_render_question_source_page", no_page)

    with pytest.raises(ValueError, match="no readable visual evidence"):
        asyncio.run(
            pdf_async._build_pcr_authoring_visual_context(
                document={},
                question={"text": "Use the following diagram to answer."},
                solution_images=[],
                gateway_context={"db": None, "is_b2c": False},
            )
        )


def test_solution_image_does_not_replace_missing_question_diagram(monkeypatch):
    import api.v1.pdf_async as pdf_async

    async def no_page(**_kwargs):
        return None

    monkeypatch.setattr(pdf_async, "_render_question_source_page", no_page)

    with pytest.raises(ValueError, match="no readable visual evidence"):
        asyncio.run(
            pdf_async._build_pcr_authoring_visual_context(
                document={},
                question={"text": "Use the diagram to answer."},
                solution_images=[
                    {"base64Data": f"data:image/png;base64,{TINY_PNG}"}
                ],
                gateway_context={"db": None, "is_b2c": False},
            )
        )


def test_visual_context_renders_canonical_pdf_page_as_fallback(tmp_path):
    import fitz

    from api.v1.pdf_async import _build_pcr_authoring_visual_context

    pdf_path = tmp_path / "paper.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=240, height=180)
    page.insert_text((20, 30), "1. Use the following graph.")
    page.draw_line((30, 130), (190, 60))
    pdf.save(pdf_path)
    pdf.close()

    visuals = asyncio.run(
        _build_pcr_authoring_visual_context(
            document={"file_path": str(pdf_path)},
            question={"text": "Use the following graph.", "page_number": 1},
            solution_images=[],
            gateway_context={"db": None, "is_b2c": False},
        )
    )

    assert visuals[0]["label"] == "Canonical question-paper page 1"
    assert visuals[0]["data_uri"].startswith("data:image/png;base64,")


def test_marking_plan_generation_sends_visuals_to_vision_model(monkeypatch):
    import openai
    import api.v1.pdf_async as pdf_async

    captured = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured["provider_call"] = kwargs
            payload = (
                '{"reference_solution":"x = 4",'
                '"method_policy":{"mode":"any_valid_method",'
                '"required_method":null,"allow_error_carried_forward":true},'
                '"marking_criteria":[{"criterion_id":"result",'
                '"description":"Finds x = 4","max_marks":1,'
                '"acceptable_evidence":"Any valid method leading to x = 4"}]}'
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
            )

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    class FakeGateway:
        def __init__(self, *_args, **_kwargs):
            pass

        async def call(self, **kwargs):
            captured["gateway"] = kwargs
            return await kwargs["call_fn"]()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeOpenAI)
    monkeypatch.setattr(pdf_async, "AIGatewayService", FakeGateway)

    draft = asyncio.run(
        pdf_async.generate_pcr_marking_plan_draft(
            document={"title": "Visual paper"},
            question={
                "id": "q1",
                "text": "Use the following diagram to find x.",
                "points": 1,
                "question_figures": [
                    {"base64Data": f"data:image/png;base64,{TINY_PNG}"}
                ],
            },
            mapped_solution=None,
            gateway_context={
                "db": None,
                "is_b2c": False,
                "user_id": "user-1",
                "tenant_id": "tenant-1",
                "document_id": "doc-1",
                "region_id": "q1",
            },
        )
    )

    assert draft["reference_solution"] == "x = 4"
    assert draft["visual_evidence_count"] == 1
    assert captured["gateway"]["input_kind"] == "multimodal"
    content = captured["provider_call"]["messages"][0]["content"]
    assert any(part.get("type") == "image_url" for part in content)
    assert captured["provider_call"]["model"] == pdf_async.PCR_AUTHORING_VISION_MODEL
