from __future__ import annotations

import base64
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest


def _module():
    from api.v1._exampen_imports import load_exampen

    return load_exampen("pcr.services.objective_answer_sheet")


def _fresh_db():
    from mongomock_motor import AsyncMongoMockClient

    return AsyncMongoMockClient()["skb_objective_test"]


class _FakeGate:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def call(self, model_id: str, prompt: str, caller_id: str, **kwargs: Any):
        self.calls.append(
            {
                "model_id": model_id,
                "prompt": prompt,
                "caller_id": caller_id,
                **kwargs,
            }
        )
        return SimpleNamespace(
            content=json.dumps(self.payload),
            usage=SimpleNamespace(
                model="gpt-5.1-2025-11-13",
                caller=caller_id,
                input_tokens=1000,
                output_tokens=500,
                cache_read_tokens=0,
                total_tokens=1500,
                estimated_cost_usd=0.01,
            ),
        )


def _region(question_number: int) -> dict[str, Any]:
    y_start = 100 + question_number * 20
    return {
        "page_number": 1,
        "x_start": 100,
        "y_start": y_start,
        "x_end": 300,
        "y_end": y_start + 15,
        "evidence": f"Q{question_number} selected mark",
        "confidence": 0.96,
    }


def _payload(*answers: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": "pcr-objective-answer-sheet-v1",
        "document": {
            "format": "omr_grid",
            "all_answer_areas_checked": True,
            "confidence": 0.97,
            "warnings": [],
        },
        "answers": list(answers),
    }


def _answer(
    question_number: int,
    *,
    state: str,
    selected_options: list[str],
    confidence: float = 0.96,
    with_evidence: bool = True,
) -> dict[str, Any]:
    return {
        "question_number": question_number,
        "state": state,
        "selected_options": selected_options,
        "confidence": confidence,
        "evidence_regions": (
            [_region(question_number)] if with_evidence else []
        ),
        "reason": "",
    }


def _synthetic_omr_jpeg(
    selected_answers: list[str],
    *,
    corrections: dict[int, str] | None = None,
) -> bytes:
    """Create a generic three-column OMR page for detector regression tests."""

    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = np.full((1600, 1200, 3), 255, dtype=np.uint8)
    groups = (
        (195, 225, 255, 285),
        (525, 555, 585, 615),
        (855, 885, 915, 945),
    )
    row_positions = tuple(650 + 29 * index for index in range(30))

    for option_positions in groups:
        # Real OMR forms align bubbles in printed vertical tracks. The light
        # guides make that geometry explicit without tying the detector to a
        # particular institution, form name, or answer sequence.
        for x_position in option_positions:
            cv2.line(
                image,
                (x_position, 625),
                (x_position, 1510),
                (190, 190, 190),
                1,
            )
        for y_position in row_positions:
            for x_position in option_positions:
                cv2.circle(
                    image,
                    (x_position, y_position),
                    6,
                    (80, 80, 80),
                    1,
                    cv2.LINE_AA,
                )

    corrections = corrections or {}
    for index, selected in enumerate(selected_answers):
        group_index = index // 30
        row_index = index % 30
        x_position = groups[group_index]["ABCD".index(selected)]
        y_position = row_positions[row_index]
        cv2.circle(
            image,
            (x_position, y_position),
            5,
            (10, 10, 10),
            -1,
            cv2.LINE_AA,
        )
        crossed_label = corrections.get(index + 1)
        if crossed_label:
            crossed_x = groups[group_index]["ABCD".index(crossed_label)]
            cv2.line(
                image,
                (crossed_x - 10, y_position - 10),
                (crossed_x + 10, y_position + 10),
                (10, 10, 10),
                2,
                cv2.LINE_AA,
            )
            cv2.line(
                image,
                (crossed_x + 10, y_position - 10),
                (crossed_x - 10, y_position + 10),
                (10, 10, 10),
                2,
                cv2.LINE_AA,
            )

    encoded, jpeg = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    assert encoded
    return jpeg.tobytes()


async def _seed(
    db: Any,
    *,
    objective: bool = True,
    submission_id: str = "SUB-OBJ-1",
) -> None:
    await db["evalpen_submissions"].insert_one(
        {
            "submission_id": submission_id,
            "exam_id": "EXAM-OBJ-1",
            "student_id": "STU-1",
            "source": "camera",
            "segmentation_status": "pending",
        }
    )
    await db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-OBJ-1",
            "exam_type": "pcr",
            "paper_version_id": "PV-1",
        }
    )
    question_type = "mcq" if objective else "subjective"
    grading_mode = "objective" if objective else "subjective"
    questions = []
    for number, correct in [(1, "B"), (2, "A"), (3, "C")]:
        questions.append(
            {
                "question_id": f"EXAM-OBJ-1::Q{number}",
                "exam_id": "EXAM-OBJ-1",
                "question_number": number,
                "question_text": f"Question {number}",
                "question_type": question_type,
                "grading_mode": grading_mode,
                "options": [
                    {"label": "A", "text": "Option A"},
                    {"label": "B", "text": "Option B"},
                    {"label": "C", "text": "Option C"},
                    {"label": "D", "text": "Option D"},
                ],
                "correct_answer": correct,
                "points": 4,
                "penalty": 1,
            }
        )
    await db["evalpen_questions"].insert_many(questions)
    # A valid 1x1 PNG data URI keeps the test independent from local/S3 storage.
    png = base64.b64encode(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
            "ASsJTYQAAAAASUVORK5CYII="
        )
    ).decode("ascii")
    await db["evalpen_answer_pages"].insert_one(
        {
            "page_id": "PAGE-OBJ-1",
            "submission_id": submission_id,
            "page_number": 1,
            "raw_image_ref": f"data:image/png;base64,{png}",
        }
    )


def test_regular_omr_grid_reads_every_row_without_using_answer_key() -> None:
    module = _module()
    selected_answers = [
        "ABCD"[(index * 3 + 1) % 4] for index in range(75)
    ]
    page_bytes = _synthetic_omr_jpeg(selected_answers)
    asset = module._PageAsset(
        page_number=1,
        image_bytes=page_bytes,
        media_type="image/jpeg",
        asset_hash=hashlib.sha256(page_bytes).hexdigest(),
    )
    questions = [
        {
            "question_number": number,
            "options": [
                {"label": label, "text": f"Option {label}"}
                for label in "ABCD"
            ],
            # Extraction must not compare pixels with the teacher key.
            "correct_answer": "D",
        }
        for number in range(1, 76)
    ]

    payload = module._extract_omr_grid_payload(questions, [asset])

    assert payload is not None
    assert payload["document"]["format"] == "omr_grid"
    assert payload["document"]["all_answer_areas_checked"] is True
    assert payload["document"]["warnings"] == []
    assert len(payload["answers"]) == 75
    assert [
        answer["selected_options"][0] for answer in payload["answers"]
    ] == selected_answers
    assert all(
        answer["state"] == "selected" for answer in payload["answers"]
    )


def test_omr_row_prefers_clean_fill_over_crossed_out_option() -> None:
    module = _module()
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    gray = np.full((100, 250), 255, dtype=np.uint8)
    x_positions = (40, 90, 140, 190)
    y_position = 50
    radius = 8
    for x_position in x_positions:
        cv2.circle(
            gray,
            (x_position, y_position),
            radius,
            80,
            1,
            cv2.LINE_AA,
        )
    cv2.circle(
        gray,
        (x_positions[1], y_position),
        6,
        10,
        -1,
        cv2.LINE_AA,
    )
    for start_x, end_x in (
        (x_positions[0] - 14, x_positions[0] + 14),
        (x_positions[0] + 14, x_positions[0] - 14),
    ):
        cv2.line(
            gray,
            (start_x, y_position - 14),
            (end_x, y_position + 14),
            10,
            2,
            cv2.LINE_AA,
        )
    row = module._OmrRow(
        y=float(y_position),
        points=tuple(
            module._OmrPoint(
                x=float(x_position),
                y=float(y_position),
                radius=float(radius),
            )
            for x_position in x_positions
        ),
    )

    state, option_indexes, confidence, reason = module._read_omr_row(
        gray,
        row,
    )

    assert state == "selected"
    assert option_indexes == [1]
    assert confidence >= 0.90
    assert reason == (
        "One clean option remains after a visible correction"
    )


@pytest.mark.asyncio
async def test_pure_mcq_uses_one_extraction_call_and_deterministic_scoring() -> None:
    module = _module()
    db = _fresh_db()
    await _seed(db)
    gate = _FakeGate(
        _payload(
            _answer(1, state="selected", selected_options=["B"]),
            _answer(2, state="selected", selected_options=["D"]),
            _answer(
                3,
                state="blank",
                selected_options=[],
                with_evidence=False,
            ),
        )
    )

    result = await module.ObjectiveAnswerSheetGradingService(
        db,
        gate,
    ).grade_submission("SUB-OBJ-1")

    assert result.handled is True
    assert result.processing_path == "objective_answer_sheet"
    assert result.response_count == 3
    assert result.evaluated_count == 3
    assert result.blocked_count == 0
    assert len(gate.calls) == 1
    call = gate.calls[0]
    assert call["caller_id"] == "pcr_objective_extraction"
    assert call["reasoning_effort"] == "low"
    # LLMGate creates the Responses API text.format wrapper. The Objective
    # caller must supply the raw schema at its root, matching every other PCR
    # structured-output caller.
    assert call["json_schema"]["type"] == "object"
    assert "schema" not in call["json_schema"]
    assert "name" not in call["json_schema"]
    serialized_input = json.dumps(call["responses_input"])
    assert "correct_answer" not in serialized_input
    assert "teacher solution" not in serialized_input.lower()

    evaluations = await db["evalpen_evaluations"].find(
        {"exam_id": "EXAM-OBJ-1"}
    ).sort("question_id", 1).to_list(length=10)
    assert [doc["total_score"] for doc in evaluations] == [4.0, -1.0, 0.0]
    assert all(
        doc["model_used"] == "deterministic-objective-scorer-v1"
        for doc in evaluations
    )
    responses = await db["evalpen_detected_responses"].find(
        {"submission_id": "SUB-OBJ-1", "eval_status": {"$ne": "superseded"}}
    ).sort("question_number", 1).to_list(length=10)
    assert [doc["detected_text"] for doc in responses] == ["B", "D", ""]
    assert [doc["answer_state"] for doc in responses] == [
        "detected",
        "detected",
        "not_attempted",
    ]


@pytest.mark.asyncio
async def test_objective_reprocess_creates_a_new_generation() -> None:
    module = _module()
    db = _fresh_db()
    await _seed(db)
    await db["exampen_processing_jobs"].insert_one(
        {
            "job_id": "JOB-OBJ-1",
            "submission_id": "SUB-OBJ-1",
            "status": "processing",
            "reprocess_count": 0,
        }
    )
    gate = _FakeGate(
        _payload(
            _answer(1, state="selected", selected_options=["B"]),
            _answer(2, state="selected", selected_options=["A"]),
            _answer(3, state="selected", selected_options=["C"]),
        )
    )
    service = module.ObjectiveAnswerSheetGradingService(db, gate)

    first = await service.grade_submission("SUB-OBJ-1")
    same_generation = await service.grade_submission("SUB-OBJ-1")
    await db["exampen_processing_jobs"].update_one(
        {"job_id": "JOB-OBJ-1"},
        {"$set": {"reprocess_count": 1}},
    )
    reprocessed = await service.grade_submission("SUB-OBJ-1")

    assert same_generation.run_id == first.run_id
    assert reprocessed.run_id != first.run_id
    assert len(gate.calls) == 2
    runs = await db["evalpen_objective_grading_runs"].find({}).to_list(length=10)
    assert {run["generation_revision"] for run in runs} == {0, 1}
    assert all(run.get("generation_fingerprint") for run in runs)
    assert await db["evalpen_detected_responses"].count_documents(
        {
            "submission_id": "SUB-OBJ-1",
            "superseded_at": {"$exists": False},
        }
    ) == 3


@pytest.mark.asyncio
async def test_ambiguous_objective_answer_blocks_only_that_question() -> None:
    module = _module()
    db = _fresh_db()
    await _seed(db)
    gate = _FakeGate(
        _payload(
            _answer(1, state="selected", selected_options=["B"]),
            _answer(2, state="multiple", selected_options=["A", "C"]),
            _answer(3, state="selected", selected_options=["C"]),
        )
    )

    result = await module.ObjectiveAnswerSheetGradingService(
        db,
        gate,
    ).grade_submission("SUB-OBJ-1")

    assert result.handled is True
    assert result.evaluated_count == 2
    assert result.blocked_count == 1
    blocked = await db["evalpen_detected_responses"].find_one(
        {"submission_id": "SUB-OBJ-1", "question_number": 2}
    )
    assert blocked["eval_status"] == "blocked"
    assert blocked["flags"][0]["flag_type"] == "objective_answer_ambiguous"
    assert await db["evalpen_evaluations"].count_documents(
        {"response_id": blocked["response_id"]}
    ) == 0


@pytest.mark.asyncio
async def test_subjective_pcr_declines_without_calling_objective_gate() -> None:
    module = _module()
    db = _fresh_db()
    await _seed(db, objective=False)
    gate = _FakeGate(_payload())

    result = await module.ObjectiveAnswerSheetGradingService(
        db,
        gate,
    ).grade_submission("SUB-OBJ-1")

    assert result.handled is False
    assert "subjective or mixed" in str(result.skipped_reason).lower()
    assert gate.calls == []
    assert await db["evalpen_detected_responses"].count_documents({}) == 0
