"""
ExamPen Test Harness — Practice boundary and tamper-proof tests.

Test IDs covered:
    I-PCR-03   Practice evaluation is stateless — no writes to
               evalpen_submissions or evalpen_evaluations
    U-TAMP-01  Conducted eval rejects client-submitted answer text
    I-TAMP-02  Server-side fetch — eval uses stored text, not request body

Spec authority:
    new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md section 8.2
    new-docs/architecture/TAMPER_PROOF_SPEC.md sections 3-4
    new-docs/architecture/DUAL_MODE_ARCHITECTURE.md section 8

Failure modes: PCR-04 (practice creates persistence), TAMP-01 (client substitution)
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_EC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "exam-conductor")
if _EC_DIR not in sys.path:
    sys.path.insert(0, _EC_DIR)

from pcr.services.eval_core import EvalCore, EvalResult
from pcr.domain.response_models import FlagSeverity


# ---------------------------------------------------------------------------
# Mock implementations of Protocols
# ---------------------------------------------------------------------------


class MockResponseReader:
    """Mock for ResponseReader protocol."""

    def __init__(self, responses: Dict[str, Dict[str, Any]] | None = None):
        self._responses = responses or {}
        self.get_response_calls: list[str] = []
        self.get_responses_by_submission_calls: list[str] = []
        self.update_eval_status_calls: list[tuple[str, str]] = []

    async def get_response(self, response_id: str) -> Optional[Dict[str, Any]]:
        self.get_response_calls.append(response_id)
        return self._responses.get(response_id)

    async def get_responses_by_submission(
        self, submission_id: str
    ) -> List[Dict[str, Any]]:
        self.get_responses_by_submission_calls.append(submission_id)
        return [
            r for r in self._responses.values()
            if r.get("submission_id") == submission_id
        ]

    async def update_eval_status(
        self, response_id: str, eval_status: str
    ) -> bool:
        self.update_eval_status_calls.append((response_id, eval_status))
        return True


class MockEvaluationWriter:
    """Mock for EvaluationWriter protocol — tracks all writes."""

    def __init__(self):
        self.insert_evaluation_calls: list[Dict[str, Any]] = []

    async def insert_evaluation(
        self, doc: Dict[str, Any]
    ) -> tuple[Dict[str, Any], bool]:
        self.insert_evaluation_calls.append(doc)
        return doc, False


class MockQuestionReader:
    """Mock for QuestionReader protocol."""

    def __init__(self, questions: Dict[str, Dict[str, Any]] | None = None):
        self._questions = questions or {}

    async def get_question(self, question_id: str) -> Optional[Dict[str, Any]]:
        return self._questions.get(question_id)


class MockSolutionCache:
    """Mock for SolutionCache — always returns a cache hit."""

    def __init__(self, reference_solution: str = "Reference answer"):
        self.reference_solution = reference_solution

    async def lookup(
        self, question_id: str, question_metadata: Dict[str, Any]
    ):
        @dataclass
        class _Result:
            hit: bool = True
            reference_solution: Optional[str] = None
            version: int = 1
            solution_source: str = "teacher"
            model_used: Optional[str] = None
            was_generated: bool = False

        return _Result(hit=True, reference_solution=self.reference_solution)


class MockGate:
    """Mock for GateProtocol — returns a fixed response, tracks calls."""

    def __init__(self, content: str = '{"step_marks": [], "total_score": 7.0, "max_score": 10.0, "overall_feedback": "Good"}'):
        self.content = content
        self.calls: list[Dict[str, Any]] = []

    async def call(
        self,
        model_id: str,
        prompt: str,
        caller_id: str,
        *,
        messages: Optional[Any] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.calls.append({
            "model_id": model_id,
            "prompt": prompt,
            "caller_id": caller_id,
            "messages": messages,
            "metadata": metadata,
        })

        @dataclass
        class _Usage:
            model: str = model_id
            caller: str = caller_id
            input_tokens: int = 500
            output_tokens: int = 200
            total_tokens: int = 700
            estimated_cost_usd: float = 0.01
            timestamp: datetime = field(
                default_factory=lambda: datetime.now(timezone.utc)
            )
            cache_read_tokens: int = 0
            cache_creation_tokens: int = 0

        @dataclass
        class _Response:
            content: str = ""
            usage: _Usage = field(default_factory=_Usage)

        return _Response(content=self.content, usage=_Usage())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def question_metadata():
    return {
        "q-001": {
            "question_id": "q-001",
            "exam_id": "exam-001",
            "subject": "Mathematics",
            "question_type": "subjective",
            "complexity": "L2",
            "eval_template": "factual_recall",
            "max_marks": 10.0,
            "question_text": "Explain the Pythagorean theorem.",
            "expects_diagram": False,
            "diagram_weight": 0.0,
        }
    }


# ===========================================================================
# I-PCR-03: Practice evaluation is stateless
# ===========================================================================


class TestIPcr03:
    """I-PCR-03: Practice call is synchronous and creates no new PCR persistence.

    Spec: PCR_EVAL_ENGINE_SPEC section 8.2
    Constraint: C3 — practice persistence remains external
    """

    def test_i_pcr_03_practice_no_submission_writes(
        self, question_metadata
    ):
        """evaluate_practice() does NOT write to evalpen_submissions."""
        async def _run():
            response_repo = MockResponseReader()
            eval_writer = MockEvaluationWriter()
            question_repo = MockQuestionReader(question_metadata)
            solution_cache = MockSolutionCache()
            gate = MockGate()

            core = EvalCore(
                response_repo=response_repo,
                eval_repo=eval_writer,
                question_repo=question_repo,
                solution_cache=solution_cache,
                gate=gate,
            )

            result = await core.evaluate_practice(
                student_response="The Pythagorean theorem states a^2 + b^2 = c^2",
                question_id="q-001",
            )

            # No writes to evalpen_evaluations
            assert len(eval_writer.insert_evaluation_calls) == 0

            # No writes to evalpen_submissions (response_repo tracks reads only)
            assert len(response_repo.get_response_calls) == 0

            # Result is returned without persistence
            assert isinstance(result, EvalResult)
            assert result.response_id == "practice"
        asyncio.run(_run())

    def test_i_pcr_03_practice_no_evaluation_writes(
        self, question_metadata
    ):
        """evaluate_practice() does NOT write to evalpen_evaluations."""
        async def _run():
            eval_writer = MockEvaluationWriter()
            core = EvalCore(
                response_repo=MockResponseReader(),
                eval_repo=eval_writer,
                question_repo=MockQuestionReader(question_metadata),
                solution_cache=MockSolutionCache(),
                gate=MockGate(),
            )

            await core.evaluate_practice(
                student_response="Test answer",
                question_id="q-001",
            )

            # Verify: ZERO writes to the evaluation collection
            assert len(eval_writer.insert_evaluation_calls) == 0
        asyncio.run(_run())

    def test_i_pcr_03_practice_uses_pcr_practice_caller(
        self, question_metadata
    ):
        """Practice path uses 'pcr_practice' caller_id, not pcr_eval_core."""
        async def _run():
            gate = MockGate()
            core = EvalCore(
                response_repo=MockResponseReader(),
                eval_repo=MockEvaluationWriter(),
                question_repo=MockQuestionReader(question_metadata),
                solution_cache=MockSolutionCache(),
                gate=gate,
            )

            await core.evaluate_practice(
                student_response="Test answer",
                question_id="q-001",
            )

            assert len(gate.calls) == 1
            assert gate.calls[0]["caller_id"] == "pcr_practice"
        asyncio.run(_run())

    def test_i_pcr_03_practice_returns_eval_result(
        self, question_metadata
    ):
        """Practice evaluation returns a well-formed EvalResult."""
        async def _run():
            core = EvalCore(
                response_repo=MockResponseReader(),
                eval_repo=MockEvaluationWriter(),
                question_repo=MockQuestionReader(question_metadata),
                solution_cache=MockSolutionCache(),
                gate=MockGate(),
            )

            result = await core.evaluate_practice(
                student_response="The Pythagorean theorem states that...",
                question_id="q-001",
            )

            assert isinstance(result, EvalResult)
            assert result.evaluation_id.startswith("PRACTICE-")
            assert result.response_id == "practice"
            assert result.question_id == "q-001"
            assert result.total_score >= 0.0
            assert result.max_score == 10.0
            assert result.raw_llm_response != ""
        asyncio.run(_run())

    def test_i_pcr_03_practice_with_inline_metadata(self):
        """Practice evaluation can accept question_metadata directly."""
        async def _run():
            core = EvalCore(
                response_repo=MockResponseReader(),
                eval_repo=MockEvaluationWriter(),
                question_repo=MockQuestionReader(),
                solution_cache=MockSolutionCache(),
                gate=MockGate(),
            )

            result = await core.evaluate_practice(
                student_response="Answer text",
                question_id="q-inline",
                question_metadata={
                    "subject": "Physics",
                    "question_type": "subjective",
                    "complexity": "L1",
                    "eval_template": "factual_recall",
                    "max_marks": 5.0,
                    "question_text": "What is gravity?",
                },
            )

            assert result.max_score == 5.0
        asyncio.run(_run())

    def test_i_pcr_03_practice_gate_token_logging_allowed(
        self, question_metadata
    ):
        """Token logging through the gate is allowed for practice (spec 11.3).

        Token logging happens inside the gate.call() — the practice path
        itself does not write to evalpen collections.
        """
        async def _run():
            gate = MockGate()
            eval_writer = MockEvaluationWriter()

            core = EvalCore(
                response_repo=MockResponseReader(),
                eval_repo=eval_writer,
                question_repo=MockQuestionReader(question_metadata),
                solution_cache=MockSolutionCache(),
                gate=gate,
            )

            result = await core.evaluate_practice(
                student_response="Test",
                question_id="q-001",
            )

            # Gate was called (token logging is a side-effect of gate.call())
            assert len(gate.calls) == 1
            # But eval_writer was NOT called
            assert len(eval_writer.insert_evaluation_calls) == 0
            # Verify metadata shows practice mode
            assert gate.calls[0]["metadata"]["mode"] == "practice"
        asyncio.run(_run())


# ===========================================================================
# U-TAMP-01: Conducted eval rejects client-submitted answer text
# ===========================================================================


class TestUTamp01:
    """U-TAMP-01: Conducted-exam evaluation rejects client-submitted
    authoritative answer text.

    Spec: TAMPER_PROOF_SPEC sections 3-4, DUAL_MODE_ARCHITECTURE section 8
    """

    def test_u_tamp_01_evaluate_response_fetches_from_storage(
        self, question_metadata
    ):
        """evaluate_response() fetches detected_text from storage, not from
        the wire request."""
        async def _run():
            stored_response = {
                "response_id": "RESP-001",
                "question_id": "q-001",
                "student_id": "stu-001",
                "detected_text": "Server-side OCR text only",
                "content_type": "TEXT_ONLY",
                "flags": [],
            }
            response_repo = MockResponseReader({"RESP-001": stored_response})
            gate = MockGate()

            core = EvalCore(
                response_repo=response_repo,
                eval_repo=MockEvaluationWriter(),
                question_repo=MockQuestionReader(question_metadata),
                solution_cache=MockSolutionCache(),
                gate=gate,
            )

            result = await core.evaluate_response("RESP-001")

            # The gate prompt should contain the server-side text
            assert len(gate.calls) == 1
            prompt = gate.calls[0]["prompt"]
            assert "Server-side OCR text only" in prompt
        asyncio.run(_run())

    def test_u_tamp_01_evaluate_response_uses_server_text(
        self, question_metadata
    ):
        """evaluate_response() uses the stored detected_text, not client-supplied
        text (TAMPER_PROOF_SPEC Layer 2)."""
        async def _run():
            server_text = "The correct server-stored answer"
            stored_response = {
                "response_id": "RESP-002",
                "question_id": "q-001",
                "student_id": "stu-001",
                "detected_text": server_text,
                "content_type": "TEXT_ONLY",
                "flags": [],
            }
            response_repo = MockResponseReader({"RESP-002": stored_response})
            gate = MockGate()

            core = EvalCore(
                response_repo=response_repo,
                eval_repo=MockEvaluationWriter(),
                question_repo=MockQuestionReader(question_metadata),
                solution_cache=MockSolutionCache(),
                gate=gate,
            )

            # Call evaluate_response — there is no way to inject client text
            # through this API since it takes only response_id
            await core.evaluate_response("RESP-002")

            prompt = gate.calls[0]["prompt"]
            assert server_text in prompt
        asyncio.run(_run())

    def test_u_tamp_01_evaluate_response_question_id_mismatch(
        self, question_metadata
    ):
        """evaluate_response() rejects when wire question_id mismatches stored."""
        async def _run():
            stored_response = {
                "response_id": "RESP-003",
                "question_id": "q-001",
                "student_id": "stu-001",
                "detected_text": "Answer",
                "content_type": "TEXT_ONLY",
                "flags": [],
            }
            response_repo = MockResponseReader({"RESP-003": stored_response})

            core = EvalCore(
                response_repo=response_repo,
                eval_repo=MockEvaluationWriter(),
                question_repo=MockQuestionReader(question_metadata),
                solution_cache=MockSolutionCache(),
                gate=MockGate(),
            )

            result = await core.evaluate_response(
                "RESP-003", question_id="q-TAMPERED"
            )

            # Should return error due to question_id mismatch
            assert result.error is not None
            assert "mismatch" in result.error.lower()
        asyncio.run(_run())

    def test_u_tamp_01_evaluate_missing_response_returns_error(self):
        """evaluate_response() for a non-existent response returns error."""
        async def _run():
            response_repo = MockResponseReader({})

            core = EvalCore(
                response_repo=response_repo,
                eval_repo=MockEvaluationWriter(),
                question_repo=MockQuestionReader(),
                solution_cache=MockSolutionCache(),
                gate=MockGate(),
            )

            result = await core.evaluate_response("RESP-NONEXISTENT")
            assert result.error is not None
            assert "not found" in result.error.lower()
        asyncio.run(_run())

    def test_u_tamp_01_evaluate_superseded_response_returns_error(self):
        """Superseded PCR detections are preserved but not evaluated."""
        async def _run():
            stored_response = {
                "response_id": "RESP-SUPERSEDED",
                "question_id": "q-001",
                "student_id": "stu-001",
                "detected_text": "Old stale answer",
                "content_type": "TEXT_ONLY",
                "eval_status": "superseded",
                "flags": [],
            }
            response_repo = MockResponseReader(
                {"RESP-SUPERSEDED": stored_response}
            )
            gate = MockGate()

            core = EvalCore(
                response_repo=response_repo,
                eval_repo=MockEvaluationWriter(),
                question_repo=MockQuestionReader(),
                solution_cache=MockSolutionCache(),
                gate=gate,
            )

            result = await core.evaluate_response("RESP-SUPERSEDED")

            assert result.error == "Response RESP-SUPERSEDED has been superseded"
            assert gate.calls == []
            assert response_repo.update_eval_status_calls == []
        asyncio.run(_run())


# ===========================================================================
# I-TAMP-02: Server-side fetch — eval uses stored text, not request body
# ===========================================================================


class TestITamp02:
    """I-TAMP-02: Conducted PCR eval fetches server-side artifact."""

    def test_i_tamp_02_server_side_fetch_happens(
        self, question_metadata
    ):
        """evaluate_response() calls get_response() (server-side fetch)."""
        async def _run():
            stored_response = {
                "response_id": "RESP-100",
                "question_id": "q-001",
                "student_id": "stu-001",
                "detected_text": "Fetched from storage",
                "content_type": "TEXT_ONLY",
                "flags": [],
            }
            response_repo = MockResponseReader({"RESP-100": stored_response})

            core = EvalCore(
                response_repo=response_repo,
                eval_repo=MockEvaluationWriter(),
                question_repo=MockQuestionReader(question_metadata),
                solution_cache=MockSolutionCache(),
                gate=MockGate(),
            )

            await core.evaluate_response("RESP-100")

            # Verify server-side fetch happened
            assert "RESP-100" in response_repo.get_response_calls
        asyncio.run(_run())

    def test_i_tamp_02_blocking_flags_prevent_auto_eval(
        self, question_metadata
    ):
        """Responses with blocking flags are skipped (I-PCR-02)."""
        async def _run():
            stored_response = {
                "response_id": "RESP-BLOCKED",
                "question_id": "q-001",
                "student_id": "stu-001",
                "detected_text": "Some text",
                "content_type": "DIAGRAM_HEAVY",
                "flags": [
                    {
                        "flag_type": "diagram_heavy_content",
                        "severity": "blocking",
                        "reason": "Less than 40% text",
                    }
                ],
            }
            response_repo = MockResponseReader(
                {"RESP-BLOCKED": stored_response}
            )
            gate = MockGate()

            core = EvalCore(
                response_repo=response_repo,
                eval_repo=MockEvaluationWriter(),
                question_repo=MockQuestionReader(question_metadata),
                solution_cache=MockSolutionCache(),
                gate=gate,
            )

            result = await core.evaluate_response("RESP-BLOCKED")

            # Auto-eval should be skipped
            assert result.skipped is True
            assert "blocking" in result.skip_reason.lower()
            # Gate should NOT have been called
            assert len(gate.calls) == 0
            # Status should be updated to 'blocked'
            assert ("RESP-BLOCKED", "blocked") in response_repo.update_eval_status_calls
        asyncio.run(_run())

    def test_i_tamp_02_conducted_eval_persists_result(
        self, question_metadata
    ):
        """Conducted-exam evaluation persists the result (unlike practice)."""
        async def _run():
            stored_response = {
                "response_id": "RESP-PERSIST",
                "question_id": "q-001",
                "student_id": "stu-001",
                "detected_text": "Answer to persist",
                "content_type": "TEXT_ONLY",
                "flags": [],
            }
            response_repo = MockResponseReader(
                {"RESP-PERSIST": stored_response}
            )
            eval_writer = MockEvaluationWriter()

            core = EvalCore(
                response_repo=response_repo,
                eval_repo=eval_writer,
                question_repo=MockQuestionReader(question_metadata),
                solution_cache=MockSolutionCache(),
                gate=MockGate(),
            )

            result = await core.evaluate_response("RESP-PERSIST")

            # Conducted-exam evaluation SHOULD persist
            assert len(eval_writer.insert_evaluation_calls) == 1
            doc = eval_writer.insert_evaluation_calls[0]
            assert doc["response_id"] == "RESP-PERSIST"
            assert doc["question_id"] == "q-001"
            assert "audit_trail" in doc
            assert len(doc["audit_trail"]) >= 1
        asyncio.run(_run())

    def test_i_tamp_02_conducted_eval_uses_pcr_eval_core_caller(
        self, question_metadata
    ):
        """Conducted-exam evaluation uses pcr_eval_core caller_id."""
        async def _run():
            stored_response = {
                "response_id": "RESP-CALLER",
                "question_id": "q-001",
                "student_id": "stu-001",
                "detected_text": "Text",
                "content_type": "TEXT_ONLY",
                "flags": [],
            }
            response_repo = MockResponseReader(
                {"RESP-CALLER": stored_response}
            )
            gate = MockGate()

            core = EvalCore(
                response_repo=response_repo,
                eval_repo=MockEvaluationWriter(),
                question_repo=MockQuestionReader(question_metadata),
                solution_cache=MockSolutionCache(),
                gate=gate,
            )

            await core.evaluate_response("RESP-CALLER")

            assert len(gate.calls) == 1
            assert gate.calls[0]["caller_id"] == "pcr_eval_core"
        asyncio.run(_run())

    def test_i_tamp_02_audit_trail_included(
        self, question_metadata
    ):
        """Evaluation result includes audit_trail with actor, timestamp, action."""
        async def _run():
            stored_response = {
                "response_id": "RESP-AUDIT",
                "question_id": "q-001",
                "student_id": "stu-001",
                "detected_text": "Answer",
                "content_type": "TEXT_ONLY",
                "flags": [],
            }
            response_repo = MockResponseReader(
                {"RESP-AUDIT": stored_response}
            )
            eval_writer = MockEvaluationWriter()

            core = EvalCore(
                response_repo=response_repo,
                eval_repo=eval_writer,
                question_repo=MockQuestionReader(question_metadata),
                solution_cache=MockSolutionCache(),
                gate=MockGate(),
            )

            await core.evaluate_response("RESP-AUDIT")

            doc = eval_writer.insert_evaluation_calls[0]
            audit_trail = doc["audit_trail"]
            assert len(audit_trail) >= 1
            entry = audit_trail[0]
            assert "actor_id" in entry
            assert "timestamp" in entry
            assert "action" in entry
            assert entry["action"] == "evaluation_created"
        asyncio.run(_run())
