import json
from types import SimpleNamespace

import jsonschema
import pytest


def load(name):
    from api.v1._exampen_imports import load_exampen
    return load_exampen(name)


def question():
    return {"question_number": 1, "question_id": "EXAM::Q1", "max_marks": 2,
            "marking_criteria": [{"criterion_id": "c1", "max_marks": 2, "description": "Correct work"}]}


def blank():
    return {"question_number": 1, "attempt_status": "not_attempted", "confidence": .86,
            "student_answer": "", "content_type": "TEXT_ONLY", "source_pages": [],
            "criterion_marks": [], "total_score": 0, "overall_feedback": "No answer found after reviewing all pages",
            "needs_review": False, "review_reason": ""}


@pytest.mark.parametrize("change", [
    {"student_answer": "No visible answer for Q3 found on the provided pages."},
    {"source_pages": [1]}, {"total_score": 1}, {"needs_review": True},
    {"criterion_marks": [{"criterion_id": "c1", "marks_awarded": 0, "confidence": .86,
                          "rationale": "Absent", "evidence": "No visible answer", "credit_basis": "no_credit"}]},
])
def test_blank_schema_and_validator_reject_contradictory_fields(change):
    whole = load("pcr.services.whole_copy_grading")
    grading = load("pcr.services.full_document_grading")
    payload = {"all_student_work_accounted": True, "questions": [{**blank(), **change}]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, whole.whole_copy_schema([question()]))
    grades, _, _ = grading._validate_ledger(whole.normalize_payload(payload), questions=[question()], page_count=1)
    assert grades[0].attempt_status == "unresolved"
    assert grades[0].total_score is None


def test_canonical_blank_accepts_absence_explanation_only_in_feedback():
    whole = load("pcr.services.whole_copy_grading")
    payload = {"all_student_work_accounted": True, "questions": [blank()]}
    jsonschema.validate(payload, whole.whole_copy_schema([question()]))
    grades, _, review = load("pcr.services.full_document_grading")._validate_ledger(
        whole.normalize_payload(payload), questions=[question()], page_count=1)
    assert grades[0].attempt_status == "not_attempted"
    assert grades[0].total_score == 0
    assert not review.required


def response(*, reason="", content="{}"):
    return SimpleNamespace(content=content, completion_status="incomplete" if reason else "completed",
                           incomplete_reason=reason, usage=SimpleNamespace(
                               model="test", input_tokens=10, output_tokens=20, total_tokens=30,
                               cache_read_tokens=0, estimated_cost_usd=.01))


class Gate:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    async def call(self, **kwargs):
        self.calls.append(kwargs)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


def request():
    return dict(model_id="test", prompt="", caller_id="pcr_eval_core", max_output_tokens=10_000,
                responses_input=[{"role": "user", "content": "original immutable copy"}],
                metadata={"pcr_stage": "student_evidence_mapping"})


def test_mapping_output_allowance_accounts_for_long_copies_and_stays_bounded():
    limit = load("pcr.services.full_document_grading")._evidence_mapping_output_limit
    assert limit(9, page_count=15) > limit(9, page_count=2)
    assert limit(9, page_count=9, reasoning_effort="high") > limit(9, page_count=9, reasoning_effort="none")
    assert limit(1000, page_count=1000) == 24_000


@pytest.mark.asyncio
async def test_mapping_recovery_keeps_evidence_and_accounts_both_requests():
    gate = Gate([response(reason="max_output_tokens"), response(content='{"questions": []}')])
    result, usage = await load("pcr.services.full_document_grading")._call_mapping_with_recovery(gate=gate, **request())
    assert result.completion_status == "completed"
    assert [c["max_output_tokens"] for c in gate.calls] == [10_000, 20_000]
    assert gate.calls[0]["responses_input"] == gate.calls[1]["responses_input"]
    assert usage["total_tokens"] == 60
    assert usage["mapping_request_count"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,content,expected_calls", [
    ("max_output_tokens", "{}", 2), ("content_filter", "{}", 1), ("", "broken JSON", 1),
])
async def test_incomplete_or_invalid_mapping_never_becomes_a_partial_map(reason, content, expected_calls):
    module = load("pcr.services.full_document_grading")
    gate = Gate([response(reason=reason, content=content)] * 2)
    with pytest.raises(module.StructuredGradingOutputError) as error:
        await module._call_mapping_with_recovery(gate=gate, **request())
    assert len(gate.calls) == expected_calls
    assert error.value.token_usage["total_tokens"] == 30 * expected_calls
    assert error.value.retryable is False


@pytest.mark.asyncio
async def test_economy_checkpoint_skips_mapping_responses_before_grading():
    batch = load("llm_gate.batch")
    class Transport:
        async def prepare_batch_responses_call(self, *args, **kwargs): return {}
        async def record_batch_response(self, **kwargs): return kwargs["metadata"]["provider_call_index"]
    gate = batch.BatchReplayGate(Transport(), response_bodies=[{"stage": "truncated-map"}, {"stage": "map"}, {"stage": "grade"}])
    gate.skip_checkpointed_calls(2)
    result = await gate.call("test", "", "pcr_eval_core", responses_input=[])
    assert result == 2
    with pytest.raises(ValueError):
        gate.skip_checkpointed_calls(1)
