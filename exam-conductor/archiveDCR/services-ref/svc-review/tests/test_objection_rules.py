"""Unit tests for domain/objection_rules.py — ZERO I/O, pure logic.

Test IDs: U-REV-RULES-01 through U-REV-RULES-10
"""

import pytest

from src.domain.objection_rules import (
    EscalationError,
    EscalationPayload,
    FilingContext,
    FilingError,
    Resolution,
    ResolutionError,
    ResolutionPayload,
    validate_escalation,
    validate_filing,
    validate_resolution,
)


# ---------------------------------------------------------------------------
# Filing validation
# ---------------------------------------------------------------------------


class TestFilingValidation:
    """U-REV-RULES-01 through U-REV-RULES-04: Filing preconditions."""

    def test_valid_filing_passes(self):
        """U-REV-RULES-01: Valid filing raises no errors."""
        ctx = FilingContext(
            role="student",
            objection_window_open=True,
            existing_objection_for_question=False,
            objection_text="I believe my answer deserves more marks because...",
        )
        validate_filing(ctx)  # should not raise

    def test_non_student_cannot_file(self):
        """U-REV-RULES-02a: Tutor/evaluator cannot file objections."""
        ctx = FilingContext(
            role="evaluator",
            objection_window_open=True,
            existing_objection_for_question=False,
            objection_text="This is an objection text long enough.",
        )
        with pytest.raises(FilingError) as exc_info:
            validate_filing(ctx)
        assert exc_info.value.code == "not_student"

    def test_parent_cannot_file(self):
        """U-REV-RULES-02b: Parent cannot file objections."""
        ctx = FilingContext(
            role="parent",
            objection_window_open=True,
            existing_objection_for_question=False,
            objection_text="This is an objection text long enough.",
        )
        with pytest.raises(FilingError) as exc_info:
            validate_filing(ctx)
        assert exc_info.value.code == "not_student"

    def test_window_closed_rejects(self):
        """U-REV-RULES-03: Filing outside objection window is rejected."""
        ctx = FilingContext(
            role="student",
            objection_window_open=False,
            existing_objection_for_question=False,
            objection_text="This is an objection text long enough.",
        )
        with pytest.raises(FilingError) as exc_info:
            validate_filing(ctx)
        assert exc_info.value.code == "window_closed"

    def test_duplicate_objection_rejected(self):
        """U-REV-RULES-04a: Max 1 objection per question per student."""
        ctx = FilingContext(
            role="student",
            objection_window_open=True,
            existing_objection_for_question=True,
            objection_text="This is an objection text long enough.",
        )
        with pytest.raises(FilingError) as exc_info:
            validate_filing(ctx)
        assert exc_info.value.code == "duplicate"

    def test_text_too_short_rejected(self):
        """U-REV-RULES-04b: Objection text must be at least 10 chars."""
        ctx = FilingContext(
            role="student",
            objection_window_open=True,
            existing_objection_for_question=False,
            objection_text="Short",
        )
        with pytest.raises(FilingError) as exc_info:
            validate_filing(ctx)
        assert exc_info.value.code == "text_too_short"

    def test_whitespace_only_text_rejected(self):
        """U-REV-RULES-04c: Whitespace-only text is rejected."""
        ctx = FilingContext(
            role="student",
            objection_window_open=True,
            existing_objection_for_question=False,
            objection_text="         ",
        )
        with pytest.raises(FilingError) as exc_info:
            validate_filing(ctx)
        assert exc_info.value.code == "text_too_short"


# ---------------------------------------------------------------------------
# Resolution validation
# ---------------------------------------------------------------------------


class TestResolutionValidation:
    """U-REV-RULES-05 through U-REV-RULES-08: Resolution rules."""

    def test_approved_with_score_passes(self):
        """U-REV-RULES-05: Approval with new_score is valid."""
        payload = ResolutionPayload(
            resolution=Resolution.APPROVED,
            reason="Answer was partially correct, awarding more marks.",
            new_score=8.5,
        )
        validate_resolution(payload)  # should not raise

    def test_rejected_with_reason_passes(self):
        """U-REV-RULES-06: Rejection with mandatory reason is valid."""
        payload = ResolutionPayload(
            resolution=Resolution.REJECTED,
            reason="The scoring is correct per the rubric.",
        )
        validate_resolution(payload)  # should not raise

    def test_approved_without_score_fails(self):
        """U-REV-RULES-07a: Approval without new_score is rejected."""
        payload = ResolutionPayload(
            resolution=Resolution.APPROVED,
            reason="Should have gotten more marks",
            new_score=None,
        )
        with pytest.raises(ResolutionError) as exc_info:
            validate_resolution(payload)
        assert exc_info.value.code == "missing_score"

    def test_approved_with_negative_score_fails(self):
        """U-REV-RULES-07b: Approval with negative score is rejected."""
        payload = ResolutionPayload(
            resolution=Resolution.APPROVED,
            reason="Correcting the score value",
            new_score=-1.0,
        )
        with pytest.raises(ResolutionError) as exc_info:
            validate_resolution(payload)
        assert exc_info.value.code == "negative_score"

    def test_reason_too_short_fails(self):
        """U-REV-RULES-08a: Reason shorter than 5 chars is rejected."""
        payload = ResolutionPayload(
            resolution=Resolution.REJECTED,
            reason="No",
        )
        with pytest.raises(ResolutionError) as exc_info:
            validate_resolution(payload)
        assert exc_info.value.code == "reason_too_short"

    def test_whitespace_reason_fails(self):
        """U-REV-RULES-08b: Whitespace-only reason is rejected."""
        payload = ResolutionPayload(
            resolution=Resolution.REJECTED,
            reason="    ",
        )
        with pytest.raises(ResolutionError) as exc_info:
            validate_resolution(payload)
        assert exc_info.value.code == "reason_too_short"

    def test_rejected_with_score_still_passes(self):
        """U-REV-RULES-05b: Rejection with new_score is accepted (score ignored)."""
        payload = ResolutionPayload(
            resolution=Resolution.REJECTED,
            reason="Score is correct, objection not valid.",
            new_score=5.0,
        )
        validate_resolution(payload)  # should not raise

    def test_approved_with_zero_score_passes(self):
        """U-REV-RULES-05c: Approval with zero score is valid."""
        payload = ResolutionPayload(
            resolution=Resolution.APPROVED,
            reason="Re-evaluated, score remains zero.",
            new_score=0.0,
        )
        validate_resolution(payload)  # should not raise


# ---------------------------------------------------------------------------
# Escalation validation
# ---------------------------------------------------------------------------


class TestEscalationValidation:
    """U-REV-RULES-09 through U-REV-RULES-10: Escalation rules."""

    def test_escalate_to_hod_passes(self):
        """U-REV-RULES-09a: Escalation to HOD is valid."""
        payload = EscalationPayload(
            escalated_to="hod",
            reason="Need department head review for this edge case.",
        )
        validate_escalation(payload)  # should not raise

    def test_escalate_to_senior_evaluator_passes(self):
        """U-REV-RULES-09b: Escalation to senior_evaluator is valid."""
        payload = EscalationPayload(
            escalated_to="senior_evaluator",
            reason="Complex multi-step answer needs expert evaluation.",
        )
        validate_escalation(payload)  # should not raise

    def test_escalate_to_invalid_target_fails(self):
        """U-REV-RULES-10a: Escalation to invalid role is rejected."""
        payload = EscalationPayload(
            escalated_to="student",
            reason="Trying to escalate to student role.",
        )
        with pytest.raises(EscalationError) as exc_info:
            validate_escalation(payload)
        assert exc_info.value.code == "invalid_target"

    def test_escalate_to_unknown_role_fails(self):
        """U-REV-RULES-10b: Escalation to unknown role is rejected."""
        payload = EscalationPayload(
            escalated_to="janitor",
            reason="This should not be allowed either.",
        )
        with pytest.raises(EscalationError) as exc_info:
            validate_escalation(payload)
        assert exc_info.value.code == "invalid_target"

    def test_escalation_reason_too_short_fails(self):
        """U-REV-RULES-10c: Escalation with short reason is rejected."""
        payload = EscalationPayload(
            escalated_to="hod",
            reason="Bad",
        )
        with pytest.raises(EscalationError) as exc_info:
            validate_escalation(payload)
        assert exc_info.value.code == "reason_too_short"
