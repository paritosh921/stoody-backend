"""
PCR Clubbed Response Detector — Detect multiple answers in one segment.

Spec authority: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md section 4.5
Failure mode:   PCR-02 (clubbed responses undetected -> multiple heuristics
                plus optional LLM-assisted topic discontinuity check)
Test ID:        U-CLUB-01

Heuristics from spec:

    H1 — multiple markers:
        More than one Q marker in one segment.
        Confidence: very high.

    H2 — length anomaly:
        Word count > expected_max_words * 2.5.
        Confidence: medium.

    H3 — missing question:
        A manifest question not represented in any segment.
        Confidence: high.

    H4 — topic discontinuity:
        LLM-assisted discontinuity check through the gate.
        Confidence: medium-high.
        NOTE: H4 is *not* executed here.  This module only defines the
        data structures and flags for it.  The actual LLM call goes
        through the gate (caller_id = pcr_clubbed_h4) per C4.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Sequence

from .flag_registry import FLAG_REGISTRY, FlagDefinition
from .response_models import (
    DetectedResponse,
    Flag,
    FlagType,
    QMarker,
)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

H2_LENGTH_MULTIPLIER: float = 2.5
"""H2: Word count must exceed expected_max_words * this multiplier to trigger."""


# ---------------------------------------------------------------------------
# Confidence levels (informational — carried in flag metadata)
# ---------------------------------------------------------------------------

H1_CONFIDENCE: str = "very_high"
H2_CONFIDENCE: str = "medium"
H3_CONFIDENCE: str = "high"
H4_CONFIDENCE: str = "medium_high"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ClubbedDetectionResult:
    """Aggregated result from all executed heuristics."""

    flags: list[Flag]
    h1_triggered: bool = False
    h2_triggered: bool = False
    h3_triggered: bool = False
    # h4_triggered is always False here — it requires gate invocation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_flag(
    flag_type: FlagType,
    response_id: str | None,
    metadata: dict | None = None,
) -> Flag:
    defn: FlagDefinition = FLAG_REGISTRY[flag_type]
    return Flag(
        flag_id=f"FLG-{uuid.uuid4().hex[:8]}",
        response_id=response_id,
        source=defn.source,
        flag_type=defn.flag_type,
        severity=defn.severity,
        reason=defn.description,
        suggested_action=defn.suggested_action,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# H1 — Multiple markers in one segment
# ---------------------------------------------------------------------------


def detect_h1_multiple_markers(
    response: DetectedResponse,
    markers_in_segment: list[QMarker],
) -> Flag | None:
    """H1: More than one Q marker in a single segment.

    Confidence: very high.
    Severity: BLOCKING (from flag registry).
    """
    if len(markers_in_segment) <= 1:
        return None

    marker_numbers = [m.question_number for m in markers_in_segment]
    return _make_flag(
        FlagType.CLUBBED_MULTIPLE_MARKERS,
        response.response_id,
        {
            "heuristic": "H1",
            "confidence": H1_CONFIDENCE,
            "marker_count": len(markers_in_segment),
            "question_numbers": marker_numbers,
        },
    )


# ---------------------------------------------------------------------------
# H2 — Length anomaly
# ---------------------------------------------------------------------------


def detect_h2_length_anomaly(
    response: DetectedResponse,
    expected_max_words: int | None,
) -> Flag | None:
    """H2: Word count greater than expected_max_words * 2.5.

    Confidence: medium.
    Severity: WARNING (from flag registry).

    Args:
        response: The detected response.
        expected_max_words: Upper bound of the expected word range for this
            question (from evalpen_questions.expected_word_range).
            If None, heuristic is skipped.
    """
    if expected_max_words is None or expected_max_words <= 0:
        return None

    threshold = expected_max_words * H2_LENGTH_MULTIPLIER
    if response.word_count <= threshold:
        return None

    return _make_flag(
        FlagType.CLUBBED_LENGTH_ANOMALY,
        response.response_id,
        {
            "heuristic": "H2",
            "confidence": H2_CONFIDENCE,
            "word_count": response.word_count,
            "expected_max_words": expected_max_words,
            "threshold": threshold,
        },
    )


# ---------------------------------------------------------------------------
# H3 — Missing question
# ---------------------------------------------------------------------------


def detect_h3_missing_questions(
    responses: list[DetectedResponse],
    manifest_question_numbers: set[int],
) -> list[Flag]:
    """H3: Manifest question not represented in any segment.

    Confidence: high.
    Severity: WARNING (from flag registry).

    Args:
        responses: All detected responses for the submission.
        manifest_question_numbers: Set of question numbers expected from the
            exam manifest (evalpen_questions for this exam).

    Returns:
        One flag per missing question number (not tied to a specific
        response since the question was never found).
    """
    represented: set[int] = set()
    for resp in responses:
        if resp.question_number is not None:
            represented.add(resp.question_number)

    missing = manifest_question_numbers - represented
    flags: list[Flag] = []
    for q_num in sorted(missing):
        flags.append(
            _make_flag(
                FlagType.CLUBBED_MISSING_QUESTION,
                None,  # no response to associate
                {
                    "heuristic": "H3",
                    "confidence": H3_CONFIDENCE,
                    "missing_question_number": q_num,
                },
            )
        )
    return flags


# ---------------------------------------------------------------------------
# H4 — Topic discontinuity (data structure only)
# ---------------------------------------------------------------------------


@dataclass
class H4Request:
    """Data payload for an H4 topic-discontinuity check.

    This is NOT executed in the domain layer.  The caller passes this to
    the LLM gate with caller_id = 'pcr_clubbed_h4'.

    See LLM_GATE_SPEC section 5, allowed callers.
    """

    response_id: str
    detected_text: str
    question_number: int | None
    sub_part: str | None


def build_h4_request(response: DetectedResponse) -> H4Request:
    """Create the data payload for an H4 gate call.

    Actual invocation happens outside this module (constraint C4).
    """
    return H4Request(
        response_id=response.response_id,
        detected_text=response.detected_text,
        question_number=response.question_number,
        sub_part=response.sub_part,
    )


def create_h4_flag(response_id: str) -> Flag:
    """Create an H4 topic-discontinuity flag after a positive gate result.

    Called by the orchestration layer after the LLM gate returns a positive
    discontinuity signal.
    """
    return _make_flag(
        FlagType.CLUBBED_TOPIC_DISCONTINUITY,
        response_id,
        {
            "heuristic": "H4",
            "confidence": H4_CONFIDENCE,
        },
    )


# ---------------------------------------------------------------------------
# Orchestrator (H1 + H2 + H3, excluding H4 which requires the gate)
# ---------------------------------------------------------------------------


def detect_clubbed_responses(
    responses: list[DetectedResponse],
    markers_by_response: dict[str, list[QMarker]],
    expected_max_words_by_question: dict[int, int | None] | None = None,
    manifest_question_numbers: set[int] | None = None,
) -> ClubbedDetectionResult:
    """Run H1, H2, and H3 heuristics across all responses.

    H4 is excluded because it requires an LLM gate call (constraint C4).
    The caller should invoke build_h4_request / create_h4_flag separately
    through the gate.

    Args:
        responses: All detected responses for the submission.
        markers_by_response: Mapping from response_id to the Q markers
            found within that response's region.
        expected_max_words_by_question: Mapping from question_number to
            the upper bound of expected word range.  Used by H2.
        manifest_question_numbers: Full set of question numbers expected
            from the exam manifest.  Used by H3.

    Returns:
        ClubbedDetectionResult with all flags and trigger indicators.
    """
    flags: list[Flag] = []
    h1_triggered = False
    h2_triggered = False
    h3_triggered = False

    expected_words = expected_max_words_by_question or {}

    for response in responses:
        # H1 — multiple markers
        markers = markers_by_response.get(response.response_id, [])
        h1_flag = detect_h1_multiple_markers(response, markers)
        if h1_flag is not None:
            flags.append(h1_flag)
            h1_triggered = True

        # H2 — length anomaly
        max_words = expected_words.get(response.question_number) if response.question_number else None
        h2_flag = detect_h2_length_anomaly(response, max_words)
        if h2_flag is not None:
            flags.append(h2_flag)
            h2_triggered = True

    # H3 — missing questions (submission-level, not per-response)
    if manifest_question_numbers:
        h3_flags = detect_h3_missing_questions(
            responses, manifest_question_numbers
        )
        if h3_flags:
            flags.extend(h3_flags)
            h3_triggered = True

    return ClubbedDetectionResult(
        flags=flags,
        h1_triggered=h1_triggered,
        h2_triggered=h2_triggered,
        h3_triggered=h3_triggered,
    )
