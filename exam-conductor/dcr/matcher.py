"""
DCR Template Matcher.

Performs character-level and semantic template matching between HWR-recognized
text and the answer key.  Produces a ``MatchOutput`` with ``match_type`` and
``score`` for each question.

Architecture: DUAL_MODE_ARCHITECTURE.md §4.3 (template matching stage)
Test ID: U-DCR-02 — exact / partial / numeric / no-match logic
Failure mode: DCR-02 — numeric tolerance and normalization rules
Failure mode: DCR-03 — keep DCR deterministic and contract-bound (no deep PCR semantics)

Hard constraints:
  - C4: Deterministic matching only.  No LLM calls.
  - C3: No practice behavior created.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Optional

from .models import (
    AnswerKey,
    MatchOutput,
    MatchType,
    RecognitionOutput,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_PARTIAL_MATCH_THRESHOLD = 0.50  # Levenshtein ratio >= 50% → partial
DEFAULT_NUMERIC_TOLERANCE = 1e-6  # absolute tolerance for float comparison
DEFAULT_PARTIAL_SCORE_FRACTION = 0.50  # fraction of max_score for partial matches


class TemplateMatcher:
    """
    Deterministic template matcher for DCR.

    Match pipeline (per question):
        1. Normalize both recognized_text and expected_text
        2. Try exact match (case-insensitive, whitespace-normalized)
        3. Try numeric match (with tolerance — DCR-02 mitigation)
        4. Try partial match (character-level Levenshtein ratio)
        5. Fallback to no_match

    No LLM calls (C4).  No practice persistence (C3).
    """

    def __init__(
        self,
        *,
        partial_match_threshold: float = DEFAULT_PARTIAL_MATCH_THRESHOLD,
        numeric_tolerance: float = DEFAULT_NUMERIC_TOLERANCE,
        partial_score_fraction: float = DEFAULT_PARTIAL_SCORE_FRACTION,
    ) -> None:
        """
        Parameters
        ----------
        partial_match_threshold
            Minimum Levenshtein similarity ratio for a partial match.
        numeric_tolerance
            Default absolute tolerance when comparing numeric answers (DCR-02).
        partial_score_fraction
            Fraction of ``max_score`` awarded for partial matches.
        """
        self.partial_match_threshold = partial_match_threshold
        self.numeric_tolerance = numeric_tolerance
        self.partial_score_fraction = partial_score_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        recognition: RecognitionOutput,
        answer_key: AnswerKey,
    ) -> MatchOutput:
        """
        Match a single recognition output against its answer key.

        Parameters
        ----------
        recognition
            HWR output for one question.
        answer_key
            Expected answer for the same question.

        Returns
        -------
        MatchOutput
            Contains ``match_type``, ``score``, and context fields.
        """
        recognized = recognition.recognized_text
        expected = answer_key.expected_text

        norm_recognized = self._normalize(recognized)
        norm_expected = self._normalize(expected)

        # Resolve numeric tolerance: per-question override or global default
        tolerance = (
            answer_key.numeric_tolerance
            if answer_key.numeric_tolerance is not None
            else self.numeric_tolerance
        )

        # Resolve match mode hints from the answer key
        match_mode = answer_key.match_mode

        # ── Step 1: Exact match ──────────────────────────────────────
        if match_mode in (None, "exact", "case_insensitive"):
            if self._is_exact_match(norm_recognized, norm_expected):
                return MatchOutput(
                    question_id=recognition.question_id,
                    match_type=MatchType.EXACT_MATCH,
                    score=answer_key.max_score,
                    max_score=answer_key.max_score,
                    recognized_text=recognized,
                    expected_text=expected,
                    confidence=recognition.confidence,
                )

        # ── Step 2: Numeric match (DCR-02 mitigation) ───────────────
        if match_mode in (None, "numeric"):
            numeric_result = self._try_numeric_match(
                norm_recognized, norm_expected, tolerance
            )
            if numeric_result:
                return MatchOutput(
                    question_id=recognition.question_id,
                    match_type=MatchType.NUMERIC_MATCH,
                    score=answer_key.max_score,
                    max_score=answer_key.max_score,
                    recognized_text=recognized,
                    expected_text=expected,
                    confidence=recognition.confidence,
                )

        # ── Step 3: Partial match ────────────────────────────────────
        if match_mode in (None, "exact", "case_insensitive"):
            ratio = self._levenshtein_ratio(norm_recognized, norm_expected)
            if ratio >= self.partial_match_threshold:
                partial_score = round(
                    answer_key.max_score * self.partial_score_fraction, 2
                )
                return MatchOutput(
                    question_id=recognition.question_id,
                    match_type=MatchType.PARTIAL_MATCH,
                    score=partial_score,
                    max_score=answer_key.max_score,
                    recognized_text=recognized,
                    expected_text=expected,
                    confidence=recognition.confidence,
                )

        # ── Step 4: No match ─────────────────────────────────────────
        return MatchOutput(
            question_id=recognition.question_id,
            match_type=MatchType.NO_MATCH,
            score=0.0,
            max_score=answer_key.max_score,
            recognized_text=recognized,
            expected_text=expected,
            confidence=recognition.confidence,
        )

    def match_batch(
        self,
        recognitions: List[RecognitionOutput],
        answer_keys: List[AnswerKey],
    ) -> List[MatchOutput]:
        """
        Match a batch of recognition outputs against their answer keys.

        The two lists are joined by ``question_id``.  Recognitions without
        a matching answer key are skipped with a warning.

        Returns
        -------
        list[MatchOutput]
        """
        key_map = {ak.question_id: ak for ak in answer_keys}
        outputs: List[MatchOutput] = []

        for rec in recognitions:
            ak = key_map.get(rec.question_id)
            if ak is None:
                logger.warning(
                    "No answer key found for question %s; skipping match.",
                    rec.question_id,
                )
                continue
            outputs.append(self.match(rec, ak))

        return outputs

    # ------------------------------------------------------------------
    # Internal matching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Normalize text for comparison.

        Steps:
          1. Unicode NFKC normalization (handles ligatures, full-width chars).
          2. Strip leading/trailing whitespace.
          3. Collapse interior whitespace to single space.
          4. Lower-case.
        """
        text = unicodedata.normalize("NFKC", text)
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        return text

    @staticmethod
    def _is_exact_match(a: str, b: str) -> bool:
        """Case-insensitive, whitespace-normalized exact match."""
        return a == b and len(a) > 0

    @staticmethod
    def _try_numeric_match(
        recognized: str,
        expected: str,
        tolerance: float,
    ) -> bool:
        """
        Attempt to parse both strings as numbers and compare within tolerance.

        Handles:
          - integers and floats
          - leading/trailing whitespace (already normalized)
          - common OCR artefacts: commas as thousands separators, trailing periods

        DCR-02 mitigation: explicit tolerance and normalization rules.
        """
        rec_num = _parse_numeric(recognized)
        exp_num = _parse_numeric(expected)

        if rec_num is None or exp_num is None:
            return False

        return abs(rec_num - exp_num) <= tolerance

    @staticmethod
    def _levenshtein_ratio(a: str, b: str) -> float:
        """
        Compute normalized Levenshtein similarity ratio in [0, 1].

        Uses the standard dynamic-programming edit distance.  Returns 1.0
        for identical strings, 0.0 for completely different strings.
        """
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0

        len_a = len(a)
        len_b = len(b)

        # Optimize: use two-row DP to keep memory O(min(len_a, len_b)).
        if len_a > len_b:
            a, b = b, a
            len_a, len_b = len_b, len_a

        prev_row = list(range(len_a + 1))
        for j in range(1, len_b + 1):
            curr_row = [j] + [0] * len_a
            for i in range(1, len_a + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr_row[i] = min(
                    curr_row[i - 1] + 1,      # insertion
                    prev_row[i] + 1,           # deletion
                    prev_row[i - 1] + cost,    # substitution
                )
            prev_row = curr_row

        distance = prev_row[len_a]
        max_len = max(len_a, len_b)
        return 1.0 - (distance / max_len)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_numeric(text: str) -> Optional[float]:
    """
    Best-effort numeric parsing.

    Handles:
      - plain integers and floats
      - thousands separators (commas): "1,000" → 1000
      - trailing dots from OCR: "42." → 42.0
      - fraction notation: "1/2" → 0.5
    """
    if not text:
        return None

    # Strip commas (thousands separators)
    cleaned = text.replace(",", "")

    # Strip trailing dot (common OCR artefact)
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]

    # Try direct float parse
    try:
        return float(cleaned)
    except (ValueError, OverflowError):
        pass

    # Try fraction notation  a/b
    fraction_match = re.match(r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$", cleaned)
    if fraction_match:
        numerator = float(fraction_match.group(1))
        denominator = float(fraction_match.group(2))
        if denominator != 0:
            return numerator / denominator

    return None
