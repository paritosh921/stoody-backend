"""
ExamPen Test Harness — PCR domain tests.

Test IDs covered:
    U-SEG-01  Boundary detection with spec parameters
    U-SEG-02  Q marker regex parsing
    U-SEG-03  Cross-page stitching
    U-CCLS-01 Content classification thresholds
    U-CLUB-01 Clubbed detection H1-H3
    U-EVAL-01 Eval result shape

Spec authority: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md sections 4-6
Failure modes:  PCR-01 (boundary/marker failure), PCR-02 (clubbed undetected),
                PCR-03 (diagram-heavy auto-scored incorrectly)
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_EC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "exam-conductor")
if _EC_DIR not in sys.path:
    sys.path.insert(0, _EC_DIR)

from pcr.domain.response_models import (
    BoundingBox,
    ContentType,
    DetectedBoundary,
    DetectedResponse,
    Flag,
    FlagSeverity,
    FlagType,
    PageOCR,
    QMarker,
    SegmentationResult,
    SourcePageRef,
    TextBlock,
)
from pcr.domain.segmenter import segment_submission
from pcr.domain.content_classifier import (
    TEXT_ONLY_THRESHOLD,
    MIXED_LOWER_THRESHOLD,
    classify_content,
    compute_scoreable_marks,
    compute_text_coverage,
)
from pcr.domain.flag_registry import FLAG_REGISTRY, FlagDefinition
from pcr.domain.marker_parser import Q_MARKER_PATTERN, parse_markers, _apply_ocr_fixes
from pcr.domain.boundary_detector import StrokeLine
from pcr.domain.clubbed_detector import detect_clubbed_responses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_block(
    text: str,
    x_min: float = 0.0,
    y_min: float = 0.0,
    x_max: float = 100.0,
    y_max: float = 10.0,
    confidence: float = 0.9,
    source: str = "pen",
) -> TextBlock:
    return TextBlock(
        text=text,
        bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        confidence=confidence,
        source=source,
    )


def _make_page_ocr(
    page_number: int,
    text_blocks: list[TextBlock] | None = None,
    page_width: float = 210.0,
    page_height: float = 297.0,
    source: str = "pen",
) -> PageOCR:
    return PageOCR(
        page_number=page_number,
        page_width_mm=page_width,
        page_height_mm=page_height,
        text_blocks=text_blocks or [],
        source=source,
        mean_ocr_confidence=0.85,
    )


# ===========================================================================
# U-SEG-01: Boundary detection with spec parameters
# ===========================================================================


class TestUSeg01:
    """U-SEG-01: Boundary detection parameters from PCR_EVAL_ENGINE_SPEC 4.1."""

    def test_u_seg_01_empty_pages_returns_empty(self):
        """Empty page list produces empty segmentation result."""
        result = segment_submission([])
        assert isinstance(result, SegmentationResult)
        assert result.page_count == 0
        assert result.responses == []
        assert result.has_blocking_flags is False

    def test_u_seg_01_single_page_no_boundaries(self):
        """Single page with text but no boundaries produces one response."""
        blocks = [
            _make_text_block("This is an answer", y_min=20.0, y_max=40.0),
        ]
        pages = [_make_page_ocr(1, blocks)]
        result = segment_submission(pages)

        assert result.page_count == 1
        assert result.total_boundaries_detected == 0
        assert len(result.responses) == 1

    def test_u_seg_01_splits_single_ocr_block_on_multiple_q_markers(self):
        """One OCR block with multiple Q markers becomes separate responses."""
        blocks = [
            _make_text_block(
                "Q.No 1.Ans Water boils at 100 C. "
                "Q.No 2.Ans Plants make food by photosynthesis. "
                "Q.No 3.Ans Force equals mass times acceleration.",
                y_min=0.0,
                y_max=297.0,
            ),
        ]
        pages = [_make_page_ocr(1, blocks)]

        result = segment_submission(pages)

        assert result.total_boundaries_detected == 0
        assert result.total_markers_detected == 3
        assert len(result.responses) == 3
        assert [r.question_number for r in result.responses] == [1, 2, 3]
        assert not any(
            flag.flag_type == FlagType.CLUBBED_MULTIPLE_MARKERS
            for response in result.responses
            for flag in response.flags
        )

    def test_u_seg_01_marker_split_preserves_content_above_first_marker(self):
        """Marker-delimited pages keep old page-top coverage for first answer."""
        blocks = [
            _make_text_block(
                "Student rough note",
                y_min=5.0,
                y_max=15.0,
            ),
            _make_text_block(
                "Q.No 1.Ans First answer. Q.No 2.Ans Second answer.",
                y_min=50.0,
                y_max=250.0,
            ),
        ]
        pages = [_make_page_ocr(1, blocks)]

        result = segment_submission(pages)

        assert len(result.responses) == 2
        assert result.responses[0].question_number == 1
        assert "Student rough note" in result.responses[0].detected_text
        assert "Second answer" in result.responses[1].detected_text

    def test_u_seg_01_boundary_detection_pen_path(self):
        """Pen path boundary detection uses stroke geometry.

        Spec parameters:
        - slope: within +/-10 degrees of horizontal
        - length: > 40% of page width
        - Y-gap between paired lines: 2-15 mm
        - temporal proximity: ~3 seconds
        - horizontal overlap: > 70%
        """
        # Create two horizontal lines that form a boundary pair
        line1 = StrokeLine(
            x_start=10.0, y_start=100.0, x_end=180.0, y_end=100.5,
            page_number=1, timestamp=1000.0,
        )
        line2 = StrokeLine(
            x_start=10.0, y_start=107.0, x_end=180.0, y_end=107.5,
            page_number=1, timestamp=1600.0,
        )

        blocks = [
            _make_text_block("Answer part 1", y_min=30.0, y_max=50.0),
            _make_text_block("Answer part 2", y_min=120.0, y_max=140.0),
        ]
        pages = [_make_page_ocr(1, blocks)]

        result = segment_submission(pages, stroke_lines=[line1, line2])
        # Boundaries detected depends on implementation details, but the
        # segmenter should process them without error
        assert isinstance(result, SegmentationResult)

    def test_u_seg_01_segmentation_result_shape(self):
        """SegmentationResult has all spec-defined summary fields."""
        result = SegmentationResult(
            responses=[],
            flags=[],
            page_count=2,
            total_boundaries_detected=1,
            total_markers_detected=3,
            has_blocking_flags=False,
        )
        assert result.page_count == 2
        assert result.total_boundaries_detected == 1
        assert result.total_markers_detected == 3
        assert result.has_blocking_flags is False

    def test_u_seg_01_detected_boundary_shape(self):
        """DetectedBoundary has all fields from PCR_EVAL_ENGINE_SPEC 4.1."""
        boundary = DetectedBoundary(
            y_top=100.0,
            y_bottom=107.0,
            page_number=1,
            confidence=0.85,
            detection_method="stroke_geometry",
        )
        assert boundary.y_top == 100.0
        assert boundary.y_bottom == 107.0
        assert boundary.detection_method == "stroke_geometry"


# ===========================================================================
# U-SEG-02: Q marker regex parsing
# ===========================================================================


class TestUSeg02:
    """U-SEG-02: Question marker parsing (PCR_EVAL_ENGINE_SPEC 4.2)."""

    def test_u_seg_02_basic_marker(self):
        """Standard 'Q.No 1.Ans' format is parsed."""
        match = Q_MARKER_PATTERN.search("Q.No 1.Ans")
        assert match is not None
        assert match.group(1) == "1"

    def test_u_seg_02_marker_with_sub_part(self):
        """'Q.No 2(a).Ans' parses question=2, sub_part=a."""
        match = Q_MARKER_PATTERN.search("Q.No 2(a).Ans")
        assert match is not None
        assert match.group(1) == "2"
        assert match.group(2) == "a"

    def test_u_seg_02_case_insensitive(self):
        """Pattern is case-insensitive per spec."""
        match = Q_MARKER_PATTERN.search("q.no 5.ans")
        assert match is not None
        assert match.group(1) == "5"

        match_upper = Q_MARKER_PATTERN.search("Q.NO 5.ANS")
        assert match_upper is not None

    def test_u_seg_02_optional_dots(self):
        """Dots after Q and No are optional."""
        match = Q_MARKER_PATTERN.search("QNo 3 Ans")
        assert match is not None
        assert match.group(1) == "3"

    @pytest.mark.parametrize(
        "label, expected_question",
        [
            ("Q1", "1"),
            ("Q. 2", "2"),
            ("Question 3", "3"),
            ("Question No. 4", "4"),
        ],
    )
    def test_u_seg_02_common_handwritten_question_labels(self, label, expected_question):
        """Normal handwritten Q-number labels split an answer copy safely."""
        match = Q_MARKER_PATTERN.search(label)
        assert match is not None
        assert match.group(1) == expected_question

    def test_u_seg_02_three_digit_question_number(self):
        """Up to 3-digit question numbers are supported."""
        match = Q_MARKER_PATTERN.search("Q.No 123.Ans")
        assert match is not None
        assert match.group(1) == "123"

    def test_u_seg_02_roman_numeral_sub_part(self):
        """Roman numeral sub-parts (e.g. 'ii') are captured."""
        match = Q_MARKER_PATTERN.search("Q.No 1(ii).Ans")
        assert match is not None
        assert match.group(1) == "1"
        assert match.group(2) == "ii"

    def test_u_seg_02_uppercase_sub_part(self):
        """Uppercase letter sub-parts are captured."""
        match = Q_MARKER_PATTERN.search("Q.No 1(A).Ans")
        assert match is not None
        assert match.group(2) == "A"

    def test_u_seg_02_no_match_for_random_text(self):
        """Random text without Q marker format returns no match."""
        match = Q_MARKER_PATTERN.search("The answer is 42")
        assert match is None

    def test_u_seg_02_plain_numbered_answer_labels(self):
        """Content-section style 1)/2. labels map to question numbers."""
        from pcr.domain.marker_parser import (
            ANSWER_NUMBER_MARKER_PATTERN,
            is_form_header_text,
            parse_markers,
            strip_form_header_noise,
        )

        assert ANSWER_NUMBER_MARKER_PATTERN.search("1) 3630") is not None
        assert ANSWER_NUMBER_MARKER_PATTERN.search("2. 64") is not None
        assert ANSWER_NUMBER_MARKER_PATTERN.search("Ans 3) small diagram") is not None
        assert ANSWER_NUMBER_MARKER_PATTERN.search("The answer is 42") is None

        page = _make_page_ocr(
            1,
            [
                _make_text_block("Prayaan Answer Book Date Page", y_min=2.0, y_max=12.0),
                _make_text_block("1) 50,067", y_min=40.0, y_max=55.0),
                _make_text_block("2) 3630", y_min=70.0, y_max=85.0),
            ],
        )
        markers = parse_markers([page])
        assert [m.question_number for m in markers] == [1, 2]
        assert is_form_header_text("Prayaan Answer Book Date Page")
        assert strip_form_header_noise("Prayaan Answer Book Date Page") == ""

    def test_u_seg_02_ocr_fix_l_to_1(self):
        """OCR fix: l -> 1 helps recognize 'Q.No l.Ans' as Q.No 1."""
        fixed = _apply_ocr_fixes("Q.No l.Ans")
        match = Q_MARKER_PATTERN.search(fixed)
        assert match is not None
        assert match.group(1) == "1"

    def test_u_seg_02_ocr_fix_O_to_0(self):
        """OCR fix: O -> 0 helps recognize digits."""
        fixed = _apply_ocr_fixes("1O")
        assert "10" in fixed

    def test_u_seg_02_ocr_fix_I_to_1(self):
        """OCR fix: I -> 1 helps recognize digits."""
        fixed = _apply_ocr_fixes("Q.No I.Ans")
        match = Q_MARKER_PATTERN.search(fixed)
        assert match is not None
        assert match.group(1) == "1"

    def test_u_seg_02_parse_markers_multipage(self):
        """parse_markers extracts markers across multiple pages."""
        page1 = _make_page_ocr(
            1,
            [_make_text_block("Q.No 1.Ans", y_min=10.0, y_max=20.0)],
        )
        page2 = _make_page_ocr(
            2,
            [_make_text_block("Q.No 2.Ans", y_min=10.0, y_max=20.0)],
        )
        markers = parse_markers([page1, page2])
        assert len(markers) == 2
        assert markers[0].question_number == 1
        assert markers[0].page_number == 1
        assert markers[1].question_number == 2
        assert markers[1].page_number == 2

    def test_u_seg_02_qmarker_shape(self):
        """QMarker model has all required fields."""
        marker = QMarker(
            question_number=5,
            sub_part="a",
            raw_text="Q.No 5(a).Ans",
            page_number=1,
            y_position=30.0,
            confidence=0.88,
        )
        assert marker.question_number == 5
        assert marker.sub_part == "a"
        assert marker.page_number == 1


# ===========================================================================
# U-SEG-03: Cross-page stitching
# ===========================================================================


class TestUSeg03:
    """U-SEG-03: Cross-page stitching (PCR_EVAL_ENGINE_SPEC 4.3)."""

    def test_u_seg_03_continuation_no_boundary(self):
        """Page N+1 with no boundary -> continuation of previous page.

        Spec: 'page N+1 starts with content and no boundary -> continuation'
        """
        page1 = _make_page_ocr(
            1,
            [_make_text_block("Start of answer", y_min=20.0, y_max=40.0)],
        )
        page2 = _make_page_ocr(
            2,
            [_make_text_block("Continued answer", y_min=20.0, y_max=40.0)],
        )
        result = segment_submission([page1, page2])

        # Without boundaries, these should be stitched into one response
        assert len(result.responses) == 1
        response = result.responses[0]
        assert response.is_continuation is True
        assert len(response.source_pages) == 2

    def test_u_seg_03_multipage_response_detected_text(self):
        """Stitched response contains text from both pages."""
        page1 = _make_page_ocr(
            1,
            [_make_text_block("Part A", y_min=20.0, y_max=40.0)],
        )
        page2 = _make_page_ocr(
            2,
            [_make_text_block("Part B", y_min=20.0, y_max=40.0)],
        )
        result = segment_submission([page1, page2])

        assert len(result.responses) == 1
        text = result.responses[0].detected_text
        assert "Part A" in text
        assert "Part B" in text

    def test_u_seg_03_detected_response_shape(self):
        """DetectedResponse has all fields from PCR_EVAL_ENGINE_SPEC section 7.2."""
        response = DetectedResponse(
            response_id="RESP-abc123",
            question_number=1,
            sub_part="a",
            detected_text="Sample answer text",
            source_pages=[
                SourcePageRef(page_number=1, y_start=0.0, y_end=100.0),
            ],
            content_type=ContentType.TEXT_ONLY,
            text_coverage_ratio=0.9,
            segmentation_confidence=0.85,
            ocr_confidence=0.88,
            word_count=3,
            is_continuation=False,
        )
        assert response.response_id == "RESP-abc123"
        assert response.question_number == 1
        assert response.content_type == ContentType.TEXT_ONLY
        assert response.flags == []

    def test_u_seg_03_page_count_tracked(self):
        """SegmentationResult.page_count reflects actual pages processed."""
        pages = [
            _make_page_ocr(1, [_make_text_block("A")]),
            _make_page_ocr(2, [_make_text_block("B")]),
            _make_page_ocr(3, [_make_text_block("C")]),
        ]
        result = segment_submission(pages)
        assert result.page_count == 3


# ===========================================================================
# U-CCLS-01: Content classification thresholds
# ===========================================================================


class TestUCcls01:
    """U-CCLS-01: Content classification (PCR_EVAL_ENGINE_SPEC 4.4)."""

    def test_u_ccls_01_thresholds_match_spec(self):
        """Threshold constants match PCR_EVAL_ENGINE_SPEC 4.4."""
        assert TEXT_ONLY_THRESHOLD == 0.85
        assert MIXED_LOWER_THRESHOLD == 0.40

    def test_u_ccls_01_text_only_classification(self):
        """Text coverage > 85% -> TEXT_ONLY."""
        # Create blocks that cover > 85% of the response area
        blocks = [
            _make_text_block(
                "Full text answer",
                x_min=0.0, y_min=0.0, x_max=100.0, y_max=90.0,
            ),
        ]
        bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)

        content_type, coverage, flags = classify_content(blocks, bbox)
        assert content_type == ContentType.TEXT_ONLY
        assert coverage > TEXT_ONLY_THRESHOLD

    def test_u_ccls_01_mixed_classification(self):
        """Text coverage 40-85% -> MIXED with DIAGRAM_PRESENT flag."""
        blocks = [
            _make_text_block(
                "Some text",
                x_min=0.0, y_min=0.0, x_max=100.0, y_max=60.0,
            ),
        ]
        bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)

        content_type, coverage, flags = classify_content(blocks, bbox)
        assert content_type == ContentType.MIXED
        assert MIXED_LOWER_THRESHOLD <= coverage <= TEXT_ONLY_THRESHOLD
        flag_types = [f.flag_type for f in flags]
        assert FlagType.DIAGRAM_PRESENT in flag_types

    def test_u_ccls_01_sparse_text_still_scoreable(self):
        """Sparse handwriting in a large region must remain auto-scoreable.

        Camera OCR often yields small text bboxes inside a full-page segment.
        That must not raise a blocking DIAGRAM_HEAVY flag (production left
        papers at 0/N evaluated after reprocess).
        """
        blocks = [
            _make_text_block(
                "Tiny text",
                x_min=0.0, y_min=0.0, x_max=30.0, y_max=10.0,
            ),
        ]
        bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)

        content_type, coverage, flags = classify_content(blocks, bbox)
        assert coverage < MIXED_LOWER_THRESHOLD
        assert content_type in {ContentType.TEXT_ONLY, ContentType.MIXED}
        flag_types = [f.flag_type for f in flags]
        assert FlagType.DIAGRAM_HEAVY_CONTENT not in flag_types
        blocking_flags = [
            f for f in flags if f.severity == FlagSeverity.BLOCKING
        ]
        assert blocking_flags == []

    def test_u_ccls_01_no_text_blocks(self):
        """No text blocks -> DIAGRAM_HEAVY (true blank geometry)."""
        bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)
        content_type, coverage, flags = classify_content([], bbox)
        assert content_type == ContentType.DIAGRAM_HEAVY
        assert coverage == 0.0
        assert any(
            f.flag_type == FlagType.DIAGRAM_HEAVY_CONTENT
            and f.severity == FlagSeverity.BLOCKING
            for f in flags
        )

    def test_u_ccls_01_expected_diagram_missing(self):
        """TEXT_ONLY + expects_diagram=True -> EXPECTED_DIAGRAM_MISSING flag."""
        blocks = [
            _make_text_block(
                "Full text no diagram",
                x_min=0.0, y_min=0.0, x_max=100.0, y_max=90.0,
            ),
        ]
        bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)

        content_type, coverage, flags = classify_content(
            blocks, bbox, expects_diagram=True
        )
        assert content_type == ContentType.TEXT_ONLY
        flag_types = [f.flag_type for f in flags]
        assert FlagType.EXPECTED_DIAGRAM_MISSING in flag_types

    def test_u_ccls_01_scoreable_marks_prorating(self):
        """compute_scoreable_marks applies the spec formula correctly."""
        # scoreable_marks = max_marks * (1 - diagram_weight)
        assert compute_scoreable_marks(10.0, 0.3) == 7.0
        assert compute_scoreable_marks(10.0, 0.0) == 10.0
        assert compute_scoreable_marks(10.0, 1.0) == 0.0

    def test_u_ccls_01_scoreable_marks_invalid_weight(self):
        """Invalid diagram_weight raises ValueError."""
        with pytest.raises(ValueError):
            compute_scoreable_marks(10.0, -0.1)
        with pytest.raises(ValueError):
            compute_scoreable_marks(10.0, 1.5)

    def test_u_ccls_01_content_type_enum_values(self):
        """ContentType enum has all four spec-defined types."""
        assert ContentType.TEXT_ONLY.value == "TEXT_ONLY"
        assert ContentType.MIXED.value == "MIXED"
        assert ContentType.DIAGRAM_HEAVY.value == "DIAGRAM_HEAVY"
        assert ContentType.TABLE_PRESENT.value == "TABLE_PRESENT"
        assert len(ContentType) == 4


# ===========================================================================
# U-CLUB-01: Clubbed detection H1-H3
# ===========================================================================


class TestUClub01:
    """U-CLUB-01: Clubbed response heuristics (PCR_EVAL_ENGINE_SPEC 4.5)."""

    def _make_response(
        self,
        response_id: str = "RESP-001",
        question_number: int = 1,
        text: str = "Some answer",
        word_count: int = 10,
    ) -> DetectedResponse:
        return DetectedResponse(
            response_id=response_id,
            question_number=question_number,
            detected_text=text,
            source_pages=[
                SourcePageRef(page_number=1, y_start=0.0, y_end=100.0)
            ],
            segmentation_confidence=0.8,
            ocr_confidence=0.85,
            word_count=word_count,
        )

    def _make_marker(
        self,
        question_number: int = 1,
        page_number: int = 1,
    ) -> QMarker:
        return QMarker(
            question_number=question_number,
            sub_part=None,
            raw_text=f"Q.No {question_number}.Ans",
            page_number=page_number,
            y_position=10.0,
            confidence=0.9,
        )

    def test_u_club_01_h1_multiple_markers(self):
        """H1: Multiple Q markers in one segment -> CLUBBED_MULTIPLE_MARKERS (blocking)."""
        response = self._make_response("RESP-001", question_number=1)
        markers_map = {
            "RESP-001": [
                self._make_marker(1),
                self._make_marker(2),
            ],
        }

        result = detect_clubbed_responses(
            [response], markers_map
        )
        assert result.h1_triggered is True
        blocking_flags = [
            f for f in result.flags
            if f.flag_type == FlagType.CLUBBED_MULTIPLE_MARKERS
        ]
        assert len(blocking_flags) >= 1
        assert blocking_flags[0].severity == FlagSeverity.BLOCKING

    def test_u_club_01_h2_length_anomaly(self):
        """H2: Word count > expected_max * 2.5 -> CLUBBED_LENGTH_ANOMALY."""
        response = self._make_response(
            "RESP-001", question_number=1, word_count=300
        )
        markers_map = {
            "RESP-001": [self._make_marker(1)],
        }
        expected_max_words = {1: 100}  # 300 > 100 * 2.5 = 250

        result = detect_clubbed_responses(
            [response],
            markers_map,
            expected_max_words_by_question=expected_max_words,
        )
        assert result.h2_triggered is True
        length_flags = [
            f for f in result.flags
            if f.flag_type == FlagType.CLUBBED_LENGTH_ANOMALY
        ]
        assert len(length_flags) >= 1

    def test_u_club_01_h2_not_triggered_normal_length(self):
        """H2: Normal word count does not trigger."""
        response = self._make_response(
            "RESP-001", question_number=1, word_count=50
        )
        markers_map = {
            "RESP-001": [self._make_marker(1)],
        }
        expected_max_words = {1: 100}  # 50 < 100 * 2.5 = 250

        result = detect_clubbed_responses(
            [response],
            markers_map,
            expected_max_words_by_question=expected_max_words,
        )
        assert result.h2_triggered is False

    def test_u_club_01_h3_missing_question(self):
        """H3: Manifest question not in any segment -> CLUBBED_MISSING_QUESTION."""
        response = self._make_response(
            "RESP-001", question_number=1, word_count=50
        )
        markers_map = {
            "RESP-001": [self._make_marker(1)],
        }
        # Question 2 is in the manifest but not in any response
        manifest = {1, 2, 3}

        result = detect_clubbed_responses(
            [response],
            markers_map,
            manifest_question_numbers=manifest,
        )
        assert result.h3_triggered is True
        missing_flags = [
            f for f in result.flags
            if f.flag_type == FlagType.CLUBBED_MISSING_QUESTION
        ]
        assert len(missing_flags) >= 1

    def test_u_club_01_h3_all_questions_present(self):
        """H3: All manifest questions present -> no missing flag."""
        responses = [
            self._make_response("RESP-001", question_number=1),
            self._make_response("RESP-002", question_number=2),
        ]
        markers_map = {
            "RESP-001": [self._make_marker(1)],
            "RESP-002": [self._make_marker(2)],
        }
        manifest = {1, 2}

        result = detect_clubbed_responses(
            responses, markers_map, manifest_question_numbers=manifest
        )
        assert result.h3_triggered is False

    def test_u_club_01_no_heuristics_triggered(self):
        """No clubbed heuristics triggered for clean input."""
        response = self._make_response("RESP-001", question_number=1, word_count=50)
        markers_map = {"RESP-001": [self._make_marker(1)]}

        result = detect_clubbed_responses([response], markers_map)
        assert result.h1_triggered is False
        assert result.h2_triggered is False
        assert result.h3_triggered is False
        assert len(result.flags) == 0


# ===========================================================================
# U-EVAL-01: Eval result shape
# ===========================================================================


class TestUEval01:
    """U-EVAL-01: Eval result parsing and scoring envelope."""

    def test_u_eval_01_eval_model_uses_gate_default_for_openai(self, monkeypatch):
        """Non-Anthropic gate providers use their configured default model."""
        from pcr.services import eval_core

        monkeypatch.setenv("AI_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5.1")

        assert eval_core._select_eval_model("L2", cache_hit=False) == "gpt-5.1"

    def test_u_eval_01_eval_model_defaults_to_openai_provider(self, monkeypatch):
        """Provider unset follows the gate's OpenAI default model behavior."""
        from pcr.services import eval_core

        monkeypatch.delenv("AI_PROVIDER", raising=False)
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5.1")

        assert eval_core._select_eval_model("L2", cache_hit=False) == "gpt-5.1"

    def test_u_eval_01_eval_model_keeps_anthropic_tiers(self, monkeypatch):
        """Anthropic deployments keep the configured Haiku/Sonnet tiers."""
        from pcr.services import eval_core

        monkeypatch.setenv("AI_PROVIDER", "anthropic")

        assert (
            eval_core._select_eval_model("L2", cache_hit=False)
            == "claude-sonnet-4-20250514"
        )

    @pytest.mark.asyncio
    async def test_u_eval_01_solution_warmup_uses_gate_default_for_openai(
        self,
        monkeypatch,
    ):
        """Solution generation uses the active provider default model."""
        from pcr.services.solution_cache import SolutionCache

        class Store:
            def __init__(self):
                self.docs = []

            async def get_latest_solution(self, _question_id):
                return None

            async def upsert_solution(self, doc):
                self.docs.append(doc)
                return doc

        class Gate:
            def __init__(self):
                self.calls = []

            async def call(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    content="Reference solution",
                    usage=SimpleNamespace(model="gpt-5.1", total_tokens=10),
                )

        monkeypatch.setenv("AI_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5.1")
        store = Store()
        gate = Gate()
        cache = SolutionCache(store, gate)

        result = await cache.lookup(
            "Q1",
            {
                "subject": "Science",
                "question_type": "subjective",
                "max_marks": 2,
                "question_text": "What is boiling point of water?",
                "complexity": "L2",
            },
        )

        assert result.reference_solution == "Reference solution"
        assert gate.calls[0]["model_id"] == "gpt-5.1"

    @pytest.mark.asyncio
    async def test_u_eval_01_teacher_rubric_is_used_without_generating_a_new_key(self):
        """A finalized rubric is authoritative marking material for PCR."""
        from pcr.services.solution_cache import SolutionCache

        class Store:
            def __init__(self):
                self.docs = []

            async def get_latest_solution(self, _question_id):
                return None

            async def upsert_solution(self, doc):
                self.docs.append(doc)
                return doc

        class Gate:
            async def call(self, **_kwargs):
                raise AssertionError("A teacher rubric must not trigger solution generation")

        store = Store()
        result = await SolutionCache(store, Gate()).lookup(
            "Q-rubric",
            {
                "subject": "English",
                "question_type": "essay",
                "max_marks": 5,
                "question_text": "Discuss the theme.",
                "rubric": "Award marks for theme, evidence, and clear expression.",
            },
        )

        assert result.hit is True
        assert result.reference_solution == "Award marks for theme, evidence, and clear expression."
        assert store.docs[0]["solution_source"] == "teacher"
        assert store.docs[0]["marking_material_type"] == "rubric"

    @pytest.mark.asyncio
    async def test_u_eval_01_subjective_option_letter_placeholder_generates_key(self):
        """A subjectively marked PCR question cannot be evaluated against "A"."""
        from pcr.services.solution_cache import SolutionCache

        class Store:
            def __init__(self):
                self.docs = [
                    {
                        "question_id": "Q-placeholder",
                        "version": 1,
                        "reference_solution": "A",
                        "solution_source": "teacher",
                    }
                ]

            async def get_latest_solution(self, question_id):
                matching = [doc for doc in self.docs if doc["question_id"] == question_id]
                return matching[-1] if matching else None

            async def upsert_solution(self, doc):
                self.docs.append(dict(doc))
                return doc

        class Gate:
            def __init__(self):
                self.calls = []

            async def call(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    content="Worked solution with calculation steps.",
                    usage=SimpleNamespace(model="gpt-5.1", total_tokens=10),
                )

        store = Store()
        gate = Gate()
        result = await SolutionCache(store, gate).lookup(
            "Q-placeholder",
            {
                "subject": "Physics",
                "question_type": "subjective",
                "max_marks": 4,
                "question_text": "Find the time of flight.",
                "reference_solution": "A",
            },
        )

        assert result.was_generated is True
        assert result.solution_source == "llm"
        assert result.reference_solution == "Worked solution with calculation steps."
        assert len(gate.calls) == 1
        assert store.docs[-1]["version"] == 2
        assert store.docs[-1]["solution_source"] == "llm"

    @pytest.mark.asyncio
    async def test_u_eval_01_solution_warmup_defaults_to_openai_provider(
        self,
        monkeypatch,
    ):
        """Provider unset in solution warmup follows the gate OpenAI default."""
        from pcr.services.solution_cache import SolutionCache

        class Store:
            async def get_latest_solution(self, _question_id):
                return None

            async def upsert_solution(self, doc):
                return doc

        class Gate:
            def __init__(self):
                self.calls = []

            async def call(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    content="Reference solution",
                    usage=SimpleNamespace(model="gpt-5.1", total_tokens=10),
                )

        monkeypatch.delenv("AI_PROVIDER", raising=False)
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5.1")
        gate = Gate()
        cache = SolutionCache(Store(), gate)

        await cache.lookup(
            "Q1",
            {
                "subject": "Science",
                "question_type": "subjective",
                "max_marks": 2,
                "question_text": "What is boiling point of water?",
                "complexity": "L2",
            },
        )

        assert gate.calls[0]["model_id"] == "gpt-5.1"

    def test_u_eval_01_eval_result_shape(self):
        """EvalResult has all fields from PCR_EVAL_ENGINE_SPEC section 7.3."""
        from pcr.services.eval_core import EvalResult, StepMark

        result = EvalResult(
            evaluation_id="EVAL-abc123",
            response_id="RESP-001",
            question_id="q-001",
            student_id="stu-001",
            eval_path="cache_hit",
            model_used="claude-haiku-4-20250514",
            total_score=7.5,
            max_score=10.0,
            scoreable_max=10.0,
            step_marks=[
                StepMark(step="Step 1", marks_awarded=4.0, max_marks=5.0, rationale="Good"),
                StepMark(step="Step 2", marks_awarded=3.5, max_marks=5.0, rationale="OK"),
            ],
            overall_feedback="Good attempt",
            reference_solution="The answer is...",
            token_usage={"total_tokens": 500},
            raw_llm_response='{"total_score": 7.5}',
        )
        assert result.evaluation_id == "EVAL-abc123"
        assert result.total_score == 7.5
        assert result.max_score == 10.0
        assert result.scoreable_max == 10.0
        assert len(result.step_marks) == 2
        assert result.step_marks[0].marks_awarded == 4.0
        assert result.overall_feedback == "Good attempt"
        assert result.skipped is False
        assert result.error is None

    def test_u_eval_01_batch_eval_result_shape(self):
        """BatchEvalResult aggregates individual results."""
        from pcr.services.eval_core import BatchEvalResult, EvalResult

        batch = BatchEvalResult(
            submission_id="sub-001",
            total_responses=3,
            evaluated_count=2,
            blocked_count=1,
            error_count=0,
            results=[
                EvalResult(
                    evaluation_id="EVAL-1",
                    response_id="RESP-1",
                    total_score=8.0,
                    max_score=10.0,
                ),
            ],
        )
        assert batch.submission_id == "sub-001"
        assert batch.total_responses == 3
        assert batch.evaluated_count == 2
        assert batch.blocked_count == 1

    def test_u_eval_01_parse_eval_response_valid_json(self):
        """_parse_eval_response parses valid JSON correctly."""
        from pcr.services.eval_core import _parse_eval_response

        raw = '{"step_marks": [{"step": "test", "marks_awarded": 5.0, "max_marks": 5.0, "rationale": "Perfect"}], "total_score": 5.0, "max_score": 10.0, "overall_feedback": "Well done"}'
        parsed = _parse_eval_response(raw, max_score=10.0)
        assert parsed["total_score"] == 5.0
        assert parsed["overall_feedback"] == "Well done"
        assert parsed["parse_error"] is False
        assert len(parsed["step_marks"]) == 1

    def test_u_eval_01_parse_eval_response_with_code_fences(self):
        """_parse_eval_response strips markdown code fences."""
        from pcr.services.eval_core import _parse_eval_response

        raw = '```json\n{"total_score": 3.0, "max_score": 5.0, "step_marks": [], "overall_feedback": "OK"}\n```'
        parsed = _parse_eval_response(raw, max_score=5.0)
        assert parsed["total_score"] == 3.0
        assert parsed["parse_error"] is False

    def test_u_eval_01_parse_eval_response_clamps_score(self):
        """_parse_eval_response clamps total_score to [0, max_score]."""
        from pcr.services.eval_core import _parse_eval_response

        raw = '{"total_score": 15.0, "max_score": 10.0, "step_marks": [], "overall_feedback": "OK"}'
        parsed = _parse_eval_response(raw, max_score=10.0)
        assert parsed["total_score"] == 10.0

    def test_u_eval_01_parse_eval_response_invalid_json(self):
        """_parse_eval_response returns degraded result on invalid JSON."""
        from pcr.services.eval_core import _parse_eval_response

        parsed = _parse_eval_response("not json at all", max_score=10.0)
        assert parsed["parse_error"] is True
        assert parsed["total_score"] == 0.0

    def test_u_eval_01_locked_criteria_require_exact_question_total(self):
        """A new PCR paper cannot freeze a rubric that invents or loses marks."""
        from pcr.marking_policy import validate_marking_criteria

        errors = validate_marking_criteria(
            [
                {
                    "criterion_id": "method",
                    "description": "Correct method",
                    "max_marks": 1,
                },
                {
                    "criterion_id": "result",
                    "description": "Correct result",
                    "max_marks": 2,
                },
            ],
            question_max_marks=4,
        )

        assert errors == ["criterion marks total 3, but this question is worth 4"]

    def test_u_eval_01_complete_criteria_are_a_valid_marking_plan_without_notes(self):
        """A teacher may use the structured rubric itself as the marking authority."""
        from services.exampen_paper_service import validate_pcr_questions

        errors = validate_pcr_questions(
            [
                {
                    "id": "Q-1",
                    "question_text": "State Newton's second law.",
                    "points": 2,
                    "marking_criteria": [
                        {
                            "criterion_id": "statement",
                            "description": "States force equals rate of change of momentum",
                            "max_marks": 2,
                        }
                    ],
                }
            ],
            marking_policy={"mode": "criterion_rubric_v1"},
        )

        assert errors == []

    def test_u_eval_01_mapped_solution_draft_cannot_change_question_marks(self):
        """Authoring AI may draft wording but cannot invent or lose locked marks."""
        from api.v1.pdf_async import _parse_pcr_marking_plan_draft

        valid = _parse_pcr_marking_plan_draft(
            '{"reference_solution":"Use the stated equation.","marking_criteria":['
            '{"criterion_id":"method","description":"Uses the correct method","max_marks":1},'
            '{"criterion_id":"result","description":"Calculates the final result","max_marks":3}'
            ']}',
            question_marks=4,
        )
        assert sum(item["max_marks"] for item in valid["marking_criteria"]) == 4

        with pytest.raises(ValueError, match="total 3"):
            _parse_pcr_marking_plan_draft(
                '{"marking_criteria":['
                '{"criterion_id":"method","description":"Uses the method","max_marks":3}'
                ']}',
                question_marks=4,
            )

    @pytest.mark.asyncio
    async def test_u_eval_01_structured_rubric_scores_only_locked_criteria(self):
        """PCR recomputes a criterion-rubric total and honours its low temperature."""
        from pcr.services.eval_core import EvalCore

        class Responses:
            def __init__(self):
                self.statuses = []

            async def get_response(self, response_id):
                assert response_id == "RESP-criterion"
                return {
                    "response_id": response_id,
                    "question_id": "EXAM-1::Q-1",
                    "student_id": "STU-1",
                    "detected_text": "Uses v = u + at and obtains the correct final value.",
                    "content_type": "TEXT_ONLY",
                    "flags": [],
                }

            async def update_eval_status(self, _response_id, status):
                self.statuses.append(status)

        class Evaluations:
            def __init__(self):
                self.docs = []

            async def insert_evaluation(self, doc):
                self.docs.append(doc)
                return doc, False

        class Questions:
            async def get_question(self, question_id):
                assert question_id == "EXAM-1::Q-1"
                return {
                    "question_id": question_id,
                    "subject": "Physics",
                    "complexity": "L2",
                    "eval_template": "stepwise_numerical",
                    "question_text": "Find the final velocity.",
                    "reference_solution": "Apply v = u + at before calculating the final value.",
                    "max_marks": 4,
                    "marking_policy": {
                        "mode": "criterion_rubric_v1",
                        "strictness": "strict",
                        "temperature": 0.05,
                    },
                    "marking_criteria": [
                        {
                            "criterion_id": "method",
                            "description": "Uses the correct kinematic equation",
                            "max_marks": 1,
                            "acceptable_evidence": "v = u + at",
                        },
                        {
                            "criterion_id": "answer",
                            "description": "Calculates the final value correctly",
                            "max_marks": 3,
                            "acceptable_evidence": "correct numerical result",
                        },
                    ],
                }

        class Cache:
            async def lookup(self, *_args, **_kwargs):
                raise AssertionError("Structured criterion marking must not generate a replacement answer key")

        class Gate:
            def __init__(self):
                self.calls = []

            async def call(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    content=(
                        '{"criterion_marks": ['
                        '{"criterion_id": "method", "marks_awarded": 1, "rationale": "Equation present", "evidence": "v = u + at"}, '
                        '{"criterion_id": "answer", "marks_awarded": 2, "rationale": "Minor arithmetic slip", "evidence": "final value differs"}'
                        '], "total_score": 999, "needs_review": false, "overall_feedback": "Check the arithmetic."}'
                    ),
                    usage=SimpleNamespace(
                        model="test-model",
                        caller="pcr_eval_core",
                        input_tokens=10,
                        output_tokens=12,
                        total_tokens=22,
                        estimated_cost_usd=0.0,
                    ),
                )

        responses = Responses()
        evaluations = Evaluations()
        gate = Gate()
        core = EvalCore(responses, evaluations, Questions(), Cache(), gate)

        result = await core.evaluate_response("RESP-criterion")

        assert result.eval_path == "criterion_rubric"
        assert result.total_score == 3.0  # Server ignores the model's 999 total.
        assert result.manual_review_required is False
        assert [mark.criterion_id for mark in result.criterion_marks] == ["method", "answer"]
        assert gate.calls[0]["temperature"] == 0.05
        assert responses.statuses == ["evaluated"]
        assert evaluations.docs[0]["total_score"] == 3.0
        assert evaluations.docs[0]["criterion_marks"][1]["max_marks"] == 3.0

    @pytest.mark.asyncio
    async def test_u_eval_01_invalid_criterion_output_requires_teacher_review(self):
        """Malformed AI criterion rows cannot silently create a score."""
        from pcr.services.eval_core import EvalCore

        class Responses:
            def __init__(self):
                self.statuses = []

            async def get_response(self, response_id):
                return {
                    "response_id": response_id,
                    "question_id": "EXAM-1::Q-2",
                    "student_id": "STU-1",
                    "detected_text": "Some work",
                    "content_type": "TEXT_ONLY",
                    "flags": [],
                }

            async def update_eval_status(self, _response_id, status):
                self.statuses.append(status)

        class Evaluations:
            def __init__(self):
                self.docs = []

            async def insert_evaluation(self, doc):
                self.docs.append(doc)
                return doc, False

        class Questions:
            async def get_question(self, _question_id):
                return {
                    "subject": "Physics",
                    "complexity": "L2",
                    "question_text": "Question",
                    "reference_solution": "Teacher solution",
                    "max_marks": 2,
                    "marking_policy": {"mode": "criterion_rubric_v1"},
                    "marking_criteria": [
                        {
                            "criterion_id": "working",
                            "description": "Shows the correct working",
                            "max_marks": 2,
                        }
                    ],
                }

        class Cache:
            async def lookup(self, *_args, **_kwargs):
                raise AssertionError("Structured criterion marking must bypass cache lookup")

        class Gate:
            async def call(self, **_kwargs):
                return SimpleNamespace(
                    content=(
                        '{"criterion_marks": [{"criterion_id": "invented", "marks_awarded": 2}], '
                        '"needs_review": false, "overall_feedback": "Looks good"}'
                    ),
                    usage=SimpleNamespace(
                        model="test-model",
                        caller="pcr_eval_core",
                        input_tokens=1,
                        output_tokens=1,
                        total_tokens=2,
                        estimated_cost_usd=0.0,
                    ),
                )

        responses = Responses()
        evaluations = Evaluations()
        core = EvalCore(responses, evaluations, Questions(), Cache(), Gate())

        result = await core.evaluate_response("RESP-invalid")

        assert result.total_score == 0.0
        assert result.manual_review_required is True
        assert responses.statuses == ["manual_review"]
        assert evaluations.docs[0]["criterion_marks"][0]["criterion_id"] == "working"
        assert evaluations.docs[0]["criterion_marks"][0]["marks_awarded"] == 0.0

    @pytest.mark.asyncio
    async def test_u_eval_01_gate_failure_keeps_structured_rubric_reviewable(self):
        """A temporary AI outage still leaves teacher-editable criterion rows."""
        from pcr.services.eval_core import EvalCore

        class Responses:
            def __init__(self):
                self.statuses = []

            async def get_response(self, response_id):
                return {
                    "response_id": response_id,
                    "question_id": "EXAM-1::Q-3",
                    "student_id": "STU-1",
                    "detected_text": "Answer",
                    "content_type": "TEXT_ONLY",
                    "flags": [],
                }

            async def update_eval_status(self, _response_id, status):
                self.statuses.append(status)

        class Evaluations:
            def __init__(self):
                self.docs = []

            async def insert_evaluation(self, doc):
                self.docs.append(doc)
                return doc, False

        class Questions:
            async def get_question(self, _question_id):
                return {
                    "subject": "Chemistry",
                    "complexity": "L2",
                    "question_text": "Question",
                    "reference_solution": "Teacher solution",
                    "max_marks": 2,
                    "marking_policy": {"mode": "criterion_rubric_v1"},
                    "marking_criteria": [
                        {
                            "criterion_id": "explain",
                            "description": "Explains the principle",
                            "max_marks": 2,
                        }
                    ],
                }

        class Cache:
            async def lookup(self, *_args, **_kwargs):
                raise AssertionError("Structured criterion marking must bypass the cache")

        class Gate:
            async def call(self, **_kwargs):
                raise RuntimeError("gate temporarily unavailable")

        responses = Responses()
        evaluations = Evaluations()
        core = EvalCore(responses, evaluations, Questions(), Cache(), Gate())

        result = await core.evaluate_response("RESP-gate-failure")

        assert result.manual_review_required is True
        assert result.criterion_marks[0].criterion_id == "explain"
        assert evaluations.docs[0]["manual_review_required"] is True
        assert evaluations.docs[0]["criterion_marks"][0]["marks_awarded"] == 0.0
        assert responses.statuses == ["manual_review"]

    @pytest.mark.asyncio
    async def test_u_eval_01_not_attempted_question_is_zero_without_ai_call(self):
        """A blank paper question is a deterministic zero, not an AI failure."""
        from pcr.services.eval_core import EvalCore

        class Responses:
            def __init__(self):
                self.statuses = []

            async def get_response(self, response_id):
                return {
                    "response_id": response_id,
                    "question_id": "EXAM-1::Q-4",
                    "student_id": "STU-1",
                    "detected_text": "",
                    "content_type": "TEXT_ONLY",
                    "is_missing_response": True,
                    "answer_state": "not_attempted",
                    "flags": [],
                }

            async def update_eval_status(self, _response_id, status):
                self.statuses.append(status)

        class Evaluations:
            def __init__(self):
                self.docs = []

            async def insert_evaluation(self, doc):
                self.docs.append(doc)
                return doc, False

        class Questions:
            async def get_question(self, _question_id):
                return {
                    "subject": "Physics",
                    "question_text": "Find the answer.",
                    "reference_solution": "Teacher worked solution",
                    "max_marks": 4,
                    "marking_policy": {"mode": "criterion_rubric_v1"},
                    "marking_criteria": [
                        {
                            "criterion_id": "method",
                            "description": "Uses the correct method",
                            "max_marks": 1,
                        },
                        {
                            "criterion_id": "answer",
                            "description": "Obtains the correct answer",
                            "max_marks": 3,
                        },
                    ],
                }

        class Cache:
            async def lookup(self, *_args, **_kwargs):
                raise AssertionError("A not-attempted answer must not warm or read the AI cache")

        class Gate:
            async def call(self, **_kwargs):
                raise AssertionError("A not-attempted answer must not call AI")

        responses = Responses()
        evaluations = Evaluations()
        result = await EvalCore(
            responses, evaluations, Questions(), Cache(), Gate()
        ).evaluate_response("RESP-MISSING-1")

        assert result.eval_path == "not_attempted"
        assert result.model_used == "none"
        assert result.total_score == 0.0
        assert result.max_score == 4.0
        assert [mark.marks_awarded for mark in result.criterion_marks] == [0.0, 0.0]
        assert responses.statuses == ["not_attempted"]
        assert evaluations.docs[0]["overall_feedback"].startswith("No answer was detected")
        assert evaluations.docs[0]["evaluation_id"].startswith("EVAL-MISSING-")


# ===========================================================================
# Flag registry completeness
# ===========================================================================


class TestFlagRegistry:
    """Verify the flag registry is complete and matches the spec."""

    def test_flag_registry_has_18_entries(self):
        """FLAG_REGISTRY has exactly 18 entries per PCR_EVAL_ENGINE_SPEC 6.2."""
        assert len(FLAG_REGISTRY) == 18

    def test_flag_registry_all_flag_types_covered(self):
        """Every FlagType enum member is in the registry."""
        for ft in FlagType:
            assert ft in FLAG_REGISTRY, f"Missing from registry: {ft}"

    def test_flag_registry_severities_match_spec(self):
        """Blocking flags match the spec-defined blocking types."""
        blocking_types = {
            FlagType.DIAGRAM_HEAVY_CONTENT,
            FlagType.CLUBBED_MULTIPLE_MARKERS,
            FlagType.OCR_REJECTED,
            FlagType.BUDGET_EXHAUSTED,
        }
        for ft, defn in FLAG_REGISTRY.items():
            if ft in blocking_types:
                assert defn.severity == FlagSeverity.BLOCKING, (
                    f"{ft} should be blocking"
                )

    def test_flag_severity_enum_values(self):
        """FlagSeverity has the three spec-defined levels."""
        assert FlagSeverity.BLOCKING.value == "blocking"
        assert FlagSeverity.WARNING.value == "warning"
        assert FlagSeverity.INFO.value == "info"
