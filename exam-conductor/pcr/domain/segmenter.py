"""
PCR Segmenter — Orchestrate boundary detection + marker parsing +
cross-page stitching.

Spec authority: new-docs/architecture/PCR_EVAL_ENGINE_SPEC.md sections 4.1-4.5
Failure mode:   PCR-01 (boundary/marker detection failure -> flags + review)
Test IDs:       U-SEG-01, U-SEG-02, U-SEG-03

Pipeline:
    boundary detect -> marker parse -> segment -> classify -> clubbed detect

Cross-page stitching rules (spec 4.3):
    - Page N ends without closing boundary: response continues on N+1
    - Page N+1 starts with boundary: previous response closed at prior page bottom
    - Page N+1 starts with content and no boundary: continuation
    - Page N+1 has marker without boundary: associate to prior response and flag
"""

from __future__ import annotations

import uuid
from typing import Sequence

from .boundary_detector import (
    DetectedBoundary,
    StrokeLine,
    detect_boundaries_camera,
    detect_boundaries_pen,
)
from .clubbed_detector import (
    ClubbedDetectionResult,
    detect_clubbed_responses,
)
from .content_classifier import classify_content
from .flag_registry import FLAG_REGISTRY
from .marker_parser import Q_MARKER_PATTERN, QMarker, parse_markers
from .response_models import (
    BoundingBox,
    ContentType,
    DetectedResponse,
    Flag,
    FlagSeverity,
    FlagType,
    PageOCR,
    SegmentationResult,
    SourcePageRef,
    TextBlock,
)


# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

LOW_SEGMENTATION_CONFIDENCE: float = 0.5
"""Below this, a LOW_SEGMENTATION_CONFIDENCE flag is raised."""

LOW_OCR_CONFIDENCE: float = 0.4
"""Below this, a LOW_OCR_CONFIDENCE flag is raised."""

OCR_REJECT_CONFIDENCE: float = 0.15
"""Below this, an OCR_REJECTED (blocking) flag is raised."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_response_id() -> str:
    return f"RESP-{uuid.uuid4().hex[:12]}"


def _generate_flag_id() -> str:
    return f"FLG-{uuid.uuid4().hex[:8]}"


def _make_flag(
    flag_type: FlagType,
    response_id: str | None = None,
    metadata: dict | None = None,
) -> Flag:
    defn = FLAG_REGISTRY[flag_type]
    return Flag(
        flag_id=_generate_flag_id(),
        response_id=response_id,
        source=defn.source,
        flag_type=defn.flag_type,
        severity=defn.severity,
        reason=defn.description,
        suggested_action=defn.suggested_action,
        metadata=metadata or {},
    )


def _blocks_in_range(
    blocks: list[TextBlock],
    y_start: float,
    y_end: float,
) -> list[TextBlock]:
    """Filter text blocks whose vertical midpoint falls within [y_start, y_end]."""
    result: list[TextBlock] = []
    for b in blocks:
        mid_y = (b.bbox.y_min + b.bbox.y_max) / 2.0
        if y_start <= mid_y <= y_end:
            result.append(b)
    return result


def _mean_confidence(blocks: list[TextBlock]) -> float:
    if not blocks:
        return 0.0
    return sum(b.confidence for b in blocks) / len(blocks)


def _word_count(text: str) -> int:
    return len(text.split())


def _concat_text(blocks: list[TextBlock]) -> str:
    """Concatenate text from blocks sorted by vertical then horizontal position."""
    sorted_blocks = sorted(blocks, key=lambda b: (b.bbox.y_min, b.bbox.x_min))
    return " ".join(b.text.strip() for b in sorted_blocks if b.text.strip())


def _text_occupied_bbox(
    text_blocks: list[TextBlock],
    fallback: BoundingBox,
) -> BoundingBox:
    """Return the region actually represented by text-only OCR evidence.

    Camera/PDF OCR commonly emits one compact text block for a handwritten
    answer even though its segment is the whole A4 page.  Comparing that small
    block to the full page incorrectly labels ordinary written answers as
    ``DIAGRAM_HEAVY`` and blocks automatic checking.  The OCR payload has no
    non-text visual evidence at this stage, so classify the occupied text
    region instead.  A genuinely image-only page still has no text blocks and
    continues down the diagram/manual-review path.
    """
    if not text_blocks:
        return fallback

    x_min = max(fallback.x_min, min(block.bbox.x_min for block in text_blocks))
    y_min = max(fallback.y_min, min(block.bbox.y_min for block in text_blocks))
    x_max = min(fallback.x_max, max(block.bbox.x_max for block in text_blocks))
    y_max = min(fallback.y_max, max(block.bbox.y_max for block in text_blocks))
    if x_max <= x_min or y_max <= y_min:
        return fallback
    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _split_blocks_on_multiple_q_markers(pages: list[PageOCR]) -> list[PageOCR]:
    """Split OCR blocks that contain several Q markers into marker blocks.

    Vision OCR can legitimately return one text block for a whole rendered pen
    page.  The text is still canonical OCR output, but segmentation needs one
    block per marker-delimited response so the PCR path can evaluate each
    answer independently instead of treating the page as a clubbed answer.
    """
    normalized_pages: list[PageOCR] = []

    for page in pages:
        normalized_blocks: list[TextBlock] = []
        changed = False

        for block in page.text_blocks:
            matches = list(Q_MARKER_PATTERN.finditer(block.text))
            if len(matches) <= 1:
                normalized_blocks.append(block)
                continue

            changed = True
            segment_count = len(matches)
            block_height = max(block.bbox.height, 1.0)
            slice_height = block_height / segment_count

            for idx, match in enumerate(matches):
                next_start = (
                    matches[idx + 1].start()
                    if idx + 1 < segment_count
                    else len(block.text)
                )
                text = block.text[match.start():next_start].strip()
                if not text:
                    continue

                y_min = block.bbox.y_min + idx * slice_height
                y_max = (
                    block.bbox.y_min + (idx + 1) * slice_height
                    if idx + 1 < segment_count
                    else block.bbox.y_max
                )
                normalized_blocks.append(
                    TextBlock(
                        text=text,
                        bbox=BoundingBox(
                            x_min=block.bbox.x_min,
                            y_min=y_min,
                            x_max=block.bbox.x_max,
                            y_max=max(y_max, y_min + 0.1),
                        ),
                        confidence=block.confidence,
                        source=block.source,
                    )
                )

        if changed:
            normalized_pages.append(
                page.model_copy(update={"text_blocks": normalized_blocks})
            )
        else:
            normalized_pages.append(page)

    return normalized_pages


def _markers_in_range(
    markers: list[QMarker],
    page_number: int,
    y_start: float,
    y_end: float,
) -> list[QMarker]:
    """Get markers on a given page within a Y range."""
    return [
        m
        for m in markers
        if m.page_number == page_number and y_start <= m.y_position <= y_end
    ]


def _build_marker_delimited_segments_for_page(
    page: PageOCR,
    page_markers: list[QMarker],
) -> list[_RawSegment]:
    """Build one segment per Q marker when no boundary lines are present."""
    segments: list[_RawSegment] = []
    sorted_markers = sorted(page_markers, key=lambda m: m.y_position)

    for idx, marker in enumerate(sorted_markers):
        y_start = 0.0 if idx == 0 else marker.y_position
        y_end = (
            sorted_markers[idx + 1].y_position
            if idx + 1 < len(sorted_markers)
            else page.page_height_mm
        )
        if y_end <= y_start:
            y_end = page.page_height_mm

        seg = _RawSegment(
            page.page_number,
            y_start,
            y_end,
            page.page_height_mm,
        )
        seg.text_blocks.extend(
            _blocks_in_range(page.text_blocks, y_start, y_end)
        )
        seg.markers.append(marker)
        seg.closed = True
        segments.append(seg)

    return segments


# ---------------------------------------------------------------------------
# Segment definition
# ---------------------------------------------------------------------------


class _RawSegment:
    """Internal mutable segment before conversion to DetectedResponse."""

    def __init__(
        self,
        page_number: int,
        y_start: float,
        y_end: float,
        page_height: float,
    ) -> None:
        self.page_spans: list[SourcePageRef] = [
            SourcePageRef(
                page_number=page_number,
                y_start=y_start,
                y_end=y_end,
            )
        ]
        self.markers: list[QMarker] = []
        self.text_blocks: list[TextBlock] = []
        self.closed: bool = False

    @property
    def first_page(self) -> int:
        return self.page_spans[0].page_number

    @property
    def last_page(self) -> int:
        return self.page_spans[-1].page_number

    def extend_to_page(
        self,
        page_number: int,
        y_start: float,
        y_end: float,
    ) -> None:
        self.page_spans.append(
            SourcePageRef(
                page_number=page_number,
                y_start=y_start,
                y_end=y_end,
            )
        )


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------


def _build_segments(
    pages: list[PageOCR],
    boundaries: list[DetectedBoundary],
    markers: list[QMarker],
) -> tuple[list[_RawSegment], list[Flag]]:
    """Build raw segments from pages, boundaries, and markers.

    Implements cross-page stitching rules from spec 4.3.
    """
    global_flags: list[Flag] = []
    segments: list[_RawSegment] = []
    current_segment: _RawSegment | None = None

    sorted_pages = sorted(pages, key=lambda p: p.page_number)

    # Index boundaries by page
    boundaries_by_page: dict[int, list[DetectedBoundary]] = {}
    for b in boundaries:
        boundaries_by_page.setdefault(b.page_number, []).append(b)
    for page_num in boundaries_by_page:
        boundaries_by_page[page_num].sort(key=lambda b: b.y_top)

    for page in sorted_pages:
        pn = page.page_number
        page_boundaries = boundaries_by_page.get(pn, [])
        page_markers = [m for m in markers if m.page_number == pn]

        # If no boundaries on this page
        if not page_boundaries:
            if current_segment is None:
                if len(page_markers) > 1:
                    segments.extend(
                        _build_marker_delimited_segments_for_page(
                            page, page_markers
                        )
                    )
                    continue

                # Start of a new segment at page top
                current_segment = _RawSegment(
                    pn, 0.0, page.page_height_mm, page.page_height_mm
                )
                # Gather blocks and markers for entire page
                current_segment.text_blocks.extend(page.text_blocks)
                current_segment.markers.extend(page_markers)
            else:
                # Continuation (spec 4.3: "page N+1 starts with content and
                # no boundary -> continuation")
                current_segment.extend_to_page(
                    pn, 0.0, page.page_height_mm
                )
                current_segment.text_blocks.extend(page.text_blocks)

                # Check for marker without boundary -> flag
                if page_markers:
                    # Spec 4.3: "page N+1 has marker without boundary ->
                    # associate to prior response and flag"
                    current_segment.markers.extend(page_markers)
                    global_flags.append(
                        _make_flag(
                            FlagType.NO_BOUNDARY_DETECTED,
                            None,
                            {
                                "page_number": pn,
                                "detail": (
                                    "Marker found on continuation page "
                                    "without boundary"
                                ),
                            },
                        )
                    )
            continue

        # Process boundaries on this page
        prev_y = 0.0

        # If page starts with a boundary, close the previous segment
        first_boundary = page_boundaries[0]
        if first_boundary.y_top < 10.0:  # boundary near top of page
            # Spec 4.3: "page N+1 starts with boundary -> previous response
            # closed at prior page bottom"
            if current_segment is not None:
                current_segment.closed = True
                segments.append(current_segment)
                current_segment = None
            prev_y = first_boundary.y_bottom
            page_boundaries = page_boundaries[1:]

        for boundary in page_boundaries:
            # Region from prev_y to boundary top is a segment
            region_y_start = prev_y
            region_y_end = boundary.y_top

            if current_segment is not None and not current_segment.closed:
                # This boundary closes the current (continued) segment
                # Extend to cover the region up to the boundary
                current_segment.extend_to_page(
                    pn, 0.0, region_y_end
                )
                blocks = _blocks_in_range(
                    page.text_blocks, 0.0, region_y_end
                )
                current_segment.text_blocks.extend(blocks)
                markers_here = _markers_in_range(
                    page_markers, pn, 0.0, region_y_end
                )
                current_segment.markers.extend(markers_here)
                current_segment.closed = True
                segments.append(current_segment)
                current_segment = None
            else:
                # New segment from prev_y to boundary
                if region_y_end > region_y_start + 1.0:
                    seg = _RawSegment(
                        pn, region_y_start, region_y_end, page.page_height_mm
                    )
                    blocks = _blocks_in_range(
                        page.text_blocks, region_y_start, region_y_end
                    )
                    seg.text_blocks.extend(blocks)
                    seg_markers = _markers_in_range(
                        page_markers, pn, region_y_start, region_y_end
                    )
                    seg.markers.extend(seg_markers)
                    seg.closed = True
                    segments.append(seg)

            prev_y = boundary.y_bottom

        # Region after last boundary to page bottom
        if prev_y < page.page_height_mm - 1.0:
            seg = _RawSegment(
                pn, prev_y, page.page_height_mm, page.page_height_mm
            )
            blocks = _blocks_in_range(
                page.text_blocks, prev_y, page.page_height_mm
            )
            seg.text_blocks.extend(blocks)
            seg_markers = _markers_in_range(
                page_markers, pn, prev_y, page.page_height_mm
            )
            seg.markers.extend(seg_markers)
            # This segment is open — may continue on next page
            current_segment = seg

    # Close any remaining open segment
    if current_segment is not None:
        current_segment.closed = True
        segments.append(current_segment)

    return segments, global_flags


def _segment_to_response(
    seg: _RawSegment,
    page_widths: dict[int, float],
) -> tuple[DetectedResponse, list[Flag]]:
    """Convert a raw segment into a DetectedResponse with classification
    and flags."""
    flags: list[Flag] = []
    response_id = _generate_response_id()

    # Determine question association from markers
    question_number: int | None = None
    sub_part: str | None = None
    if seg.markers:
        # Use the first marker for association
        primary = seg.markers[0]
        question_number = primary.question_number
        sub_part = primary.sub_part

    # Compute bounding box for the response across all page spans
    # For classification, use the first page span as reference
    first_span = seg.page_spans[0]
    page_width = page_widths.get(first_span.page_number, 210.0)  # A4 default
    response_bbox = BoundingBox(
        x_min=0.0,
        y_min=first_span.y_start,
        x_max=page_width,
        y_max=first_span.y_end,
    )

    # Text and OCR confidence
    text = _concat_text(seg.text_blocks)
    ocr_conf = _mean_confidence(seg.text_blocks)
    wc = _word_count(text)

    # OCR confidence flags
    if ocr_conf < OCR_REJECT_CONFIDENCE and seg.text_blocks:
        flags.append(
            _make_flag(
                FlagType.OCR_REJECTED,
                response_id,
                {"ocr_confidence": round(ocr_conf, 4)},
            )
        )
    elif ocr_conf < LOW_OCR_CONFIDENCE and seg.text_blocks:
        flags.append(
            _make_flag(
                FlagType.LOW_OCR_CONFIDENCE,
                response_id,
                {"ocr_confidence": round(ocr_conf, 4)},
            )
        )

    # Content classification.  OCR currently carries text blocks but no
    # reliable non-text/diagram geometry.  Use the text-occupied region so a
    # normal compact answer on a large page is not falsely blocked as a
    # diagram.  Empty OCR stays diagram-heavy and requires review.
    classification_bbox = _text_occupied_bbox(seg.text_blocks, response_bbox)
    content_type, text_coverage, class_flags = classify_content(
        seg.text_blocks,
        classification_bbox,
        response_id,
    )
    flags.extend(class_flags)

    # Segmentation confidence heuristic
    has_marker = len(seg.markers) > 0
    has_boundary = seg.closed
    if has_marker and has_boundary:
        seg_confidence = 0.95
    elif has_marker or has_boundary:
        seg_confidence = 0.70
    else:
        seg_confidence = 0.40

    if seg_confidence < LOW_SEGMENTATION_CONFIDENCE:
        flags.append(
            _make_flag(
                FlagType.LOW_SEGMENTATION_CONFIDENCE,
                response_id,
                {"segmentation_confidence": seg_confidence},
            )
        )

    # Marker-related flags
    if not has_marker:
        if has_boundary:
            flags.append(
                _make_flag(FlagType.BOUNDARY_ONLY_NO_MARKER, response_id)
            )
        else:
            flags.append(
                _make_flag(FlagType.NO_QUESTION_MARKER, response_id)
            )

    is_continuation = len(seg.page_spans) > 1

    response = DetectedResponse(
        response_id=response_id,
        question_number=question_number,
        sub_part=sub_part,
        detected_text=text,
        source_pages=seg.page_spans,
        content_type=content_type,
        text_coverage_ratio=text_coverage,
        segmentation_confidence=seg_confidence,
        ocr_confidence=ocr_conf,
        flags=flags,
        word_count=wc,
        is_continuation=is_continuation,
    )

    return response, flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment_submission(
    pages: list[PageOCR],
    stroke_lines: list[StrokeLine] | None = None,
    hough_lines: list[StrokeLine] | None = None,
    manifest_question_numbers: set[int] | None = None,
    expected_max_words_by_question: dict[int, int | None] | None = None,
) -> SegmentationResult:
    """Run the full segmentation pipeline on a submission.

    Pipeline:
        boundary detect -> marker parse -> segment -> classify -> clubbed detect

    Args:
        pages: All PageOCR objects for the submission, in page order.
        stroke_lines: Pre-extracted candidate lines from pen strokes.
            Supply for pen-originated submissions.
        hough_lines: Lines from HoughLinesP in mm-space.
            Supply for camera-originated submissions.
        manifest_question_numbers: Expected question numbers from the exam
            manifest.  Used by clubbed H3 heuristic.
        expected_max_words_by_question: Upper-bound word counts per question
            number.  Used by clubbed H2 heuristic.

    Returns:
        SegmentationResult with responses, flags, and summary stats.
    """
    if not pages:
        return SegmentationResult(
            responses=[],
            flags=[],
            page_count=0,
            total_boundaries_detected=0,
            total_markers_detected=0,
            has_blocking_flags=False,
        )

    sorted_pages = sorted(
        _split_blocks_on_multiple_q_markers(pages),
        key=lambda p: p.page_number,
    )
    page_widths = {p.page_number: p.page_width_mm for p in sorted_pages}

    # Step 1: Boundary detection
    is_pen = any(p.source == "pen" for p in sorted_pages)
    if is_pen and stroke_lines:
        boundaries = detect_boundaries_pen(sorted_pages, stroke_lines)
    elif hough_lines:
        boundaries = detect_boundaries_camera(sorted_pages, hough_lines)
    else:
        boundaries = []

    # Step 2: Marker parsing
    markers = parse_markers(sorted_pages)

    # Step 3: Segment + cross-page stitch
    raw_segments, global_flags = _build_segments(
        sorted_pages, boundaries, markers
    )

    # Step 4: Convert to DetectedResponses (includes classification)
    responses: list[DetectedResponse] = []
    all_response_flags: list[Flag] = []
    markers_by_response: dict[str, list[QMarker]] = {}

    for seg in raw_segments:
        # Skip empty segments (no text blocks)
        if not seg.text_blocks:
            continue
        response, resp_flags = _segment_to_response(seg, page_widths)
        responses.append(response)
        all_response_flags.extend(resp_flags)
        markers_by_response[response.response_id] = seg.markers

    # Step 5: Clubbed detection (H1 + H2 + H3; H4 excluded per C4)
    clubbed_result: ClubbedDetectionResult = detect_clubbed_responses(
        responses,
        markers_by_response,
        expected_max_words_by_question=expected_max_words_by_question,
        manifest_question_numbers=manifest_question_numbers,
    )

    # Attach clubbed flags to the relevant responses
    for flag in clubbed_result.flags:
        if flag.response_id:
            for resp in responses:
                if resp.response_id == flag.response_id:
                    resp.flags.append(flag)
                    break
        else:
            global_flags.append(flag)

    # Compute summary
    all_flags = global_flags + [
        f for resp in responses for f in resp.flags
    ]
    has_blocking = any(
        f.severity == FlagSeverity.BLOCKING for f in all_flags
    )

    return SegmentationResult(
        responses=responses,
        flags=global_flags,
        page_count=len(sorted_pages),
        total_boundaries_detected=len(boundaries),
        total_markers_detected=len(markers),
        has_blocking_flags=has_blocking,
    )
