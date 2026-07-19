"""Document-level answer mapping for conducted PCR answer copies.

The page segmenter is deliberately conservative: it can only trust explicit
question markers and physical boundaries.  A real handwritten copy frequently
contains neither.  This service handles that separate problem before marking:
it looks at the *complete submitted document*, identifies the answer regions
that actually belong to each immutable paper question, and returns only the
associations it can support with evidence.

It never awards marks.  If a region/question association is not reliable, the
region is retained as an unassigned response for teacher review and the caller
must not manufacture ``not attempted`` zero rows for the rest of the paper.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

from ..domain.response_models import (
    ContentType,
    DetectedResponse,
    Flag,
    FlagSeverity,
    FlagType,
    PageOCR,
    SourcePageRef,
    TextBlock,
)
from .ocr_service import (
    VisionGateProtocol,
    _detect_media_type,
    _get_ocr_vision_model,
    _resolve_image_base64,
)

logger = logging.getLogger(__name__)

_CALLER_ID = "pcr_eval_core"
_PROMPT_VERSION = "exampen-document-answer-map-v1"
_LAYOUT_PROMPT_VERSION = "exampen-same-paper-page-order-v2"
_ACCEPT_CONFIDENCE = 0.82
_WARN_CONFIDENCE = 0.90
_BLANK_ACCEPT_CONFIDENCE = 0.90
_MAX_QUESTION_TEXT_CHARS = 650
_MAX_REFERENCE_TEXT_CHARS = 450
_MAX_RUBRIC_TEXT_CHARS = 450
_MAX_CRITERIA_PER_QUESTION = 8
_MAX_CRITERION_TEXT_CHARS = 300
_MAX_OCR_TEXT_CHARS_PER_PAGE = 7500


@dataclass
class DocumentAnswerMappingResult:
    """The evidence-backed output of document-level answer association."""

    responses: List[DetectedResponse] = field(default_factory=list)
    assignment_details_by_response: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )
    coverage_is_reliable: bool = False
    manual_review_required: bool = False
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _AnswerCandidate:
    """A validated association proposed by the document vision model.

    A response can legitimately span several pages.  Keeping the normalized
    regions with the original model payload lets us merge a proven continuation
    before we create any scoreable response row.
    """

    question_number: int
    confidence: float
    regions: List[SourcePageRef]
    answer: Dict[str, Any]


@dataclass(frozen=True)
class _QuestionLayoutAnchor:
    """One immutable question region from the finalized paper snapshot."""

    question_number: int
    page_number: int
    x_percent: float
    y_percent: float
    width_percent: float
    height_percent: float
    question: Dict[str, Any]


@dataclass(frozen=True)
class _SamePaperLayoutMatch:
    """Deterministic proof that an upload preserves the finalized paper layout."""

    anchors: tuple[_QuestionLayoutAnchor, ...]
    confidence: float
    matched_question_numbers: tuple[int, ...]
    matched_page_numbers: tuple[int, ...]


@runtime_checkable
class DocumentAnswerMapperProtocol(Protocol):
    """Contract used by :class:`SubmissionService` and test doubles."""

    async def map_submission(
        self,
        *,
        pages: List[PageOCR],
        answer_pages: List[Dict[str, Any]],
        numbered_questions: List[tuple[int, Dict[str, Any]]],
        source: str,
    ) -> DocumentAnswerMappingResult:
        ...  # pragma: no cover


def has_reliable_marker_coverage(
    segmented_responses: Iterable[DetectedResponse],
    numbered_questions: List[tuple[int, Dict[str, Any]]],
) -> bool:
    """Return True when marker-based segments already own distinct paper Qs.

    Content-section mapping trusts numbered answer labels when they line up
    with the paper.  PCR must do the same: once the segmenter has split a copy
    into unique, non-empty answers with question numbers, full-document vision
    remapping must not replace those associations (that was overwriting correct
    ``1)``/``2)`` splits with header chrome or wrong Q ownership).
    """
    responses = list(segmented_responses)
    if len(responses) < 2 or len(numbered_questions) < 2:
        return False

    valid_numbers = {int(number) for number, _question in numbered_questions}
    marked: List[DetectedResponse] = []
    for response in responses:
        number = getattr(response, "question_number", None)
        text = str(getattr(response, "detected_text", "") or "").strip()
        if number is None or int(number) not in valid_numbers:
            continue
        if not text:
            continue
        # Reject pure form-header detections so "Answer Book Date Page" never
        # counts as a reliable Q1 mapping.
        from ..domain.marker_parser import is_form_header_text, strip_form_header_noise

        cleaned = strip_form_header_noise(text)
        if not cleaned or is_form_header_text(cleaned):
            continue
        marked.append(response)

    if len(marked) < 2:
        return False

    numbers = [int(response.question_number) for response in marked]  # type: ignore[arg-type]
    if len(numbers) != len(set(numbers)):
        return False

    # Leftover unmarked handwriting still needs document association.
    from ..domain.marker_parser import strip_form_header_noise as _strip_headers

    unmarked_body = [
        response
        for response in responses
        if response.question_number is None
        and _strip_headers(str(response.detected_text or ""))
    ]
    if unmarked_body:
        return False

    paper_size = len(valid_numbers)
    # Only a complete set of explicit labels proves coverage.  A partial set
    # cannot prove that later, unlabelled handwriting was not appended to the
    # final labelled response by OCR.  Full-document mapping must account for
    # the remaining page evidence before absent questions can become zeros.
    return len(numbers) == paper_size and not _responses_reuse_evidence(marked)


def _responses_reuse_evidence(responses: Iterable[DetectedResponse]) -> bool:
    """Detect duplicate text/region ownership across proposed answers."""

    seen_text: set[str] = set()
    accepted_regions: Dict[int, List[tuple[float, float, int]]] = {}
    for index, response in enumerate(responses):
        text_key = _normalized_evidence_text(response.detected_text)
        if len(text_key) >= 24:
            if text_key in seen_text:
                return True
            seen_text.add(text_key)
        if _materially_overlaps_other_assignment(
            list(response.source_pages), accepted_regions
        ):
            return True
        question_number = int(response.question_number or index + 1)
        for region in response.source_pages:
            accepted_regions.setdefault(region.page_number, []).append(
                (region.y_start, region.y_end, question_number)
            )
    return False


def needs_document_answer_mapping(
    *,
    pages: List[PageOCR],
    segmented_responses: Iterable[DetectedResponse],
    numbered_questions: List[tuple[int, Dict[str, Any]]],
    source: str = "camera",
) -> bool:
    """Return whether a copy needs document-level answer ownership mapping.

    A multi-page uploaded camera/PDF copy is one document, not a sequence of
    independently scoreable OCR blobs.  The page segmenter is still useful for
    boundary hints, but it cannot prove that every handwritten region has been
    assigned exactly once.  Therefore every multi-page camera copy for a
    meaningful paper takes the document-association path before any missing
    answer slots or marks are created — **unless** marker coverage is already
    reliable (content-section style numbered answers).

    Pen submissions are deliberately left on their existing stroke-aware path
    until they expose the same canonical page-image evidence.  A normal short,
    one-page uploaded answer remains on the deterministic matcher unless it
    shows signs that several answers were collapsed together.
    """
    responses = list(segmented_responses)
    if len(numbered_questions) < 2 or not pages:
        return False

    # Strong numbered / Q-marker splits already mirror the content-section
    # answer mapper.  Do not throw them away for a second vision pass that has
    # been observed to reattach the form header to Q1.
    if has_reliable_marker_coverage(responses, numbered_questions):
        return False

    # Student-uploaded PDF/image copies have canonical private image pages, so
    # their full document can be mapped safely.  Do this even if a few OCR
    # markers were found: otherwise an unmarked third answer can still be
    # silently lost and turned into a false zero.
    if source == "camera":
        # The document vision mapper works from the private source images, not
        # only from OCR text.  A handwriting OCR miss is therefore not proof
        # that the page is blank.  Route a multi-page copy, or a copy for which
        # the page segmenter produced no response at all, through full-document
        # association before declaring it unreadable/not attempted.
        return True

    # A single huge response on a multi-question paper is also suspicious,
    # especially when the segmenter has already reported clubbing/low
    # segmentation confidence.  This captures one-page mixed answers without
    # treating every longer answer as a failure.
    if len(responses) == 1 and len(numbered_questions) >= 3:
        response = responses[0]
        has_segment_warning = any(
            flag.flag_type
            in {
                FlagType.CLUBBED_LENGTH_ANOMALY,
                FlagType.CLUBBED_MULTIPLE_MARKERS,
                FlagType.CLUBBED_TOPIC_DISCONTINUITY,
                FlagType.LOW_SEGMENTATION_CONFIDENCE,
                FlagType.NO_BOUNDARY_DETECTED,
            }
            for flag in response.flags
        )
        if has_segment_warning or response.word_count >= 180:
            return True

    return False


class DocumentAnswerMapper:
    """Map free-form handwritten answer regions to paper question numbers.

    The mapper receives private page artefacts from ingest and resolves them
    only through the existing private S3 resolver.  It does not create local
    copies, public URLs, or a second storage system.
    """

    def __init__(self, gate: VisionGateProtocol) -> None:
        self._gate = gate

    async def map_submission(
        self,
        *,
        pages: List[PageOCR],
        answer_pages: List[Dict[str, Any]],
        numbered_questions: List[tuple[int, Dict[str, Any]]],
        source: str,
    ) -> DocumentAnswerMappingResult:
        if not numbered_questions:
            return _unsafe_result("No immutable paper questions were available")

        # A student may write directly on the same printed question paper that
        # was finalized for the exam.  In that format, the immutable source
        # page order is stronger ownership evidence than guessed document-wide
        # OCR/vision boundaries. Activate this path only after answer evidence
        # is found on the expected pages in monotonic question order. A separate
        # or jumbled answer book therefore continues through the flexible
        # document mapper below.
        layout_match = _detect_same_paper_layout(
            pages=pages,
            numbered_questions=numbered_questions,
        )
        if layout_match is not None:
            logger.info(
                "Answer copy matches finalized paper layout: %d/%d ownership anchors "
                "across %d page(s), confidence=%.3f",
                len(layout_match.matched_question_numbers),
                len(numbered_questions),
                len(layout_match.matched_page_numbers),
                layout_match.confidence,
            )
            return await self._map_same_paper_layout(
                pages=pages,
                answer_pages=answer_pages,
                layout_match=layout_match,
                source=source,
            )

        # Prefer content-section style numbered extraction before vision.
        try:
            from .answer_book_extractor import try_extract_answer_book_responses

            book = try_extract_answer_book_responses(pages, numbered_questions)
        except Exception:
            book = None
            logger.exception("Answer-book numbered extract failed during document mapping")
        if book is not None:
            book_responses, book_assignment = book
            if has_reliable_marker_coverage(book_responses, numbered_questions):
                mapped = {
                    int(r.question_number)
                    for r in book_responses
                    if r.question_number is not None
                }
                result = DocumentAnswerMappingResult(
                    responses=book_responses,
                    assignment_details_by_response=book_assignment,
                    coverage_is_reliable=True,
                    manual_review_required=False,
                    reason=None,
                    metadata={
                        "mapping_strategy": "answer_book_numbered_extract",
                        "mapped_question_numbers": sorted(mapped),
                        "coverage_proof": "all_paper_question_labels_detected",
                    },
                )
                logger.info(
                    "Document answer mapping used complete answer-book numbered extract: "
                    "%d response(s)",
                    len(book_responses),
                )
                return result
            logger.info(
                "Answer-book numbered extract was partial; requiring full-document "
                "coverage before creating absent-answer rows"
            )

        # Prefer content-section style deterministic numbered-block mapping
        # before paying for a multi-page vision remap.  Student answer books
        # almost always use ``1)`` / ``2.`` labels.
        deterministic = _deterministic_numbered_mapping(
            pages=pages,
            numbered_questions=numbered_questions,
        )
        if deterministic is not None and deterministic.responses:
            logger.info(
                "Document answer mapping used deterministic numbered anchors: "
                "%d response(s), reliable=%s",
                len(deterministic.responses),
                deterministic.coverage_is_reliable,
            )
            # When deterministic coverage is reliable, skip vision entirely.
            if deterministic.coverage_is_reliable:
                return deterministic

        messages, unresolved_page_numbers = await _build_document_messages(
            pages=pages,
            answer_pages=answer_pages,
            numbered_questions=numbered_questions,
        )
        if not messages:
            if deterministic is not None and deterministic.responses:
                return deterministic
            return _unsafe_result(
                "The answer copy could not be loaded for document-level mapping",
                metadata={"unreadable_pages": unresolved_page_numbers},
            )

        try:
            gate_response = await self._gate.call(
                model_id=_get_ocr_vision_model(),
                prompt="",
                caller_id=_CALLER_ID,
                messages=messages,
                max_output_tokens=6000,
                temperature=0.0,
                metadata={
                    "pcr_stage": "document_answer_mapping",
                    "mapping_prompt_version": _PROMPT_VERSION,
                    "page_count": len(pages),
                    "question_count": len(numbered_questions),
                    "source": source,
                },
            )
        except Exception as exc:
            logger.exception("Document answer mapping vision request failed")
            if deterministic is not None and deterministic.responses:
                return deterministic
            return _unsafe_result(
                "Document-level answer mapping was unavailable; teacher review is required",
                metadata={"error": str(exc)[:500]},
            )

        payload = _parse_mapping_payload(getattr(gate_response, "content", ""))
        if payload is None:
            if deterministic is not None and deterministic.responses:
                return deterministic
            return _unsafe_result(
                "The document mapper returned an invalid response; teacher review is required"
            )

        result = _build_mapping_result(
            payload=payload,
            pages=pages,
            numbered_questions=numbered_questions,
            unreadable_pages=unresolved_page_numbers,
        )
        # Prefer the mapping that owns more real student text without header junk.
        if deterministic is not None and deterministic.responses:
            result = _prefer_better_mapping(deterministic, result)
        logger.info(
            "Document answer mapping completed: %d mapped response(s), reliable=%s, review=%s",
            len(result.responses),
            result.coverage_is_reliable,
            result.manual_review_required,
        )
        return result

    async def _map_same_paper_layout(
        self,
        *,
        pages: List[PageOCR],
        answer_pages: List[Dict[str, Any]],
        layout_match: _SamePaperLayoutMatch,
        source: str,
    ) -> DocumentAnswerMappingResult:
        """Transcribe one proven paper page at a time in immutable Q order."""

        page_requests, unreadable_page_numbers = await _build_layout_page_requests(
            pages=pages,
            answer_pages=answer_pages,
            anchors=list(layout_match.anchors),
        )
        if unreadable_page_numbers or not page_requests:
            return _layout_mapping_failure(
                anchors=list(layout_match.anchors),
                pages=pages,
                reason=(
                    "One or more verified paper pages could not be loaded "
                    "from the submitted answer copy"
                ),
                unreadable_page_numbers=unreadable_page_numbers,
                layout_match_confidence=layout_match.confidence,
            )

        records: List[Dict[str, Any]] = []
        coverage_confidences: List[float] = []
        coverage_complete = True
        for request_index, request in enumerate(page_requests, start=1):
            messages = _build_layout_page_messages(request)
            try:
                gate_response = await self._gate.call(
                    model_id=_get_ocr_vision_model(),
                    prompt="",
                    caller_id=_CALLER_ID,
                    messages=messages,
                    max_output_tokens=min(
                        8000,
                        1400 + 900 * len(request[1]),
                    ),
                    temperature=0.0,
                    metadata={
                        "pcr_stage": "same_paper_layout_transcription",
                        "layout_prompt_version": _LAYOUT_PROMPT_VERSION,
                        "page_number": request[0].page_number,
                        "page_index": request_index,
                        "page_count": len(page_requests),
                        "question_count": len(request[1]),
                        "source": source,
                    },
                )
            except Exception as exc:
                logger.exception(
                    "Verified paper-page transcription failed for page %d",
                    request[0].page_number,
                )
                return _layout_mapping_failure(
                    anchors=list(layout_match.anchors),
                    pages=pages,
                    reason=(
                        "Verified paper pages could not be transcribed; "
                        "teacher review is required"
                    ),
                    layout_match_confidence=layout_match.confidence,
                    metadata={"error": str(exc)[:500]},
                )

            payload = _parse_mapping_payload(getattr(gate_response, "content", ""))
            if payload is None:
                return _layout_mapping_failure(
                    anchors=list(layout_match.anchors),
                    pages=pages,
                    reason=(
                        "Verified paper-page transcription returned an invalid "
                        "response; teacher review is required"
                    ),
                    layout_match_confidence=layout_match.confidence,
                )
            raw_records = payload.get("questions")
            if not isinstance(raw_records, list):
                raw_records = []
            records.extend(item for item in raw_records if isinstance(item, dict))
            coverage = payload.get("page_coverage")
            if not isinstance(coverage, dict):
                coverage_complete = False
                coverage_confidences.append(0.0)
            else:
                coverage_complete = coverage_complete and bool(coverage.get("complete"))
                coverage_confidences.append(_confidence(coverage.get("confidence")))

        return _build_layout_page_mapping_result(
            records=records,
            anchors=list(layout_match.anchors),
            pages=pages,
            coverage_complete=coverage_complete,
            coverage_confidence=(
                min(coverage_confidences) if coverage_confidences else 0.0
            ),
            layout_match_confidence=layout_match.confidence,
        )


_LAYOUT_PROMPT_STOP_WORDS = {
    "and",
    "answer",
    "calculate",
    "find",
    "for",
    "from",
    "given",
    "into",
    "question",
    "show",
    "that",
    "the",
    "then",
    "this",
    "use",
    "using",
    "what",
    "with",
    "work",
    "working",
}


def _detect_same_paper_layout(
    *,
    pages: List[PageOCR],
    numbered_questions: List[tuple[int, Dict[str, Any]]],
) -> Optional[_SamePaperLayoutMatch]:
    """Return verified layout evidence only when printed prompts align.

    Question count or page count alone is never enough.  A free-form answer
    book can have both.  We require several immutable question prompts (or
    their explicit number plus distinctive content) to be present in granular
    OCR blocks on the exact finalized page and inside the finalized region.
    """

    anchors = _validated_layout_anchors(pages, numbered_questions)
    if not anchors:
        return None

    pages_by_number = {page.page_number: page for page in pages}
    matched_questions: List[int] = []
    matched_positions_by_page: Dict[int, List[tuple[int, float]]] = {}
    for anchor in anchors:
        page = pages_by_number[anchor.page_number]
        matching_blocks = sorted(
            (
                block
                for block in page.text_blocks
                if _block_matches_layout_prompt(block, page, anchor)
            ),
            key=lambda block: (
                -_layout_block_match_score(block, anchor),
                block.bbox.y_min,
                block.bbox.x_min,
            ),
        )
        if matching_blocks:
            block = matching_blocks[0]
            matched_questions.append(anchor.question_number)
            matched_positions_by_page.setdefault(anchor.page_number, []).append(
                (
                    anchor.question_number,
                    (block.bbox.y_min + block.bbox.y_max) / 2.0,
                )
            )

    # The expected questions must occur top-to-bottom on their finalized page.
    # This rejects a jumbled answer book that happens to reuse the same page
    # count and values while still accepting normal handwriting-only OCR.
    matched_pages: set[int] = set()
    for page_number, positions in matched_positions_by_page.items():
        ordered = sorted(positions, key=lambda item: item[0])
        if all(
            right[1] + pages_by_number[page_number].page_height_mm * 0.02 >= left[1]
            for left, right in zip(ordered, ordered[1:])
        ):
            matched_pages.add(page_number)

    layout_pages = {anchor.page_number for anchor in anchors}
    required_questions = max(2, math.ceil(len(anchors) * 0.45))
    required_pages = max(1, math.ceil(len(layout_pages) * 0.75))
    if (
        len(matched_questions) < required_questions
        or len(matched_pages) < required_pages
    ):
        return None

    question_ratio = len(matched_questions) / len(anchors)
    page_ratio = len(matched_pages) / len(layout_pages)
    confidence = round(min(0.99, 0.78 + 0.14 * question_ratio + 0.07 * page_ratio), 4)
    return _SamePaperLayoutMatch(
        anchors=tuple(anchors),
        confidence=confidence,
        matched_question_numbers=tuple(sorted(matched_questions)),
        matched_page_numbers=tuple(sorted(matched_pages)),
    )


def _validated_layout_anchors(
    pages: List[PageOCR],
    numbered_questions: List[tuple[int, Dict[str, Any]]],
) -> List[_QuestionLayoutAnchor]:
    pages_by_number = {page.page_number: page for page in pages}
    anchors: List[_QuestionLayoutAnchor] = []
    for question_number, question in numbered_questions:
        layout = question.get("question_layout")
        if not isinstance(layout, dict):
            layout = {}
        bbox = question.get("source_bbox_percent") or layout.get("bbox_percent")
        if not isinstance(bbox, dict):
            return []
        try:
            page_number = int(
                question.get("source_page_number") or layout.get("page_number")
            )
            x = float(bbox.get("x"))
            y = float(bbox.get("y"))
            width = float(bbox.get("width"))
            height = float(bbox.get("height"))
        except (TypeError, ValueError):
            return []
        if page_number not in pages_by_number:
            return []
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > 100.001
            or y + height > 100.001
        ):
            return []
        anchors.append(
            _QuestionLayoutAnchor(
                question_number=int(question_number),
                page_number=page_number,
                x_percent=x,
                y_percent=y,
                width_percent=width,
                height_percent=height,
                question=question,
            )
        )

    if len(anchors) != len(numbered_questions) or not anchors:
        return []
    expected_pages = set(range(1, max(anchor.page_number for anchor in anchors) + 1))
    if set(pages_by_number) != expected_pages:
        return []

    # Finalization normally enforces this already.  Revalidate at consumption
    # so a malformed legacy record can never give two questions the same ink.
    for index, left in enumerate(anchors):
        for right in anchors[index + 1 :]:
            if left.page_number != right.page_number:
                continue
            overlap_width = max(
                0.0,
                min(left.x_percent + left.width_percent, right.x_percent + right.width_percent)
                - max(left.x_percent, right.x_percent),
            )
            overlap_height = max(
                0.0,
                min(left.y_percent + left.height_percent, right.y_percent + right.height_percent)
                - max(left.y_percent, right.y_percent),
            )
            smaller_area = min(
                left.width_percent * left.height_percent,
                right.width_percent * right.height_percent,
            )
            if smaller_area > 0 and overlap_width * overlap_height / smaller_area > 0.02:
                return []
    return sorted(anchors, key=lambda anchor: anchor.question_number)


def _layout_prompt_tokens(value: Any) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    normalized = str(value or "").casefold()
    normalized = re.sub(r"\blcm\b", "lowest common multiple", normalized)
    normalized = re.sub(r"\bhcf\b", "highest common factor", normalized)
    raw = re.findall(r"[a-zA-Z]{3,}|-?\d+(?:\.\d+)?", normalized)
    words = [
        token
        for token in raw
        if token[0].isalpha() and token not in _LAYOUT_PROMPT_STOP_WORDS
    ]
    numbers = {token.lstrip("-") for token in raw if not token[0].isalpha()}
    return set(words), numbers, set(zip(words, words[1:]))


def _block_matches_layout_prompt(
    block: TextBlock,
    page: PageOCR,
    anchor: _QuestionLayoutAnchor,
) -> bool:
    text = str(block.text or "").strip()
    if not text or block.bbox.height >= page.page_height_mm * 0.72:
        return False

    return _layout_block_match_score(block, anchor) > 0


def _layout_block_match_score(
    block: TextBlock,
    anchor: _QuestionLayoutAnchor,
) -> float:
    text = str(block.text or "").strip()
    if not text:
        return 0.0

    question_words, question_numbers, question_bigrams = _layout_prompt_tokens(
        " ".join(
            (
                _question_text(anchor.question),
                _question_reference_solution(anchor.question),
                _question_rubric(anchor.question),
            )
        )
    )
    block_words, block_numbers, block_bigrams = _layout_prompt_tokens(text)
    shared_words = question_words & block_words
    shared_numbers = question_numbers & block_numbers
    shared_bigrams = question_bigrams & block_bigrams
    word_coverage = len(shared_words) / max(1, len(question_words))
    lexical_match = (
        len(shared_words) >= 3
        or (
            len(shared_words) >= 2
            and (word_coverage >= 0.25 or bool(shared_bigrams))
        )
        or (
            len(shared_words) >= 1
            and len(shared_numbers) >= 1
        )
        or len(shared_numbers) >= 2
    )
    marker_match = bool(
        re.match(
            rf"^\s*(?:q(?:uestion)?\s*\.?\s*)?{anchor.question_number}\s*(?:[\)\]:]|\.(?=\s))",
            text,
            flags=re.IGNORECASE,
        )
    )
    if not lexical_match and not marker_match:
        return 0.0
    return (
        (100.0 if marker_match else 0.0)
        + 6.0 * len(shared_words)
        + 4.0 * len(shared_numbers)
        + 10.0 * len(shared_bigrams)
        + 5.0 * word_coverage
    )


async def _build_layout_page_requests(
    *,
    pages: List[PageOCR],
    answer_pages: List[Dict[str, Any]],
    anchors: List[_QuestionLayoutAnchor],
) -> tuple[
    List[tuple[PageOCR, List[_QuestionLayoutAnchor], str, str]],
    List[int],
]:
    artefacts = {
        int(page.get("page_number") or 0): page
        for page in answer_pages
        if int(page.get("page_number") or 0) > 0
    }
    anchors_by_page: Dict[int, List[_QuestionLayoutAnchor]] = {}
    for anchor in anchors:
        anchors_by_page.setdefault(anchor.page_number, []).append(anchor)

    requests: List[tuple[PageOCR, List[_QuestionLayoutAnchor], str, str]] = []
    unreadable: List[int] = []
    for page in sorted(pages, key=lambda item: item.page_number):
        page_anchors = sorted(
            anchors_by_page.get(page.page_number, []),
            key=lambda anchor: anchor.question_number,
        )
        if not page_anchors:
            continue
        artefact = artefacts.get(page.page_number, {})
        raw_image_ref = artefact.get("raw_image_ref")
        if not isinstance(raw_image_ref, str) or not raw_image_ref.strip():
            unreadable.append(page.page_number)
            continue
        expected_sha256 = artefact.get("asset_sha256")
        image_b64 = (
            await _resolve_image_base64(raw_image_ref, expected_sha256=expected_sha256)
            if expected_sha256
            else await _resolve_image_base64(raw_image_ref)
        )
        if not image_b64:
            unreadable.append(page.page_number)
            continue
        requests.append(
            (
                page,
                page_anchors,
                image_b64,
                _detect_media_type(raw_image_ref),
            )
        )
    return requests, sorted(set(unreadable))


def _build_layout_page_messages(
    request: tuple[PageOCR, List[_QuestionLayoutAnchor], str, str],
) -> List[Dict[str, Any]]:
    page, anchors, image_b64, media_type = request
    catalog = [
        {
            "question_number": anchor.question_number,
            "question": _question_text(anchor.question)[:_MAX_QUESTION_TEXT_CHARS],
        }
        for anchor in anchors
    ]
    prompt = (
        "This is one page from a student's submission whose page grouping and "
        "top-to-bottom question order have already been verified against the "
        "finalized printed paper. Segment and transcribe ONLY the student's marks "
        "for the listed questions on this page. Never grade and never move work to "
        "a different question or page.\n\n"
        "The catalog order is authoritative. Answers must occur top-to-bottom in "
        "that same order, but their physical bands are not pre-assumed because a "
        "scan may be shifted or stretched. Extract handwriting, filled blanks, "
        "calculations, diagrams, ticks, and corrections. Ignore printed question "
        "wording, printed diagrams, page furniture, and empty lines. Printed content "
        "alone means attempted=false. Preserve mathematical signs. Describe student "
        "additions to a diagram/table in transcribed_text.\n\n"
        "Return one item for EVERY catalog question. Regions are normalized page-Y "
        "coordinates 0..1000 and must contain only that student's work. Regions for "
        "different questions must not overlap and must follow catalog order. A blank "
        "question has attempted=false, transcribed_text='', and regions=[]. If work "
        "visibly carries a conflicting question label, set ownership_conflict=true.\n\n"
        "Return ONLY JSON in this exact shape:\n"
        "{\n"
        '  "page_coverage": {"complete": true, "confidence": 0.0},\n'
        '  "questions": [{"question_number": 1, "attempted": true, '
        '"confidence": 0.0, "regions": [{"page_number": 1, "y_start": 0, '
        '"y_end": 1000}], "transcribed_text": "student work only", '
        '"ownership_conflict": false, "reason": ""}]\n'
        "}\n\n"
        f"Page number: {page.page_number}\n"
        f"Ordered catalog: {json.dumps(catalog, ensure_ascii=False)}\n"
        f"OCR handwriting layout hints: {_page_layout_evidence(page)}"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_b64}",
                        "detail": "high",
                    },
                },
            ],
        }
    ]


def _build_layout_page_mapping_result(
    *,
    records: List[Dict[str, Any]],
    anchors: List[_QuestionLayoutAnchor],
    pages: List[PageOCR],
    coverage_complete: bool,
    coverage_confidence: float,
    layout_match_confidence: float,
) -> DocumentAnswerMappingResult:
    pages_by_number = {page.page_number: page for page in pages}
    records_by_number: Dict[int, List[Dict[str, Any]]] = {}
    invalid_record_count = 0
    for record in records:
        number = _as_int(record.get("question_number"))
        if number is None:
            invalid_record_count += 1
            continue
        records_by_number.setdefault(number, []).append(record)

    responses: List[DetectedResponse] = []
    assignment_details: Dict[str, Dict[str, Any]] = {}
    unresolved: List[Dict[str, Any]] = []
    processed_numbers: set[int] = set()
    verified_blank_numbers: set[int] = set()
    anchor_numbers = {anchor.question_number for anchor in anchors}
    accepted_regions_by_page: Dict[int, List[tuple[float, float, int]]] = {}
    last_region_start_by_page: Dict[int, float] = {}

    for anchor in anchors:
        number = anchor.question_number
        candidates = records_by_number.get(number, [])
        if len(candidates) != 1:
            unresolved.append(
                {
                    "reason": (
                        f"Q{number} page returned no transcription record"
                        if not candidates
                        else f"Q{number} page returned duplicate transcription records"
                    ),
                    "page_numbers": [anchor.page_number],
                }
            )
            continue

        record = candidates[0]
        attempted = record.get("attempted")
        confidence = _confidence(record.get("confidence"))
        text = str(record.get("transcribed_text") or "").strip()
        ownership_conflict = record.get("ownership_conflict") is True
        regions = _normalise_regions(record.get("regions"), pages_by_number)
        if any(region.page_number != anchor.page_number for region in regions):
            ownership_conflict = True
            record = {
                **record,
                "reason": f"Q{number} returned a region on the wrong source page",
            }
        if not isinstance(attempted, bool):
            unresolved.append(
                {
                    "reason": f"Q{number} did not return a valid attempted state",
                    "regions": record.get("regions") or [],
                    "transcribed_text": text,
                }
            )
            continue
        if ownership_conflict:
            unresolved.append(
                {
                    "reason": (
                        str(record.get("reason") or "").strip()
                        or f"Q{number} contains work whose ownership is ambiguous"
                    ),
                    "regions": record.get("regions") or [],
                    "transcribed_text": text,
                }
            )
            continue
        threshold = _ACCEPT_CONFIDENCE if attempted else _BLANK_ACCEPT_CONFIDENCE
        if confidence < threshold:
            unresolved.append(
                {
                    "reason": (
                        f"Q{number} confidence {confidence:.2f} is below the "
                        f"required {threshold:.2f}"
                    ),
                    "regions": record.get("regions") or [],
                    "transcribed_text": text,
                }
            )
            continue
        if not attempted:
            if _normalized_evidence_text(text) or regions:
                unresolved.append(
                    {
                        "reason": f"Q{number} was labelled blank but also returned student evidence",
                        "regions": record.get("regions") or [],
                        "transcribed_text": text,
                    }
                )
                continue
            processed_numbers.add(number)
            verified_blank_numbers.add(number)
            continue
        if not regions or not _normalized_evidence_text(text):
            unresolved.append(
                {
                    "reason": f"Q{number} was labelled attempted but lacks a usable region or transcription",
                    "regions": record.get("regions") or [],
                    "transcribed_text": text,
                }
            )
            continue
        if _materially_overlaps_other_assignment(regions, accepted_regions_by_page):
            unresolved.append(
                {
                    "reason": f"Q{number} materially overlaps another question on the same page",
                    "regions": record.get("regions") or [],
                    "transcribed_text": text,
                }
            )
            continue
        region_start = min(region.y_start for region in regions)
        previous_start = last_region_start_by_page.get(anchor.page_number)
        if previous_start is not None and region_start + 3.0 < previous_start:
            unresolved.append(
                {
                    "reason": f"Q{number} appears above an earlier question, violating verified page order",
                    "regions": record.get("regions") or [],
                    "transcribed_text": text,
                }
            )
            continue

        processed_numbers.add(number)
        last_region_start_by_page[anchor.page_number] = region_start
        for region in regions:
            accepted_regions_by_page.setdefault(region.page_number, []).append(
                (region.y_start, region.y_end, number)
            )
        answer = {
            "question_number": number,
            "confidence": confidence,
            "mapping_basis": "same_paper_layout",
            "transcribed_text": text,
        }
        response = _response_from_answer(
            answer,
            number,
            confidence,
            regions,
            pages_by_number,
            prefer_vision_transcription=True,
        )
        responses.append(response)
        assignment_details[str(response.response_id)] = {
            "method": "verified_paper_page_order",
            "question_number": number,
            "confidence": confidence,
            "mapping_basis": "same_paper_layout",
            "prompt_version": _LAYOUT_PROMPT_VERSION,
            "manual_review_required": False,
            "evidence_version": 1,
            "evidence_atom_ids": sorted(_region_atom_ids(regions)),
            "evidence_source": "verified_page_order_regions",
            "layout_match_confidence": layout_match_confidence,
        }

    unexpected_numbers = sorted(set(records_by_number) - anchor_numbers)
    if invalid_record_count or unexpected_numbers:
        unresolved.append(
            {"reason": "Paper-page transcription returned invalid or out-of-catalog records"}
        )

    coverage_is_reliable = (
        coverage_complete
        and coverage_confidence >= _ACCEPT_CONFIDENCE
        and len(processed_numbers) == len(anchors)
        and not unresolved
    )
    if unresolved:
        unresolved_response = _unresolved_response(unresolved, pages_by_number)
        responses.append(unresolved_response)
        assignment_details[str(unresolved_response.response_id)] = {
            "method": "verified_paper_layout_unresolved",
            "confidence": min(coverage_confidence, layout_match_confidence),
            "reason": _compact_unresolved_reason(unresolved),
            "prompt_version": _LAYOUT_PROMPT_VERSION,
            "manual_review_required": True,
        }

    return DocumentAnswerMappingResult(
        responses=responses,
        assignment_details_by_response=assignment_details,
        coverage_is_reliable=coverage_is_reliable,
        manual_review_required=not coverage_is_reliable,
        reason=None if coverage_is_reliable else _compact_unresolved_reason(unresolved),
        metadata={
            "mapping_strategy": "verified_same_paper_page_order",
            "layout_prompt_version": _LAYOUT_PROMPT_VERSION,
            "layout_match_confidence": layout_match_confidence,
            "page_coverage_confidence": coverage_confidence,
            "mapped_question_numbers": sorted(
                response.question_number
                for response in responses
                if response.question_number is not None
            ),
            "verified_blank_question_numbers": sorted(verified_blank_numbers),
            "unresolved_region_count": len(unresolved),
        },
    )


def _layout_mapping_failure(
    *,
    anchors: List[_QuestionLayoutAnchor],
    pages: List[PageOCR],
    reason: str,
    layout_match_confidence: float,
    unreadable_page_numbers: Optional[List[int]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DocumentAnswerMappingResult:
    pages_by_number = {page.page_number: page for page in pages}
    target_pages = sorted(
        set(unreadable_page_numbers or [anchor.page_number for anchor in anchors])
    )
    unresolved = [
        {
            "reason": reason,
            "regions": [
                {"page_number": page_number, "y_start": 0, "y_end": 1000}
            ],
        }
        for page_number in target_pages
        if page_number in pages_by_number
    ] or [{"reason": reason}]
    response = _unresolved_response(unresolved, pages_by_number)
    detail = {
        "method": "verified_paper_layout_unresolved",
        "confidence": layout_match_confidence,
        "reason": reason,
        "prompt_version": _LAYOUT_PROMPT_VERSION,
        "manual_review_required": True,
    }
    result_metadata = {
        "mapping_strategy": "verified_same_paper_page_order",
        "layout_match_confidence": layout_match_confidence,
        "unreadable_page_numbers": target_pages,
    }
    result_metadata.update(metadata or {})
    return DocumentAnswerMappingResult(
        responses=[response],
        assignment_details_by_response={str(response.response_id): detail},
        coverage_is_reliable=False,
        manual_review_required=True,
        reason=reason,
        metadata=result_metadata,
    )


async def _build_document_messages(
    *,
    pages: List[PageOCR],
    answer_pages: List[Dict[str, Any]],
    numbered_questions: List[tuple[int, Dict[str, Any]]],
) -> tuple[List[Dict[str, Any]], List[int]]:
    """Build one private, multi-page vision request with OCR layout evidence."""
    by_number = {
        int(page.get("page_number") or 0): page
        for page in answer_pages
        if int(page.get("page_number") or 0) > 0
    }
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": _build_mapping_prompt(pages, numbered_questions),
        }
    ]
    unreadable: List[int] = []

    for page in sorted(pages, key=lambda value: value.page_number):
        artefact = by_number.get(page.page_number, {})
        raw_image_ref = artefact.get("raw_image_ref")
        if not isinstance(raw_image_ref, str) or not raw_image_ref.strip():
            unreadable.append(page.page_number)
            continue
        expected_sha256 = artefact.get("asset_sha256")
        image_b64 = (
            await _resolve_image_base64(
                raw_image_ref,
                expected_sha256=expected_sha256,
            )
            if expected_sha256
            else await _resolve_image_base64(raw_image_ref)
        )
        if not image_b64:
            unreadable.append(page.page_number)
            continue
        content.append(
            {
            "type": "text",
            "text": (
                f"\n--- Page {page.page_number} image follows. OCR layout evidence for this page: ---\n"
                + _page_layout_evidence(page)
            ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{_detect_media_type(raw_image_ref)};base64,{image_b64}",
                    "detail": "high",
                },
            }
        )

    if len(content) <= 1:
        return [], unreadable
    return [{"role": "user", "content": content}], unreadable


def _build_mapping_prompt(
    pages: List[PageOCR],
    numbered_questions: List[tuple[int, Dict[str, Any]]],
) -> str:
    catalog: List[Dict[str, Any]] = []
    for question_number, question in numbered_questions:
        question_text = _question_text(question)
        reference = _question_reference_solution(question)
        rubric = _question_rubric(question)
        criteria = _question_marking_criteria(question)
        catalog.append(
            {
                "question_number": question_number,
                "question": question_text[:_MAX_QUESTION_TEXT_CHARS],
                "reference_solution": reference[:_MAX_REFERENCE_TEXT_CHARS],
                "rubric": rubric[:_MAX_RUBRIC_TEXT_CHARS],
                "marking_criteria": criteria,
            }
        )
    return (
        "You are the document-association stage of an exam marking system. "
        "You are given every page of ONE student's handwritten answer copy and "
        "the immutable question paper catalog. Your only job is to identify which "
        "visible answer regions belong to which question. Do NOT award marks, do "
        "not infer a missing answer, and do not use the question paper itself as "
        "student evidence. Answers may be intermixed, continue across pages, omit "
        "Q labels, or appear in a different order. Use visible handwriting, layout, "
        "equations, and semantic content together.\n\n"
        "CRITICAL: Ignore printed answer-book form chrome such as Name, Date, Page, "
        "Class, Roll No, Answer Book, school logos, and the student name header. "
        "Never map a form-header band to any question. Prefer handwritten body text "
        "and numbered labels like '1)', '2.', 'Q1', 'Ans 3)'.\n\n"
        "First make a private ledger of every distinct worked solution visible on "
        "the answer pages. Match each ledger item against the catalog using its "
        "given values, method, equations, requested result, teacher reference "
        "solution/rubric, and visible label when there is one. The teacher reference "
        "material is an identity anchor only: it never proves that the student wrote "
        "an answer and it must not be used to award marks. Never copy the teacher "
        "reference / answer key text into transcribed_text. A projectile-motion "
        "solution, a smooth-wedge solution, and "
        "a work-energy solution are distinct answers even if the student wrote them "
        "without Q numbers. Never attach a whole mixed page to the first question "
        "just because that question appears first in the catalog.\n\n"
        "Return ONLY valid JSON in exactly this shape:\n"
        "{\n"
        '  "document_coverage": {"complete": true, "confidence": 0.0},\n'
        '  "answers": [\n'
        "    {\n"
        '      "question_number": 1,\n'
        '      "confidence": 0.0,\n'
        '      "mapping_basis": "explicit_label|layout_and_semantics|continuation",\n'
        '      "regions": [{"page_number": 1, "y_start": 0, "y_end": 1000}],\n'
        '      "transcribed_text": "only the handwriting inside these regions"\n'
        "    }\n"
        "  ],\n"
        '  "unresolved_regions": [{"page_number": 1, "y_start": 0, "y_end": 1000, "reason": "...", "transcribed_text": "..."}]\n'
        "}\n\n"
        "Coordinates MUST be normalized page Y coordinates from 0 (top) to 1000 "
        "(bottom). An answer may contain multiple regions/pages. Include only an "
        "answer when its question_number is in this catalog and confidence is at "
        "least 0.50. Put uncertain or unreadable handwriting in unresolved_regions. "
        "A region may be assigned to only one question. Do not assign one large "
        "page-wide region to Q1 merely because it is the first catalog question. "
        "Separate visible working for different questions into separate regions. If "
        "one answer continues on another page, return it as one answer with every "
        "continuation region included; use mapping_basis=continuation. Set "
        "document_coverage.complete=true ONLY if every nonblank handwriting region "
        "in the submitted copy has been assigned exactly once; absent questions do "
        "not need an answer entry.\n\n"
        f"Question catalog:\n{json.dumps(catalog, ensure_ascii=False)}\n"
        f"Page count: {len(pages)}"
    )


def _as_text(value: Any) -> str:
    """Return a scalar text value without serialising arbitrary metadata."""

    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value).strip()
    return ""


def _question_text(question: Dict[str, Any]) -> str:
    """Use the immutable question wording, including older authoring aliases."""

    for key in ("question_text", "text", "question", "content"):
        value = _as_text(question.get(key))
        if value:
            return value
    metadata = question.get("metadata")
    if isinstance(metadata, dict):
        for key in ("question_text", "text", "question", "content"):
            value = _as_text(metadata.get(key))
            if value:
                return value
    return ""


def _question_reference_solution(question: Dict[str, Any]) -> str:
    """Read canonical and legacy approved-answer fields from a snapshot.

    Finalized PCR questions normally expose ``reference_solution``.  Older
    finalized papers and imported content can still use one of the legacy
    aliases below.  Mapping needs the same semantic anchor as scoring, but it
    must never depend on a mutable content-management object.
    """

    for key in (
        "reference_solution",
        "teacher_reference_solution",
        "reference_answer",
        "solution",
        "answer",
        "correct_answer",
        "correctAnswer",
        "final_answer_text",
    ):
        value = _as_text(question.get(key))
        if value:
            return value
    metadata = question.get("metadata")
    if isinstance(metadata, dict):
        for key in (
            "reference_solution",
            "teacher_reference_solution",
            "reference_answer",
            "solution",
            "answer",
            "correct_answer",
            "correctAnswer",
            "final_answer_text",
        ):
            value = _as_text(metadata.get(key))
            if value:
                return value
    return ""


def _question_rubric(question: Dict[str, Any]) -> str:
    """Return only free-text rubric guidance; structured rows stay structured."""

    for key in ("rubric", "marking_scheme", "marking_criteria", "explanation"):
        raw = question.get(key)
        if isinstance(raw, (list, dict)):
            continue
        value = _as_text(raw)
        if value:
            return value
    metadata = question.get("metadata")
    if isinstance(metadata, dict):
        for key in ("rubric", "marking_scheme", "marking_criteria", "explanation"):
            raw = metadata.get(key)
            if isinstance(raw, (list, dict)):
                continue
            value = _as_text(raw)
            if value:
                return value
    return ""


def _question_marking_criteria(question: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compact immutable criterion rows into semantic mapping anchors.

    The mapping model only needs to tell one answer's method from another.  It
    receives the criterion description/evidence, not marks to award.  This is
    deliberately bounded so a long authoring rubric cannot crowd out the page
    images that are the actual evidence.
    """

    raw: Any = question.get("marking_criteria")
    if raw is None and isinstance(question.get("metadata"), dict):
        raw = question["metadata"].get("marking_criteria")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    if not isinstance(raw, list):
        return []

    criteria: List[Dict[str, Any]] = []
    for item in raw[:_MAX_CRITERIA_PER_QUESTION]:
        if not isinstance(item, dict):
            continue
        description = _as_text(
            item.get("description") or item.get("criterion") or item.get("step")
        )
        evidence = _as_text(
            item.get("acceptable_evidence")
            or item.get("evidence")
            or item.get("expected_evidence")
        )
        if not description and not evidence:
            continue
        criteria.append(
            {
                "description": description[:_MAX_CRITERION_TEXT_CHARS],
                "acceptable_evidence": evidence[:_MAX_CRITERION_TEXT_CHARS],
            }
        )
    return criteria


def _page_layout_evidence(page: PageOCR) -> str:
    blocks: List[Dict[str, Any]] = []
    for block in sorted(page.text_blocks, key=lambda value: (value.bbox.y_min, value.bbox.x_min)):
        blocks.append(
            {
                "text": str(block.text)[:2000],
                "confidence": round(float(block.confidence), 3),
                "bbox_mm": {
                    "x_min": round(block.bbox.x_min, 2),
                    "y_min": round(block.bbox.y_min, 2),
                    "x_max": round(block.bbox.x_max, 2),
                    "y_max": round(block.bbox.y_max, 2),
                },
            }
        )
    raw = json.dumps(blocks, ensure_ascii=False)
    return raw[:_MAX_OCR_TEXT_CHARS_PER_PAGE]


def _parse_mapping_payload(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except (TypeError, ValueError, json.JSONDecodeError):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    return parsed if isinstance(parsed, dict) else None


def _build_mapping_result(
    *,
    payload: Dict[str, Any],
    pages: List[PageOCR],
    numbered_questions: List[tuple[int, Dict[str, Any]]],
    unreadable_pages: List[int],
) -> DocumentAnswerMappingResult:
    valid_numbers = {number for number, _question in numbered_questions}
    pages_by_number = {page.page_number: page for page in pages}
    used_numbers: set[int] = set()
    accepted_regions_by_page: Dict[int, List[tuple[float, float, int]]] = {}
    accepted_block_owners: Dict[str, int] = {}
    accepted_text_owners: Dict[str, int] = {}
    responses: List[DetectedResponse] = []
    assignment_details: Dict[str, Dict[str, Any]] = {}
    unresolved: List[Dict[str, Any]] = []

    candidates, candidate_unresolved = _normalise_answer_candidates(
        payload.get("answers"),
        valid_numbers=valid_numbers,
        pages_by_number=pages_by_number,
    )
    unresolved.extend(candidate_unresolved)

    for candidate in candidates:
        number = candidate.question_number
        confidence = candidate.confidence
        regions = candidate.regions
        answer = candidate.answer
        if _materially_overlaps_other_assignment(
            regions,
            accepted_regions_by_page,
        ):
            unresolved.append(
                {
                    "reason": (
                        f"The proposed region for Q{number} materially overlaps a region "
                        "already assigned to another question"
                    ),
                    "answer": answer,
                }
            )
            continue
        if confidence < _ACCEPT_CONFIDENCE:
            unresolved.append(
                {
                    "reason": f"Question association confidence {confidence:.2f} is below the automatic threshold",
                    "answer": answer,
                }
            )
            continue
        block_atom_ids = _block_atom_ids_for_regions(regions, pages_by_number)
        reused_atoms = sorted(
            atom_id for atom_id in block_atom_ids if atom_id in accepted_block_owners
        )
        if reused_atoms:
            unresolved.append(
                {
                    "reason": (
                        f"The proposed answer for Q{number} reuses OCR evidence "
                        "already owned by another question"
                    ),
                    "answer": answer,
                    "conflicting_question_numbers": sorted(
                        {accepted_block_owners[atom_id] for atom_id in reused_atoms}
                    ),
                    "evidence_atom_ids": reused_atoms,
                }
            )
            continue
        response = _response_from_answer(answer, number, confidence, regions, pages_by_number)
        if not response.detected_text.strip():
            unresolved.append(
                {
                    "reason": (
                        f"The proposed region for Q{number} contained no usable "
                        "student-answer transcription"
                    ),
                    "answer": answer,
                }
            )
            continue
        normalized_text = _normalized_evidence_text(response.detected_text)
        if len(normalized_text) >= 24 and normalized_text in accepted_text_owners:
            unresolved.append(
                {
                    "reason": (
                        f"The proposed answer for Q{number} duplicates answer text "
                        "already assigned to another question"
                    ),
                    "answer": answer,
                    "conflicting_question_numbers": [
                        accepted_text_owners[normalized_text]
                    ],
                }
            )
            continue
        used_numbers.add(number)
        for region in regions:
            accepted_regions_by_page.setdefault(region.page_number, []).append(
                (region.y_start, region.y_end, number)
            )
        responses.append(response)
        evidence_atom_ids = block_atom_ids or _region_atom_ids(regions)
        for atom_id in block_atom_ids:
            accepted_block_owners[atom_id] = number
        if len(normalized_text) >= 24:
            accepted_text_owners[normalized_text] = number
        assignment_details[str(response.response_id)] = {
            "method": "document_vision_mapping",
            "question_number": number,
            "confidence": confidence,
            "mapping_basis": str(answer.get("mapping_basis") or "layout_and_semantics"),
            "continuation_segment_count": int(
                answer.get("_continuation_segment_count") or 1
            ),
            "prompt_version": _PROMPT_VERSION,
            "manual_review_required": False,
            "evidence_version": 1,
            "evidence_atom_ids": sorted(evidence_atom_ids),
            "evidence_source": "ocr_blocks" if block_atom_ids else "vision_regions",
        }

    # The model's coverage assertion is useful but not sufficient on its own.
    # When OCR has real layout boxes, independently check that each meaningful
    # text block is owned by one mapped answer.  This prevents an omitted lower
    # half of a page from silently becoming a not-attempted zero.
    unresolved.extend(
        _uncovered_layout_regions(
            pages_by_number=pages_by_number,
            accepted_regions_by_page=accepted_regions_by_page,
        )
    )

    raw_unresolved = payload.get("unresolved_regions")
    if isinstance(raw_unresolved, list):
        unresolved.extend(item for item in raw_unresolved if isinstance(item, dict))
    if unreadable_pages:
        unresolved.append(
            {
                "reason": "One or more answer-copy pages could not be read",
                "page_numbers": unreadable_pages,
            }
        )

    coverage = payload.get("document_coverage")
    coverage_complete = bool(coverage.get("complete")) if isinstance(coverage, dict) else False
    coverage_confidence = _confidence(coverage.get("confidence")) if isinstance(coverage, dict) else 0.0
    coverage_is_reliable = (
        coverage_complete
        and coverage_confidence >= _ACCEPT_CONFIDENCE
        and not unresolved
        and bool(responses)
    )

    if unresolved:
        unresolved_response = _unresolved_response(unresolved, pages_by_number)
        responses.append(unresolved_response)
        assignment_details[str(unresolved_response.response_id)] = {
            "method": "document_mapping_unresolved",
            "confidence": coverage_confidence,
            "reason": _compact_unresolved_reason(unresolved),
            "prompt_version": _PROMPT_VERSION,
            "manual_review_required": True,
        }

    if not responses:
        return _unsafe_result(
            "No reliable answer regions could be associated with paper questions",
            metadata={"coverage_confidence": coverage_confidence},
        )

    return DocumentAnswerMappingResult(
        responses=responses,
        assignment_details_by_response=assignment_details,
        coverage_is_reliable=coverage_is_reliable,
        manual_review_required=not coverage_is_reliable,
        reason=None if coverage_is_reliable else _compact_unresolved_reason(unresolved),
        metadata={
            "prompt_version": _PROMPT_VERSION,
            "document_coverage_confidence": coverage_confidence,
            "mapped_question_numbers": sorted(used_numbers),
            "unresolved_region_count": len(unresolved),
        },
    )


def _response_from_answer(
    answer: Dict[str, Any],
    question_number: int,
    confidence: float,
    regions: List[SourcePageRef],
    pages_by_number: Dict[int, PageOCR],
    *,
    prefer_vision_transcription: bool = False,
) -> DetectedResponse:
    from ..domain.marker_parser import is_form_header_text, strip_form_header_noise

    selected_block_pairs = _blocks_for_regions(regions, pages_by_number)
    selected_blocks = [
        block
        for _page_number, block in selected_block_pairs
        if not is_form_header_text(block.text)
    ]
    text = " ".join(block.text.strip() for block in selected_blocks if block.text.strip())
    text = strip_form_header_noise(text)
    used_vision_transcription = False
    fallback_text = strip_form_header_noise(str(answer.get("transcribed_text") or "").strip())
    if (
        prefer_vision_transcription
        and fallback_text
        and not is_form_header_text(fallback_text)
    ) or (
        (not text or _contains_only_full_page_blocks(
            selected_block_pairs,
            pages_by_number,
        ))
        and fallback_text
        and not is_form_header_text(fallback_text)
    ):
        text = fallback_text
        used_vision_transcription = True
    text = text.strip()
    # Never persist pure form chrome as a scoreable answer.
    if is_form_header_text(text):
        text = ""
    response_id = f"RESP-MAP-{uuid.uuid4().hex[:12]}"
    flags: List[Flag] = []
    mapping_basis = str(answer.get("mapping_basis") or "layout_and_semantics")
    if mapping_basis not in {"explicit_label", "same_paper_layout"}:
        flags.append(
            _make_flag(
                response_id,
                FlagType.NO_QUESTION_MARKER,
                FlagSeverity.WARNING,
                "Question ownership was inferred from the full answer-copy layout and content because no explicit Q marker was reliable.",
                "Review the highlighted source region before publishing if this answer is high stakes.",
                {"mapping_basis": mapping_basis, "confidence": confidence},
            )
        )
    if confidence < _WARN_CONFIDENCE or (
        used_vision_transcription and mapping_basis != "same_paper_layout"
    ):
        flags.append(
            _make_flag(
                response_id,
                FlagType.LOW_SEGMENTATION_CONFIDENCE,
                FlagSeverity.WARNING,
                "Document-level answer mapping used region transcription or had below-high confidence.",
                "Review the mapped region if the automated score looks unexpected.",
                {
                    "mapping_confidence": confidence,
                    "used_vision_region_transcription": used_vision_transcription,
                },
            )
        )
    mean_ocr = (
        sum(block.confidence for block in selected_blocks) / len(selected_blocks)
        if selected_blocks
        else confidence
    )
    return DetectedResponse(
        response_id=response_id,
        question_number=question_number,
        sub_part=None,
        detected_text=text,
        source_pages=regions,
        content_type=ContentType.TEXT_ONLY,
        text_coverage_ratio=1.0 if text else 0.0,
        segmentation_confidence=confidence,
        ocr_confidence=round(max(0.0, min(1.0, mean_ocr)), 4),
        flags=flags,
        word_count=len(text.split()),
        is_continuation=(
            len({region.page_number for region in regions}) > 1
            or int(answer.get("_continuation_segment_count") or 1) > 1
        ),
    )


def _normalise_answer_candidates(
    raw_answers: Any,
    *,
    valid_numbers: set[int],
    pages_by_number: Dict[int, PageOCR],
) -> tuple[List[_AnswerCandidate], List[Dict[str, Any]]]:
    """Validate and safely coalesce answer regions by question number.

    The vision model should normally return a single item with multiple regions
    for a continued solution.  Real model output may instead emit one item per
    page.  We merge only non-overlapping, clearly continuous claims.  Duplicate
    or competing claims remain visible for a teacher instead of being scored as
    two attempts or silently dropped.
    """
    answers = raw_answers if isinstance(raw_answers, list) else []
    by_question: Dict[int, List[_AnswerCandidate]] = {}
    unresolved: List[Dict[str, Any]] = []

    for answer in answers:
        if not isinstance(answer, dict):
            continue
        number = _as_int(answer.get("question_number"))
        regions = _normalise_regions(answer.get("regions"), pages_by_number)
        if number not in valid_numbers or not regions:
            unresolved.append(
                {
                    "reason": "The mapper returned an invalid question or answer region",
                    "answer": answer,
                }
            )
            continue
        by_question.setdefault(number, []).append(
            _AnswerCandidate(
                question_number=number,
                confidence=_confidence(answer.get("confidence")),
                regions=regions,
                answer=dict(answer),
            )
        )

    merged: List[_AnswerCandidate] = []
    for number in sorted(by_question):
        candidates = by_question[number]
        coalesced, group_unresolved = _coalesce_question_candidates(candidates)
        merged.extend(coalesced)
        unresolved.extend(group_unresolved)
    return merged, unresolved


def _coalesce_question_candidates(
    candidates: List[_AnswerCandidate],
) -> tuple[List[_AnswerCandidate], List[Dict[str, Any]]]:
    """Return one safe answer candidate per question, or teacher-review data."""
    if len(candidates) <= 1:
        return candidates, []

    # Some providers repeat the same region in their JSON.  Identical claims
    # for the same question are harmless; merge them deterministically.
    distinct: List[_AnswerCandidate] = []
    for candidate in candidates:
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(distinct)
                if _regions_equivalent(existing.regions, candidate.regions)
            ),
            None,
        )
        if duplicate_index is None:
            distinct.append(candidate)
        else:
            distinct[duplicate_index] = _merge_candidates(
                [distinct[duplicate_index], candidate],
                continuation=False,
            )

    if len(distinct) == 1:
        return distinct, []

    if _candidates_materially_overlap(distinct):
        return [], _candidate_group_unresolved(
            distinct,
            "Multiple proposed regions for this question overlap, so the system "
            "cannot prove whether they are a continuation or competing work.",
        )

    # Cross-page, non-overlapping fragments are a normal handwritten answer
    # continuation.  Same-page fragments need an explicit continuation signal;
    # otherwise they may be separate attempts and should be reviewed.
    page_sets = [{region.page_number for region in candidate.regions} for candidate in distinct]
    spans_multiple_pages = len(set().union(*page_sets)) > 1
    explicit_continuation = any(
        str(candidate.answer.get("mapping_basis") or "").strip().lower()
        == "continuation"
        for candidate in distinct
    )
    if spans_multiple_pages or explicit_continuation:
        return [_merge_candidates(distinct, continuation=True)], []

    return [], _candidate_group_unresolved(
        distinct,
        "More than one separate region claimed this question without evidence "
        "that the regions form one continued answer.",
    )


def _merge_candidates(
    candidates: List[_AnswerCandidate],
    *,
    continuation: bool,
) -> _AnswerCandidate:
    first = candidates[0]
    merged_answer = dict(first.answer)
    merged_answer["mapping_basis"] = (
        "continuation"
        if continuation
        else str(merged_answer.get("mapping_basis") or "layout_and_semantics")
    )
    merged_answer["transcribed_text"] = _merge_transcriptions(
        [str(candidate.answer.get("transcribed_text") or "") for candidate in candidates]
    )
    merged_answer["_continuation_segment_count"] = (
        sum(
            max(1, int(candidate.answer.get("_continuation_segment_count") or 1))
            for candidate in candidates
        )
        if continuation
        else 1
    )
    return _AnswerCandidate(
        question_number=first.question_number,
        confidence=min(candidate.confidence for candidate in candidates),
        regions=_dedupe_regions(
            [region for candidate in candidates for region in candidate.regions]
        ),
        answer=merged_answer,
    )


def _candidate_group_unresolved(
    candidates: List[_AnswerCandidate],
    reason: str,
) -> List[Dict[str, Any]]:
    return [{"reason": reason, "answer": candidate.answer} for candidate in candidates]


def _regions_equivalent(
    left: List[SourcePageRef],
    right: List[SourcePageRef],
) -> bool:
    left_keys = {
        (region.page_number, round(region.y_start, 1), round(region.y_end, 1))
        for region in left
    }
    right_keys = {
        (region.page_number, round(region.y_start, 1), round(region.y_end, 1))
        for region in right
    }
    return bool(left_keys) and left_keys == right_keys


def _candidates_materially_overlap(candidates: List[_AnswerCandidate]) -> bool:
    for index, candidate in enumerate(candidates):
        for other_candidate in candidates[index + 1 :]:
            for region in candidate.regions:
                for other in other_candidate.regions:
                    if region.page_number != other.page_number:
                        continue
                    overlap = max(
                        0.0,
                        min(region.y_end, other.y_end)
                        - max(region.y_start, other.y_start),
                    )
                    shortest = min(
                        region.y_end - region.y_start,
                        other.y_end - other.y_start,
                    )
                    if shortest > 0 and overlap / shortest >= 0.35:
                        return True
    return False


def _merge_transcriptions(values: List[str]) -> str:
    unique: List[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        if normalized and normalized.casefold() not in seen:
            seen.add(normalized.casefold())
            unique.append(value.strip())
    return "\n\n".join(unique)


def _uncovered_layout_regions(
    *,
    pages_by_number: Dict[int, PageOCR],
    accepted_regions_by_page: Dict[int, List[tuple[float, float, int]]],
) -> List[Dict[str, Any]]:
    """Return meaningful OCR blocks that no accepted question owns.

    OCR fallback can produce one page-sized block with no layout fidelity.  Such
    a block is deliberately ignored here: the vision mapper's image evidence is
    the authority in that case.  Granular blocks are independent evidence and
    must be accounted for before absent questions may be assigned zero rows.
    """
    unresolved: List[Dict[str, Any]] = []
    for page_number, page in pages_by_number.items():
        assignments = accepted_regions_by_page.get(page_number, [])
        for block in page.text_blocks:
            if not _is_layout_informative_block(block, page):
                continue
            if _block_is_covered_by_assignment(block, assignments):
                continue
            unresolved.append(
                {
                    "reason": "A visible OCR region was not assigned to any paper question",
                    "regions": [_block_as_normalized_region(page, block)],
                    "transcribed_text": block.text.strip(),
                }
            )
    return unresolved


def _is_layout_informative_block(block: TextBlock, page: PageOCR) -> bool:
    text = block.text.strip()
    if len(text) < 6:
        return False
    # A page-wide OCR fallback tells us text exists, not where it is.  Do not
    # mistake it for a separate unassigned answer region.
    return (
        block.bbox.height < page.page_height_mm * 0.82
        and block.bbox.width < page.page_width_mm * 0.98
    )


def _block_is_covered_by_assignment(
    block: TextBlock,
    assignments: List[tuple[float, float, int]],
) -> bool:
    midpoint = (block.bbox.y_min + block.bbox.y_max) / 2.0
    for start, end, _question_number in assignments:
        overlap = max(0.0, min(block.bbox.y_max, end) - max(block.bbox.y_min, start))
        if (
            overlap >= min(max(block.bbox.height * 0.5, 1.0), 8.0)
            or start <= midpoint <= end
        ):
            return True
    return False


def _block_as_normalized_region(page: PageOCR, block: TextBlock) -> Dict[str, float | int]:
    return {
        "page_number": page.page_number,
        "y_start": round((block.bbox.y_min / page.page_height_mm) * 1000.0, 2),
        "y_end": round((block.bbox.y_max / page.page_height_mm) * 1000.0, 2),
    }


def _unresolved_response(
    unresolved: List[Dict[str, Any]],
    pages_by_number: Dict[int, PageOCR],
) -> DetectedResponse:
    regions: List[SourcePageRef] = []
    text_parts: List[str] = []
    for item in unresolved:
        regions.extend(_normalise_regions(item.get("regions"), pages_by_number))
        candidate = str(item.get("transcribed_text") or "").strip()
        if candidate:
            text_parts.append(candidate)
    if not regions:
        regions = [
            SourcePageRef(page_number=page.page_number, y_start=0.0, y_end=page.page_height_mm)
            for page in sorted(pages_by_number.values(), key=lambda value: value.page_number)
        ]
    if not text_parts:
        text_parts = [
            block.text.strip()
            for page in pages_by_number.values()
            for block in page.text_blocks
            if block.text.strip()
        ]
    response_id = f"RESP-MAP-UNRESOLVED-{uuid.uuid4().hex[:10]}"
    return DetectedResponse(
        response_id=response_id,
        question_number=None,
        sub_part=None,
        detected_text=" ".join(text_parts).strip(),
        source_pages=_dedupe_regions(regions),
        content_type=ContentType.TEXT_ONLY,
        text_coverage_ratio=1.0 if text_parts else 0.0,
        segmentation_confidence=0.0,
        ocr_confidence=0.0,
        flags=[
            _make_flag(
                response_id,
                FlagType.LOW_SEGMENTATION_CONFIDENCE,
                FlagSeverity.BLOCKING,
                "Document-level mapping could not safely identify every answer region.",
                "Teacher review is required; do not assign absent-answer zero marks from this unresolved copy.",
                {"unresolved_region_count": len(unresolved)},
            )
        ],
        word_count=len(" ".join(text_parts).split()),
        is_continuation=len({region.page_number for region in regions}) > 1,
    )


def _normalise_regions(
    value: Any,
    pages_by_number: Dict[int, PageOCR],
) -> List[SourcePageRef]:
    if not isinstance(value, list):
        return []
    regions: List[SourcePageRef] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        page_number = _as_int(item.get("page_number"))
        page = pages_by_number.get(page_number)
        if page is None:
            continue
        start = _normalised_y_to_mm(item.get("y_start"), page.page_height_mm)
        end = _normalised_y_to_mm(item.get("y_end"), page.page_height_mm)
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start
        if end - start < 1.0:
            continue
        regions.append(SourcePageRef(page_number=page_number, y_start=start, y_end=end))
    return _dedupe_regions(regions)


def _normalised_y_to_mm(value: Any, page_height_mm: float) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= number <= 1.0:
        number *= 1000.0
    number = max(0.0, min(1000.0, number))
    return round((number / 1000.0) * page_height_mm, 3)


def _blocks_for_regions(
    regions: List[SourcePageRef],
    pages_by_number: Dict[int, PageOCR],
) -> List[tuple[int, TextBlock]]:
    selected: List[tuple[int, TextBlock]] = []
    seen: set[tuple[int, float, float, float, float, str]] = set()
    for region in regions:
        page = pages_by_number.get(region.page_number)
        if page is None:
            continue
        for block in page.text_blocks:
            overlap = max(0.0, min(block.bbox.y_max, region.y_end) - max(block.bbox.y_min, region.y_start))
            midpoint = (block.bbox.y_min + block.bbox.y_max) / 2.0
            if overlap >= min(max(block.bbox.height * 0.25, 1.0), 8.0) or region.y_start <= midpoint <= region.y_end:
                key = (
                    page.page_number,
                    round(block.bbox.x_min, 3),
                    round(block.bbox.y_min, 3),
                    round(block.bbox.x_max, 3),
                    round(block.bbox.y_max, 3),
                    block.text,
                )
                if key not in seen:
                    seen.add(key)
                    selected.append((page.page_number, block))
    return sorted(
        selected,
        key=lambda item: (item[0], item[1].bbox.y_min, item[1].bbox.x_min),
    )


def _normalized_evidence_text(value: Any) -> str:
    """Normalize meaningful text for conservative duplicate detection."""

    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _block_atom_id(page_number: int, block: TextBlock) -> str:
    payload = "\x1f".join(
        (
            str(page_number),
            f"{block.bbox.x_min:.3f}",
            f"{block.bbox.y_min:.3f}",
            f"{block.bbox.x_max:.3f}",
            f"{block.bbox.y_max:.3f}",
            _normalized_evidence_text(block.text),
        )
    )
    return f"ocr-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _block_atom_ids_for_regions(
    regions: List[SourcePageRef],
    pages_by_number: Dict[int, PageOCR],
) -> set[str]:
    return {
        _block_atom_id(page_number, block)
        for page_number, block in _blocks_for_regions(regions, pages_by_number)
    }


def _region_atom_ids(regions: List[SourcePageRef]) -> set[str]:
    atom_ids: set[str] = set()
    for region in regions:
        payload = (
            f"{region.page_number}\x1f{region.y_start:.3f}\x1f{region.y_end:.3f}"
        )
        atom_ids.add(
            f"region-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"
        )
    return atom_ids


def _contains_only_full_page_blocks(
    blocks: List[tuple[int, TextBlock]],
    pages_by_number: Dict[int, PageOCR],
) -> bool:
    if not blocks:
        return False
    # A parser fallback that created one page-sized block has no usable region
    # evidence.  Do not duplicate that whole transcription into every mapped
    # answer; use the mapper's region transcription instead.
    for page_number, block in blocks:
        matching_page = pages_by_number.get(page_number)
        if matching_page is None:
            return False
        if block.bbox.height < matching_page.page_height_mm * 0.88:
            return False
    return True


def _materially_overlaps_other_assignment(
    regions: List[SourcePageRef],
    accepted_regions_by_page: Dict[int, List[tuple[float, float, int]]],
) -> bool:
    """Reject ownership claims that reuse most of another question's region."""
    for region in regions:
        for existing_start, existing_end, _question_number in accepted_regions_by_page.get(
            region.page_number,
            [],
        ):
            overlap = max(
                0.0,
                min(region.y_end, existing_end) - max(region.y_start, existing_start),
            )
            shortest = min(region.y_end - region.y_start, existing_end - existing_start)
            if shortest > 0 and overlap / shortest >= 0.35:
                return True
    return False


def _dedupe_regions(regions: List[SourcePageRef]) -> List[SourcePageRef]:
    seen: set[tuple[int, float, float]] = set()
    result: List[SourcePageRef] = []
    for region in sorted(regions, key=lambda value: (value.page_number, value.y_start, value.y_end)):
        key = (region.page_number, round(region.y_start, 2), round(region.y_end, 2))
        if key not in seen:
            seen.add(key)
            result.append(region)
    return result


def _make_flag(
    response_id: str,
    flag_type: FlagType,
    severity: FlagSeverity,
    reason: str,
    suggested_action: str,
    metadata: Dict[str, Any],
) -> Flag:
    return Flag(
        flag_id=f"FLG-MAP-{uuid.uuid4().hex[:10]}",
        response_id=response_id,
        source="document_answer_mapper",
        flag_type=flag_type,
        severity=severity,
        reason=reason,
        suggested_action=suggested_action,
        metadata=metadata,
    )


def _unsafe_result(reason: str, *, metadata: Optional[Dict[str, Any]] = None) -> DocumentAnswerMappingResult:
    return DocumentAnswerMappingResult(
        coverage_is_reliable=False,
        manual_review_required=True,
        reason=reason,
        metadata=metadata or {},
    )


def _prefer_better_mapping(
    primary: DocumentAnswerMappingResult,
    secondary: DocumentAnswerMappingResult,
) -> DocumentAnswerMappingResult:
    """Choose the mapping with more usable, non-header student text."""
    from ..domain.marker_parser import is_form_header_text, strip_form_header_noise

    def _score(result: DocumentAnswerMappingResult) -> tuple[int, int, int]:
        usable = 0
        chars = 0
        for response in result.responses:
            text = strip_form_header_noise(str(response.detected_text or ""))
            if not text or is_form_header_text(text):
                continue
            if response.question_number is None:
                continue
            usable += 1
            chars += len(text)
        return (
            usable,
            1 if result.coverage_is_reliable else 0,
            chars,
        )

    if primary.coverage_is_reliable != secondary.coverage_is_reliable:
        return primary if primary.coverage_is_reliable else secondary

    def _has_unassigned_evidence(result: DocumentAnswerMappingResult) -> bool:
        return any(response.question_number is None for response in result.responses)

    # When neither candidate proves coverage, preserve the result that keeps
    # unowned handwriting explicit and blocking.  Never trade that evidence
    # away merely because another candidate assigned more snippets.
    if not primary.coverage_is_reliable:
        primary_unassigned = _has_unassigned_evidence(primary)
        secondary_unassigned = _has_unassigned_evidence(secondary)
        if primary_unassigned != secondary_unassigned:
            return primary if primary_unassigned else secondary

    return primary if _score(primary) >= _score(secondary) else secondary


def _deterministic_numbered_mapping(
    *,
    pages: List[PageOCR],
    numbered_questions: List[tuple[int, Dict[str, Any]]],
) -> Optional[DocumentAnswerMappingResult]:
    """Map answers using content-section style numbered labels from OCR.

    Mirrors ``AnswerSheetBlockNormalizer`` / ``AnswerQuestionMappingService``:
    collect blocks that start with ``1)``, ``2.``, ``Ans 3)``, or Q markers,
    then bind each unique number to the matching paper question.
    """
    from ..domain.marker_parser import (
        is_form_header_text,
        parse_markers,
        strip_form_header_noise,
    )

    if not pages or not numbered_questions:
        return None

    valid_numbers = {int(number) for number, _question in numbered_questions}
    pages_by_number = {page.page_number: page for page in pages}
    markers = parse_markers(pages)
    if len(markers) < 2:
        return None

    # Keep first (top-most) occurrence of each number across the document.
    first_by_number: Dict[int, Any] = {}
    for marker in markers:
        number = int(marker.question_number)
        if number not in valid_numbers:
            continue
        if number not in first_by_number:
            first_by_number[number] = marker

    if len(first_by_number) < 2:
        return None

    ordered = sorted(
        first_by_number.values(),
        key=lambda marker: (marker.page_number, marker.y_position),
    )
    responses: List[DetectedResponse] = []
    assignment_details: Dict[str, Dict[str, Any]] = {}
    owned_block_atoms: set[str] = set()

    for index, marker in enumerate(ordered):
        number = int(marker.question_number)
        page = pages_by_number.get(marker.page_number)
        if page is None:
            continue

        # Region ends at the next marker on the same page, else page bottom.
        # Markers on later pages own their own regions.
        y_start = max(0.0, float(marker.y_position) - 3.0)
        if float(marker.y_position) > 35.0:
            y_start = max(35.0, y_start)

        y_end = page.page_height_mm
        next_same_page = None
        for later in ordered[index + 1 :]:
            if later.page_number == marker.page_number:
                next_same_page = later
                break
            if later.page_number > marker.page_number:
                break
        if next_same_page is not None:
            y_end = float(next_same_page.y_position)

        # Continuation: if the next marker is on a later page, still only take
        # this page's span for the current answer (continuation is rare for
        # numbered short answers).  Body text on intermediate pages without a
        # marker remains unassigned for vision / teacher review.
        region = SourcePageRef(
            page_number=marker.page_number,
            y_start=y_start,
            y_end=max(y_end, y_start + 1.0),
        )
        selected_block_pairs = _blocks_for_regions([region], pages_by_number)
        block_atom_ids = {
            _block_atom_id(page_number, block)
            for page_number, block in selected_block_pairs
        }
        if block_atom_ids & owned_block_atoms:
            # A coarse OCR block spanning several labelled regions cannot be
            # treated as distinct evidence for each label. Vision/teacher
            # review must split the physical handwriting first.
            continue
        selected_blocks = [
            block
            for _page_number, block in selected_block_pairs
            if not is_form_header_text(block.text)
        ]
        text = strip_form_header_noise(
            " ".join(block.text.strip() for block in selected_blocks if block.text.strip())
        )
        if not text:
            continue

        response_id = f"RESP-NUM-{uuid.uuid4().hex[:12]}"
        mean_ocr = (
            sum(block.confidence for block in selected_blocks) / len(selected_blocks)
            if selected_blocks
            else 0.85
        )
        response = DetectedResponse(
            response_id=response_id,
            question_number=number,
            sub_part=None,
            detected_text=text,
            source_pages=[region],
            content_type=ContentType.TEXT_ONLY,
            text_coverage_ratio=1.0,
            segmentation_confidence=0.92,
            ocr_confidence=round(max(0.0, min(1.0, mean_ocr)), 4),
            flags=[],
            word_count=len(text.split()),
            is_continuation=False,
        )
        responses.append(response)
        owned_block_atoms.update(block_atom_ids)
        assignment_details[response_id] = {
            "method": "deterministic_answer_number",
            "question_number": number,
            "confidence": 0.92,
            "mapping_basis": "explicit_label",
            "prompt_version": "content-section-numbered-v1",
            "manual_review_required": False,
            "evidence_version": 1,
            "evidence_atom_ids": sorted(block_atom_ids or _region_atom_ids([region])),
            "evidence_source": "ocr_blocks" if block_atom_ids else "numbered_region",
        }

    if len(responses) < 2:
        return None

    mapped_numbers = {
        int(response.question_number)
        for response in responses
        if response.question_number is not None
    }
    paper_size = len(valid_numbers)
    coverage_is_reliable = (
        len(mapped_numbers) == paper_size
        and len(mapped_numbers) == len(responses)
    )
    if not coverage_is_reliable:
        for detail in assignment_details.values():
            detail["manual_review_required"] = True
            detail["reason"] = (
                "Explicit labels were found, but full answer-copy coverage was not proven"
            )
    return DocumentAnswerMappingResult(
        responses=responses,
        assignment_details_by_response=assignment_details,
        coverage_is_reliable=coverage_is_reliable,
        manual_review_required=not coverage_is_reliable,
        reason=None if coverage_is_reliable else "Some paper questions lack numbered answer labels",
        metadata={
            "mapping_strategy": "deterministic_answer_number",
            "mapped_question_numbers": sorted(mapped_numbers),
            "paper_question_count": paper_size,
        },
    )


def _compact_unresolved_reason(unresolved: List[Dict[str, Any]]) -> str:
    if not unresolved:
        return "The answer-copy coverage could not be proven"
    reasons = [str(item.get("reason") or "unresolved answer region").strip() for item in unresolved]
    return "; ".join(reasons[:3])[:700]


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
