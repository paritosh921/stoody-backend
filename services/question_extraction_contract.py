"""Versioned contract for question-paper model output.

The OpenAI request and every legacy-provider response meet at this boundary so
the OCR pipeline never depends on prompt-only JSON shapes.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError


QUESTION_EXTRACTION_CONTRACT_VERSION = "question-paper-visual-v2"


class QuestionExtractionContractError(ValueError):
    """A stable, user-visible failure at the model-output boundary."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class QuestionExtractionBBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x0: float
    y0: float
    x1: float
    y1: float


class QuestionExtractionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: Optional[float]
    printed_text: Optional[str]
    page: Optional[int]
    bbox: Optional[QuestionExtractionBBox]
    confidence: float = Field(ge=0.0, le=1.0)


class QuestionExtractionDiagramRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int = Field(ge=0)
    bbox: QuestionExtractionBBox


class QuestionExtractionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: str = Field(min_length=1)
    text: str = Field(min_length=1)
    options: List[str]
    page: int = Field(ge=0)
    continuation_pages: List[int]
    has_figure: bool
    diagram_regions: List[QuestionExtractionDiagramRegion]
    max_marks: Optional[float]
    marks_evidence: Optional[QuestionExtractionEvidence]


class QuestionPaperExtractionPayload(BaseModel):
    """Strict OpenAI response schema and canonical internal extraction DTO."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[QUESTION_EXTRACTION_CONTRACT_VERSION]
    paper_total_marks: Optional[QuestionExtractionEvidence]
    questions: List[QuestionExtractionItem] = Field(min_length=1)


def question_extraction_response_format() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "question_paper_extraction",
            "strict": True,
            "schema": QuestionPaperExtractionPayload.model_json_schema(),
        },
    }


def _normalise_option(value: Any) -> Optional[str]:
    """Normalise known legacy option shapes at one explicit boundary."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in ("text", "option_text", "content", "value", "answer"):
            candidate = value.get(key)
            if isinstance(candidate, (str, int, float, bool)):
                rendered = str(candidate).strip()
                if rendered:
                    return rendered
        if len(value) == 1:
            return _normalise_option(next(iter(value.values())))
    return None


def _normalise_bbox(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, dict):
        return None
    try:
        return {
            key: float(value[key])
            for key in ("x0", "y0", "x1", "y1")
        }
    except (KeyError, TypeError, ValueError):
        return None


def _normalise_evidence(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, dict):
        return None

    def _optional_float(raw: Any) -> Optional[float]:
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    def _optional_int(raw: Any) -> Optional[int]:
        try:
            return max(0, int(raw)) if raw is not None else None
        except (TypeError, ValueError):
            return None

    confidence = _optional_float(value.get("confidence"))
    return {
        "value": _optional_float(value.get("value")),
        "printed_text": (
            str(value.get("printed_text")).strip()
            if value.get("printed_text") is not None
            else None
        ) or None,
        "page": _optional_int(value.get("page")),
        "bbox": _normalise_bbox(value.get("bbox")),
        "confidence": max(0.0, min(1.0, confidence if confidence is not None else 0.0)),
    }


def normalize_question_extraction_payload(value: Any) -> Dict[str, Any]:
    """Convert provider output to the versioned canonical extraction contract."""
    if not isinstance(value, dict) or not isinstance(value.get("questions"), list):
        raise QuestionExtractionContractError(
            "invalid_model_output",
            "Question extraction returned an invalid response shape.",
        )

    canonical_questions: List[Dict[str, Any]] = []
    for index, raw_question in enumerate(value["questions"], start=1):
        if not isinstance(raw_question, dict):
            raise QuestionExtractionContractError(
                "invalid_model_output",
                f"Question {index} has an invalid response shape.",
            )
        text = str(raw_question.get("text") or "").strip()
        if not text:
            raise QuestionExtractionContractError(
                "incomplete_model_output",
                f"Question {index} was returned without question text.",
            )
        raw_options = raw_question.get("options") or []
        if not isinstance(raw_options, list):
            raw_options = [raw_options]
        options = [
            option
            for option in (_normalise_option(item) for item in raw_options)
            if option
        ]
        try:
            page = max(0, int(raw_question.get("page") or 0))
        except (TypeError, ValueError):
            page = 0
        continuation_pages: List[int] = []
        for raw_page in raw_question.get("continuation_pages") or []:
            try:
                continuation_pages.append(max(0, int(raw_page)))
            except (TypeError, ValueError):
                continue
        diagram_regions: List[Dict[str, Any]] = []
        for raw_region in raw_question.get("diagram_regions") or []:
            if not isinstance(raw_region, dict):
                continue
            bbox = _normalise_bbox(raw_region.get("bbox"))
            if bbox is None:
                continue
            try:
                region_page = max(0, int(raw_region.get("page") or page))
            except (TypeError, ValueError):
                region_page = page
            diagram_regions.append({"page": region_page, "bbox": bbox})
        try:
            max_marks = (
                float(raw_question.get("max_marks"))
                if raw_question.get("max_marks") is not None
                else None
            )
        except (TypeError, ValueError):
            max_marks = None
        canonical_questions.append({
            "number": str(raw_question.get("number") or index).strip() or str(index),
            "text": text,
            "options": options,
            "page": page,
            "continuation_pages": sorted(set(continuation_pages)),
            "has_figure": bool(raw_question.get("has_figure")),
            "diagram_regions": diagram_regions,
            "max_marks": max_marks,
            "marks_evidence": _normalise_evidence(raw_question.get("marks_evidence")),
        })

    canonical = {
        "contract_version": QUESTION_EXTRACTION_CONTRACT_VERSION,
        "paper_total_marks": _normalise_evidence(value.get("paper_total_marks")),
        "questions": canonical_questions,
    }
    try:
        return QuestionPaperExtractionPayload.model_validate(canonical).model_dump(mode="json")
    except ValidationError as exc:
        raise QuestionExtractionContractError(
            "invalid_model_output",
            "Question extraction returned data that does not match the required contract.",
        ) from exc
