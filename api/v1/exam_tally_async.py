from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime
from difflib import SequenceMatcher
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager
from core.ocr_service import get_ocr_service
from services.tally_question_map_service import build_tally_question_map

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/exam-tally", tags=["Exam Tally"])


class TallyStudentContext(BaseModel):
    student_id: Optional[str] = None
    name: Optional[str] = None
    username: Optional[str] = None
    roll_no: Optional[str] = None


class TallyMarkingRange(BaseModel):
    from_: int = Field(..., alias="from")
    to: int = Field(..., alias="to")
    marks: float


class TallyQuestionMapItem(BaseModel):
    question_number: int
    question_id: Optional[str] = None
    question_text: Optional[str] = None
    question_text_preview: Optional[str] = None
    sub_topic: Optional[str] = None
    question_type: Optional[str] = None
    max_marks: Optional[float] = None
    confidence: Optional[float] = None
    source: Optional[str] = None


class TallyDocumentContext(BaseModel):
    document_id: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    standard: Optional[str] = None
    section: Optional[str] = None
    num_questions: Optional[int] = None
    max_marks_per_question: Optional[float] = None
    marking_scheme: List[TallyMarkingRange] = Field(default_factory=list)
    validate_paper_set: bool = False
    expected_paper_set: Optional[str] = None
    question_source_document_id: Optional[str] = None
    question_map: List[TallyQuestionMapItem] = Field(default_factory=list)


class TallyExtractRequest(BaseModel):
    image_b64: str = Field(..., description="Full-page canvas PNG data URL or raw base64")
    document: Optional[TallyDocumentContext] = None
    student: Optional[TallyStudentContext] = None
    copy_id: Optional[str] = None


class TallyValidationIssue(BaseModel):
    severity: str
    code: str
    message: str
    row_index: Optional[int] = None
    column: Optional[str] = None
    question_number: Optional[int] = None
    expected: Optional[str] = None
    actual: Optional[str] = None


class TallyExtractResponse(BaseModel):
    success: bool
    extraction_id: str
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    validation_issues: List[TallyValidationIssue] = Field(default_factory=list)
    confidence: Optional[float] = None
    raw_text: Optional[str] = None


class TallyExportRequest(BaseModel):
    extraction_id: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    filename: Optional[str] = None
    document: Optional[TallyDocumentContext] = None
    student: Optional[TallyStudentContext] = None
    allow_validation_errors: bool = False


class TallyValidateRequest(BaseModel):
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    document: Optional[TallyDocumentContext] = None
    student: Optional[TallyStudentContext] = None


class TallyValidateResponse(BaseModel):
    success: bool
    validation_issues: List[TallyValidationIssue] = Field(default_factory=list)


class TallyTemplateSaveRequest(BaseModel):
    image_b64: str = Field(..., description="Saved full-page tally template PNG data URL or raw base64")
    template_copy_id: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


class TallyTemplateResponse(BaseModel):
    success: bool
    document_id: str
    image_b64: Optional[str] = None
    template_copy_id: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class TallyTemplateSummary(BaseModel):
    document_id: str
    title: Optional[str] = None
    subject: Optional[str] = None
    standard: Optional[str] = None
    section: Optional[str] = None
    image_b64: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None


class TallyTemplateListResponse(BaseModel):
    success: bool
    templates: List[TallyTemplateSummary] = Field(default_factory=list)


class TallyQuestionMapBuildRequest(BaseModel):
    source_document_id: Optional[str] = None
    force: bool = False


class TallyQuestionMapSaveRequest(BaseModel):
    source_document_id: Optional[str] = None
    items: List[TallyQuestionMapItem] = Field(default_factory=list)


class TallyQuestionMapResponse(BaseModel):
    success: bool
    tally_document_id: str
    source_document_id: Optional[str] = None
    status: str = "none"
    items: List[TallyQuestionMapItem] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    generated_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TallyQuestionSourcePreviewItem(BaseModel):
    question_number: int
    max_marks: Optional[float] = None
    max_marks_source: str = "default"
    question_type: Optional[str] = None
    text_preview: Optional[str] = None


class TallyQuestionSourcePreviewResponse(BaseModel):
    success: bool
    question_count: int = 0
    marking_scheme: List[TallyMarkingRange] = Field(default_factory=list)
    items: List[TallyQuestionSourcePreviewItem] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


def _require_admin_or_tutor(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if current_user.get("user_type") not in {"admin", "tutor", "b2c_admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or tutor access required for exam tally",
        )
    return current_user


async def _tenant_db(db: DatabaseManager, current_user: Dict[str, Any]) -> Any:
    db_name = current_user.get("db_name")
    if not db_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context missing from authentication token",
        )
    tenant_db = await db.get_tenant_db(db_name)
    if tenant_db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant database not available",
        )
    return tenant_db


def _strip_code_fence(text: str) -> str:
    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
    return clean.strip()


def _parse_llm_json(text: str) -> Dict[str, Any]:
    clean = _strip_code_fence(text)
    try:
        parsed = json.loads(clean)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(clean[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _stringify_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _flatten_row(row: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in (row or {}).items():
        label = str(key or "").strip()
        if not label:
            continue
        full_key = f"{prefix}.{label}" if prefix else label
        if isinstance(value, dict):
            for nested_key, nested_value in _flatten_row(value, full_key).items():
                flat[nested_key] = nested_value
        else:
            flat[full_key] = _stringify_cell(value)
    return flat


def _column_label(column: Any) -> str:
    if isinstance(column, dict):
        return str(column.get("label") or column.get("key") or column.get("name") or "").strip()
    return str(column or "").strip()


def _normalise_table(parsed: Dict[str, Any]) -> tuple[List[str], List[Dict[str, Any]], List[str], Optional[float]]:
    label_by_key: Dict[str, str] = {}
    for column in parsed.get("columns") or []:
        if isinstance(column, dict):
            key = str(column.get("key") or column.get("name") or "").strip()
            label = _column_label(column)
            if key and label:
                label_by_key[key] = label

    raw_rows = parsed.get("rows")
    if not isinstance(raw_rows, list):
        raw_fields = parsed.get("fields") or parsed.get("values") or parsed.get("data")
        raw_rows = [raw_fields] if isinstance(raw_fields, dict) else []

    rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        if isinstance(row, dict):
            flat = _flatten_row(row)
            rows.append({label_by_key.get(key, key): value for key, value in flat.items()})

    columns: List[str] = []
    seen = set()
    for column in parsed.get("columns") or []:
        label = _column_label(column)
        if label and label not in seen:
            columns.append(label)
            seen.add(label)

    for row in rows:
        for key in row.keys():
            if key not in seen:
                columns.append(key)
                seen.add(key)

    warnings = parsed.get("warnings") if isinstance(parsed.get("warnings"), list) else []
    warnings = [str(w) for w in warnings if str(w).strip()]

    confidence = parsed.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    return columns, rows, warnings, confidence


def _format_marks(value: float) -> str:
    return f"{value:g}"


def _filter_tally_warnings(warnings: List[str], document: TallyDocumentContext) -> List[str]:
    filtered: List[str] = []
    for warning in warnings:
        text = str(warning or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if "roll" in lowered:
            continue
        if not document.validate_paper_set and ("paper set" in lowered or "paper code" in lowered):
            continue
        filtered.append(text)
    return filtered


def _format_marking_scheme(scheme: List[TallyMarkingRange]) -> str:
    parts: List[str] = []
    for item in scheme:
        if item.from_ <= 0 or item.to <= 0 or item.marks <= 0 or item.from_ > item.to:
            continue
        question_range = f"Q{item.from_}" if item.from_ == item.to else f"Q{item.from_}-Q{item.to}"
        parts.append(f"{question_range} max {_format_marks(item.marks)}")
    return ", ".join(parts)


def _normalise_compare(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _cell_text(value: Any) -> str:
    return str(_stringify_cell(value) or "").strip()


def _looks_like_student_name_label(label: str) -> bool:
    normalised = _normalise_compare(label)
    if not normalised:
        return False
    if any(blocked in normalised for blocked in ("exam", "paper", "subject", "school", "class")):
        return False
    return normalised == "name" or normalised in {"studentname", "nameofstudent"}


def _looks_like_roll_label(label: str) -> bool:
    normalised = _normalise_compare(label)
    if not normalised:
        return False
    return (
        "roll" in normalised
        or normalised in {"studentid", "studentcode", "admissionno", "admissionnumber", "username"}
    )


def _looks_like_paper_set_label(label: str) -> bool:
    normalised = _normalise_compare(label)
    if not normalised:
        return False
    return normalised in {"paperset", "papercode", "setcode"} or (
        "paper" in normalised and ("set" in normalised or "code" in normalised)
    )


def _find_first_field(
    columns: List[str],
    rows: List[Dict[str, Any]],
    predicate,
) -> Tuple[Optional[str], Optional[str]]:
    ordered_columns = list(columns)
    seen = set(ordered_columns)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                ordered_columns.append(key)
                seen.add(key)

    for column in ordered_columns:
        if not predicate(str(column)):
            continue
        for row in rows:
            value = _cell_text(row.get(column))
            if value:
                return str(column), value
    return None, None


def _text_matches(actual: str, expected: str) -> bool:
    actual_norm = _normalise_compare(actual)
    expected_norm = _normalise_compare(expected)
    if not actual_norm or not expected_norm:
        return False
    if actual_norm == expected_norm:
        return True
    short, long = sorted((actual_norm, expected_norm), key=len)
    if len(short) >= max(4, int(len(long) * 0.75)) and short in long:
        return True
    return SequenceMatcher(None, actual_norm, expected_norm).ratio() >= 0.78


def _question_number_from_label(label: Any) -> Optional[int]:
    text = str(label or "").strip().lower()
    if not text:
        return None
    normalised = _normalise_compare(text)
    if any(blocked in normalised for blocked in ("roll", "name", "paper", "total", "class", "subject")):
        return None

    for pattern in (
        r"\bq(?:uestion)?\s*0*(\d{1,3})\b",
        r"\bquestion\s*0*(\d{1,3})\b",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    match = re.fullmatch(r"q0*(\d{1,3})", normalised)
    if match:
        return int(match.group(1))
    if re.fullmatch(r"\d{1,3}", normalised):
        return int(normalised)
    return None


def _parse_mark_value(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in {"-", "--", "na", "n/a", "absent", "ab"}:
        return None
    match = re.search(r"(?<!\d)-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _format_question_ranges(question_numbers: List[int]) -> str:
    unique = sorted({int(number) for number in question_numbers if int(number) > 0})
    if not unique:
        return ""

    ranges: List[str] = []
    start = unique[0]
    previous = unique[0]
    for number in unique[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(f"Q{start}" if start == previous else f"Q{start}-Q{previous}")
        start = previous = number
    ranges.append(f"Q{start}" if start == previous else f"Q{start}-Q{previous}")
    return ", ".join(ranges)


def _configured_question_count(document: TallyDocumentContext) -> Optional[int]:
    counts: List[int] = []
    if document.num_questions and document.num_questions > 0:
        counts.append(int(document.num_questions))
    for item in document.marking_scheme:
        if item.to > 0:
            counts.append(int(item.to))
    return max(counts) if counts else None


def _expected_marks_for_question(document: TallyDocumentContext, question_number: int) -> Optional[float]:
    for item in document.marking_scheme:
        if item.from_ <= question_number <= item.to and item.marks > 0:
            return float(item.marks)
    if document.marking_scheme:
        return None
    if document.max_marks_per_question and document.max_marks_per_question > 0:
        return float(document.max_marks_per_question)
    return None


def _configured_question_numbers(document: TallyDocumentContext) -> List[int]:
    max_question = _configured_question_count(document)
    if max_question and max_question > 0:
        return list(range(1, max_question + 1))

    question_numbers = [
        int(item.question_number)
        for item in document.question_map
        if item.question_number and item.question_number > 0
    ]
    return sorted(set(question_numbers))


def _format_marking_context(document: TallyDocumentContext) -> str:
    ranges: List[str] = []
    for item in document.marking_scheme:
        if item.from_ > 0 and item.to >= item.from_ and item.marks > 0:
            label = f"Q{item.from_}" if item.from_ == item.to else f"Q{item.from_}-Q{item.to}"
            ranges.append(f"{label}: 0 to {_format_marks(float(item.marks))}")

    if ranges:
        return "; ".join(ranges)
    if document.max_marks_per_question and document.max_marks_per_question > 0:
        return f"Each question: 0 to {_format_marks(float(document.max_marks_per_question))}"
    return "Use visible numeric marks; preserve blank only when no mark is written."


def _format_ocr_question_context(document: TallyDocumentContext) -> str:
    question_numbers = _configured_question_numbers(document)
    if not question_numbers:
        return "Question count is not configured. Read every visible Q column in the marks grid."

    question_range = _format_question_ranges(question_numbers)
    return (
        f"Configured question cells: {question_range}. "
        f"Allowed mark ranges: {_format_marking_context(document)}."
    )


def _question_column_by_number(
    columns: List[str],
    rows: List[Dict[str, Any]],
) -> Dict[int, str]:
    question_column_by_number: Dict[int, str] = {}
    seen_columns = set()
    for column in columns:
        question_number = _question_number_from_label(column)
        if question_number is not None and column not in seen_columns:
            question_column_by_number.setdefault(question_number, column)
            seen_columns.add(column)

    for row in rows:
        for column in row.keys():
            if column in seen_columns:
                continue
            question_number = _question_number_from_label(column)
            if question_number is not None:
                question_column_by_number.setdefault(question_number, column)
                seen_columns.add(column)

    return question_column_by_number


def _missing_questions_by_row(
    columns: List[str],
    rows: List[Dict[str, Any]],
    document: TallyDocumentContext,
) -> Dict[int, List[int]]:
    question_numbers = _configured_question_numbers(document)
    if not question_numbers or not rows:
        return {}

    question_columns = _question_column_by_number(columns, rows)
    missing_by_row: Dict[int, List[int]] = {}
    for row_index, row in enumerate(rows):
        missing = [
            question_number
            for question_number in question_numbers
            if not _cell_text(row.get(question_columns.get(question_number, "")))
        ]
        if missing:
            missing_by_row[row_index] = missing

    return missing_by_row


def _validate_tally_result(
    columns: List[str],
    rows: List[Dict[str, Any]],
    document: TallyDocumentContext,
    student: TallyStudentContext,
) -> List[TallyValidationIssue]:
    issues: List[TallyValidationIssue] = []

    if student.name:
        name_column, extracted_name = _find_first_field(columns, rows, _looks_like_student_name_label)
        if extracted_name:
            if not _text_matches(extracted_name, student.name):
                issues.append(
                    TallyValidationIssue(
                        severity="error",
                        code="student_name_mismatch",
                        message=f"Student name mismatch: selected {student.name}, sheet shows {extracted_name}.",
                        column=name_column,
                        expected=student.name,
                        actual=extracted_name,
                    )
                )
        else:
            issues.append(
                TallyValidationIssue(
                    severity="warning",
                    code="student_name_missing",
                    message="Student name was not confidently detected on the tally sheet.",
                    expected=student.name,
                )
            )

    if document.validate_paper_set:
        expected_paper_set = (document.expected_paper_set or "").strip()
        paper_column, extracted_paper_set = _find_first_field(columns, rows, _looks_like_paper_set_label)
        if expected_paper_set and extracted_paper_set:
            if not _text_matches(extracted_paper_set, expected_paper_set):
                issues.append(
                    TallyValidationIssue(
                        severity="error",
                        code="paper_set_mismatch",
                        message=(
                            f"Paper set/code mismatch: expected {expected_paper_set}, "
                            f"sheet shows {extracted_paper_set}."
                        ),
                        column=paper_column,
                        expected=expected_paper_set,
                        actual=extracted_paper_set,
                    )
                )
        elif expected_paper_set:
            issues.append(
                TallyValidationIssue(
                    severity="error",
                    code="paper_set_missing",
                    message="Paper set/code was not confidently detected on the tally sheet.",
                    column=paper_column,
                    expected=expected_paper_set,
                )
            )
        else:
            issues.append(
                TallyValidationIssue(
                    severity="warning",
                    code="paper_set_expected_missing",
                    message="Paper set/code validation is enabled but no expected value was configured.",
                )
            )

    max_question = _configured_question_count(document)
    question_columns: List[Tuple[str, int]] = []
    question_column_by_number: Dict[int, str] = {}
    seen_columns = set()
    for column in columns:
        question_number = _question_number_from_label(column)
        if question_number is not None and column not in seen_columns:
            question_columns.append((column, question_number))
            question_column_by_number.setdefault(question_number, column)
            seen_columns.add(column)

    for row in rows:
        for column in row.keys():
            if column in seen_columns:
                continue
            question_number = _question_number_from_label(column)
            if question_number is not None:
                question_columns.append((column, question_number))
                question_column_by_number.setdefault(question_number, column)
                seen_columns.add(column)

    if max_question and not question_columns:
        issues.append(
            TallyValidationIssue(
                severity="error",
                code="question_columns_missing",
                message=f"No question marks were confidently detected. Expected marks for Q1-Q{max_question}.",
                expected=f"Q1-Q{max_question}",
            )
        )

    if max_question and document.marking_scheme:
        covered_questions = {
            question_number
            for item in document.marking_scheme
            for question_number in range(max(1, item.from_), min(max_question, item.to) + 1)
            if item.marks > 0
        }
        missing_config = [
            question_number
            for question_number in range(1, max_question + 1)
            if question_number not in covered_questions
        ]
        if missing_config:
            missing_label = _format_question_ranges(missing_config)
            issues.append(
                TallyValidationIssue(
                    severity="error",
                    code="marking_scheme_incomplete",
                    message=f"No max marks are configured for {missing_label}.",
                    column=missing_label,
                    question_number=missing_config[0],
                    expected=f"Q1-Q{max_question}",
                    actual=f"Missing {missing_label}",
                )
            )

    for row_index, row in enumerate(rows):
        if max_question and question_columns:
            missing_questions: List[int] = []
            for question_number in range(1, max_question + 1):
                column = question_column_by_number.get(question_number)
                if not column or not _cell_text(row.get(column)):
                    missing_questions.append(question_number)

            if missing_questions:
                missing_label = _format_question_ranges(missing_questions)
                issues.append(
                    TallyValidationIssue(
                        severity="error",
                        code="missing_question_marks",
                        message=(
                            f"Marks are missing for {missing_label}. "
                            f"Expected marks for all {max_question} questions."
                        ),
                        row_index=row_index,
                        column=missing_label,
                        question_number=missing_questions[0],
                        expected=f"Q1-Q{max_question}",
                        actual=f"Missing {missing_label}",
                    )
                )

        for column, question_number in question_columns:
            raw_value = _cell_text(row.get(column))
            if not raw_value:
                continue

            if max_question and question_number > max_question:
                issues.append(
                    TallyValidationIssue(
                        severity="error",
                        code="question_out_of_range",
                        message=f"{column} is outside the configured {max_question} question limit.",
                        row_index=row_index,
                        column=column,
                        question_number=question_number,
                        expected=f"Q1-Q{max_question}",
                        actual=column,
                    )
                )
                continue

            mark = _parse_mark_value(raw_value)
            if mark is None:
                issues.append(
                    TallyValidationIssue(
                        severity="warning",
                        code="mark_not_numeric",
                        message=f"{column} value '{raw_value}' is not a readable numeric mark.",
                        row_index=row_index,
                        column=column,
                        question_number=question_number,
                        actual=raw_value,
                    )
                )
                continue

            if mark < 0:
                issues.append(
                    TallyValidationIssue(
                        severity="error",
                        code="negative_mark",
                        message=f"{column} has a negative mark ({_format_marks(mark)}).",
                        row_index=row_index,
                        column=column,
                        question_number=question_number,
                        expected="0 or more",
                        actual=_format_marks(mark),
                    )
                )
                continue

            expected_marks = _expected_marks_for_question(document, question_number)
            if expected_marks is not None and mark > expected_marks + 1e-6:
                issues.append(
                    TallyValidationIssue(
                        severity="error",
                        code="mark_above_max",
                        message=(
                            f"{column} has {_format_marks(mark)} marks, "
                            f"but the configured max is {_format_marks(expected_marks)}."
                        ),
                        row_index=row_index,
                        column=column,
                        question_number=question_number,
                        expected=_format_marks(expected_marks),
                        actual=_format_marks(mark),
                    )
                )

    return issues


def _has_validation_errors(issues: List[Any]) -> bool:
    for issue in issues:
        if isinstance(issue, TallyValidationIssue):
            severity = issue.severity
        elif isinstance(issue, dict):
            severity = issue.get("severity")
        else:
            severity = None
        if str(severity or "").lower() == "error":
            return True
    return False


def _item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, TallyQuestionMapItem):
        return getattr(item, key, default)
    if isinstance(item, dict):
        return item.get(key, default)
    return default


def _question_map_items_for_export(
    document: TallyDocumentContext,
    map_doc: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if document.question_map:
        return _normalise_question_map_items(
            [item.model_dump(exclude_none=True) for item in document.question_map]
        )
    if map_doc:
        return _normalise_question_map_items(map_doc.get("items") or [])
    return []


def _document_max_marks_for_question(
    document: TallyDocumentContext,
    question_number: int,
) -> Optional[float]:
    for item in document.marking_scheme:
        if item.from_ <= question_number <= item.to:
            return float(item.marks)
    return float(document.max_marks_per_question) if document.max_marks_per_question else None


def _question_max_marks(
    item: Dict[str, Any],
    document: TallyDocumentContext,
    question_number: int,
) -> float:
    max_marks = _parse_mark_value(item.get("max_marks"))
    if max_marks is not None:
        return float(max_marks)
    fallback = _document_max_marks_for_question(document, question_number)
    return float(fallback) if fallback is not None else 0.0


async def _load_export_question_map(
    tenant_db: Any,
    document: TallyDocumentContext,
) -> Optional[Dict[str, Any]]:
    document_id = (document.document_id or "").strip()
    if not document_id:
        return None
    return await _load_tally_question_map(tenant_db, document_id)


def _find_question_columns(columns: List[str], rows: List[Dict[str, Any]]) -> Dict[int, str]:
    by_number: Dict[int, str] = {}
    ordered_columns = list(columns)
    seen = set(ordered_columns)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                ordered_columns.append(key)
                seen.add(key)
    for column in ordered_columns:
        question_number = _question_number_from_label(column)
        if question_number is not None:
            by_number.setdefault(question_number, column)
    return by_number


def _student_label(row: Dict[str, Any], row_index: int) -> str:
    for key in ("Selected Student", "NAME", "Name", "Student", "Student Name"):
        value = _cell_text(row.get(key))
        if value:
            return value
    return f"Student {row_index + 1}"


def _percentage(obtained: float, maximum: float) -> Optional[float]:
    if maximum <= 0:
        return None
    return round((obtained / maximum) * 100, 2)


def _format_percentage(value: Optional[float]) -> str:
    return "" if value is None else f"{value:g}%"


def _pick_strengths(stats: Dict[str, Dict[str, float]], overall_pct: Optional[float]) -> Tuple[str, str]:
    scored = [
        (label, _percentage(values.get("obtained", 0.0), values.get("max", 0.0)))
        for label, values in stats.items()
        if values.get("max", 0.0) > 0
    ]
    scored = [(label, pct) for label, pct in scored if pct is not None]
    if not scored:
        return "", ""
    scored.sort(key=lambda item: item[1])
    weak_label, weak_pct = scored[0]
    strong_label, _ = scored[-1]
    weak = ""
    if weak_pct < 60:
        weak = weak_label
    elif overall_pct is not None and weak_pct <= overall_pct - 15:
        weak = weak_label
    return weak, strong_label


def _build_analysis_rows(
    rows: List[Dict[str, Any]],
    ordered_columns: List[str],
    question_map_items: List[Dict[str, Any]],
    document: TallyDocumentContext,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not question_map_items:
        return [], [], [], []

    question_columns = _find_question_columns(ordered_columns, rows)
    map_by_number = {
        int(item["question_number"]): item
        for item in question_map_items
        if item.get("question_number")
    }
    summary_rows: List[Dict[str, Any]] = []
    topic_rows: List[Dict[str, Any]] = []
    class_topic_rows: List[Dict[str, Any]] = []
    question_rows: List[Dict[str, Any]] = []
    class_topic_stats: Dict[str, Dict[str, Any]] = {}
    class_label = str(document.standard or "").strip()
    section_label = str(document.section or "").strip()
    subject_label = str(document.subject or "").strip()

    for row_index, row in enumerate(rows):
        student = _student_label(row, row_index)
        student_id = _cell_text(row.get("Selected Student ID"))
        student_key = student_id or student
        total_obtained = 0.0
        total_max = 0.0
        topic_stats: Dict[str, Dict[str, float]] = {}

        for question_number in sorted(map_by_number.keys()):
            item = map_by_number[question_number]
            column = question_columns.get(question_number) or f"Q{question_number}"
            raw_value = row.get(column, "")
            mark = _parse_mark_value(raw_value)
            obtained = float(mark) if mark is not None else 0.0
            max_marks = _question_max_marks(item, document, question_number)
            topic = str(item.get("sub_topic") or "Unmapped").strip() or "Unmapped"

            total_obtained += obtained
            total_max += max_marks
            topic_bucket = topic_stats.setdefault(topic, {"obtained": 0.0, "max": 0.0, "questions": 0})
            topic_bucket["obtained"] += obtained
            topic_bucket["max"] += max_marks
            topic_bucket["questions"] += 1
            class_bucket = class_topic_stats.setdefault(
                topic,
                {"obtained": 0.0, "max": 0.0, "questions": 0, "students": set()},
            )
            class_bucket["obtained"] += obtained
            class_bucket["max"] += max_marks
            class_bucket["questions"] += 1
            class_bucket["students"].add(student_key)

            question_rows.append(
                {
                    "Student": student,
                    "Student ID": student_id,
                    "Class": class_label,
                    "Section": section_label,
                    "Subject": subject_label,
                    "Question": f"Q{question_number}",
                    "Marks Obtained": obtained,
                    "Max Marks": max_marks,
                    "Percentage": _format_percentage(_percentage(obtained, max_marks)),
                    "Sub-topic": topic,
                }
            )

        overall_pct = _percentage(total_obtained, total_max)
        weak_topic, strong_topic = _pick_strengths(topic_stats, overall_pct)
        summary_rows.append(
            {
                "Student": student,
                "Student ID": student_id,
                "Class": class_label,
                "Section": section_label,
                "Subject": subject_label,
                "Total Obtained": round(total_obtained, 2),
                "Total Max": round(total_max, 2),
                "Percentage": _format_percentage(overall_pct),
                "Weak Sub-topic": weak_topic,
                "Strong Sub-topic": strong_topic,
            }
        )

        for topic, values in sorted(topic_stats.items()):
            topic_rows.append(
                {
                    "Student": student,
                    "Student ID": student_id,
                    "Class": class_label,
                    "Section": section_label,
                    "Subject": subject_label,
                    "Sub-topic": topic,
                    "Marks Obtained": round(values["obtained"], 2),
                    "Max Marks": round(values["max"], 2),
                    "Percentage": _format_percentage(_percentage(values["obtained"], values["max"])),
                    "Questions": int(values["questions"]),
                }
            )

    for topic, values in sorted(class_topic_stats.items()):
        percentage = _percentage(values["obtained"], values["max"])
        if percentage is None:
            status_label = ""
        elif percentage < 60:
            status_label = "Needs attention"
        elif percentage >= 80:
            status_label = "Strong"
        else:
            status_label = "Developing"
        class_topic_rows.append(
            {
                "Class": class_label,
                "Section": section_label,
                "Subject": subject_label,
                "Sub-topic": topic,
                "Students": len(values["students"]),
                "Marks Obtained": round(values["obtained"], 2),
                "Max Marks": round(values["max"], 2),
                "Percentage": _format_percentage(percentage),
                "Question Attempts": int(values["questions"]),
                "Class Status": status_label,
            }
        )

    return summary_rows, topic_rows, class_topic_rows, question_rows


def _build_prompt(payload: TallyExtractRequest) -> str:
    document = payload.document or TallyDocumentContext()
    student = payload.student or TallyStudentContext()
    marking_scheme = _format_marking_scheme(document.marking_scheme)
    marking_rule = (
        f"\n9. Use this marking scheme for validation only: {marking_scheme}. "
        "If a recognized mark is greater than the allowed max for that question, keep the value and add a warning."
        if marking_scheme
        else ""
    )
    question_context = _format_ocr_question_context(document)
    context = {
        "document": document.model_dump(exclude_none=True, by_alias=True),
        "selected_student": student.model_dump(exclude_none=True),
        "copy_id": payload.copy_id,
        "question_context": question_context,
    }
    return f"""
You are reading a full-page handwritten exam tally sheet drawn on a digital canvas.
The sheet may contain hand-drawn table borders and handwritten headings/values.
This is an exam tally marks grid, not a generic spreadsheet. The important cells are the evaluator marks under Q1, Q2, Q3... headings.

Task:
1. Detect the table structure from the image.
2. Read all headings exactly as intended, including labels like NAME, ROLL NO., PAPER SET, Q1, Q01, Q2, TOTAL, MAX MARKS.
3. Pair each value with the correct heading/cell.
4. If the sheet is a single-student form, return one row.
5. If the sheet has multiple student rows, return all rows.
6. Preserve blank cells as empty strings.
7. Normalize question headings to Q1, Q2, Q3... where obvious.
8. Do not invent marks or names. If uncertain, keep the cell empty and add a warning.{marking_rule}

Question mark grid rules:
- {question_context}
- Read values by the physical cell position below each Q heading. Do not shift marks left or right just because an earlier cell is hard to read.
- A single black vertical handwritten stroke inside a question mark cell is a valid mark of "1", even when it is close to a blue printed grid line.
- Blue table borders and printed labels are not marks. Black handwritten strokes inside the white mark area are marks.
- Valid mark cells normally contain small values like 0, 1, 2, 0.5, or blanks. Preserve the exact numeric value as a string.
- Return blank only when the cell truly has no handwritten mark.

Context from the UI, for disambiguation only:
{json.dumps(context, ensure_ascii=False)}

Return ONLY strict JSON in this shape:
{{
  "columns": ["NAME", "ROLL NO.", "PAPER SET", "Q1", "Q2"],
  "rows": [
    {{"NAME": "", "ROLL NO.": "", "PAPER SET": "", "Q1": "", "Q2": ""}}
  ],
  "cell_confidence": {{"Q1": 0.0, "Q2": 0.0}},
  "uncertain_cells": ["Q1"],
  "warnings": [],
  "confidence": 0.0
}}
""".strip()


def _build_missing_marks_recheck_prompt(
    payload: TallyExtractRequest,
    missing_by_row: Dict[int, List[int]],
) -> str:
    document = payload.document or TallyDocumentContext()
    student = payload.student or TallyStudentContext()
    targets = [
        {
            "row_index": row_index,
            "questions": [f"Q{number}" for number in question_numbers],
        }
        for row_index, question_numbers in sorted(missing_by_row.items())
    ]
    context = {
        "selected_student": student.model_dump(exclude_none=True),
        "question_context": _format_ocr_question_context(document),
        "targets": targets,
    }
    return f"""
You are doing a second OCR pass on the same exam tally sheet.
The first pass left some question mark cells blank. Only inspect the target cells listed below.

Critical reading rules:
- Read the physical mark cell below each target Q heading in the evaluator marks grid.
- A single black vertical handwritten stroke inside the white mark area is the numeric mark "1".
- Do not confuse black handwritten "1" marks with blue printed table borders.
- Do not shift values from neighboring cells. Each value must stay with its own Q heading.
- If a target cell contains a visible numeric mark, return that numeric mark as a string.
- If a target cell is genuinely empty or unreadable after careful inspection, return an empty string.
- Do not fill cells that are not listed in targets.

Context:
{json.dumps(context, ensure_ascii=False)}

Return ONLY strict JSON in this shape:
{{
  "rows": [
    {{"row_index": 0, "values": {{"Q1": "1", "Q2": ""}}}}
  ],
  "warnings": [],
  "confidence": 0.0
}}
""".strip()


def _coerce_recheck_row_index(
    row: Dict[str, Any],
    fallback_index: int,
    target_rows: List[int],
) -> int:
    for key in ("row_index", "row"):
        if key in row:
            try:
                return int(row.get(key))
            except (TypeError, ValueError):
                break

    if "row_number" in row:
        try:
            row_number = int(row.get("row_number"))
            return max(0, row_number - 1)
        except (TypeError, ValueError):
            pass

    if len(target_rows) == 1:
        return target_rows[0]
    return fallback_index


def _parse_rechecked_marks(
    parsed: Dict[str, Any],
    missing_by_row: Dict[int, List[int]],
) -> Tuple[Dict[int, Dict[int, Any]], List[str], Optional[float]]:
    target_rows = sorted(missing_by_row.keys())
    rechecked: Dict[int, Dict[int, Any]] = {}

    def collect_values(row_index: int, values: Dict[str, Any]) -> None:
        target_questions = set(missing_by_row.get(row_index, []))
        if not target_questions:
            return
        for key, value in values.items():
            question_number = _question_number_from_label(key)
            if question_number in target_questions:
                rechecked.setdefault(row_index, {})[question_number] = value

    raw_rows = parsed.get("rows")
    if isinstance(raw_rows, list):
        for fallback_index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, dict):
                continue
            row_index = _coerce_recheck_row_index(raw_row, fallback_index, target_rows)
            raw_values = raw_row.get("values")
            values = raw_values if isinstance(raw_values, dict) else raw_row
            collect_values(row_index, values)
    else:
        raw_values = parsed.get("values")
        values = raw_values if isinstance(raw_values, dict) else parsed
        if isinstance(values, dict) and target_rows:
            collect_values(target_rows[0], values)

    warnings = parsed.get("warnings") if isinstance(parsed.get("warnings"), list) else []
    warnings = [str(warning) for warning in warnings if str(warning).strip()]
    confidence = parsed.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    return rechecked, warnings, confidence


def _format_filled_recheck_ranges(filled_by_row: Dict[int, List[int]]) -> str:
    parts = []
    for row_index, question_numbers in sorted(filled_by_row.items()):
        label = _format_question_ranges(question_numbers)
        if label:
            parts.append(f"row {row_index + 1}: {label}")
    return "; ".join(parts)


def _merge_rechecked_marks(
    columns: List[str],
    rows: List[Dict[str, Any]],
    rechecked: Dict[int, Dict[int, Any]],
    missing_by_row: Dict[int, List[int]],
    document: TallyDocumentContext,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[int, List[int]], List[str]]:
    next_columns = list(columns)
    next_rows = [dict(row) for row in rows]
    question_columns = _question_column_by_number(next_columns, next_rows)
    filled_by_row: Dict[int, List[int]] = {}
    warnings: List[str] = []

    for row_index, values in sorted(rechecked.items()):
        if row_index < 0 or row_index >= len(next_rows):
            continue
        row = next_rows[row_index]
        target_questions = set(missing_by_row.get(row_index, []))
        if not target_questions:
            continue

        for question_number, raw_value in sorted(values.items()):
            if question_number not in target_questions:
                continue
            raw_text = _cell_text(raw_value)
            if not raw_text:
                continue
            mark = _parse_mark_value(raw_text)
            if mark is None:
                warnings.append(
                    f"OCR recheck saw Q{question_number} as '{raw_text}', but it was not numeric."
                )
                continue
            if mark < 0:
                warnings.append(
                    f"OCR recheck saw Q{question_number} as {_format_marks(mark)}, but negative marks are invalid."
                )
                continue

            expected_marks = _expected_marks_for_question(document, question_number)
            if expected_marks is not None and mark > expected_marks + 1e-6:
                warnings.append(
                    "OCR recheck saw "
                    f"Q{question_number} as {_format_marks(mark)}, "
                    f"above configured max {_format_marks(expected_marks)}."
                )
                continue

            column = question_columns.get(question_number)
            if not column:
                column = f"Q{question_number}"
                question_columns[question_number] = column
                if column not in next_columns:
                    next_columns.append(column)

            if _cell_text(row.get(column)):
                continue

            row[column] = _format_marks(mark)
            filled_by_row.setdefault(row_index, []).append(question_number)

    return next_columns, next_rows, filled_by_row, warnings


def _safe_filename(value: Optional[str], fallback: str) -> str:
    base = (value or fallback).strip() or fallback
    base = re.sub(r"[^a-zA-Z0-9_.-]+", "-", base).strip("-")
    return base or fallback


def _safe_document_id(value: str) -> str:
    document_id = (value or "").strip()
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id is required")
    if len(document_id) > 240 or not re.fullmatch(r"[A-Za-z0-9_.:-]+", document_id):
        raise HTTPException(status_code=400, detail="Invalid document_id")
    return document_id


def _question_sort_key(question: Dict[str, Any]) -> Tuple[int, int, str]:
    try:
        explicit_order = int(question.get("extraction_order") or question.get("question_number") or 0)
    except (TypeError, ValueError):
        explicit_order = 0
    if explicit_order > 0:
        return (0, explicit_order, str(question.get("id") or ""))
    return (
        1,
        int(question.get("page_number") or 0),
        str(question.get("id") or ""),
    )


def _extract_leading_int(value: Any) -> Optional[int]:
    text = str(value or "").strip()
    digits = ""
    for char in text:
        if char.isdigit():
            digits += char
        elif digits:
            break
    if not digits:
        return None
    try:
        parsed = int(digits)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _preview_question_number(question: Any, fallback: int) -> int:
    metadata = getattr(question, "metadata", None)
    if not isinstance(metadata, dict) and isinstance(question, dict):
        metadata = question.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    for value in (
        metadata.get("question_number"),
        getattr(question, "question_number", None),
        question.get("question_number") if isinstance(question, dict) else None,
        fallback,
    ):
        parsed = _extract_leading_int(value)
        if parsed is not None:
            return parsed
    return fallback


def _preview_question_text(question: Any) -> str:
    if isinstance(question, dict):
        return str(question.get("text") or question.get("question_text") or "").strip()
    return str(getattr(question, "text", "") or "").strip()


def _preview_question_type(question: Any) -> str:
    if isinstance(question, dict):
        options = question.get("options") or []
        raw_type = question.get("question_type")
    else:
        options = getattr(question, "options", []) or []
        raw_type = getattr(question, "question_type", None)
    if raw_type:
        return str(raw_type).strip()[:40] or "other"
    return "objective" if isinstance(options, list) and len(options) >= 2 else "subjective"


def _preview_marking_scheme(
    questions: List[Any],
) -> Tuple[List[TallyMarkingRange], List[TallyQuestionSourcePreviewItem], List[str]]:
    warnings: List[str] = []
    items: List[TallyQuestionSourcePreviewItem] = []
    defaulted_marks = 0

    for index, question in enumerate(questions or [], start=1):
        question_number = _preview_question_number(question, index)
        metadata = getattr(question, "metadata", None)
        if not isinstance(metadata, dict) and isinstance(question, dict):
            metadata = question.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        mark = _parse_mark_value(getattr(question, "points", None) if not isinstance(question, dict) else question.get("points"))
        mark_source = "paper" if metadata.get("max_marks_extracted") and mark is not None else "default"
        if mark_source == "default":
            mark = 1.0
            defaulted_marks += 1
        text = _preview_question_text(question)
        items.append(
            TallyQuestionSourcePreviewItem(
                question_number=question_number,
                max_marks=mark,
                max_marks_source=mark_source,
                question_type=_preview_question_type(question),
                text_preview=text[:180],
            )
        )

    items.sort(key=lambda item: item.question_number)
    if not items:
        return [], [], ["No questions could be extracted from the uploaded paper."]

    if defaulted_marks:
        warnings.append(
            f"Marks were not printed clearly for {defaulted_marks} question(s); defaulted those questions to 1 mark."
        )

    ranges: List[TallyMarkingRange] = []
    start = items[0].question_number
    end = start
    current_marks = float(items[0].max_marks or 1)
    for item in items[1:]:
        item_marks = float(item.max_marks or 1)
        if item.question_number == end + 1 and item_marks == current_marks:
            end = item.question_number
            continue
        ranges.append(
            TallyMarkingRange.model_validate(
                {"from": start, "to": end, "marks": current_marks}
            )
        )
        start = item.question_number
        end = item.question_number
        current_marks = item_marks
    ranges.append(
        TallyMarkingRange.model_validate(
            {"from": start, "to": end, "marks": current_marks}
        )
    )
    return ranges, items, warnings


def _normalise_question_map_items(items: List[Any]) -> List[Dict[str, Any]]:
    normalised: List[Dict[str, Any]] = []
    for item in items or []:
        if isinstance(item, TallyQuestionMapItem):
            data = item.model_dump(exclude_none=True)
        elif isinstance(item, dict):
            data = dict(item)
        else:
            continue
        try:
            question_number = int(data.get("question_number"))
        except (TypeError, ValueError):
            continue
        if question_number <= 0:
            continue
        max_marks = _parse_mark_value(data.get("max_marks"))
        confidence = data.get("confidence")
        try:
            confidence_value = max(0.0, min(1.0, float(confidence))) if confidence is not None else None
        except (TypeError, ValueError):
            confidence_value = None
        normalised.append(
            {
                "question_number": question_number,
                "question_id": str(data.get("question_id") or ""),
                "question_text": str(data.get("question_text") or ""),
                "question_text_preview": str(data.get("question_text_preview") or data.get("question_text") or "")[:240],
                "sub_topic": str(data.get("sub_topic") or "General").strip()[:80] or "General",
                "question_type": str(data.get("question_type") or "other").strip()[:40] or "other",
                "max_marks": max_marks,
                "confidence": confidence_value,
                "source": str(data.get("source") or "manual").strip()[:40] or "manual",
            }
        )
    return sorted(normalised, key=lambda item: item["question_number"])


def _question_map_response(
    tally_document_id: str,
    doc: Optional[Dict[str, Any]],
) -> TallyQuestionMapResponse:
    if not doc:
        return TallyQuestionMapResponse(
            success=True,
            tally_document_id=tally_document_id,
            status="none",
            items=[],
        )
    return TallyQuestionMapResponse(
        success=True,
        tally_document_id=tally_document_id,
        source_document_id=doc.get("source_document_id"),
        status=doc.get("status") or "none",
        items=[
            TallyQuestionMapItem(**item)
            for item in _normalise_question_map_items(doc.get("items") or [])
        ],
        warnings=[str(w) for w in doc.get("warnings") or [] if str(w).strip()],
        generated_at=doc.get("generated_at"),
        updated_at=doc.get("updated_at"),
    )


async def _load_tally_question_map(
    tenant_db: Any,
    tally_document_id: str,
) -> Optional[Dict[str, Any]]:
    return await tenant_db["exam_tally_question_maps"].find_one(
        {"tally_document_id": tally_document_id}
    )


async def _build_and_store_tally_question_map(
    tenant_db: Any,
    *,
    tally_document: Dict[str, Any],
    source_document_id: str,
    current_user: Dict[str, Any],
) -> Dict[str, Any]:
    tally_document_id = str(tally_document.get("document_id") or "")
    source_document = await tenant_db["documents"].find_one({"document_id": source_document_id})
    if not source_document:
        raise HTTPException(status_code=404, detail="Question source document not found")

    questions = await tenant_db["questions"].find(
        {"document_id": source_document_id}
    ).to_list(length=500)
    questions = sorted(questions, key=_question_sort_key)

    marking_scheme = [
        {"from": item.from_, "to": item.to, "marks": item.marks}
        for item in (tally_document.get("tally_marking_scheme") or [])
        if isinstance(item, TallyMarkingRange)
    ]
    if not marking_scheme:
        marking_scheme = [
            {
                "from": item.get("from"),
                "to": item.get("to"),
                "marks": item.get("marks"),
            }
            for item in (tally_document.get("tally_marking_scheme") or [])
            if isinstance(item, dict)
        ]

    map_doc = await build_tally_question_map(
        tally_document_id=tally_document_id,
        source_document_id=source_document_id,
        questions=questions,
        subject=tally_document.get("subject") or source_document.get("subject"),
        standard=tally_document.get("standard") or source_document.get("standard"),
        course_plan=tally_document.get("course_plan") or source_document.get("course_plan"),
        marking_scheme=marking_scheme,
        fallback_max_marks=tally_document.get("tally_max_marks_per_question"),
        generated_by=current_user.get("user_id"),
    )
    await tenant_db["exam_tally_question_maps"].update_one(
        {"tally_document_id": tally_document_id},
        {"$set": map_doc},
        upsert=True,
    )
    await tenant_db["documents"].update_one(
        {"document_id": tally_document_id},
        {
            "$set": {
                "tally_question_source_mode": (
                    "upload" if source_document_id == tally_document_id else "existing"
                ),
                "tally_question_source_document_id": source_document_id,
                "tally_question_map_status": map_doc.get("status"),
                "tally_question_map_updated_at": map_doc.get("updated_at"),
            }
        },
    )
    return map_doc


@router.post("/question-source/preview", response_model=TallyQuestionSourcePreviewResponse)
async def preview_tally_question_source(
    file: UploadFile = File(...),
    subject: Optional[str] = Form(None),
    difficulty: Optional[str] = Form("medium"),
    current_user: Dict[str, Any] = Depends(_require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    filename = file.filename or "question-paper.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Question source must be a PDF")

    file_content = await file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="Question source PDF is empty")
    if len(file_content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Question source PDF must be 25 MB or smaller")

    try:
        from api.v1.pdf_async import (
            _augment_ocr_with_pymupdf,
            _build_ai_gateway_context,
            _count_pdf_pages,
            call_sarvam_ocr,
            extract_questions_with_gpt,
            is_b2c_admin,
        )
        from services.document_layout_provider import DocumentLayoutProvider, compact_layout_context

        preview_document_id = f"tally-preview-{uuid4().hex}"
        layout_report = await DocumentLayoutProvider().analyze(
            pdf_bytes=file_content,
            document_id=preview_document_id,
            mode="question_paper",
        )
        layout_context = compact_layout_context(layout_report)
        page_count = int(layout_report.get("page_count") or _count_pdf_pages(file_content) or 1)
        gateway_context = _build_ai_gateway_context(
            current_user=current_user,
            db=db,
            document_id=preview_document_id,
            region_scope="tally_question_source_preview",
            is_b2c=is_b2c_admin(current_user),
        )
        ocr_result = await call_sarvam_ocr(
            file_content,
            gateway_context=gateway_context,
            page_count=page_count,
        )
        await asyncio.get_event_loop().run_in_executor(
            None,
            _augment_ocr_with_pymupdf,
            ocr_result,
            file_content,
        )
        questions = await extract_questions_with_gpt(
            ocr_result,
            subject or "General",
            difficulty or "medium",
            skip_option_extraction=True,
            gateway_context=gateway_context,
            layout_report=layout_context,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Tally question source preview failed")
        raise HTTPException(
            status_code=502,
            detail=f"Could not analyze question source PDF: {exc}",
        ) from exc

    marking_scheme, items, warnings = _preview_marking_scheme(questions)
    question_count = max([item.question_number for item in items], default=0)
    return TallyQuestionSourcePreviewResponse(
        success=True,
        question_count=question_count,
        marking_scheme=marking_scheme,
        items=items,
        warnings=warnings,
    )


@router.get("/templates", response_model=TallyTemplateListResponse)
async def list_tally_templates(
    current_user: Dict[str, Any] = Depends(_require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    tenant_db = await _tenant_db(db, current_user)
    template_docs = await tenant_db["exam_tally_templates"].find(
        {"image_b64": {"$exists": True, "$ne": ""}},
        {
            "document_id": 1,
            "image_b64": 1,
            "width": 1,
            "height": 1,
            "updated_at": 1,
            "updated_by": 1,
        },
    ).sort("updated_at", -1).to_list(length=200)

    document_ids = [doc.get("document_id") for doc in template_docs if doc.get("document_id")]
    document_map: Dict[str, Dict[str, Any]] = {}
    if document_ids:
        documents = await tenant_db["documents"].find(
            {"document_id": {"$in": document_ids}},
            {
                "document_id": 1,
                "title": 1,
                "subject": 1,
                "standard": 1,
                "section": 1,
            },
        ).to_list(length=len(document_ids))
        document_map = {str(doc.get("document_id")): doc for doc in documents if doc.get("document_id")}

    summaries: List[TallyTemplateSummary] = []
    for template in template_docs:
        doc_id = str(template.get("document_id") or "")
        if not doc_id:
            continue
        source_doc = document_map.get(doc_id, {})
        summaries.append(
            TallyTemplateSummary(
                document_id=doc_id,
                title=source_doc.get("title") or doc_id,
                subject=source_doc.get("subject"),
                standard=source_doc.get("standard"),
                section=source_doc.get("section"),
                image_b64=template.get("image_b64"),
                width=template.get("width"),
                height=template.get("height"),
                updated_at=template.get("updated_at"),
                updated_by=template.get("updated_by"),
            )
        )

    return TallyTemplateListResponse(success=True, templates=summaries)


@router.get("/templates/{document_id}", response_model=TallyTemplateResponse)
async def get_tally_template(
    document_id: str,
    current_user: Dict[str, Any] = Depends(_require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    safe_document_id = _safe_document_id(document_id)
    tenant_db = await _tenant_db(db, current_user)
    doc = await tenant_db["exam_tally_templates"].find_one({"document_id": safe_document_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Tally template not found")

    return TallyTemplateResponse(
        success=True,
        document_id=safe_document_id,
        image_b64=doc.get("image_b64"),
        template_copy_id=doc.get("template_copy_id"),
        width=doc.get("width"),
        height=doc.get("height"),
        updated_at=doc.get("updated_at"),
        updated_by=doc.get("updated_by"),
    )


@router.put("/templates/{document_id}", response_model=TallyTemplateResponse)
async def save_tally_template(
    document_id: str,
    payload: TallyTemplateSaveRequest,
    current_user: Dict[str, Any] = Depends(_require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    safe_document_id = _safe_document_id(document_id)
    if not payload.image_b64:
        raise HTTPException(status_code=400, detail="image_b64 is required")
    if len(payload.image_b64) > 12 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Template image too large")

    tenant_db = await _tenant_db(db, current_user)
    now = datetime.utcnow()
    update_doc = {
        "document_id": safe_document_id,
        "image_b64": payload.image_b64,
        "template_copy_id": payload.template_copy_id,
        "width": payload.width,
        "height": payload.height,
        "updated_by": current_user.get("user_id"),
        "updated_by_type": current_user.get("user_type"),
        "updated_at": now,
    }
    await tenant_db["exam_tally_templates"].update_one(
        {"document_id": safe_document_id},
        {
            "$set": update_doc,
            "$setOnInsert": {"created_at": now, "created_by": current_user.get("user_id")},
        },
        upsert=True,
    )

    return TallyTemplateResponse(
        success=True,
        document_id=safe_document_id,
        image_b64=payload.image_b64,
        template_copy_id=payload.template_copy_id,
        width=payload.width,
        height=payload.height,
        updated_at=now,
        updated_by=current_user.get("user_id"),
    )


@router.get("/question-map/{document_id}", response_model=TallyQuestionMapResponse)
async def get_tally_question_map(
    document_id: str,
    current_user: Dict[str, Any] = Depends(_require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    tenant_db = await _tenant_db(db, current_user)
    safe_document_id = _safe_document_id(document_id)
    doc = await _load_tally_question_map(tenant_db, safe_document_id)
    return _question_map_response(safe_document_id, doc)


@router.post("/question-map/{document_id}/build", response_model=TallyQuestionMapResponse)
async def build_tally_question_map_endpoint(
    document_id: str,
    payload: TallyQuestionMapBuildRequest,
    current_user: Dict[str, Any] = Depends(_require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    tenant_db = await _tenant_db(db, current_user)
    safe_document_id = _safe_document_id(document_id)
    tally_document = await tenant_db["documents"].find_one({"document_id": safe_document_id})
    if not tally_document:
        raise HTTPException(status_code=404, detail="Tally document not found")

    existing = await _load_tally_question_map(tenant_db, safe_document_id)
    if existing and existing.get("status") == "ready" and not payload.force:
        return _question_map_response(safe_document_id, existing)

    source_document_id = (
        payload.source_document_id
        or tally_document.get("tally_question_source_document_id")
        or safe_document_id
    )
    source_document_id = _safe_document_id(str(source_document_id))
    map_doc = await _build_and_store_tally_question_map(
        tenant_db,
        tally_document=tally_document,
        source_document_id=source_document_id,
        current_user=current_user,
    )
    return _question_map_response(safe_document_id, map_doc)


@router.put("/question-map/{document_id}", response_model=TallyQuestionMapResponse)
async def save_tally_question_map(
    document_id: str,
    payload: TallyQuestionMapSaveRequest,
    current_user: Dict[str, Any] = Depends(_require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    tenant_db = await _tenant_db(db, current_user)
    safe_document_id = _safe_document_id(document_id)
    source_document_id = (
        _safe_document_id(payload.source_document_id)
        if payload.source_document_id
        else None
    )
    now = datetime.utcnow()
    items = _normalise_question_map_items(payload.items)
    map_doc = {
        "_id": safe_document_id,
        "tally_document_id": safe_document_id,
        "source_document_id": source_document_id,
        "status": "ready" if items else "empty",
        "items": items,
        "warnings": [],
        "updated_at": now,
        "updated_by": current_user.get("user_id"),
    }
    await tenant_db["exam_tally_question_maps"].update_one(
        {"tally_document_id": safe_document_id},
        {"$set": map_doc},
        upsert=True,
    )
    await tenant_db["documents"].update_one(
        {"document_id": safe_document_id},
        {
            "$set": {
                "tally_question_source_document_id": source_document_id,
                **(
                    {"tally_question_source_mode": "manual"}
                    if source_document_id is None and items
                    else {}
                ),
                "tally_question_map_status": map_doc["status"],
                "tally_question_map_updated_at": now,
            }
        },
    )
    return _question_map_response(safe_document_id, map_doc)


@router.post("/extract", response_model=TallyExtractResponse)
async def extract_tally(
    request: Request,
    payload: TallyExtractRequest,
    current_user: Dict[str, Any] = Depends(_require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    if not payload.image_b64:
        raise HTTPException(status_code=400, detail="image_b64 is required")
    if len(payload.image_b64) > 12 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Canvas image too large")

    tenant_db = await _tenant_db(db, current_user)
    prompt = _build_prompt(payload)

    try:
        result = await get_ocr_service().analyze_image(
            image_b64=payload.image_b64,
            prompt=prompt,
            tenant_db=tenant_db,
            max_tokens=4096,
        )
    except Exception as exc:
        logger.exception("Exam tally extraction failed")
        raise HTTPException(status_code=502, detail=f"Exam tally OCR failed: {exc}") from exc

    raw_text = result.get("text", "")
    parsed = _parse_llm_json(raw_text)
    columns, rows, warnings, confidence = _normalise_table(parsed)
    document_context = payload.document or TallyDocumentContext()
    student_context = payload.student or TallyStudentContext()
    warnings = _filter_tally_warnings(warnings, document_context)
    if not rows:
        warnings.append("No table rows were confidently detected.")
    validation_issues = _validate_tally_result(columns, rows, document_context, student_context)
    missing_by_row = _missing_questions_by_row(columns, rows, document_context)
    recheck_raw_text: Optional[str] = None
    recheck_provider: Optional[str] = None
    recheck_confidence: Optional[float] = None

    if missing_by_row:
        try:
            recheck_result = await get_ocr_service().analyze_image(
                image_b64=payload.image_b64,
                prompt=_build_missing_marks_recheck_prompt(payload, missing_by_row),
                tenant_db=tenant_db,
                max_tokens=2048,
            )
            recheck_provider = recheck_result.get("provider")
            recheck_raw_text = recheck_result.get("text", "")
            recheck_parsed = _parse_llm_json(recheck_raw_text)
            rechecked, recheck_warnings, recheck_confidence = _parse_rechecked_marks(
                recheck_parsed,
                missing_by_row,
            )
            columns, rows, filled_by_row, merge_warnings = _merge_rechecked_marks(
                columns,
                rows,
                rechecked,
                missing_by_row,
                document_context,
            )

            warnings.extend(recheck_warnings)
            warnings.extend(merge_warnings)
            filled_label = _format_filled_recheck_ranges(filled_by_row)
            if filled_label:
                warnings.append(f"OCR recheck filled missing marks for {filled_label}.")
                validation_issues = _validate_tally_result(
                    columns,
                    rows,
                    document_context,
                    student_context,
                )
            else:
                missing_label = _format_filled_recheck_ranges(missing_by_row)
                if missing_label:
                    warnings.append(
                        f"OCR recheck could not confidently read missing marks for {missing_label}."
                    )
        except Exception as exc:
            logger.warning("Exam tally missing-mark OCR recheck failed: %s", exc)
            warnings.append("OCR recheck for missing marks failed; please review the flagged cells manually.")

    extraction_id = uuid4().hex
    doc = {
        "_id": extraction_id,
        "document": document_context.model_dump(exclude_none=True, by_alias=True),
        "student": student_context.model_dump(exclude_none=True),
        "copy_id": payload.copy_id,
        "columns": columns,
        "rows": rows,
        "warnings": warnings,
        "validation_issues": [
            issue.model_dump(exclude_none=True) for issue in validation_issues
        ],
        "confidence": confidence,
        "raw_text": raw_text,
        "recheck_raw_text": recheck_raw_text,
        "recheck_provider": recheck_provider,
        "recheck_confidence": recheck_confidence,
        "provider": result.get("provider"),
        "created_by": current_user.get("user_id"),
        "created_by_type": current_user.get("user_type"),
        "created_at": datetime.utcnow(),
    }
    await tenant_db["exam_tally_extractions"].insert_one(doc)

    return TallyExtractResponse(
        success=True,
        extraction_id=extraction_id,
        columns=columns,
        rows=rows,
        warnings=warnings,
        validation_issues=validation_issues,
        confidence=confidence,
        raw_text=raw_text,
    )


@router.post("/validate", response_model=TallyValidateResponse)
async def validate_tally(
    payload: TallyValidateRequest,
    current_user: Dict[str, Any] = Depends(_require_admin_or_tutor),
):
    normalised_rows = [_flatten_row(row) for row in payload.rows]
    validation_issues = _validate_tally_result(
        payload.columns,
        normalised_rows,
        payload.document or TallyDocumentContext(),
        payload.student or TallyStudentContext(),
    )
    return TallyValidateResponse(success=True, validation_issues=validation_issues)


@router.post("/export")
async def export_tally(
    payload: TallyExportRequest,
    current_user: Dict[str, Any] = Depends(_require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    tenant_db = await _tenant_db(db, current_user)
    columns = payload.columns
    rows = payload.rows
    validation_issues: List[Any] = []

    if payload.extraction_id and not rows:
        saved = await tenant_db["exam_tally_extractions"].find_one({"_id": payload.extraction_id})
        if not saved:
            raise HTTPException(status_code=404, detail="Extraction not found")
        columns = saved.get("columns") or []
        rows = saved.get("rows") or []
        validation_issues = saved.get("validation_issues") or []

    if not rows:
        raise HTTPException(status_code=400, detail="No rows available to export")

    normalised_rows = [_flatten_row(row) for row in rows]
    if not validation_issues:
        validation_issues = _validate_tally_result(
            columns,
            normalised_rows,
            payload.document or TallyDocumentContext(),
            payload.student or TallyStudentContext(),
        )
    if _has_validation_errors(validation_issues) and not payload.allow_validation_errors:
        raise HTTPException(
            status_code=400,
            detail="Fix exam tally red flags before exporting Excel",
        )

    seen = set()
    ordered_columns: List[str] = []
    for column in columns:
        label = str(column or "").strip()
        if label and label not in seen:
            ordered_columns.append(label)
            seen.add(label)
    for row in normalised_rows:
        for key in row.keys():
            if key not in seen:
                ordered_columns.append(key)
                seen.add(key)

    document_context = payload.document or TallyDocumentContext()
    map_doc = await _load_export_question_map(tenant_db, document_context)
    question_map_items = _question_map_items_for_export(document_context, map_doc)
    question_map_rows = []
    for item in question_map_items:
        question_number = int(item.get("question_number") or 0)
        max_marks = _question_max_marks(item, document_context, question_number)
        question_map_rows.append(
            {
                "Question": f"Q{question_number}",
                "Question ID": item.get("question_id") or "",
                "Sub-topic": item.get("sub_topic") or "Unmapped",
                "Max Marks": max_marks if max_marks > 0 else "",
                "Confidence": item.get("confidence") if item.get("confidence") is not None else "",
                "Source": item.get("source") or "",
                "Question Preview": item.get("question_text_preview") or "",
            }
        )
    summary_rows, topic_rows, class_topic_rows, question_rows = _build_analysis_rows(
        normalised_rows,
        ordered_columns,
        question_map_items,
        document_context,
    )

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df = pd.DataFrame(normalised_rows, columns=ordered_columns)
        df.to_excel(writer, sheet_name="Exam Tally", index=False)
        workbook = writer.book
        worksheet = writer.sheets["Exam Tally"]
        header_format = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        for col_idx, col_name in enumerate(ordered_columns):
            worksheet.write(0, col_idx, col_name, header_format)
            max_len = max([len(str(col_name))] + [len(str(row.get(col_name, ""))) for row in normalised_rows])
            worksheet.set_column(col_idx, col_idx, min(max(max_len + 2, 10), 32))

        def write_sheet(sheet_name: str, sheet_rows: List[Dict[str, Any]]) -> None:
            if not sheet_rows:
                return
            sheet_columns = list(sheet_rows[0].keys())
            sheet_df = pd.DataFrame(sheet_rows, columns=sheet_columns)
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            sheet = writer.sheets[sheet_name]
            for col_idx, col_name in enumerate(sheet_columns):
                sheet.write(0, col_idx, col_name, header_format)
                max_len = max([len(str(col_name))] + [len(str(row.get(col_name, ""))) for row in sheet_rows])
                sheet.set_column(col_idx, col_idx, min(max(max_len + 2, 10), 42))

        write_sheet("Question Map", question_map_rows)
        write_sheet("Student Summary", summary_rows)
        write_sheet("Topic Analysis", topic_rows)
        write_sheet("Class Topic Analysis", class_topic_rows)
        write_sheet("Question Analysis", question_rows)

    output.seek(0)
    title = payload.filename or (payload.document.title if payload.document else None)
    filename = f"{_safe_filename(title, 'exam-tally')}.xlsx"
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
