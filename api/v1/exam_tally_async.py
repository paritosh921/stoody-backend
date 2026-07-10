from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import math
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
from core.upload_security.service import secure_upload
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
    topic: Optional[str] = None
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


class TallyOcrImage(BaseModel):
    label: str
    image_b64: str = Field(..., description="PNG data URL or raw base64")
    description: Optional[str] = None


class TallyExtractRequest(BaseModel):
    image_b64: Optional[str] = Field(None, description="Legacy full-page canvas PNG data URL or raw base64")
    images: List[TallyOcrImage] = Field(default_factory=list)
    document: Optional[TallyDocumentContext] = None
    student: Optional[TallyStudentContext] = None
    copy_id: Optional[str] = None
    debug: bool = False


class TallyValidationIssue(BaseModel):
    severity: str
    code: str
    message: str
    row_index: Optional[int] = None
    column: Optional[str] = None
    question_number: Optional[int] = None
    expected: Optional[str] = None
    actual: Optional[str] = None


class TallyQuestionEvidence(BaseModel):
    row_index: int = 0
    question_number: int
    column: str
    crop_hash: Optional[str] = None
    crop_box: Dict[str, int] = Field(default_factory=dict)


class TallyTargetedRecheck(BaseModel):
    id: str
    row_index: int = 0
    question_number: int
    column: str
    original_value: Optional[str] = None
    configured_max: Optional[str] = None
    candidate_value: Optional[str] = None
    confidence: Optional[float] = None
    status: str
    reason: Optional[str] = None
    crop_hash: Optional[str] = None
    crop_box: Dict[str, int] = Field(default_factory=dict)


class TallyAppliedCorrection(BaseModel):
    row_index: int = 0
    question_number: int
    column: str
    original_ocr_value: Optional[str] = None
    approved_value: Optional[str] = None
    decision: str = "set"
    crop_hash: Optional[str] = None
    evidence_scope: str = "cell"
    source_extraction_id: Optional[str] = None
    targeted_candidate: Optional[str] = None
    reason: Optional[str] = None
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    resolution_source: str = "teacher_override"


class TallyAutoResolvedMark(BaseModel):
    row_index: int = 0
    question_number: int
    column: str
    original_ocr_value: Optional[str] = None
    resolved_value: Optional[str] = None
    crop_hash: Optional[str] = None
    evidence_scope: str = "cell"
    confidence: Optional[float] = None
    reason: Optional[str] = None
    source_extraction_id: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_source: str = "focused_ocr"


class TallyMarkCorrectionSaveRequest(BaseModel):
    source_extraction_id: str
    row_index: int = Field(0, ge=0)
    question_number: int = Field(..., ge=1)
    crop_hash: str = Field(..., min_length=16, max_length=128)
    evidence_scope: str = "cell"
    decision: str = "set"
    approved_value: Optional[float] = None
    reason: Optional[str] = Field(None, max_length=500)


class TallyMarkCorrectionResponse(BaseModel):
    success: bool
    document_id: str
    student_id: str
    revision: int = 0
    corrections: List[TallyAppliedCorrection] = Field(default_factory=list)
    applied_corrections: List[TallyAppliedCorrection] = Field(default_factory=list)
    stale_corrections: List[TallyAppliedCorrection] = Field(default_factory=list)
    auto_resolved_marks: List[TallyAutoResolvedMark] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    validation_issues: List[TallyValidationIssue] = Field(default_factory=list)


class TallyTargetedRecheckDebug(BaseModel):
    id: str
    row_index: int = 0
    question_number: int
    prompt: Optional[str] = None
    raw_text: Optional[str] = None
    provider: Optional[str] = None
    images: List[TallyOcrImage] = Field(default_factory=list)


class TallyExtractDebugResponse(BaseModel):
    prompt: Optional[str] = None
    raw_text: Optional[str] = None
    provider: Optional[str] = None
    recheck_prompt: Optional[str] = None
    recheck_raw_text: Optional[str] = None
    recheck_provider: Optional[str] = None
    image_labels: List[str] = Field(default_factory=list)
    targeted_rechecks: List[TallyTargetedRecheckDebug] = Field(default_factory=list)


class TallyExtractResponse(BaseModel):
    success: bool
    extraction_id: str
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    validation_issues: List[TallyValidationIssue] = Field(default_factory=list)
    confidence: Optional[float] = None
    raw_text: Optional[str] = None
    raw_rows: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_hash: Optional[str] = None
    question_evidence: List[TallyQuestionEvidence] = Field(default_factory=list)
    targeted_rechecks: List[TallyTargetedRecheck] = Field(default_factory=list)
    auto_resolved_marks: List[TallyAutoResolvedMark] = Field(default_factory=list)
    applied_corrections: List[TallyAppliedCorrection] = Field(default_factory=list)
    stale_corrections: List[TallyAppliedCorrection] = Field(default_factory=list)
    debug: Optional[TallyExtractDebugResponse] = None


class TallyExportRequest(BaseModel):
    extraction_id: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    filename: Optional[str] = None
    document: Optional[TallyDocumentContext] = None
    student: Optional[TallyStudentContext] = None
    allow_validation_errors: bool = False
    corrections: List[Dict[str, Any]] = Field(default_factory=list)


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
    topic: Optional[str] = None
    sub_topic: Optional[str] = None


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


async def _assert_tally_context_access(
    tenant_db: Any,
    current_user: Dict[str, Any],
    document: Optional[TallyDocumentContext],
    student: Optional[TallyStudentContext],
) -> None:
    if current_user.get("user_type") != "tutor":
        return

    tutor_id = str(current_user.get("tutor_id") or "")
    user_id = str(current_user.get("user_id") or "")
    if not tutor_id and not user_id:
        raise HTTPException(status_code=403, detail="Tutor identity is missing")

    if document and document.document_id:
        tally_document = await tenant_db["documents"].find_one(
            {"document_id": document.document_id}
        )
        if not tally_document:
            raise HTTPException(status_code=404, detail="Tally document not found")
        teacher_ids = {str(value) for value in (tally_document.get("teacher_ids") or [])}
        if teacher_ids and tutor_id not in teacher_ids and user_id not in teacher_ids:
            raise HTTPException(status_code=403, detail="Tutor is not assigned to this tally document")

    student_id = _tally_student_identity(student)
    if not student_id:
        return
    student_doc = await tenant_db["students"].find_one({"student_id": student_id})
    if not student_doc:
        raise HTTPException(status_code=404, detail="Selected student was not found")
    teacher_ids = {str(value) for value in (student_doc.get("teacher_ids") or [])}
    if tutor_id in teacher_ids or user_id in teacher_ids:
        return

    tutor_doc = await tenant_db["tutors"].find_one({"tutor_id": tutor_id}) if tutor_id else None
    assigned_student_ids = {
        str(value) for value in ((tutor_doc or {}).get("assigned_student_ids") or [])
    }
    # Some tenants scope tutors by class/section rather than explicit mappings;
    # do not reject those legacy records when neither relation is configured.
    if not teacher_ids and not assigned_student_ids:
        return
    if student_id not in assigned_student_ids:
        raise HTTPException(status_code=403, detail="Tutor is not assigned to this student")


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


def _normalise_uncertain_question_cells(
    parsed: Dict[str, Any],
) -> Dict[int, List[int]]:
    uncertain_by_row: Dict[int, List[int]] = {}

    def add(row_index: Any, column: Any) -> None:
        try:
            resolved_row_index = max(0, int(row_index or 0))
        except (TypeError, ValueError):
            resolved_row_index = 0
        question_number = _question_number_from_label(column)
        if question_number is not None and question_number > 0:
            uncertain_by_row.setdefault(resolved_row_index, []).append(question_number)

    raw_uncertain = parsed.get("uncertain_cells") or []
    if isinstance(raw_uncertain, list):
        for item in raw_uncertain:
            if isinstance(item, dict):
                add(item.get("row_index", item.get("row", 0)), item.get("column") or item.get("question"))
            else:
                add(0, item)

    return {
        row_index: sorted(set(question_numbers))
        for row_index, question_numbers in uncertain_by_row.items()
    }


def _uncertain_question_validation_issues(
    uncertain_by_row: Dict[int, List[int]],
    document: TallyDocumentContext,
) -> List[TallyValidationIssue]:
    issues: List[TallyValidationIssue] = []
    for row_index, question_numbers in sorted(uncertain_by_row.items()):
        for question_number in question_numbers:
            expected_max = _expected_marks_for_question(document, question_number)
            issues.append(
                TallyValidationIssue(
                    severity="error",
                    code="ocr_uncertain",
                    message=f"Q{question_number} contains handwriting that OCR could not read unambiguously.",
                    row_index=row_index,
                    column=f"Q{question_number}",
                    question_number=question_number,
                    expected=_format_marks(expected_max) if expected_max is not None else None,
                )
            )
    return issues


def _tally_mark_review_issues(
    validation_issues: List[TallyValidationIssue],
    columns: List[str],
    rows: List[Dict[str, Any]],
    document: TallyDocumentContext,
) -> List[TallyValidationIssue]:
    review_issues = [
        issue
        for issue in validation_issues
        if issue.code in {"mark_above_max", "mark_unreadable", "ocr_uncertain"}
        and issue.question_number is not None
    ]
    for row_index, question_numbers in _missing_questions_by_row(columns, rows, document).items():
        for question_number in question_numbers:
            expected_max = _expected_marks_for_question(document, question_number)
            review_issues.append(
                TallyValidationIssue(
                    severity="error",
                    code="missing_mark",
                    message=f"Q{question_number} did not have a readable OCR mark.",
                    row_index=row_index,
                    column=f"Q{question_number}",
                    question_number=question_number,
                    expected=_format_marks(expected_max) if expected_max is not None else None,
                    actual="",
                )
            )
    return review_issues


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


def _line_clusters_from_counts(
    counts: Any,
    threshold: int,
    *,
    max_gap: int = 3,
    min_size: int = 1,
) -> List[Dict[str, int]]:
    active = [index for index, value in enumerate(counts) if int(value) >= threshold]
    if not active:
        return []

    raw_clusters: List[Tuple[int, int]] = []
    start = previous = active[0]
    for index in active[1:]:
        if index - previous <= max_gap + 1:
            previous = index
            continue
        raw_clusters.append((start, previous))
        start = previous = index
    raw_clusters.append((start, previous))

    clusters: List[Dict[str, int]] = []
    for start, end in raw_clusters:
        if end - start + 1 < min_size:
            continue
        weight_sum = 0
        weighted_position_sum = 0
        max_count = 0
        for index in range(start, end + 1):
            weight = int(counts[index])
            weight_sum += weight
            weighted_position_sum += index * weight
            max_count = max(max_count, weight)
        center = int(round(weighted_position_sum / weight_sum)) if weight_sum else (start + end) // 2
        clusters.append(
            {
                "start": start,
                "end": end,
                "center": center,
                "max_count": max_count,
            }
        )
    return clusters


def _even_line_score(centers: List[int]) -> Optional[float]:
    if len(centers) < 2:
        return None
    gaps = [centers[index + 1] - centers[index] for index in range(len(centers) - 1)]
    if not gaps or any(gap <= 0 for gap in gaps):
        return None
    average = sum(gaps) / len(gaps)
    if average <= 1:
        return None
    return sum(abs(gap - average) for gap in gaps) / len(gaps) / average


def _best_even_line_run(
    clusters: List[Dict[str, int]],
    desired_count: int,
    *,
    min_span: int = 0,
) -> Optional[List[Dict[str, int]]]:
    best_run: Optional[List[Dict[str, int]]] = None
    best_score: Optional[float] = None
    for start in range(0, len(clusters) - desired_count + 1):
        run = clusters[start : start + desired_count]
        centers = [int(item["center"]) for item in run]
        if min_span and centers[-1] - centers[0] < min_span:
            continue
        score = _even_line_score(centers)
        if score is None:
            continue
        if best_score is None or score < best_score:
            best_score = score
            best_run = run
    if best_run is None or best_score is None or best_score > 0.22:
        return None
    return best_run


def _decode_tally_image_for_mark_detection(image_b64: str) -> Optional[Any]:
    try:
        from PIL import Image
    except Exception as exc:
        logger.warning("Pillow unavailable for tally mark detection: %s", exc)
        return None

    try:
        raw = str(image_b64 or "")
        if "," in raw:
            prefix, payload = raw.split(",", 1)
            if "svg+xml" in prefix:
                return None
            raw = payload
        image_bytes = base64.b64decode(raw, validate=False)
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning("Could not decode tally image for mark detection: %s", exc)
        return None


def _detect_tally_question_grid(image: Any, document: TallyDocumentContext) -> Optional[Dict[str, List[int]]]:
    try:
        import numpy as np
    except Exception as exc:
        logger.warning("NumPy unavailable for tally mark detection: %s", exc)
        return None

    arr = np.asarray(image.convert("RGB"))
    height, width = arr.shape[:2]
    if width < 300 or height < 240:
        return None

    rgb = arr.astype("int16")
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    blue_mask = (
        (blue > 80)
        & (blue < 240)
        & (red < 190)
        & (green < 220)
        & ((blue - red) > 25)
    )

    horizontal_clusters = _line_clusters_from_counts(
        blue_mask.sum(axis=1),
        max(60, int(width * 0.35)),
        max_gap=3,
    )
    if len(horizontal_clusters) < 3:
        return None

    question_count = _configured_question_count(document) or 40
    required_bands = min(4, max(1, (int(question_count) + 9) // 10))
    required_lines = 1 + required_bands * 2
    max_window = min(len(horizontal_clusters), max(required_lines, 9))
    window_lengths = sorted(range(required_lines, max_window + 1), reverse=True)

    best: Optional[Tuple[float, List[int], List[int]]] = None
    for window_len in window_lengths:
        for start in range(0, len(horizontal_clusters) - window_len + 1):
            h_run = horizontal_clusters[start : start + window_len]
            h_centers = [int(item["center"]) for item in h_run]
            if h_centers[-1] - h_centers[0] < max(80, int(height * 0.08)):
                continue

            y0 = max(0, h_centers[0] - 2)
            y1 = min(height, h_centers[-1] + 2)
            if y1 <= y0:
                continue

            vertical_counts = blue_mask[y0:y1, :].sum(axis=0)
            vertical_threshold = max(18, int((y1 - y0) * 0.24))
            vertical_clusters = _line_clusters_from_counts(
                vertical_counts,
                vertical_threshold,
                max_gap=3,
            )
            strong_vertical_threshold = max(vertical_threshold, int((y1 - y0) * 0.55))
            strong_vertical_clusters = [
                cluster
                for cluster in vertical_clusters
                if int(cluster.get("max_count") or 0) >= strong_vertical_threshold
            ]
            v_run = _best_even_line_run(
                strong_vertical_clusters,
                11,
                min_span=int(width * 0.45),
            ) or _best_even_line_run(
                vertical_clusters,
                11,
                min_span=int(width * 0.45),
            )
            if not v_run:
                continue

            v_centers = [int(item["center"]) for item in v_run]
            v_score = _even_line_score(v_centers)
            h_score = _even_line_score(h_centers)
            if v_score is None:
                continue
            total_score = (
                v_score
                + (min(h_score or 0.0, 1.0) * 0.25)
                + ((window_len - required_lines) * 0.003)
                + ((h_centers[0] / max(1, height)) * 0.35)
            )
            if best is None or total_score < best[0]:
                best = (total_score, h_centers, v_centers)

    if not best:
        return None

    horizontal = list(best[1])
    vertical = list(best[2])
    channel_range = np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])
    ink_mask = (
        (red < 160)
        & (green < 160)
        & (blue < 160)
        & (channel_range < 100)
    )
    if len(horizontal) >= required_lines + 1 and len(vertical) >= 2:
        x0 = max(0, int(vertical[0]) + 5)
        x1 = min(width, int(vertical[-1]) - 5)
        for index in range(0, len(horizontal) - required_lines + 1):
            label_top = int(horizontal[index]) + 3
            label_bottom = int(horizontal[index + 1]) - 3
            mark_top = int(horizontal[index + 1]) + 3
            mark_bottom = int(horizontal[index + 2]) - 3
            if label_bottom <= label_top or mark_bottom <= mark_top or x1 <= x0:
                continue
            label_area = max(1, (label_bottom - label_top) * (x1 - x0))
            mark_area = max(1, (mark_bottom - mark_top) * (x1 - x0))
            label_ink_ratio = float(ink_mask[label_top:label_bottom, x0:x1].sum()) / label_area
            mark_ink_ratio = float(ink_mask[mark_top:mark_bottom, x0:x1].sum()) / mark_area
            if label_ink_ratio <= 0.004 and mark_ink_ratio >= 0.006:
                horizontal = horizontal[index:]
                break

    return {"horizontal": horizontal, "vertical": vertical}


def _tally_ocr_layer_sources(payload: TallyExtractRequest) -> Dict[str, str]:
    sources: Dict[str, str] = {}
    for image in payload.images:
        label = str(image.label or "").strip().lower().replace("-", "_")
        if not label or not image.image_b64:
            continue
        if label == "template":
            sources.setdefault("template", image.image_b64)
        elif label in {"filled_sheet", "filled", "combined", "final"}:
            sources.setdefault("filled_sheet", image.image_b64)
        elif label in {"strokes_only", "strokes", "ink", "handwriting"}:
            sources.setdefault("strokes_only", image.image_b64)

    if payload.image_b64:
        sources.setdefault("filled_sheet", payload.image_b64)
    return sources


def _decode_tally_ocr_layers(payload: TallyExtractRequest) -> Dict[str, Any]:
    decoded: Dict[str, Any] = {}
    for label, image_b64 in _tally_ocr_layer_sources(payload).items():
        image = _decode_tally_image_for_mark_detection(image_b64)
        if image is not None:
            decoded[label] = image
    return decoded


def _clip_tally_crop_box(
    box: Dict[str, int],
    *,
    width: int,
    height: int,
) -> Optional[Dict[str, int]]:
    left = max(0, min(width, int(box.get("left", 0))))
    top = max(0, min(height, int(box.get("top", 0))))
    right = max(0, min(width, int(box.get("right", 0))))
    bottom = max(0, min(height, int(box.get("bottom", 0))))
    if right <= left or bottom <= top:
        return None
    return {"left": left, "top": top, "right": right, "bottom": bottom}


def _tally_question_crop_boxes(
    grid: Dict[str, List[int]],
    question_number: int,
    *,
    width: int,
    height: int,
) -> Optional[Tuple[Dict[str, int], Dict[str, int]]]:
    horizontal = grid.get("horizontal") or []
    vertical = grid.get("vertical") or []
    if question_number <= 0:
        return None

    band_index = (question_number - 1) // 10
    column_index = (question_number - 1) % 10
    question_top_index = band_index * 2
    mark_top_index = question_top_index + 1
    mark_bottom_index = question_top_index + 2
    if (
        mark_bottom_index >= len(horizontal)
        or column_index + 1 >= len(vertical)
    ):
        return None

    left = int(vertical[column_index])
    right = int(vertical[column_index + 1])
    mark_top = int(horizontal[mark_top_index])
    mark_bottom = int(horizontal[mark_bottom_index])
    cell_width = max(1, right - left)
    cell_height = max(1, mark_bottom - mark_top)
    tight_x_pad = max(2, int(cell_width * 0.07))
    tight_y_pad = max(2, int(cell_height * 0.12))

    tight = _clip_tally_crop_box(
        {
            "left": left + tight_x_pad,
            "top": mark_top + tight_y_pad,
            "right": right - tight_x_pad,
            "bottom": mark_bottom - tight_y_pad,
        },
        width=width,
        height=height,
    )
    if tight is None:
        return None

    context_x_pad = max(2, int(cell_width * 0.08))
    context_y_pad = max(2, int(cell_height * 0.12))
    context = _clip_tally_crop_box(
        {
            "left": left - context_x_pad,
            "top": int(horizontal[question_top_index]) - context_y_pad,
            "right": right + context_x_pad,
            "bottom": mark_bottom + context_y_pad,
        },
        width=width,
        height=height,
    )
    return tight, context


def _scale_tally_crop_box(
    box: Dict[str, int],
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> Optional[Dict[str, int]]:
    if source_width <= 0 or source_height <= 0:
        return None
    scaled = {
        "left": round(int(box["left"]) * target_width / source_width),
        "top": round(int(box["top"]) * target_height / source_height),
        "right": round(int(box["right"]) * target_width / source_width),
        "bottom": round(int(box["bottom"]) * target_height / source_height),
    }
    return _clip_tally_crop_box(scaled, width=target_width, height=target_height)


def _tally_crop_hash(image: Any, box: Dict[str, int]) -> str:
    crop = image.crop((box["left"], box["top"], box["right"], box["bottom"]))
    digest = hashlib.sha256()
    digest.update(f"{crop.mode}:{crop.width}x{crop.height}".encode("ascii"))
    digest.update(crop.tobytes())
    return digest.hexdigest()


def _tally_image_hash(image: Any) -> str:
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.width}x{image.height}".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _combine_tally_evidence_hash(*parts: Optional[str]) -> Optional[str]:
    clean_parts = [str(part) for part in parts if part]
    if not clean_parts:
        return None
    return hashlib.sha256("|".join(clean_parts).encode("ascii")).hexdigest()


def _tally_crop_data_url(
    image: Any,
    box: Dict[str, int],
    *,
    max_dimension: int = 960,
) -> str:
    crop = image.crop((box["left"], box["top"], box["right"], box["bottom"]))
    if max(crop.width, crop.height) > max_dimension:
        crop.thumbnail((max_dimension, max_dimension))
    output = BytesIO()
    crop.save(output, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _build_tally_question_evidence(
    payload: TallyExtractRequest,
    document: TallyDocumentContext,
    columns: List[str],
    rows: List[Dict[str, Any]],
) -> Tuple[List[TallyQuestionEvidence], Dict[int, Dict[str, Any]], Optional[str]]:
    question_numbers = _configured_question_numbers(document)
    if not question_numbers:
        question_numbers = sorted(_question_column_by_number(columns, rows).keys())
    question_columns = _question_column_by_number(columns, rows)
    layers = _decode_tally_ocr_layers(payload)
    reference = layers.get("template") or layers.get("filled_sheet")
    hash_layer = layers.get("strokes_only") or layers.get("filled_sheet")
    template_layer = layers.get("template") or reference
    full_evidence_hash = _combine_tally_evidence_hash(
        _tally_image_hash(hash_layer) if hash_layer is not None else None,
        _tally_image_hash(template_layer) if template_layer is not None else None,
    )
    evidence: List[TallyQuestionEvidence] = []
    crop_assets: Dict[int, Dict[str, Any]] = {}

    if reference is None:
        return [
            TallyQuestionEvidence(
                question_number=question_number,
                column=question_columns.get(question_number, f"Q{question_number}"),
            )
            for question_number in question_numbers
        ], crop_assets, full_evidence_hash

    grid = _detect_tally_question_grid(reference, document)
    if not grid:
        logger.info("Question-cell evidence unavailable because the tally grid was not detected")
        return [
            TallyQuestionEvidence(
                question_number=question_number,
                column=question_columns.get(question_number, f"Q{question_number}"),
            )
            for question_number in question_numbers
        ], crop_assets, full_evidence_hash

    for question_number in question_numbers:
        boxes = _tally_question_crop_boxes(
            grid,
            question_number,
            width=reference.width,
            height=reference.height,
        )
        if not boxes:
            evidence.append(
                TallyQuestionEvidence(
                    question_number=question_number,
                    column=question_columns.get(question_number, f"Q{question_number}"),
                )
            )
            continue

        tight_box, context_box = boxes
        hash_box = (
            _scale_tally_crop_box(
                context_box,
                source_width=reference.width,
                source_height=reference.height,
                target_width=hash_layer.width,
                target_height=hash_layer.height,
            )
            if hash_layer is not None
            else None
        )
        template_hash_box = (
            _scale_tally_crop_box(
                context_box,
                source_width=reference.width,
                source_height=reference.height,
                target_width=template_layer.width,
                target_height=template_layer.height,
            )
            if template_layer is not None
            else None
        )
        crop_hash = _combine_tally_evidence_hash(
            _tally_crop_hash(hash_layer, hash_box) if hash_layer is not None and hash_box else None,
            _tally_crop_hash(template_layer, template_hash_box)
            if template_layer is not None and template_hash_box
            else None,
        )
        evidence.append(
            TallyQuestionEvidence(
                question_number=question_number,
                column=question_columns.get(question_number, f"Q{question_number}"),
                crop_hash=crop_hash,
                crop_box=context_box,
            )
        )
        crop_assets[question_number] = {
            "reference_width": reference.width,
            "reference_height": reference.height,
            "tight_box": tight_box,
            "context_box": context_box,
            "layers": layers,
        }

    return evidence, crop_assets, full_evidence_hash


def _targeted_recheck_images(
    question_number: int,
    crop_asset: Dict[str, Any],
) -> List[TallyOcrImage]:
    images: List[TallyOcrImage] = []
    reference_width = int(crop_asset["reference_width"])
    reference_height = int(crop_asset["reference_height"])
    descriptions = {
        "template": "Clean printed template. Use only for the Q label, cell boundary, and layout.",
        "filled_sheet": "The printed tally sheet with the handwritten mark in this exact question cell.",
        "strokes_only": "Only the teacher's handwritten strokes for this exact question cell.",
    }
    for layer_name in ("template", "filled_sheet", "strokes_only"):
        image = crop_asset.get("layers", {}).get(layer_name)
        if image is None:
            continue
        for crop_kind in ("context", "mark"):
            source_box = crop_asset["context_box"] if crop_kind == "context" else crop_asset["tight_box"]
            box = _scale_tally_crop_box(
                source_box,
                source_width=reference_width,
                source_height=reference_height,
                target_width=image.width,
                target_height=image.height,
            )
            if not box:
                continue
            images.append(
                TallyOcrImage(
                    label=f"q{question_number}_{layer_name}_{crop_kind}",
                    image_b64=_tally_crop_data_url(image, box),
                    description=f"{descriptions[layer_name]} {crop_kind.title()} crop.",
                )
            )
    return images


def _strict_ocr_mark_value(value: Any) -> Optional[str]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        text = str(value).strip()
        if not re.fullmatch(r"\d+(?:\.\d+)?", text):
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
    if not math.isfinite(numeric) or numeric < 0:
        return None
    return _format_marks(numeric)


def _build_targeted_mark_recheck_prompt(
    *,
    question_number: int,
    original_value: str,
    configured_max: Optional[str],
) -> str:
    return f"""
You are reviewing one exact handwritten exam-tally mark cell: Q{question_number}.
The attached crops are grouped by label: template, filled_sheet, and strokes_only. The context crops include the printed Q label; the mark crops contain only the answer area.

Read only the handwriting visible inside this exact Q{question_number} mark cell. The first full-sheet OCR read it as {json.dumps(original_value)}. The configured maximum is {json.dumps(configured_max)} and is validation information only: it is never evidence of what was written.

Rules:
- Do not change a visible digit merely because it is above the configured maximum.
- Do not combine a border, printed text, neighbouring cell, erased stroke, or crossed-out stroke into a mark.
- If the remaining strokes could mean more than one number, or an overwritten correction cannot be read unambiguously, return unreadable. Do not guess the teacher's intended correction.
- Return a numeric value only when the visible handwriting itself clearly supports one exact non-negative number.

Return ONLY strict JSON:
{{
  "status": "readable" | "unreadable",
  "value": null,
  "confidence": 0.0,
  "reason": "short visual reason"
}}
""".strip()


def _parse_targeted_mark_recheck(raw_text: str) -> Tuple[Optional[str], Optional[float], str, str]:
    parsed = _parse_llm_json(raw_text)
    status_value = str(parsed.get("status") or "").strip().lower()
    reason = str(parsed.get("reason") or "").strip()[:500]
    try:
        confidence = float(parsed.get("confidence"))
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))

    candidate = _strict_ocr_mark_value(parsed.get("value"))
    if status_value != "readable" or candidate is None:
        return None, confidence, "unreadable", reason or "The focused OCR pass could not read one unambiguous value."
    return candidate, confidence, "resolved", reason or "Focused OCR resolved one exact visible mark."


async def _run_targeted_mark_rechecks(
    *,
    tenant_db: Any,
    document: TallyDocumentContext,
    raw_rows: List[Dict[str, Any]],
    issues: List[TallyValidationIssue],
    crop_assets: Dict[int, Dict[str, Any]],
    evidence_by_question: Dict[int, TallyQuestionEvidence],
    include_debug: bool,
) -> Tuple[List[TallyTargetedRecheck], List[TallyTargetedRecheckDebug]]:
    targeted: List[TallyTargetedRecheck] = []
    debug_entries: List[TallyTargetedRecheckDebug] = []
    seen_targets = set()
    # Every invalid Q-cell gets its own focused evidence pass. The sheet's
    # actual unresolved cells, not an arbitrary cap, determine the work.
    max_rechecks = len(
        {
            (int(issue.row_index or 0), int(issue.question_number))
            for issue in issues
            if issue.question_number is not None
            and issue.code in {"mark_above_max", "mark_unreadable", "missing_mark", "ocr_uncertain"}
        }
    )

    for issue in issues:
        if issue.code not in {"mark_above_max", "mark_unreadable", "missing_mark", "ocr_uncertain"} or issue.question_number is None:
            continue
        row_index = int(issue.row_index or 0)
        question_number = int(issue.question_number)
        target_key = (row_index, question_number)
        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        column = issue.column or f"Q{question_number}"
        original_value = issue.actual or (
            _cell_text(raw_rows[row_index].get(column))
            if 0 <= row_index < len(raw_rows)
            else ""
        )
        evidence = evidence_by_question.get(question_number)
        crop_asset = crop_assets.get(question_number) if row_index == 0 else None
        recheck_id = uuid4().hex

        if len(targeted) >= max_rechecks:
            targeted.append(
                TallyTargetedRecheck(
                    id=recheck_id,
                    row_index=row_index,
                    question_number=question_number,
                    column=column,
                    original_value=original_value,
                    configured_max=issue.expected,
                    status="manual_review_required",
                    reason="Focused OCR recheck limit reached; inspect and confirm this cell manually.",
                    crop_hash=evidence.crop_hash if evidence else None,
                    crop_box=evidence.crop_box if evidence else {},
                )
            )
            continue

        if not crop_asset or not evidence or not evidence.crop_hash:
            targeted.append(
                TallyTargetedRecheck(
                    id=recheck_id,
                    row_index=row_index,
                    question_number=question_number,
                    column=column,
                    original_value=original_value,
                    configured_max=issue.expected,
                    status="crop_unavailable",
                    reason="The Q-cell grid could not be isolated. Review this mark manually; no value was assumed.",
                    crop_hash=evidence.crop_hash if evidence else None,
                    crop_box=evidence.crop_box if evidence else {},
                )
            )
            continue

        images = _targeted_recheck_images(question_number, crop_asset)
        if not images:
            targeted.append(
                TallyTargetedRecheck(
                    id=recheck_id,
                    row_index=row_index,
                    question_number=question_number,
                    column=column,
                    original_value=original_value,
                    configured_max=issue.expected,
                    status="crop_unavailable",
                    reason="No readable image crop could be made for this cell.",
                    crop_hash=evidence.crop_hash,
                    crop_box=evidence.crop_box,
                )
            )
            continue

        prompt = _build_targeted_mark_recheck_prompt(
            question_number=question_number,
            original_value=original_value,
            configured_max=issue.expected,
        )
        raw_text = ""
        provider: Optional[str] = None
        try:
            response = await get_ocr_service().analyze_images(
                images=[image.model_dump(exclude_none=True) for image in images],
                prompt=prompt,
                tenant_db=tenant_db,
                max_tokens=512,
                temperature=0.0,
            )
            raw_text = str(response.get("text") or "")
            provider = response.get("provider")
            candidate, confidence, recheck_status, reason = _parse_targeted_mark_recheck(raw_text)
        except Exception as exc:
            logger.warning("Targeted tally mark recheck failed for Q%s: %s", question_number, exc)
            candidate = None
            confidence = None
            recheck_status = "recheck_failed"
            reason = "Focused OCR recheck failed. Review this cell manually; no value was assumed."

        targeted.append(
            TallyTargetedRecheck(
                id=recheck_id,
                row_index=row_index,
                question_number=question_number,
                column=column,
                original_value=original_value,
                configured_max=issue.expected,
                candidate_value=candidate,
                confidence=confidence,
                status=recheck_status,
                reason=reason,
                crop_hash=evidence.crop_hash,
                crop_box=evidence.crop_box,
            )
        )
        if include_debug:
            debug_entries.append(
                TallyTargetedRecheckDebug(
                    id=recheck_id,
                    row_index=row_index,
                    question_number=question_number,
                    prompt=prompt,
                    raw_text=raw_text,
                    provider=provider,
                    images=images,
                )
            )

    return targeted, debug_entries


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

            strict_mark = _strict_ocr_mark_value(raw_value)
            mark = float(strict_mark) if strict_mark is not None else None
            if mark is None:
                issues.append(
                    TallyValidationIssue(
                        severity="error",
                        code="mark_unreadable",
                        message=f"{column} value '{raw_value}' is not one exact readable numeric mark.",
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
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    question_columns = _find_question_columns(ordered_columns, rows)
    map_by_number = {
        int(item["question_number"]): item
        for item in question_map_items
        if item.get("question_number")
    }
    if not map_by_number:
        fallback_numbers = sorted(question_columns.keys())
        if not fallback_numbers and document.num_questions and document.num_questions > 0:
            fallback_numbers = list(range(1, int(document.num_questions) + 1))
        map_by_number = {
            question_number: {
                "question_number": question_number,
                "topic": "Overall",
                "sub_topic": "Overall",
                "max_marks": _document_max_marks_for_question(document, question_number),
                "source": "fallback",
            }
            for question_number in fallback_numbers
        }
    summary_rows: List[Dict[str, Any]] = []
    topic_rows: List[Dict[str, Any]] = []
    subtopic_rows: List[Dict[str, Any]] = []
    class_topic_rows: List[Dict[str, Any]] = []
    class_subtopic_rows: List[Dict[str, Any]] = []
    question_rows: List[Dict[str, Any]] = []
    intervention_rows: List[Dict[str, Any]] = []
    class_topic_stats: Dict[str, Dict[str, Any]] = {}
    class_subtopic_stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
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
        subtopic_stats: Dict[Tuple[str, str], Dict[str, float]] = {}

        for question_number in sorted(map_by_number.keys()):
            item = map_by_number[question_number]
            column = question_columns.get(question_number) or f"Q{question_number}"
            raw_value = row.get(column, "")
            strict_mark = _strict_ocr_mark_value(raw_value)
            mark = float(strict_mark) if strict_mark is not None else None
            obtained = float(mark) if mark is not None else 0.0
            max_marks = _question_max_marks(item, document, question_number)
            topic = str(item.get("topic") or "Unmapped").strip() or "Unmapped"
            sub_topic = str(item.get("sub_topic") or "General").strip() or "General"

            total_obtained += obtained
            total_max += max_marks
            topic_bucket = topic_stats.setdefault(topic, {"obtained": 0.0, "max": 0.0, "questions": 0})
            topic_bucket["obtained"] += obtained
            topic_bucket["max"] += max_marks
            topic_bucket["questions"] += 1
            subtopic_bucket = subtopic_stats.setdefault(
                (topic, sub_topic),
                {"obtained": 0.0, "max": 0.0, "questions": 0},
            )
            subtopic_bucket["obtained"] += obtained
            subtopic_bucket["max"] += max_marks
            subtopic_bucket["questions"] += 1
            class_bucket = class_topic_stats.setdefault(
                topic,
                {"obtained": 0.0, "max": 0.0, "questions": 0, "students": set(), "question_numbers": set()},
            )
            class_bucket["obtained"] += obtained
            class_bucket["max"] += max_marks
            class_bucket["questions"] += 1
            class_bucket["students"].add(student_key)
            class_bucket["question_numbers"].add(question_number)
            class_subtopic_bucket = class_subtopic_stats.setdefault(
                (topic, sub_topic),
                {"obtained": 0.0, "max": 0.0, "questions": 0, "students": set(), "question_numbers": set()},
            )
            class_subtopic_bucket["obtained"] += obtained
            class_subtopic_bucket["max"] += max_marks
            class_subtopic_bucket["questions"] += 1
            class_subtopic_bucket["students"].add(student_key)
            class_subtopic_bucket["question_numbers"].add(question_number)

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
                    "Percentage": _percentage(obtained, max_marks),
                    "Mark Status": "Recorded" if mark is not None else "Not recorded",
                    "Topic": topic,
                    "Sub-topic": sub_topic,
                }
            )

        overall_pct = _percentage(total_obtained, total_max)
        weak_topic, strong_topic = _pick_strengths(topic_stats, overall_pct)
        weak_subtopic, strong_subtopic = _pick_strengths(
            {f"{topic} - {sub_topic}": values for (topic, sub_topic), values in subtopic_stats.items()},
            overall_pct,
        )
        summary_rows.append(
            {
                "Student": student,
                "Student ID": student_id,
                "Class": class_label,
                "Section": section_label,
                "Subject": subject_label,
                "Total Obtained": round(total_obtained, 2),
                "Total Max": round(total_max, 2),
                "Percentage": overall_pct,
                "Weak Topic": weak_topic,
                "Weak Sub-topic": weak_subtopic,
                "Strong Topic": strong_topic,
                "Strong Sub-topic": strong_subtopic,
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
                    "Topic": topic,
                    "Marks Obtained": round(values["obtained"], 2),
                    "Max Marks": round(values["max"], 2),
                    "Percentage": _percentage(values["obtained"], values["max"]),
                    "Question Count": int(values["questions"]),
                }
            )

        for (topic, sub_topic), values in sorted(subtopic_stats.items()):
            percentage = _percentage(values["obtained"], values["max"])
            subtopic_rows.append(
                {
                    "Student": student,
                    "Student ID": student_id,
                    "Class": class_label,
                    "Section": section_label,
                    "Subject": subject_label,
                    "Topic": topic,
                    "Sub-topic": sub_topic,
                    "Marks Obtained": round(values["obtained"], 2),
                    "Max Marks": round(values["max"], 2),
                    "Percentage": percentage,
                    "Question Count": int(values["questions"]),
                }
            )
            if percentage is not None and percentage < 60:
                intervention_rows.append(
                    {
                        "Student": student,
                        "Student ID": student_id,
                        "Class": class_label,
                        "Section": section_label,
                        "Subject": subject_label,
                        "Topic": topic,
                        "Sub-topic": sub_topic,
                        "Percentage": percentage,
                        "Priority": "High" if percentage < 40 else "Medium",
                        "Suggested Action": "Re-teach and assign targeted practice",
                    }
                )

    def append_class_rows(
        stats: Dict[Any, Dict[str, Any]],
        target_rows: List[Dict[str, Any]],
        include_sub_topic: bool,
    ) -> None:
        for key, values in sorted(stats.items()):
            topic, sub_topic = key if include_sub_topic else (key, None)
            percentage = _percentage(values["obtained"], values["max"])
            student_count = len(values["students"])
            average_obtained = values["obtained"] / student_count if student_count else 0.0
            average_max = values["max"] / student_count if student_count else 0.0
            status_label = (
                "Needs attention" if percentage is not None and percentage < 60
                else "Strong" if percentage is not None and percentage >= 80
                else "Developing" if percentage is not None
                else ""
            )
            row = {
                "Class": class_label,
                "Section": section_label,
                "Subject": subject_label,
                "Topic": topic,
                "Students": student_count,
                "Marks Obtained": round(values["obtained"], 2),
                "Max Marks": round(values["max"], 2),
                "Average Marks": round(average_obtained, 2),
                "Average Max Marks": round(average_max, 2),
                "Percentage": percentage,
                "Question Count": len(values["question_numbers"]),
                "Scored Opportunities": int(values["questions"]),
                "Class Status": status_label,
            }
            if include_sub_topic:
                row["Sub-topic"] = sub_topic
            target_rows.append(row)

    append_class_rows(class_topic_stats, class_topic_rows, False)
    append_class_rows(class_subtopic_stats, class_subtopic_rows, True)

    return (
        summary_rows,
        topic_rows,
        subtopic_rows,
        class_topic_rows,
        class_subtopic_rows,
        question_rows,
        intervention_rows,
    )


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
    image_instructions = (
        """
You are reading three labeled OCR images from the same exam tally sheet:
- template: the clean printed tally template. Use it only to understand the layout, headings, blue grid lines, and Q cell positions.
- filled_sheet: the template plus the teacher's handwritten marks.
- strokes_only: only the teacher's handwritten strokes on a white background, rendered in high contrast.

Compare filled_sheet against template to identify handwritten content. If a mark is unclear in filled_sheet, use strokes_only to decide what was written. Do not treat anything that exists only in template as a handwritten mark.
""".strip()
        if payload.images
        else """
You are reading one flattened image exported from a digital canvas.
The image contains a clean printed/template tally sheet underneath and imperfect black handwritten teacher marks written on top.
Use the printed/template text, blue grid lines, and Q labels only to locate the correct cells.
""".strip()
    )
    return f"""
{image_instructions}
This is an exam tally marks grid, not a generic spreadsheet. The important cells are the evaluator marks under the printed question headings.

Task:
1. Detect the table structure from the image.
2. Read all headings exactly as intended, including student details, question labels, totals, and maximum-marks labels.
3. Pair each value with the correct heading/cell.
4. If the sheet is a single-student form, return one row.
5. If the sheet has multiple student rows, return all rows.
6. Preserve truly blank cells as empty strings.
7. Normalize question headings to the Q<number> form where obvious.
8. Do not invent names or header text. For each Q mark cell, return a numeric value only when the visible handwriting supports one exact value. If a visible mark is ambiguous, return an empty string and include that exact Q label in uncertain_cells. Do not choose a closest value.{marking_rule}

Question mark grid rules:
- {question_context}
- Read values by the physical cell position below each Q heading. Do not shift marks left or right just because an earlier cell is hard to read.
- Interpret a handwritten mark only from the digit shape visibly written inside its own cell, even when it is close to a blue printed grid line.
- Printed/template text, blue table borders, and printed Q labels are not marks. Black handwritten strokes inside the white mark area are marks.
- Extract Q marks only from handwritten content inside each Q cell; never treat template content as a mark value.
- Preserve the exact numeric value as a string only when the written digit sequence is clear. The configured maximum must never turn an unclear mark into a value.
- Do not turn a messy stroke, overwritten correction, grid-line overlap, or border-adjacent mark into a different value. Mark it uncertain instead when the written value is not exact.
- Never combine strokes from neighboring cells or printed Q labels into a two-digit mark.
- For Q cells, do not leave a visible handwritten mark blank just because it is small, faint, near a border, or slightly messy when it is still clearly readable.
- Return blank and list the cell in uncertain_cells when a visible mark could mean more than one value, is overwritten, or cannot be read exactly. Return blank without uncertain_cells only when there is no visible handwritten mark inside that cell.

Context from the UI, for disambiguation only:
{json.dumps(context, ensure_ascii=False)}

Return ONLY strict JSON in this shape:
{{
  "columns": [],
  "rows": [],
  "cell_confidence": {{}},
  "uncertain_cells": [],
  "warnings": [],
  "confidence": 0.0
}}
""".strip()


def _format_filled_recheck_ranges(filled_by_row: Dict[int, List[int]]) -> str:
    parts = []
    for row_index, question_numbers in sorted(filled_by_row.items()):
        label = _format_question_ranges(question_numbers)
        if label:
            parts.append(f"row {row_index + 1}: {label}")
    return "; ".join(parts)


def _active_tally_ocr_images(payload: TallyExtractRequest) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []
    for image in payload.images:
        if image.image_b64:
            images.append(image.model_dump(exclude_none=True))
    return images


def _primary_tally_ocr_image(payload: TallyExtractRequest) -> Optional[str]:
    if payload.image_b64:
        return payload.image_b64
    preferred_labels = {"filled_sheet", "filled", "combined", "final"}
    for image in payload.images:
        if image.label in preferred_labels and image.image_b64:
            return image.image_b64
    for image in payload.images:
        if image.image_b64:
            return image.image_b64
    return None


async def _analyze_tally_ocr(
    *,
    tenant_db: Any,
    payload: TallyExtractRequest,
    prompt: str,
    max_tokens: int,
) -> Dict[str, Any]:
    images = _active_tally_ocr_images(payload)
    if images:
        return await get_ocr_service().analyze_images(
            images=images,
            prompt=prompt,
            tenant_db=tenant_db,
            max_tokens=max_tokens,
            temperature=0.0,
        )

    primary_image = _primary_tally_ocr_image(payload)
    if not primary_image:
        raise HTTPException(status_code=400, detail="image_b64 or images are required")

    return await get_ocr_service().analyze_image(
        image_b64=primary_image,
        prompt=prompt,
        tenant_db=tenant_db,
        max_tokens=max_tokens,
        temperature=0.0,
    )


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


def _safe_tally_student_id(value: str) -> str:
    student_id = (value or "").strip()
    if not student_id:
        raise HTTPException(status_code=400, detail="student_id is required")
    if len(student_id) > 240 or "/" in student_id or "\\" in student_id:
        raise HTTPException(status_code=400, detail="Invalid student_id")
    return student_id


def _tally_student_identity(student: Any) -> str:
    if isinstance(student, TallyStudentContext):
        values = (student.student_id, student.username, student.roll_no, student.name)
    elif isinstance(student, dict):
        values = (
            student.get("student_id"),
            student.get("username"),
            student.get("roll_no"),
            student.get("name"),
        )
    else:
        values = ()
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _tally_review_id(document_id: str, student_id: str) -> str:
    source = f"{document_id}\x00{student_id}".encode("utf-8")
    return f"tally-review-{hashlib.sha256(source).hexdigest()}"


def _correction_key(row_index: int, question_number: int) -> str:
    return f"{int(row_index)}:Q{int(question_number)}"


def _correction_as_response(record: Dict[str, Any]) -> TallyAppliedCorrection:
    approved_value = record.get("approved_value")
    if approved_value is not None:
        approved_value = _strict_ocr_mark_value(approved_value)
    return TallyAppliedCorrection(
        row_index=int(record.get("row_index") or 0),
        question_number=int(record.get("question_number") or 0),
        column=str(record.get("column") or f"Q{record.get('question_number') or ''}"),
        original_ocr_value=(
            str(record.get("original_ocr_value"))
            if record.get("original_ocr_value") is not None
            else None
        ),
        approved_value=approved_value,
        decision=str(record.get("decision") or "set"),
        crop_hash=record.get("crop_hash"),
        evidence_scope=str(record.get("evidence_scope") or "cell"),
        source_extraction_id=record.get("source_extraction_id"),
        targeted_candidate=(
            str(record.get("targeted_candidate"))
            if record.get("targeted_candidate") is not None
            else None
        ),
        reason=record.get("reason"),
        approved_at=record.get("approved_at"),
        approved_by=record.get("approved_by"),
        resolution_source=str(record.get("resolution_source") or "teacher_override"),
    )


def _auto_resolution_as_response(record: Dict[str, Any]) -> Optional[TallyAutoResolvedMark]:
    resolved_value = _strict_ocr_mark_value(
        record.get("resolved_value", record.get("approved_value"))
    )
    if resolved_value is None:
        return None
    try:
        row_index = int(record.get("row_index") or 0)
        question_number = int(record.get("question_number") or 0)
    except (TypeError, ValueError):
        return None
    if row_index < 0 or question_number <= 0:
        return None
    confidence = record.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))
    return TallyAutoResolvedMark(
        row_index=row_index,
        question_number=question_number,
        column=str(record.get("column") or f"Q{question_number}"),
        original_ocr_value=(
            str(record.get("original_ocr_value"))
            if record.get("original_ocr_value") is not None
            else None
        ),
        resolved_value=resolved_value,
        crop_hash=record.get("crop_hash"),
        evidence_scope=str(record.get("evidence_scope") or "cell"),
        confidence=confidence,
        reason=record.get("reason"),
        source_extraction_id=record.get("source_extraction_id"),
        resolved_at=record.get("resolved_at"),
        resolution_source=str(record.get("resolution_source") or "focused_ocr"),
    )


def _focused_rechecks_to_auto_resolutions(
    rechecks: List[TallyTargetedRecheck],
    *,
    document: TallyDocumentContext,
    copy_id: Optional[str],
) -> List[Dict[str, Any]]:
    resolutions: List[Dict[str, Any]] = []
    seen_keys = set()
    now = datetime.utcnow()
    for recheck in rechecks:
        if recheck.status != "resolved" or not recheck.crop_hash:
            continue
        resolved_value = _strict_ocr_mark_value(recheck.candidate_value)
        if resolved_value is None:
            continue
        expected_max = _expected_marks_for_question(document, recheck.question_number)
        if expected_max is not None and float(resolved_value) > expected_max + 1e-6:
            continue
        key = (int(recheck.row_index), int(recheck.question_number))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        resolutions.append(
            {
                "row_index": recheck.row_index,
                "question_number": recheck.question_number,
                "column": recheck.column,
                "original_ocr_value": recheck.original_value,
                "resolved_value": resolved_value,
                "crop_hash": recheck.crop_hash,
                "evidence_scope": "cell",
                "confidence": recheck.confidence,
                "reason": recheck.reason or "Focused OCR resolved one exact visible mark.",
                "copy_id": copy_id,
                "resolved_at": now,
                "resolution_source": "focused_ocr",
            }
        )
    return resolutions


def _apply_tally_auto_resolutions(
    *,
    columns: List[str],
    raw_rows: List[Dict[str, Any]],
    resolutions: List[Dict[str, Any]],
    document: TallyDocumentContext,
    question_evidence: List[TallyQuestionEvidence],
    copy_id: Optional[str],
    full_evidence_hash: Optional[str] = None,
) -> Tuple[List[str], List[Dict[str, Any]], List[TallyAutoResolvedMark]]:
    correction_records: List[Dict[str, Any]] = []
    resolution_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for raw_resolution in resolutions:
        response = _auto_resolution_as_response(raw_resolution)
        if response is None:
            continue
        expected_max = _expected_marks_for_question(document, response.question_number)
        if expected_max is not None and float(response.resolved_value or 0) > expected_max + 1e-6:
            continue
        key = (response.row_index, response.question_number)
        if key in resolution_by_key:
            continue
        resolution = {
            **raw_resolution,
            "row_index": response.row_index,
            "question_number": response.question_number,
            "column": response.column,
            "approved_value": response.resolved_value,
            "decision": "set",
            "crop_hash": response.crop_hash,
            "evidence_scope": response.evidence_scope,
            "copy_id": raw_resolution.get("copy_id", copy_id),
        }
        correction_records.append(resolution)
        resolution_by_key[key] = resolution

    effective_columns, effective_rows, applied, _ = _apply_tally_mark_corrections(
        columns=columns,
        raw_rows=raw_rows,
        corrections=correction_records,
        question_evidence=question_evidence,
        copy_id=copy_id,
        full_evidence_hash=full_evidence_hash,
    )
    applied_keys = {(item.row_index, item.question_number) for item in applied}
    applied_resolutions = [
        response
        for key, record in resolution_by_key.items()
        if key in applied_keys
        for response in [_auto_resolution_as_response(record)]
        if response is not None
    ]
    return effective_columns, effective_rows, applied_resolutions


def _active_tally_corrections(review: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not review:
        return []
    corrections_by_key = review.get("active_corrections_by_key")
    if isinstance(corrections_by_key, dict):
        return [
            dict(corrections_by_key[key])
            for key in sorted(corrections_by_key)
            if isinstance(corrections_by_key.get(key), dict)
        ]
    corrections = review.get("active_corrections") or []
    if not isinstance(corrections, list):
        return []
    return [item for item in corrections if isinstance(item, dict)]


def _apply_tally_mark_corrections(
    *,
    columns: List[str],
    raw_rows: List[Dict[str, Any]],
    corrections: List[Dict[str, Any]],
    question_evidence: List[TallyQuestionEvidence],
    copy_id: Optional[str],
    full_evidence_hash: Optional[str] = None,
) -> Tuple[List[str], List[Dict[str, Any]], List[TallyAppliedCorrection], List[TallyAppliedCorrection]]:
    effective_columns = list(columns)
    effective_rows = [dict(row) for row in raw_rows]
    question_columns = _question_column_by_number(effective_columns, effective_rows)
    evidence_by_key = {
        (int(item.row_index), int(item.question_number)): item
        for item in question_evidence
        if item.question_number > 0
    }
    applied: List[TallyAppliedCorrection] = []
    stale: List[TallyAppliedCorrection] = []

    for correction in corrections:
        try:
            row_index = int(correction.get("row_index") or 0)
            question_number = int(correction.get("question_number") or 0)
        except (TypeError, ValueError):
            continue
        if row_index < 0 or question_number <= 0 or row_index >= len(effective_rows):
            continue

        response = _correction_as_response(correction)
        evidence = evidence_by_key.get((row_index, question_number))
        saved_hash = str(correction.get("crop_hash") or "")
        evidence_scope = str(correction.get("evidence_scope") or "cell").lower()
        saved_copy_id = str(correction.get("copy_id") or "")
        current_copy_id = str(copy_id or "")
        evidence_matches = (
            bool(full_evidence_hash)
            and bool(saved_hash)
            and full_evidence_hash == saved_hash
            if evidence_scope == "full_sheet"
            else bool(evidence and evidence.crop_hash and saved_hash and evidence.crop_hash == saved_hash)
        )
        if not evidence_matches or (saved_copy_id and current_copy_id and saved_copy_id != current_copy_id):
            stale.append(response)
            continue

        column = question_columns.get(question_number) or response.column or f"Q{question_number}"
        if column not in effective_columns:
            effective_columns.append(column)
        question_columns[question_number] = column
        decision = str(correction.get("decision") or "set").lower()
        if decision == "clear":
            effective_rows[row_index][column] = ""
            applied.append(response)
            continue

        approved_value = _strict_ocr_mark_value(correction.get("approved_value"))
        if approved_value is None:
            stale.append(response)
            continue
        effective_rows[row_index][column] = approved_value
        applied.append(response)

    return effective_columns, effective_rows, applied, stale


def _validate_tally_mark_correction(
    *,
    request: TallyMarkCorrectionSaveRequest,
    document: TallyDocumentContext,
) -> Optional[str]:
    decision = str(request.decision or "").strip().lower()
    if decision not in {"set", "clear"}:
        raise HTTPException(status_code=422, detail="Correction decision must be 'set' or 'clear'")

    configured_question_count = _configured_question_count(document)
    if configured_question_count and request.question_number > configured_question_count:
        raise HTTPException(
            status_code=422,
            detail=f"Q{request.question_number} is outside the configured question range",
        )
    if decision == "clear":
        if request.approved_value is not None:
            raise HTTPException(status_code=422, detail="A cleared mark cannot include a numeric value")
        return None

    approved_value = _strict_ocr_mark_value(request.approved_value)
    if approved_value is None:
        raise HTTPException(status_code=422, detail="Approved mark must be one exact non-negative number")
    expected_max = _expected_marks_for_question(document, request.question_number)
    if expected_max is None:
        raise HTTPException(
            status_code=422,
            detail=f"No configured maximum marks found for Q{request.question_number}",
        )
    if float(approved_value) > expected_max + 1e-6:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Approved mark {_format_marks(float(approved_value))} exceeds the configured "
                f"maximum {_format_marks(expected_max)} for Q{request.question_number}"
            ),
        )
    return approved_value


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
                "topic": str(data.get("topic") or "").strip()[:80],
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
    standard: Optional[str] = Form(None),
    course_plan: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(_require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    clean_upload = await secure_upload(
        file=file,
        policy_id="tally_question_source_pdf",
        actor=current_user,
        db=db,
        purpose_metadata={
            "purpose": "tally_question_source_pdf",
            "region_scope": "tally_question_source_preview",
            "created_by": current_user.get("user_id"),
        },
        authorization_subject=f"tally_question_source:{current_user.get('user_id', 'unknown')}",
    )
    file_content = clean_upload.bytes or b""

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
    preview_questions = [
        question.model_dump() if hasattr(question, "model_dump") else dict(question)
        for question in questions
        if isinstance(question, dict) or hasattr(question, "model_dump")
    ]
    if preview_questions:
        preview_map = await build_tally_question_map(
            tally_document_id=preview_document_id,
            source_document_id=preview_document_id,
            questions=preview_questions,
            subject=subject,
            standard=standard,
            course_plan=course_plan,
        )
        classifications = {
            int(item["question_number"]): item
            for item in preview_map.get("items") or []
            if item.get("question_number")
        }
        for item in items:
            classification = classifications.get(item.question_number)
            if classification:
                item.topic = classification.get("topic") or "Unmapped"
                item.sub_topic = classification.get("sub_topic") or "General"
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


@router.get(
    "/mark-corrections/{document_id}/{student_id}",
    response_model=TallyMarkCorrectionResponse,
)
async def get_tally_mark_corrections(
    document_id: str,
    student_id: str,
    current_user: Dict[str, Any] = Depends(_require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    safe_document_id = _safe_document_id(document_id)
    safe_student_id = _safe_tally_student_id(student_id)
    tenant_db = await _tenant_db(db, current_user)
    await _assert_tally_context_access(
        tenant_db,
        current_user,
        TallyDocumentContext(document_id=safe_document_id),
        TallyStudentContext(student_id=safe_student_id),
    )
    review = await tenant_db["exam_tally_mark_reviews"].find_one(
        {"_id": _tally_review_id(safe_document_id, safe_student_id)}
    )
    return TallyMarkCorrectionResponse(
        success=True,
        document_id=safe_document_id,
        student_id=safe_student_id,
        revision=int((review or {}).get("revision") or 0),
        corrections=[_correction_as_response(item) for item in _active_tally_corrections(review)],
    )


@router.put(
    "/mark-corrections/{document_id}/{student_id}",
    response_model=TallyMarkCorrectionResponse,
)
async def save_tally_mark_correction(
    document_id: str,
    student_id: str,
    payload: TallyMarkCorrectionSaveRequest,
    current_user: Dict[str, Any] = Depends(_require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    safe_document_id = _safe_document_id(document_id)
    safe_student_id = _safe_tally_student_id(student_id)
    tenant_db = await _tenant_db(db, current_user)
    extraction = await tenant_db["exam_tally_extractions"].find_one(
        {"_id": payload.source_extraction_id}
    )
    if not extraction:
        raise HTTPException(status_code=404, detail="OCR extraction evidence was not found")

    raw_document = extraction.get("document") or {}
    try:
        document_context = TallyDocumentContext.model_validate(raw_document)
    except Exception as exc:
        raise HTTPException(status_code=409, detail="OCR extraction has invalid document context") from exc
    extraction_document_id = str(document_context.document_id or raw_document.get("document_id") or "")
    if extraction_document_id != safe_document_id:
        raise HTTPException(status_code=409, detail="OCR extraction belongs to a different tally document")
    try:
        student_context = TallyStudentContext.model_validate(extraction.get("student") or {})
    except Exception as exc:
        raise HTTPException(status_code=409, detail="OCR extraction has invalid student context") from exc
    extraction_student_id = _tally_student_identity(student_context)
    if extraction_student_id != safe_student_id:
        raise HTTPException(status_code=409, detail="OCR extraction belongs to a different student")
    await _assert_tally_context_access(
        tenant_db,
        current_user,
        document_context,
        student_context,
    )

    evidence_items: List[TallyQuestionEvidence] = []
    for item in extraction.get("question_evidence") or []:
        try:
            evidence_items.append(TallyQuestionEvidence.model_validate(item))
        except Exception:
            continue
    evidence = next(
        (
            item
            for item in evidence_items
            if item.row_index == payload.row_index and item.question_number == payload.question_number
        ),
        None,
    )
    evidence_scope = str(payload.evidence_scope or "").strip().lower()
    if evidence_scope not in {"cell", "full_sheet"}:
        raise HTTPException(status_code=422, detail="Evidence scope must be 'cell' or 'full_sheet'")
    if evidence_scope == "cell":
        if not evidence or not evidence.crop_hash:
            raise HTTPException(
                status_code=409,
                detail="The handwritten Q-cell evidence is unavailable. Use the audited full-sheet review path or re-run OCR with a detectable tally grid.",
            )
        expected_evidence_hash = evidence.crop_hash
    else:
        expected_evidence_hash = str(extraction.get("full_evidence_hash") or "")
        if not expected_evidence_hash:
            raise HTTPException(
                status_code=409,
                detail="Full-sheet handwriting evidence is unavailable. Re-run review before confirming a mark.",
            )
    if not hmac.compare_digest(expected_evidence_hash, payload.crop_hash):
        raise HTTPException(
            status_code=409,
            detail="This correction does not match the current handwritten evidence. Re-run review first.",
        )

    approved_value = _validate_tally_mark_correction(
        request=payload,
        document=document_context,
    )
    raw_columns = [str(column) for column in extraction.get("columns") or []]
    raw_rows = [dict(row) for row in extraction.get("rows") or [] if isinstance(row, dict)]
    if payload.row_index >= len(raw_rows):
        raise HTTPException(status_code=422, detail="Correction row is outside the OCR extraction")
    question_columns = _question_column_by_number(raw_columns, raw_rows)
    column = question_columns.get(payload.question_number, f"Q{payload.question_number}")
    original_ocr_value = _cell_text(raw_rows[payload.row_index].get(column)) or None
    targeted_record = next(
        (
            item
            for item in extraction.get("targeted_rechecks") or []
            if isinstance(item, dict)
            and int(item.get("row_index") or 0) == payload.row_index
            and int(item.get("question_number") or 0) == payload.question_number
        ),
        None,
    )

    review_id = _tally_review_id(safe_document_id, safe_student_id)
    review = await tenant_db["exam_tally_mark_reviews"].find_one({"_id": review_id})
    if not review:
        now = datetime.utcnow()
        await tenant_db["exam_tally_mark_reviews"].update_one(
            {"_id": review_id},
            {
                "$setOnInsert": {
                    "document_id": safe_document_id,
                    "student_id": safe_student_id,
                    "active_corrections_by_key": {},
                    "revision": 0,
                    "created_at": now,
                    "created_by": str(current_user.get("user_id") or ""),
                }
            },
            upsert=True,
        )
        review = await tenant_db["exam_tally_mark_reviews"].find_one({"_id": review_id})
    active_by_key = {
        _correction_key(int(item.get("row_index") or 0), int(item.get("question_number") or 0)): dict(item)
        for item in _active_tally_corrections(review)
    }
    now = datetime.utcnow()
    correction_key = _correction_key(payload.row_index, payload.question_number)
    previous_correction = active_by_key.get(correction_key)
    correction = {
        "row_index": payload.row_index,
        "question_number": payload.question_number,
        "column": column,
        "original_ocr_value": original_ocr_value,
        "approved_value": approved_value,
        "decision": str(payload.decision).strip().lower(),
        "crop_hash": expected_evidence_hash,
        "evidence_scope": evidence_scope,
        "crop_box": evidence.crop_box if evidence else {},
        "source_extraction_id": payload.source_extraction_id,
        "targeted_candidate": (targeted_record or {}).get("candidate_value"),
        "reason": (payload.reason or "teacher_confirmed").strip() or "teacher_confirmed",
        "copy_id": extraction.get("copy_id"),
        "approved_at": now,
        "approved_by": str(current_user.get("user_id") or ""),
        "approved_by_type": current_user.get("user_type"),
        "resolution_source": "teacher_override",
    }
    active_by_key[correction_key] = correction
    expected_revision = int((review or {}).get("revision") or 0)
    revision = expected_revision + 1
    audit_event = {
        "event_id": uuid4().hex,
        "event_type": "mark_correction_confirmed",
        "revision": revision,
        "correction": correction,
        "previous_correction": previous_correction,
        "created_at": now,
        "created_by": str(current_user.get("user_id") or ""),
        "created_by_type": current_user.get("user_type"),
    }
    review_update_fields = {
        "document_id": safe_document_id,
        "student_id": safe_student_id,
        "copy_id": extraction.get("copy_id"),
        "updated_at": now,
        "updated_by": str(current_user.get("user_id") or ""),
        "updated_by_type": current_user.get("user_type"),
    }
    if isinstance((review or {}).get("active_corrections_by_key"), dict):
        review_update_fields[f"active_corrections_by_key.{correction_key}"] = correction
    else:
        # One-time migration from the older list representation. Later writes
        # Update only one map key, so concurrent Q-cell confirmations cannot
        # erase one another.
        review_update_fields["active_corrections_by_key"] = active_by_key

    await tenant_db["exam_tally_mark_reviews"].update_one(
        {"_id": review_id},
        {
            "$set": review_update_fields,
            "$setOnInsert": {"created_at": now, "created_by": str(current_user.get("user_id") or "")},
            "$inc": {"revision": 1},
            "$push": {"audit_events": audit_event},
        },
        upsert=True,
    )
    persisted_review = await tenant_db["exam_tally_mark_reviews"].find_one({"_id": review_id})
    active_corrections = _active_tally_corrections(persisted_review)
    revision = int((persisted_review or {}).get("revision") or revision)

    stored_auto_resolutions = [
        dict(item)
        for item in extraction.get("auto_resolved_marks") or []
        if isinstance(item, dict)
    ]
    auto_columns, auto_rows, auto_resolved_marks = _apply_tally_auto_resolutions(
        columns=raw_columns,
        raw_rows=raw_rows,
        resolutions=stored_auto_resolutions,
        document=document_context,
        question_evidence=evidence_items,
        copy_id=extraction.get("copy_id"),
        full_evidence_hash=extraction.get("full_evidence_hash"),
    )
    effective_columns, effective_rows, applied, stale = _apply_tally_mark_corrections(
        columns=auto_columns,
        raw_rows=auto_rows,
        corrections=active_corrections,
        question_evidence=evidence_items,
        copy_id=extraction.get("copy_id"),
        full_evidence_hash=extraction.get("full_evidence_hash"),
    )
    validation_issues = _validate_tally_result(
        effective_columns,
        effective_rows,
        document_context,
        student_context,
    )
    applied_correction_keys = {
        (item.row_index, item.question_number) for item in applied
    }
    auto_resolved_keys = {
        (item.row_index, item.question_number) for item in auto_resolved_marks
    }
    resolved_question_keys = applied_correction_keys.union(auto_resolved_keys)
    for issue_data in extraction.get("raw_validation_issues") or []:
        try:
            issue = TallyValidationIssue.model_validate(issue_data)
        except Exception:
            continue
        if issue.code != "ocr_uncertain":
            continue
        if (int(issue.row_index or 0), int(issue.question_number or 0)) not in resolved_question_keys:
            validation_issues.append(issue)
    await tenant_db["exam_tally_extractions"].update_one(
        {"_id": payload.source_extraction_id},
        {
            "$set": {
                "effective_columns": effective_columns,
                "effective_rows": effective_rows,
                "validation_issues": [
                    issue.model_dump(exclude_none=True) for issue in validation_issues
                ],
                "applied_corrections": [
                    item.model_dump(exclude_none=True) for item in applied
                ],
                "auto_resolved_marks": [
                    item.model_dump(exclude_none=True) for item in auto_resolved_marks
                ],
                "effective_updated_at": now,
            }
        },
    )
    return TallyMarkCorrectionResponse(
        success=True,
        document_id=safe_document_id,
        student_id=safe_student_id,
        revision=revision,
        corrections=[_correction_as_response(item) for item in active_corrections],
        applied_corrections=applied,
        stale_corrections=stale,
        auto_resolved_marks=auto_resolved_marks,
        rows=effective_rows,
        validation_issues=validation_issues,
    )


@router.post("/extract", response_model=TallyExtractResponse)
async def extract_tally(
    request: Request,
    payload: TallyExtractRequest,
    current_user: Dict[str, Any] = Depends(_require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    ocr_images = _active_tally_ocr_images(payload)
    primary_image_b64 = _primary_tally_ocr_image(payload)
    if not primary_image_b64:
        raise HTTPException(status_code=400, detail="image_b64 or images are required")
    if payload.image_b64 and len(payload.image_b64) > 12 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Canvas image too large")
    total_image_bytes = 0
    for image in payload.images:
        image_size = len(image.image_b64 or "")
        total_image_bytes += image_size
        if image_size > 12 * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"{image.label} image too large")
    if total_image_bytes > 30 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Exam tally OCR images are too large")

    tenant_db = await _tenant_db(db, current_user)
    await _assert_tally_context_access(tenant_db, current_user, payload.document, payload.student)
    prompt = _build_prompt(payload)

    try:
        result = await _analyze_tally_ocr(
            tenant_db=tenant_db,
            payload=payload,
            prompt=prompt,
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
    initial_ocr_columns = list(columns)
    initial_ocr_rows = [dict(row) for row in rows]
    uncertain_by_row = _normalise_uncertain_question_cells(parsed)

    if not rows:
        warnings.append("No table rows were confidently detected.")
    missing_by_row = _missing_questions_by_row(columns, rows, document_context)
    recheck_raw_text: Optional[str] = None
    recheck_provider: Optional[str] = None
    recheck_confidence: Optional[float] = None
    recheck_prompt: Optional[str] = None
    missing_label = _format_filled_recheck_ranges(missing_by_row)
    if missing_label:
        warnings.append(
            f"Initial full-sheet OCR left marks unresolved for {missing_label}; focused cell evidence was reviewed."
        )

    # The initial OCR result remains immutable evidence. A focused recheck may
    # resolve one exact cell only when its own image evidence is readable; a
    # teacher correction remains the final override when one exists.
    raw_rows = initial_ocr_rows
    raw_columns = initial_ocr_columns
    raw_validation_issues = _validate_tally_result(
        raw_columns,
        raw_rows,
        document_context,
        student_context,
    )
    uncertain_validation_issues = _uncertain_question_validation_issues(
        uncertain_by_row,
        document_context,
    )
    raw_validation_issues.extend(uncertain_validation_issues)
    question_evidence, crop_assets, full_evidence_hash = _build_tally_question_evidence(
        payload,
        document_context,
        raw_columns,
        raw_rows,
    )

    review: Optional[Dict[str, Any]] = None
    active_corrections: List[Dict[str, Any]] = []
    student_identity = _tally_student_identity(student_context)
    if document_context.document_id and student_identity:
        review = await tenant_db["exam_tally_mark_reviews"].find_one(
            {"_id": _tally_review_id(document_context.document_id, student_identity)}
        )
        active_corrections = _active_tally_corrections(review)

    _, _, initially_applied_corrections, _ = _apply_tally_mark_corrections(
        columns=raw_columns,
        raw_rows=raw_rows,
        corrections=active_corrections,
        question_evidence=question_evidence,
        copy_id=payload.copy_id,
        full_evidence_hash=full_evidence_hash,
    )
    initially_applied_correction_keys = {
        (item.row_index, item.question_number) for item in initially_applied_corrections
    }
    targeted_rechecks, targeted_recheck_debug = await _run_targeted_mark_rechecks(
        tenant_db=tenant_db,
        document=document_context,
        raw_rows=raw_rows,
        issues=[
            issue
            for issue in _tally_mark_review_issues(
                raw_validation_issues,
                raw_columns,
                raw_rows,
                document_context,
            )
            if (int(issue.row_index or 0), int(issue.question_number or 0))
            not in initially_applied_correction_keys
        ],
        crop_assets=crop_assets,
        evidence_by_question={item.question_number: item for item in question_evidence},
        include_debug=payload.debug,
    )
    extraction_id = uuid4().hex
    auto_resolution_records = _focused_rechecks_to_auto_resolutions(
        targeted_rechecks,
        document=document_context,
        copy_id=payload.copy_id,
    )
    for record in auto_resolution_records:
        record["source_extraction_id"] = extraction_id
    auto_columns, auto_rows, auto_resolved_marks = _apply_tally_auto_resolutions(
        columns=raw_columns,
        raw_rows=raw_rows,
        resolutions=auto_resolution_records,
        document=document_context,
        question_evidence=question_evidence,
        copy_id=payload.copy_id,
        full_evidence_hash=full_evidence_hash,
    )
    effective_columns, effective_rows, applied_corrections, stale_corrections = _apply_tally_mark_corrections(
        columns=auto_columns,
        raw_rows=auto_rows,
        corrections=active_corrections,
        question_evidence=question_evidence,
        copy_id=payload.copy_id,
        full_evidence_hash=full_evidence_hash,
    )
    applied_correction_keys = {
        (item.row_index, item.question_number) for item in applied_corrections
    }
    auto_resolved_keys = {
        (item.row_index, item.question_number) for item in auto_resolved_marks
    }
    overridden_auto_keys = applied_correction_keys.intersection(auto_resolved_keys)
    if overridden_auto_keys:
        auto_resolved_marks = [
            item
            for item in auto_resolved_marks
            if (item.row_index, item.question_number) not in overridden_auto_keys
        ]
        auto_resolved_keys -= overridden_auto_keys
    if stale_corrections:
        stale_labels = _format_question_ranges(
            [item.question_number for item in stale_corrections]
        )
        if stale_labels:
            warnings.append(
                f"Previous teacher correction is stale because its handwritten cell changed: {stale_labels}."
            )
    resolved_question_keys = applied_correction_keys.union(auto_resolved_keys)
    validation_issues = _validate_tally_result(
        effective_columns,
        effective_rows,
        document_context,
        student_context,
    )
    validation_issues.extend(
        issue
        for issue in uncertain_validation_issues
        if (int(issue.row_index or 0), int(issue.question_number or 0))
        not in resolved_question_keys
    )

    doc = {
        "_id": extraction_id,
        "document": document_context.model_dump(exclude_none=True, by_alias=True),
        "student": student_context.model_dump(exclude_none=True),
        "copy_id": payload.copy_id,
        "columns": raw_columns,
        "rows": raw_rows,
        "effective_columns": effective_columns,
        "effective_rows": effective_rows,
        "warnings": warnings,
        "validation_issues": [
            issue.model_dump(exclude_none=True) for issue in validation_issues
        ],
        "raw_validation_issues": [
            issue.model_dump(exclude_none=True) for issue in raw_validation_issues
        ],
        "question_evidence": [
            item.model_dump(exclude_none=True) for item in question_evidence
        ],
        "full_evidence_hash": full_evidence_hash,
        "targeted_rechecks": [
            item.model_dump(exclude_none=True) for item in targeted_rechecks
        ],
        "auto_resolved_marks": [
            item.model_dump(exclude_none=True) for item in auto_resolved_marks
        ],
        "applied_corrections": [
            item.model_dump(exclude_none=True) for item in applied_corrections
        ],
        "stale_corrections": [
            item.model_dump(exclude_none=True) for item in stale_corrections
        ],
        "confidence": confidence,
        "raw_text": raw_text,
        "recheck_raw_text": recheck_raw_text,
        "recheck_provider": recheck_provider,
        "recheck_confidence": recheck_confidence,
        "image_labels": [image.get("label") for image in ocr_images],
        "provider": result.get("provider"),
        "created_by": current_user.get("user_id"),
        "created_by_type": current_user.get("user_type"),
        "created_at": datetime.utcnow(),
    }
    await tenant_db["exam_tally_extractions"].insert_one(doc)

    return TallyExtractResponse(
        success=True,
        extraction_id=extraction_id,
        columns=effective_columns,
        rows=effective_rows,
        warnings=warnings,
        validation_issues=validation_issues,
        confidence=confidence,
        raw_text=raw_text,
        raw_rows=raw_rows,
        evidence_hash=full_evidence_hash,
        question_evidence=question_evidence,
        targeted_rechecks=[
            item for item in targeted_rechecks if item.status != "resolved"
        ],
        auto_resolved_marks=auto_resolved_marks,
        applied_corrections=applied_corrections,
        stale_corrections=stale_corrections,
        debug=TallyExtractDebugResponse(
            prompt=prompt,
            raw_text=raw_text,
            provider=result.get("provider"),
            recheck_prompt=recheck_prompt,
            recheck_raw_text=recheck_raw_text,
            recheck_provider=recheck_provider,
            image_labels=[image.get("label", "") for image in ocr_images],
            targeted_rechecks=targeted_recheck_debug,
        ) if payload.debug else None,
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
    correction_rows = [dict(item) for item in payload.corrections if isinstance(item, dict)]

    if payload.extraction_id and not rows:
        saved = await tenant_db["exam_tally_extractions"].find_one({"_id": payload.extraction_id})
        if not saved:
            raise HTTPException(status_code=404, detail="Extraction not found")
        columns = saved.get("effective_columns") or saved.get("columns") or []
        rows = saved.get("effective_rows") or saved.get("rows") or []
        validation_issues = saved.get("validation_issues") or []
        if not correction_rows:
            correction_rows = [
                dict(item)
                for item in saved.get("auto_resolved_marks") or []
                if isinstance(item, dict)
            ] + [
                dict(item)
                for item in saved.get("applied_corrections") or []
                if isinstance(item, dict)
            ]

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
                "Topic": item.get("topic") or "Unmapped",
                "Sub-topic": item.get("sub_topic") or "Unmapped",
                "Max Marks": max_marks if max_marks > 0 else "",
                "Confidence": item.get("confidence") if item.get("confidence") is not None else "",
                "Source": item.get("source") or "",
                "Question Preview": item.get("question_text_preview") or "",
            }
        )
    (
        summary_rows,
        topic_rows,
        subtopic_rows,
        class_topic_rows,
        class_subtopic_rows,
        question_rows,
        intervention_rows,
    ) = _build_analysis_rows(
        normalised_rows,
        ordered_columns,
        question_map_items,
        document_context,
    )
    student_percentages = [
        row["Percentage"]
        for row in summary_rows
        if isinstance(row.get("Percentage"), (int, float))
    ]
    correction_audit_rows = []
    for correction in correction_rows:
        is_focused_ocr = (
            str(correction.get("resolution_source") or "").strip().lower() == "focused_ocr"
            or "resolved_value" in correction
        )
        resolution_source = "Focused OCR" if is_focused_ocr else "Teacher Override"
        resolved_mark = (
            correction.get("resolved_value")
            if is_focused_ocr
            else correction.get("approved_value")
        )
        question_number = correction.get("question_number")
        question_label = (
            f"Q{int(question_number)}"
            if isinstance(question_number, (int, float)) or str(question_number or "").isdigit()
            else str(correction.get("column") or "")
        )
        correction_audit_rows.append(
            {
                "Student": correction.get("Selected Student") or correction.get("student_name") or "",
                "Student ID": correction.get("Selected Student ID") or correction.get("student_id") or "",
                "Question": question_label,
                "OCR Reading": correction.get("original_ocr_value") or "",
                "Resolution Source": resolution_source,
                "Resolved Mark": resolved_mark if resolved_mark is not None else "",
                "Focused OCR Candidate": (
                    correction.get("resolved_value")
                    if is_focused_ocr
                    else correction.get("targeted_candidate") or ""
                ),
                "Reason": correction.get("reason") or "",
                "Recorded At": correction.get("resolved_at") or correction.get("approved_at") or "",
                "Evidence Hash": correction.get("crop_hash") or "",
            }
        )
    focused_ocr_count = sum(
        1
        for correction in correction_rows
        if str(correction.get("resolution_source") or "").strip().lower() == "focused_ocr"
        or "resolved_value" in correction
    )
    teacher_override_count = len(correction_rows) - focused_ocr_count
    overview_rows = [
        {"Metric": "Exam", "Value": document_context.title or "Exam Tally"},
        {"Metric": "Subject", "Value": document_context.subject or ""},
        {"Metric": "Class", "Value": document_context.standard or ""},
        {"Metric": "Section", "Value": document_context.section or ""},
        {"Metric": "Students", "Value": len(summary_rows)},
        {"Metric": "Questions", "Value": len(question_map_items) or len(_find_question_columns(ordered_columns, normalised_rows))},
        {
            "Metric": "Class Average (%)",
            "Value": round(sum(student_percentages) / len(student_percentages), 2) if student_percentages else "",
        },
        {"Metric": "Students Needing Support", "Value": len({row["Student ID"] or row["Student"] for row in intervention_rows})},
        {"Metric": "Focused OCR resolutions", "Value": focused_ocr_count},
        {"Metric": "Teacher overrides", "Value": teacher_override_count},
    ]

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        header_format = workbook.add_format(
            {"bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "border": 0}
        )
        percentage_format = workbook.add_format({"num_format": '0.0"%"'})
        sheet_index = 0

        def write_sheet(sheet_name: str, sheet_rows: List[Dict[str, Any]]) -> None:
            nonlocal sheet_index
            if not sheet_rows:
                sheet = workbook.add_worksheet(sheet_name)
                sheet.write(0, 0, "No data available")
                return
            sheet_columns = list(sheet_rows[0].keys())
            sheet_df = pd.DataFrame(sheet_rows, columns=sheet_columns)
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            sheet = writer.sheets[sheet_name]
            sheet.freeze_panes(1, 0)
            for col_idx, col_name in enumerate(sheet_columns):
                sheet.write(0, col_idx, col_name, header_format)
                max_len = max([len(str(col_name))] + [len(str(row.get(col_name, ""))) for row in sheet_rows])
                cell_format = percentage_format if col_name == "Percentage" else None
                sheet.set_column(col_idx, col_idx, min(max(max_len + 2, 10), 42), cell_format)
                if col_name == "Percentage":
                    sheet.conditional_format(
                        1,
                        col_idx,
                        len(sheet_rows),
                        col_idx,
                        {
                            "type": "3_color_scale",
                            "min_color": "#F4CCCC",
                            "mid_color": "#FFF2CC",
                            "max_color": "#D9EAD3",
                            "min_value": 0,
                            "mid_value": 60,
                            "max_value": 100,
                        },
                    )
            sheet.add_table(
                0,
                0,
                len(sheet_rows),
                len(sheet_columns) - 1,
                {
                    "name": f"TallyTable{sheet_index}",
                    "columns": [{"header": column} for column in sheet_columns],
                    "style": "Table Style Medium 2",
                },
            )
            sheet_index += 1

        write_sheet("Overview", overview_rows)
        write_sheet("Student Summary", summary_rows)
        write_sheet("Class Topic Analysis", class_topic_rows)
        write_sheet("Class Sub-topic Analysis", class_subtopic_rows)
        write_sheet("Intervention Plan", intervention_rows)
        write_sheet("Topic Analysis", topic_rows)
        write_sheet("Sub-topic Analysis", subtopic_rows)
        write_sheet("Question Analysis", question_rows)
        write_sheet("Question Map", question_map_rows)
        write_sheet("OCR Corrections", correction_audit_rows)
        write_sheet("Exam Tally", normalised_rows)

        if class_topic_rows:
            chart = workbook.add_chart({"type": "column"})
            chart.add_series(
                {
                    "name": "Class performance by topic",
                    "categories": ["Class Topic Analysis", 1, 3, len(class_topic_rows), 3],
                    "values": ["Class Topic Analysis", 1, 9, len(class_topic_rows), 9],
                }
            )
            chart.set_title({"name": "Class performance by topic"})
            chart.set_y_axis({"name": "Percentage", "min": 0, "max": 100})
            chart.set_legend({"none": True})
            writer.sheets["Overview"].insert_chart("D2", chart, {"x_scale": 1.35, "y_scale": 1.2})

    output.seek(0)
    title = payload.filename or (payload.document.title if payload.document else None)
    filename = f"{_safe_filename(title, 'exam-tally')}.xlsx"
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
