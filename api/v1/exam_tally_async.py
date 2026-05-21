from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from difflib import SequenceMatcher
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.v1.auth_async import get_current_user, get_database
from core.database import DatabaseManager
from core.ocr_service import get_ocr_service

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
    if document.max_marks_per_question and document.max_marks_per_question > 0:
        return float(document.max_marks_per_question)
    return None


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
    seen_columns = set()
    for column in columns:
        question_number = _question_number_from_label(column)
        if question_number is not None and column not in seen_columns:
            question_columns.append((column, question_number))
            seen_columns.add(column)

    for row in rows:
        for column in row.keys():
            if column in seen_columns:
                continue
            question_number = _question_number_from_label(column)
            if question_number is not None:
                question_columns.append((column, question_number))
                seen_columns.add(column)

    if max_question and not question_columns:
        issues.append(
            TallyValidationIssue(
                severity="warning",
                code="question_columns_missing",
                message="No question columns were confidently detected for validation.",
            )
        )

    for row_index, row in enumerate(rows):
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
    context = {
        "document": document.model_dump(exclude_none=True, by_alias=True),
        "selected_student": student.model_dump(exclude_none=True),
        "copy_id": payload.copy_id,
    }
    return f"""
You are reading a full-page handwritten exam tally sheet drawn on a digital canvas.
The sheet may contain hand-drawn table borders and handwritten headings/values.

Task:
1. Detect the table structure from the image.
2. Read all headings exactly as intended, including labels like NAME, ROLL NO., PAPER SET, Q1, Q01, Q2, TOTAL, MAX MARKS.
3. Pair each value with the correct heading/cell.
4. If the sheet is a single-student form, return one row.
5. If the sheet has multiple student rows, return all rows.
6. Preserve blank cells as empty strings.
7. Normalize question headings to Q1, Q2, Q3... where obvious.
8. Do not invent marks or names. If uncertain, keep the cell empty and add a warning.{marking_rule}

Context from the UI, for disambiguation only:
{json.dumps(context, ensure_ascii=False)}

Return ONLY strict JSON in this shape:
{{
  "columns": ["NAME", "ROLL NO.", "PAPER SET", "Q1", "Q2"],
  "rows": [
    {{"NAME": "", "ROLL NO.": "", "PAPER SET": "", "Q1": "", "Q2": ""}}
  ],
  "warnings": [],
  "confidence": 0.0
}}
""".strip()


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

    output.seek(0)
    title = payload.filename or (payload.document.title if payload.document else None)
    filename = f"{_safe_filename(title, 'exam-tally')}.xlsx"
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
