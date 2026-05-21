from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional
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


class TallyExtractRequest(BaseModel):
    image_b64: str = Field(..., description="Full-page canvas PNG data URL or raw base64")
    document: Optional[TallyDocumentContext] = None
    student: Optional[TallyStudentContext] = None
    copy_id: Optional[str] = None


class TallyExtractResponse(BaseModel):
    success: bool
    extraction_id: str
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    raw_text: Optional[str] = None


class TallyExportRequest(BaseModel):
    extraction_id: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    filename: Optional[str] = None
    document: Optional[TallyDocumentContext] = None
    student: Optional[TallyStudentContext] = None


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


def _format_marking_scheme(scheme: List[TallyMarkingRange]) -> str:
    parts: List[str] = []
    for item in scheme:
        if item.from_ <= 0 or item.to <= 0 or item.marks <= 0 or item.from_ > item.to:
            continue
        question_range = f"Q{item.from_}" if item.from_ == item.to else f"Q{item.from_}-Q{item.to}"
        parts.append(f"{question_range} max {_format_marks(item.marks)}")
    return ", ".join(parts)


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
    if not rows:
        warnings.append("No table rows were confidently detected.")

    extraction_id = uuid4().hex
    doc = {
        "_id": extraction_id,
        "document": (payload.document or TallyDocumentContext()).model_dump(exclude_none=True, by_alias=True),
        "student": (payload.student or TallyStudentContext()).model_dump(exclude_none=True),
        "copy_id": payload.copy_id,
        "columns": columns,
        "rows": rows,
        "warnings": warnings,
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
        confidence=confidence,
        raw_text=raw_text,
    )


@router.post("/export")
async def export_tally(
    payload: TallyExportRequest,
    current_user: Dict[str, Any] = Depends(_require_admin_or_tutor),
    db: DatabaseManager = Depends(get_database),
):
    tenant_db = await _tenant_db(db, current_user)
    columns = payload.columns
    rows = payload.rows

    if payload.extraction_id and not rows:
        saved = await tenant_db["exam_tally_extractions"].find_one({"_id": payload.extraction_id})
        if not saved:
            raise HTTPException(status_code=404, detail="Extraction not found")
        columns = saved.get("columns") or []
        rows = saved.get("rows") or []

    if not rows:
        raise HTTPException(status_code=400, detail="No rows available to export")

    normalised_rows = [_flatten_row(row) for row in rows]
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
