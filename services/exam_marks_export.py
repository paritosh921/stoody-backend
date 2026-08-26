"""Create a teacher-facing Excel workbook from canonical ExamPen results."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from io import BytesIO
from typing import Any, Iterable, Mapping, Optional

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


_STATUS_LABELS = {
    "expected": "Not submitted",
    "submitted": "Submitted",
    "evaluating": "Checking",
    "blocked": "Needs attention",
    "review": "Under review",
    "ready": "Ready to publish",
    "published": "Published",
}


def _safe_excel_text(value: Any) -> str:
    """Prevent user-controlled identifiers/names from becoming Excel formulas."""

    text = str(value or "")
    if text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def _filename_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", normalized).strip("-").lower()
    return slug[:80] or "exam"


def exam_marks_filename(exam_title: str) -> str:
    return f"{_filename_slug(exam_title)}-student-marks.xlsx"


def build_exam_marks_workbook(
    *,
    exam_id: str,
    exam_title: str,
    class_label: Optional[str],
    roster_rows: Iterable[Mapping[str, Any]],
    result_rows: Iterable[Mapping[str, Any]],
    generated_at: datetime,
) -> bytes:
    """Return an .xlsx containing every roster row and its current safe score."""

    roster = list(roster_rows)
    results_by_student = {
        str(row.get("student_id") or ""): row
        for row in result_rows
        if str(row.get("student_id") or "")
    }

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Student marks"
    sheet.sheet_view.showGridLines = False
    workbook.properties.title = f"{exam_title} student marks"
    workbook.properties.subject = "Stoody ExamPen student marks export"
    workbook.properties.creator = "Stoody"

    headers = [
        "S.No.",
        "Student name",
        "Student ID",
        "Submission status",
        "Marks obtained",
        "Maximum marks",
        "Percentage",
        "PCR marks",
        "PCR maximum",
        "DCR marks",
        "DCR maximum",
        "Result status",
        "Open rechecks",
    ]
    final_column = get_column_letter(len(headers))

    sheet.merge_cells(f"A1:{final_column}1")
    title_cell = sheet["A1"]
    title_cell.value = _safe_excel_text(exam_title)
    title_cell.font = Font(size=16, bold=True, color="FFFFFF")
    title_cell.fill = PatternFill("solid", fgColor="0F766E")
    title_cell.alignment = Alignment(vertical="center")
    sheet.row_dimensions[1].height = 28

    sheet.merge_cells("A2:F2")
    sheet["A2"] = f"Exam ID: {_safe_excel_text(exam_id)}"
    sheet.merge_cells(f"G2:{final_column}2")
    sheet["G2"] = f"Class: {_safe_excel_text(class_label or 'Not specified')}"
    sheet.merge_cells(f"A3:{final_column}3")
    sheet["A3"] = f"Generated: {generated_at.astimezone().strftime('%d %b %Y, %I:%M %p %Z')}"
    for row_number in (2, 3):
        for cell in sheet[row_number]:
            cell.font = Font(size=10, color="475569")
            cell.alignment = Alignment(vertical="center")

    header_row = 5
    header_fill = PatternFill("solid", fgColor="DFF5EE")
    header_font = Font(bold=True, color="134E4A")
    thin_border = Border(bottom=Side(style="thin", color="99C9BA"))
    for column, value in enumerate(headers, start=1):
        cell = sheet.cell(row=header_row, column=column, value=value)
        cell.fill = header_fill
        cell.font = header_font
        cell.border = thin_border
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    sheet.row_dimensions[header_row].height = 30

    for sequence, roster_row in enumerate(roster, start=1):
        student_id = str(roster_row.get("student_id") or "")
        result = results_by_student.get(student_id)
        roster_status = str(roster_row.get("status") or "expected").lower()
        publication_status = str((result or {}).get("publication_status") or "").lower()
        score_state = str((result or {}).get("score_state") or "available").lower()
        score_available = bool(
            result
            and (
                publication_status == "published"
                or score_state == "available"
            )
            and float((result or {}).get("combined_max") or 0) > 0
        )

        if publication_status == "published":
            result_status = "Published"
        elif result and score_state == "processing":
            result_status = "Checking"
        elif result and score_state == "unavailable":
            result_status = "Unavailable"
        else:
            result_status = _STATUS_LABELS.get(
                roster_status,
                roster_status.replace("_", " ").title(),
            )

        values: list[Any] = [
            sequence,
            _safe_excel_text(roster_row.get("student_name") or student_id),
            _safe_excel_text(student_id),
            _STATUS_LABELS.get(
                roster_status,
                roster_status.replace("_", " ").title(),
            ),
            float(result.get("combined_total") or 0) if score_available and result else None,
            float(result.get("combined_max") or 0) if score_available and result else None,
            (
                float(result.get("combined_total") or 0)
                / float(result.get("combined_max") or 0)
                if score_available and result
                else None
            ),
            float(result.get("pcr_total_score") or 0) if score_available and result else None,
            float(result.get("pcr_max_score") or 0) if score_available and result else None,
            float(result.get("dcr_total_score") or 0) if score_available and result else None,
            float(result.get("dcr_max_score") or 0) if score_available and result else None,
            result_status,
            int(roster_row.get("open_recheck_count") or 0),
        ]
        worksheet_row = header_row + sequence
        for column, value in enumerate(values, start=1):
            cell = sheet.cell(row=worksheet_row, column=column, value=value)
            cell.alignment = Alignment(
                vertical="center",
                horizontal="right" if column in {1, 5, 6, 7, 8, 9, 10, 11, 13} else "left",
            )
            if worksheet_row % 2 == 0:
                cell.fill = PatternFill("solid", fgColor="F8FAFC")
        sheet.cell(row=worksheet_row, column=7).number_format = "0.00%"
        for column in (5, 6, 8, 9, 10, 11):
            sheet.cell(row=worksheet_row, column=column).number_format = "0.00"

    last_row = header_row + len(roster)
    sheet.freeze_panes = f"A{header_row + 1}"
    sheet.auto_filter.ref = f"A{header_row}:{final_column}{max(last_row, header_row)}"
    sheet.print_title_rows = f"1:{header_row}"
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.fitToWidth = 1
    sheet.sheet_properties.pageSetUpPr.fitToPage = True

    widths = {
        1: 8,
        2: 24,
        3: 22,
        4: 20,
        5: 16,
        6: 16,
        7: 13,
        8: 13,
        9: 15,
        10: 13,
        11: 15,
        12: 18,
        13: 15,
    }
    for column, width in widths.items():
        sheet.column_dimensions[get_column_letter(column)].width = width

    output = BytesIO()
    workbook.save(output)
    return output.getvalue()
