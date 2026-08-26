from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import AsyncMock, patch

from openpyxl import load_workbook
import pytest


def test_exam_marks_workbook_includes_full_roster_and_hides_stale_scores():
    from services.exam_marks_export import build_exam_marks_workbook

    content = build_exam_marks_workbook(
        exam_id="EXAM-1",
        exam_title="Vectors and Projectile",
        class_label="Class 10 - Section A",
        generated_at=datetime(2026, 8, 27, 10, 30, tzinfo=timezone.utc),
        roster_rows=[
            {
                "student_id": "aaradhya",
                "student_name": "Aaradhya Tomar",
                "status": "published",
                "open_recheck_count": 0,
            },
            {
                "student_id": "missing-student",
                "student_name": "Missing Student",
                "status": "expected",
                "open_recheck_count": 0,
            },
            {
                "student_id": "stale-student",
                "student_name": "=HYPERLINK(\"https://invalid.example\")",
                "status": "blocked",
                "open_recheck_count": 1,
            },
        ],
        result_rows=[
            {
                "student_id": "aaradhya",
                "pcr_total_score": 17,
                "pcr_max_score": 25,
                "dcr_total_score": 0,
                "dcr_max_score": 0,
                "combined_total": 17,
                "combined_max": 25,
                "publication_status": "published",
                "score_state": "available",
            },
            {
                "student_id": "stale-student",
                "pcr_total_score": 14,
                "pcr_max_score": 25,
                "dcr_total_score": 0,
                "dcr_max_score": 0,
                "combined_total": 14,
                "combined_max": 25,
                "publication_status": "pending",
                "score_state": "unavailable",
            },
        ],
    )

    workbook = load_workbook(BytesIO(content), data_only=False)
    sheet = workbook["Student marks"]

    assert sheet["A1"].value == "Vectors and Projectile"
    assert sheet["G2"].value == "Class: Class 10 - Section A"
    assert sheet["A5"].value == "S.No."
    assert sheet["M5"].value == "Open rechecks"
    assert sheet.freeze_panes == "A6"
    assert sheet.auto_filter.ref == "A5:M8"

    assert sheet["B6"].value == "Aaradhya Tomar"
    assert sheet["E6"].value == 17
    assert sheet["F6"].value == 25
    assert sheet["G6"].value == 17 / 25
    assert sheet["L6"].value == "Published"

    assert sheet["D7"].value == "Not submitted"
    assert sheet["E7"].value is None
    assert sheet["L7"].value == "Not submitted"

    assert sheet["B8"].value.startswith("'=")
    assert sheet["E8"].value is None
    assert sheet["F8"].value is None
    assert sheet["L8"].value == "Unavailable"
    assert sheet["M8"].value == 1


def test_exam_marks_filename_is_safe_and_readable():
    from services.exam_marks_export import exam_marks_filename

    assert (
        exam_marks_filename("Vectors & Projectile / Class 10")
        == "vectors-projectile-class-10-student-marks.xlsx"
    )


@pytest.mark.asyncio
async def test_export_endpoint_returns_a_real_workbook_for_the_authorized_roster():
    from mongomock_motor import AsyncMongoMockClient

    from api.v1.evalpen_review_async import (
        CollectionRosterItemAPI,
        ExamResultStudentAPI,
        ExamResultsAPI,
        ExamRosterAPI,
        export_exam_results,
    )

    tenant_db = AsyncMongoMockClient()["skb_test"]
    await tenant_db["exampen_exams"].insert_one(
        {
            "exam_id": "EXAM-EXPORT",
            "title": "Vectors and Projectile",
            "class_name": "10",
            "section_name": "A",
        }
    )
    roster = ExamRosterAPI(
        exam_id="EXAM-EXPORT",
        expected_students=[
            CollectionRosterItemAPI(
                student_id="student-1",
                student_name="Student One",
                status="published",
            )
        ],
        total_expected=1,
        total_submitted=1,
        total_published=1,
    )
    results = ExamResultsAPI(
        exam_id="EXAM-EXPORT",
        students=[
            ExamResultStudentAPI(
                student_id="student-1",
                combined_total=22,
                combined_max=25,
                pcr_total_score=22,
                pcr_max_score=25,
                publication_status="published",
            )
        ],
        total_students=1,
    )

    with (
        patch(
            "api.v1.evalpen_review_async.get_exam_roster",
            new=AsyncMock(return_value=roster),
        ),
        patch(
            "api.v1.evalpen_review_async.get_exam_results",
            new=AsyncMock(return_value=results),
        ),
        patch(
            "api.v1.evalpen_review_async._get_tenant_db",
            new=AsyncMock(return_value=tenant_db),
        ),
    ):
        response = await export_exam_results(
            exam_id="EXAM-EXPORT",
            current_user={
                "user_id": "admin-1",
                "user_type": "admin",
                "db_name": "skb_test",
            },
            db=None,
        )

    content = b"".join([chunk async for chunk in response.body_iterator])
    workbook = load_workbook(BytesIO(content))
    sheet = workbook["Student marks"]

    assert response.media_type == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert (
        response.headers["content-disposition"]
        == 'attachment; filename="vectors-and-projectile-student-marks.xlsx"'
    )
    assert sheet["B6"].value == "Student One"
    assert sheet["E6"].value == 22
    assert sheet["F6"].value == 25
