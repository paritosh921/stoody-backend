from __future__ import annotations

import sys
from pathlib import Path
from inspect import signature, getsource
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def test_create_student_request_requires_section():
    from api.v1.admin_async import CreateStudentRequest

    with pytest.raises(ValidationError):
        CreateStudentRequest(full_name="Test Student", password="secret123", grade="10")

    with pytest.raises(ValidationError):
        CreateStudentRequest(
            full_name="Test Student",
            password="secret123",
            grade="10",
            section="   ",
        )


def test_update_student_request_rejects_blank_section_when_present():
    from api.v1.admin_async import UpdateStudentRequest

    with pytest.raises(ValidationError):
        UpdateStudentRequest(section=" ")


def test_bulk_student_section_validation_rejects_missing_section():
    from api.v1.student_bulk_upload import validate_section_for_class

    assert validate_section_for_class("", "10", ["A", "B"], {"10": ["A", "B"]}) is False
    assert validate_section_for_class("A", "10", ["A", "B"], {"10": ["A", "B"]}) is True


def test_bulk_student_grade_validation_requires_exact_settings_format():
    from api.v1 import student_bulk_upload as bulk

    assert bulk.validate_grade("IV", ["IV"]) is True
    assert bulk.validate_grade("4", ["IV"]) is False
    assert bulk.validate_grade("IV", ["4"]) is False

    formatter = getattr(bulk, "format_invalid_grade_message", None)
    assert callable(formatter), "format_invalid_grade_message should explain exact class format"

    message = formatter("4", ["IV"])
    assert "Invalid grade '4'" in message
    assert "Allowed classes: IV" in message
    assert "exact class format configured in Settings" in message
    assert "roman numerals" in message
    assert "numeric values" in message


def test_school_settings_class_values_strip_display_prefix():
    from core.school_settings_format import clean_class_sections, clean_class_values, validate_class_values

    assert clean_class_values(["Class X", "Nursery", " class iv ", "4", "Class X"]) == ["X", "IV", "4"]
    with pytest.raises(ValueError, match="Nursery"):
        validate_class_values(["Class Nursery"])
    assert clean_class_sections(
        {"Class X": ["A", "Z"], "Class 4": ["B"], "Old": ["A"]},
        classes=["X", "4"],
        sections=["A", "B"],
    ) == {"X": ["A"], "4": ["B"]}


@pytest.mark.asyncio
async def test_school_settings_update_rejects_non_numeric_non_roman_classes():
    from fastapi import HTTPException
    from api.v1.settings_async import SchoolSettingsRequest, update_school_settings

    class FakeDB:
        async def mongo_find_one(self, *args, **kwargs):
            return None

        async def mongo_update_one(self, *args, **kwargs):
            raise AssertionError("invalid class settings should not be saved")

    class FakeCache:
        async def delete(self, *args, **kwargs):
            return None

    with pytest.raises(HTTPException) as exc:
        await update_school_settings.__wrapped__(
            request=SimpleNamespace(),
            settings_data=SchoolSettingsRequest(classes=["Class Nursery"]),
            current_user={"user_id": "690c55fd3c6e1a875ea134e8", "user_type": "admin"},
            db=FakeDB(),
            cache=FakeCache(),
        )

    assert exc.value.status_code == 400
    assert "number or roman numeral" in exc.value.detail


def test_bulk_student_upload_does_not_accept_default_class_section_or_subject():
    from api.v1 import student_bulk_upload as bulk

    preview_params = signature(bulk.preview_bulk_upload).parameters
    import_params = signature(bulk.import_bulk_students).parameters

    for params in (preview_params, import_params):
        assert "default_grade" not in params
        assert "default_section" not in params
        assert "default_stream" not in params
        assert "default_subject" not in params

    import_source = getsource(bulk.import_bulk_students)
    assert "settings_subjects" not in import_source
    assert "If subjects is empty, assign ALL valid subjects from settings" not in import_source


@pytest.mark.asyncio
async def test_bulk_student_preview_derives_excel_file_type_from_upload_filename(monkeypatch):
    from api.v1 import student_bulk_upload as bulk

    class FakeSettingsCollection:
        async def find_one(self, query):
            return {
                "classes": ["X", "10"],
                "sections": ["A", "B"],
                "class_sections": {"X": ["A"], "10": ["A", "B"]},
                "subjects": [],
                "plan_types": [],
                "school_info": {"school_name": "CIEL"},
            }

    class FakeTenantDB:
        def __getitem__(self, name):
            assert name == "school_settings"
            return FakeSettingsCollection()

    class FakeDB:
        async def mongo_find(self, *args, **kwargs):
            return []

    async def fake_parse_upload_file(file, *, current_user, db, purpose):
        return pd.DataFrame(
            [
                {"full_name": "Rahul Sharma", "username": "Rahul", "grade": "X", "section": "A"},
                {"full_name": "Priya Patel", "username": "Priya", "grade": "10", "section": "A"},
            ]
        )

    async def fake_get_tenant_db_or_403(db, current_user):
        return FakeTenantDB()

    monkeypatch.setattr(bulk, "parse_upload_file", fake_parse_upload_file)
    monkeypatch.setattr(bulk, "get_tenant_db_or_403", fake_get_tenant_db_or_403)

    response = await bulk.preview_bulk_upload.__wrapped__(
        request=SimpleNamespace(),
        file=SimpleNamespace(filename="bulk_upload.xlsx"),
        current_user={"user_id": "690c55fd3c6e1a875ea134e8"},
        db=FakeDB(),
    )

    assert response.file_type == "Excel"
    assert response.valid_rows == 2


@pytest.mark.asyncio
async def test_bulk_student_preview_uses_objectid_school_settings_for_grade_validation(monkeypatch):
    from bson import ObjectId
    from api.v1 import student_bulk_upload as bulk

    admin_id = "690c55fd3c6e1a875ea134e8"

    class FakeSettingsCollection:
        async def find_one(self, query):
            if query == {"admin_id": admin_id}:
                return None
            if query == {"admin_id": ObjectId(admin_id)}:
                return {
                    "classes": ["IV"],
                    "sections": ["A"],
                    "class_sections": {"IV": ["A"]},
                    "subjects": [],
                    "plan_types": [],
                    "school_info": {"school_name": "CIEL"},
                }
            return None

    class FakeTenantDB:
        def __getitem__(self, name):
            assert name == "school_settings"
            return FakeSettingsCollection()

    class FakeDB:
        async def mongo_find(self, *args, **kwargs):
            return []

    async def fake_parse_upload_file(file, *, current_user, db, purpose):
        return pd.DataFrame(
            [{"full_name": "Priya Patel", "username": "Priya", "grade": "10", "section": "A"}]
        )

    async def fake_get_tenant_db_or_403(db, current_user):
        return FakeTenantDB()

    monkeypatch.setattr(bulk, "parse_upload_file", fake_parse_upload_file)
    monkeypatch.setattr(bulk, "get_tenant_db_or_403", fake_get_tenant_db_or_403)

    response = await bulk.preview_bulk_upload.__wrapped__(
        request=SimpleNamespace(),
        file=SimpleNamespace(filename="bulk_upload.xlsx"),
        current_user={"user_id": admin_id},
        db=FakeDB(),
    )

    assert response.success is False
    assert response.valid_rows == 0
    assert response.errors[0].field == "grade"
    assert "Allowed classes: IV" in response.errors[0].message
