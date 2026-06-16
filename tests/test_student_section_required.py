from __future__ import annotations

import sys
from pathlib import Path

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
