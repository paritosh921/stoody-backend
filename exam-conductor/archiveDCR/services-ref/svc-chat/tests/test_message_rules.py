"""Unit tests for domain/message_rules.py -- ZERO I/O, pure logic.

Test IDs: U-CHAT-MR-01 through U-CHAT-MR-12
"""

from src.domain.message_rules import (
    ALLOWED_SENDER_ROLES,
    ALLOWED_STUDENT_ROLE,
    MAX_CONTENT_LENGTH,
    check_rbac,
    check_sender_role,
    is_teacher_role,
    validate_message,
)


# -- U-CHAT-MR-01: Valid message passes validation -------------------------


def test_valid_message_passes():
    """U-CHAT-MR-01: Well-formed message returns valid=True."""
    result = validate_message(
        sender_id="teacher-1",
        recipient_id="student-1",
        exam_id="exam-abc",
        content="Your answer for Q3 needs more detail.",
    )
    assert result.valid is True
    assert result.error is None


# -- U-CHAT-MR-02: Empty fields rejected ----------------------------------


def test_empty_sender_id_rejected():
    """U-CHAT-MR-02a: Empty sender_id is invalid."""
    result = validate_message("", "student-1", "exam-1", "hello")
    assert result.valid is False
    assert "sender_id" in result.error


def test_empty_recipient_id_rejected():
    """U-CHAT-MR-02b: Empty recipient_id is invalid."""
    result = validate_message("teacher-1", "", "exam-1", "hello")
    assert result.valid is False
    assert "recipient_id" in result.error


def test_empty_exam_id_rejected():
    """U-CHAT-MR-02c: Empty exam_id is invalid."""
    result = validate_message("teacher-1", "student-1", "", "hello")
    assert result.valid is False
    assert "exam_id" in result.error


def test_empty_content_rejected():
    """U-CHAT-MR-02d: Empty content is invalid."""
    result = validate_message("teacher-1", "student-1", "exam-1", "")
    assert result.valid is False
    assert "content" in result.error.lower()


def test_whitespace_only_content_rejected():
    """U-CHAT-MR-02e: Whitespace-only content is invalid."""
    result = validate_message("teacher-1", "student-1", "exam-1", "   ")
    assert result.valid is False


# -- U-CHAT-MR-03: Self-message rejected ----------------------------------


def test_self_message_rejected():
    """U-CHAT-MR-03: sender_id == recipient_id is invalid."""
    result = validate_message("user-1", "user-1", "exam-1", "hello")
    assert result.valid is False
    assert "self" in result.error.lower()


# -- U-CHAT-MR-04: Content length enforcement -----------------------------


def test_content_at_limit_passes():
    """U-CHAT-MR-04a: Content at exactly MAX_CONTENT_LENGTH passes."""
    content = "x" * MAX_CONTENT_LENGTH
    result = validate_message("t", "s", "e", content)
    assert result.valid is True


def test_content_over_limit_rejected():
    """U-CHAT-MR-04b: Content exceeding MAX_CONTENT_LENGTH is rejected."""
    content = "x" * (MAX_CONTENT_LENGTH + 1)
    result = validate_message("t", "s", "e", content)
    assert result.valid is False
    assert "2000" in result.error


# -- U-CHAT-MR-05: Sender role validation ---------------------------------


def test_teacher_role_allowed():
    """U-CHAT-MR-05a: teacher role can send messages."""
    assert check_sender_role("teacher").valid is True


def test_evaluator_role_allowed():
    """U-CHAT-MR-05b: evaluator role can send messages."""
    assert check_sender_role("evaluator").valid is True


def test_tutor_role_allowed():
    """U-CHAT-MR-05c: tutor role can send messages."""
    assert check_sender_role("tutor").valid is True


def test_student_role_allowed():
    """U-CHAT-MR-05d: student role can send messages."""
    assert check_sender_role("student").valid is True


def test_parent_role_blocked():
    """U-CHAT-MR-05e: parent role cannot send messages."""
    result = check_sender_role("parent")
    assert result.valid is False
    assert "parent" in result.error


def test_admin_role_blocked():
    """U-CHAT-MR-05f: admin/principal role cannot send messages."""
    assert check_sender_role("principal").valid is False
    assert check_sender_role("hod").valid is False


# -- U-CHAT-MR-06: is_teacher_role ----------------------------------------


def test_is_teacher_role_true():
    """U-CHAT-MR-06a: teacher-like roles return True."""
    for role in ALLOWED_SENDER_ROLES:
        assert is_teacher_role(role) is True


def test_is_teacher_role_false_for_student():
    """U-CHAT-MR-06b: student is not a teacher role."""
    assert is_teacher_role("student") is False


# -- U-CHAT-MR-07: RBAC — teacher can message own students ----------------


def test_teacher_can_message_own_student():
    """U-CHAT-MR-07a: Teacher messages a student in their exam."""
    result = check_rbac(
        sender_role="teacher",
        sender_id="teacher-1",
        recipient_id="student-1",
        teacher_ids=["teacher-1"],
        student_ids=["student-1", "student-2"],
    )
    assert result.valid is True


def test_teacher_cannot_message_foreign_student():
    """U-CHAT-MR-07b: Teacher cannot message a student not in their exam."""
    result = check_rbac(
        sender_role="teacher",
        sender_id="teacher-1",
        recipient_id="student-99",
        teacher_ids=["teacher-1"],
        student_ids=["student-1", "student-2"],
    )
    assert result.valid is False
    assert "students in their exam" in result.error


# -- U-CHAT-MR-08: RBAC — student can message exam tutors -----------------


def test_student_can_message_exam_tutor():
    """U-CHAT-MR-08a: Student messages a tutor assigned to the exam."""
    result = check_rbac(
        sender_role="student",
        sender_id="student-1",
        recipient_id="teacher-1",
        teacher_ids=["teacher-1", "teacher-2"],
        student_ids=[],
    )
    assert result.valid is True


def test_student_cannot_message_non_exam_tutor():
    """U-CHAT-MR-08b: Student cannot message a tutor not on the exam."""
    result = check_rbac(
        sender_role="student",
        sender_id="student-1",
        recipient_id="teacher-99",
        teacher_ids=["teacher-1"],
        student_ids=[],
    )
    assert result.valid is False
    assert "tutors of the exam" in result.error


# -- U-CHAT-MR-09: RBAC — disallowed role ---------------------------------


def test_parent_rbac_blocked():
    """U-CHAT-MR-09: Parent role fails RBAC check."""
    result = check_rbac(
        sender_role="parent",
        sender_id="parent-1",
        recipient_id="teacher-1",
        teacher_ids=["teacher-1"],
        student_ids=[],
    )
    assert result.valid is False
