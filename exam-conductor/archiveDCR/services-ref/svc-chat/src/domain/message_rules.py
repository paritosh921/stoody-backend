"""Chat message validation and RBAC rules -- ZERO I/O, pure domain logic.

APPEND-ONLY contract: no UPDATE, no DELETE at any layer.
This module validates message content and enforces role-based access:
- Only teachers and students may send messages.
- Teachers can message their own students only.
- Students can message exam tutors only.

DPDPA audit safety: minors' chat data is never edited or deleted
through the application layer.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_CONTENT_LENGTH: int = 2000

ALLOWED_SENDER_ROLES: frozenset[str] = frozenset({"teacher", "evaluator", "tutor"})
ALLOWED_STUDENT_ROLE: str = "student"


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of message validation."""

    valid: bool
    error: str | None = None


def validate_message(
    sender_id: str,
    recipient_id: str,
    exam_id: str,
    content: str,
) -> ValidationResult:
    """Validate a chat message before persistence.

    Checks:
    1. sender_id and recipient_id must be non-empty and distinct.
    2. exam_id must be non-empty.
    3. content must be non-empty and within MAX_CONTENT_LENGTH chars.

    Returns a ``ValidationResult`` indicating pass/fail with an error
    message on failure.
    """
    if not sender_id or not sender_id.strip():
        return ValidationResult(valid=False, error="sender_id is required")

    if not recipient_id or not recipient_id.strip():
        return ValidationResult(valid=False, error="recipient_id is required")

    if sender_id == recipient_id:
        return ValidationResult(valid=False, error="Cannot send message to self")

    if not exam_id or not exam_id.strip():
        return ValidationResult(valid=False, error="exam_id is required")

    if not content or not content.strip():
        return ValidationResult(valid=False, error="Message content is required")

    if len(content) > MAX_CONTENT_LENGTH:
        return ValidationResult(
            valid=False,
            error=f"Content exceeds {MAX_CONTENT_LENGTH} character limit",
        )

    return ValidationResult(valid=True)


def check_sender_role(sender_role: str) -> ValidationResult:
    """Verify the sender has a role allowed to send chat messages.

    Only teachers (teacher/evaluator/tutor) and students may send.
    Parents, admins, and other roles are blocked.
    """
    if sender_role == ALLOWED_STUDENT_ROLE:
        return ValidationResult(valid=True)
    if sender_role in ALLOWED_SENDER_ROLES:
        return ValidationResult(valid=True)
    return ValidationResult(
        valid=False,
        error=f"Role '{sender_role}' is not allowed to send chat messages",
    )


def is_teacher_role(role: str) -> bool:
    """Return True if the role is a teacher-like role."""
    return role in ALLOWED_SENDER_ROLES


def check_rbac(
    sender_role: str,
    sender_id: str,
    recipient_id: str,
    teacher_ids: list[str],
    student_ids: list[str],
) -> ValidationResult:
    """Enforce RBAC constraints on who can message whom.

    Rules:
    - Teacher can message only students assigned to their exam.
    - Student can message only tutors assigned to the exam.
    """
    role_check = check_sender_role(sender_role)
    if not role_check.valid:
        return role_check

    if is_teacher_role(sender_role):
        # Teacher -> must target one of their own students
        if recipient_id not in student_ids:
            return ValidationResult(
                valid=False,
                error="Teacher can only message students in their exam",
            )
        return ValidationResult(valid=True)

    if sender_role == ALLOWED_STUDENT_ROLE:
        # Student -> must target a tutor assigned to the exam
        if recipient_id not in teacher_ids:
            return ValidationResult(
                valid=False,
                error="Student can only message tutors of the exam",
            )
        return ValidationResult(valid=True)

    return ValidationResult(valid=False, error="Unknown sender role")
