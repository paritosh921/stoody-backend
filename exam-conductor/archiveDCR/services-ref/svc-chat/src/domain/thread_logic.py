"""Chat thread identity and participation rules -- ZERO I/O, pure domain logic.

A thread is uniquely identified by the triple
``(exam_id, teacher_id, student_id)``. This module provides the
``ThreadKey`` data structure and participation-checking logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ThreadKey:
    """Unique identity of a chat thread.

    Every message belongs to exactly one thread, determined by the
    exam, the teacher, and the student involved in the conversation.
    """

    exam_id: str
    teacher_id: str
    student_id: str


def build_thread_key(
    exam_id: str,
    sender_id: str,
    recipient_id: str,
    sender_role: str,
) -> ThreadKey:
    """Derive the canonical ThreadKey from message participants.

    The teacher_id and student_id slots are filled based on the
    sender's role so that both directions of a conversation share
    the same thread.
    """
    from src.domain.message_rules import is_teacher_role

    if is_teacher_role(sender_role):
        return ThreadKey(
            exam_id=exam_id,
            teacher_id=sender_id,
            student_id=recipient_id,
        )
    # Student is sender
    return ThreadKey(
        exam_id=exam_id,
        teacher_id=recipient_id,
        student_id=sender_id,
    )


def can_participate(
    user_id: str,
    user_role: str,
    thread_key: ThreadKey,
) -> bool:
    """Check whether *user_id* with *user_role* may access *thread_key*.

    Rules:
    - A teacher may participate if they are the teacher_id in the thread.
    - A student may participate if they are the student_id in the thread.
    - All other roles are excluded.
    """
    from src.domain.message_rules import is_teacher_role

    if is_teacher_role(user_role):
        return user_id == thread_key.teacher_id
    if user_role == "student":
        return user_id == thread_key.student_id
    return False


def resolve_other_user_id(
    current_user_id: str,
    current_role: str,
    other_user_id: str,
) -> tuple[str, str]:
    """Given the current user and the ``other_user_id`` from the URL,
    return ``(teacher_id, student_id)`` for the thread.

    This maps the REST URL pattern
    ``/threads/{exam_id}/{other_user_id}`` to the canonical thread
    key fields regardless of who is making the request.
    """
    from src.domain.message_rules import is_teacher_role

    if is_teacher_role(current_role):
        # Current user is the teacher; other_user_id is the student
        return current_user_id, other_user_id
    # Current user is the student; other_user_id is the teacher
    return other_user_id, current_user_id
