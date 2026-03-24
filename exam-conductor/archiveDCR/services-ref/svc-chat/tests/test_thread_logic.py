"""Unit tests for domain/thread_logic.py -- ZERO I/O, pure logic.

Test IDs: U-CHAT-TL-01 through U-CHAT-TL-08
"""

from src.domain.thread_logic import (
    ThreadKey,
    build_thread_key,
    can_participate,
    resolve_other_user_id,
)


# -- U-CHAT-TL-01: ThreadKey identity -------------------------------------


def test_thread_key_is_frozen():
    """U-CHAT-TL-01a: ThreadKey is immutable."""
    tk = ThreadKey(exam_id="e1", teacher_id="t1", student_id="s1")
    try:
        tk.exam_id = "e2"  # type: ignore[misc]
        assert False, "Should have raised"
    except AttributeError:
        pass


def test_thread_key_equality():
    """U-CHAT-TL-01b: Same components produce equal ThreadKeys."""
    a = ThreadKey(exam_id="e1", teacher_id="t1", student_id="s1")
    b = ThreadKey(exam_id="e1", teacher_id="t1", student_id="s1")
    assert a == b


def test_thread_key_inequality():
    """U-CHAT-TL-01c: Different components produce unequal ThreadKeys."""
    a = ThreadKey(exam_id="e1", teacher_id="t1", student_id="s1")
    b = ThreadKey(exam_id="e1", teacher_id="t1", student_id="s2")
    assert a != b


# -- U-CHAT-TL-02: build_thread_key normalizes direction ------------------


def test_build_from_teacher_sender():
    """U-CHAT-TL-02a: Teacher sending places teacher as teacher_id."""
    tk = build_thread_key(
        exam_id="e1",
        sender_id="teacher-1",
        recipient_id="student-1",
        sender_role="teacher",
    )
    assert tk.teacher_id == "teacher-1"
    assert tk.student_id == "student-1"
    assert tk.exam_id == "e1"


def test_build_from_student_sender():
    """U-CHAT-TL-02b: Student sending places teacher as teacher_id."""
    tk = build_thread_key(
        exam_id="e1",
        sender_id="student-1",
        recipient_id="teacher-1",
        sender_role="student",
    )
    assert tk.teacher_id == "teacher-1"
    assert tk.student_id == "student-1"


def test_build_both_directions_same_key():
    """U-CHAT-TL-02c: Teacher->student and student->teacher produce same key."""
    tk_teacher = build_thread_key("e1", "t1", "s1", "teacher")
    tk_student = build_thread_key("e1", "s1", "t1", "student")
    assert tk_teacher == tk_student


def test_build_evaluator_role():
    """U-CHAT-TL-02d: evaluator role is treated as teacher."""
    tk = build_thread_key("e1", "eval-1", "s1", "evaluator")
    assert tk.teacher_id == "eval-1"
    assert tk.student_id == "s1"


# -- U-CHAT-TL-03: can_participate — teacher access ------------------------


def test_teacher_can_participate_in_own_thread():
    """U-CHAT-TL-03a: Teacher is participant in their own thread."""
    tk = ThreadKey(exam_id="e1", teacher_id="t1", student_id="s1")
    assert can_participate("t1", "teacher", tk) is True


def test_teacher_cannot_participate_in_other_thread():
    """U-CHAT-TL-03b: Teacher not in thread is denied."""
    tk = ThreadKey(exam_id="e1", teacher_id="t2", student_id="s1")
    assert can_participate("t1", "teacher", tk) is False


# -- U-CHAT-TL-04: can_participate — student access ------------------------


def test_student_can_participate_in_own_thread():
    """U-CHAT-TL-04a: Student is participant in their own thread."""
    tk = ThreadKey(exam_id="e1", teacher_id="t1", student_id="s1")
    assert can_participate("s1", "student", tk) is True


def test_student_cannot_participate_in_other_thread():
    """U-CHAT-TL-04b: Student not in thread is denied."""
    tk = ThreadKey(exam_id="e1", teacher_id="t1", student_id="s2")
    assert can_participate("s1", "student", tk) is False


# -- U-CHAT-TL-05: can_participate — disallowed roles ---------------------


def test_parent_cannot_participate():
    """U-CHAT-TL-05a: Parent role is always denied."""
    tk = ThreadKey(exam_id="e1", teacher_id="t1", student_id="s1")
    assert can_participate("s1", "parent", tk) is False


def test_admin_cannot_participate():
    """U-CHAT-TL-05b: Admin role is always denied."""
    tk = ThreadKey(exam_id="e1", teacher_id="t1", student_id="s1")
    assert can_participate("t1", "principal", tk) is False


# -- U-CHAT-TL-06: resolve_other_user_id ----------------------------------


def test_resolve_teacher_perspective():
    """U-CHAT-TL-06a: Teacher sees other_user_id as the student."""
    teacher_id, student_id = resolve_other_user_id(
        current_user_id="t1",
        current_role="teacher",
        other_user_id="s1",
    )
    assert teacher_id == "t1"
    assert student_id == "s1"


def test_resolve_student_perspective():
    """U-CHAT-TL-06b: Student sees other_user_id as the teacher."""
    teacher_id, student_id = resolve_other_user_id(
        current_user_id="s1",
        current_role="student",
        other_user_id="t1",
    )
    assert teacher_id == "t1"
    assert student_id == "s1"


def test_resolve_evaluator_perspective():
    """U-CHAT-TL-06c: Evaluator is treated same as teacher."""
    teacher_id, student_id = resolve_other_user_id(
        current_user_id="eval-1",
        current_role="evaluator",
        other_user_id="s1",
    )
    assert teacher_id == "eval-1"
    assert student_id == "s1"
