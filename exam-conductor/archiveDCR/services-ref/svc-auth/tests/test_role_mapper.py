"""Unit tests for domain/role_mapper.py — ZERO I/O, pure logic.

Test IDs: U-AUTH-RM-01 through U-AUTH-RM-08
"""

from src.domain.role_mapper import (
    DEFAULT_ROLE_MAP,
    NO_ACCESS_SENTINEL,
    ROLE_HIERARCHY,
    has_minimum_role,
    map_roles,
    role_rank,
)


# -- U-AUTH-RM-01: Default mapping for known Stoody roles ------------------

def test_tutor_maps_to_evaluator():
    """U-AUTH-RM-01a: Stoody tutor -> ExamPen [evaluator]."""
    assert map_roles("tutor") == ["evaluator"]


def test_student_maps_to_student():
    """U-AUTH-RM-01b: Stoody student -> ExamPen [student]."""
    assert map_roles("student") == ["student"]


def test_parent_maps_to_parent():
    """U-AUTH-RM-01c: Stoody parent -> ExamPen [parent]."""
    assert map_roles("parent") == ["parent"]


def test_admin_maps_to_principal():
    """U-AUTH-RM-01d: Stoody admin -> ExamPen [principal]."""
    assert map_roles("admin") == ["principal"]


def test_hod_maps_to_hod():
    """U-AUTH-RM-01e: Stoody hod -> ExamPen [hod]."""
    assert map_roles("hod") == ["hod"]


def test_super_admin_maps_to_super_admin():
    """U-AUTH-RM-01f: Stoody super_admin -> ExamPen [super_admin]."""
    assert map_roles("super_admin") == ["super_admin"]


# -- U-AUTH-RM-02: Unknown role returns sentinel ---------------------------

def test_unknown_role_returns_no_access():
    """U-AUTH-RM-02: Unknown Stoody role -> no_exampen_access."""
    result = map_roles("janitor")
    assert result == [NO_ACCESS_SENTINEL]


def test_empty_role_returns_no_access():
    """U-AUTH-RM-02b: Empty string Stoody role -> no_exampen_access."""
    result = map_roles("")
    assert result == [NO_ACCESS_SENTINEL]


# -- U-AUTH-RM-03: DB overrides take precedence ----------------------------

def test_override_replaces_default():
    """U-AUTH-RM-03: DB override replaces default mapping."""
    overrides = {"tutor": ["invigilator", "evaluator"]}
    result = map_roles("tutor", overrides=overrides)
    assert result == ["invigilator", "evaluator"]


def test_override_adds_new_role():
    """U-AUTH-RM-03b: DB override can map a previously unknown role."""
    overrides = {"coordinator": ["hod"]}
    result = map_roles("coordinator", overrides=overrides)
    assert result == ["hod"]


# -- U-AUTH-RM-04: Role hierarchy ranking ----------------------------------

def test_hierarchy_order():
    """U-AUTH-RM-04: super_admin > principal > ... > parent."""
    assert role_rank("super_admin") > role_rank("principal")
    assert role_rank("principal") > role_rank("hod")
    assert role_rank("hod") > role_rank("tutor")
    assert role_rank("evaluator") > role_rank("invigilator")
    assert role_rank("invigilator") > role_rank("student")
    assert role_rank("student") > role_rank("parent")


def test_unknown_role_rank_is_negative():
    """U-AUTH-RM-04b: Unknown roles return rank -1."""
    assert role_rank("janitor") == -1


# -- U-AUTH-RM-05: has_minimum_role ----------------------------------------

def test_principal_meets_evaluator():
    """U-AUTH-RM-05: principal meets minimum evaluator requirement."""
    assert has_minimum_role(["principal"], "evaluator") is True


def test_student_does_not_meet_evaluator():
    """U-AUTH-RM-05b: student does NOT meet evaluator requirement."""
    assert has_minimum_role(["student"], "evaluator") is False


def test_multiple_roles_any_match():
    """U-AUTH-RM-05c: If any role meets the minimum, returns True."""
    assert has_minimum_role(["student", "evaluator"], "evaluator") is True


def test_no_access_meets_nothing():
    """U-AUTH-RM-05d: no_exampen_access meets no role requirement."""
    assert has_minimum_role([NO_ACCESS_SENTINEL], "parent") is False
