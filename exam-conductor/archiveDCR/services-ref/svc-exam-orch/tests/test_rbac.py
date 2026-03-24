"""Unit tests for RBAC domain logic.

Test IDs: U-RBAC-01 through U-RBAC-08.

Exercises the pure domain logic in ``src/domain/rbac.py`` with ZERO I/O.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from exampen_common.auth import ExamPenUser
from src.domain.rbac import (
    has_any_role,
    require_minimum_role,
    require_role,
    require_transition_role,
    role_rank,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user(
    roles: list[str],
    user_id: str = "user-1",
    tenant_id: str = "tenant-1",
    stoody_role: str = "tutor",
) -> ExamPenUser:
    return ExamPenUser(
        user_id=user_id,
        tenant_id=tenant_id,
        stoody_role=stoody_role,
        exampen_roles=roles,
    )


# ---------------------------------------------------------------------------
# U-RBAC-01: role_rank ordering
# ---------------------------------------------------------------------------


class TestRoleRank:
    def test_super_admin_is_highest(self) -> None:
        assert role_rank("super_admin") > role_rank("principal")
        assert role_rank("principal") > role_rank("hod")
        assert role_rank("hod") > role_rank("evaluator")

    def test_student_above_parent(self) -> None:
        assert role_rank("student") > role_rank("parent")

    def test_unknown_role_returns_negative(self) -> None:
        assert role_rank("alien") == -1


# ---------------------------------------------------------------------------
# U-RBAC-02: has_any_role
# ---------------------------------------------------------------------------


class TestHasAnyRole:
    def test_match(self) -> None:
        user = _user(["evaluator"])
        assert has_any_role(user, frozenset({"evaluator", "hod"})) is True

    def test_no_match(self) -> None:
        user = _user(["student"])
        assert has_any_role(user, frozenset({"evaluator", "hod"})) is False


# ---------------------------------------------------------------------------
# U-RBAC-03: require_role
# ---------------------------------------------------------------------------


class TestRequireRole:
    def test_allowed(self) -> None:
        user = _user(["evaluator"])
        require_role(user, "evaluator", "hod")  # should not raise

    def test_denied(self) -> None:
        user = _user(["student"])
        with pytest.raises(HTTPException) as exc_info:
            require_role(user, "evaluator", "hod")
        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# U-RBAC-04: require_minimum_role
# ---------------------------------------------------------------------------


class TestRequireMinimumRole:
    def test_exact_match(self) -> None:
        user = _user(["evaluator"])
        require_minimum_role(user, "evaluator")  # should not raise

    def test_above_minimum(self) -> None:
        user = _user(["super_admin"])
        require_minimum_role(user, "evaluator")  # should not raise

    def test_below_minimum(self) -> None:
        user = _user(["student"])
        with pytest.raises(HTTPException) as exc_info:
            require_minimum_role(user, "evaluator")
        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# U-RBAC-05: require_transition_role — invigilator transitions
# ---------------------------------------------------------------------------


class TestTransitionRoleInvigilator:
    def test_assigned_invigilator_allowed(self) -> None:
        user = _user(["invigilator"], user_id="inv-1")
        require_transition_role(user, "armed", frozenset({"inv-1"}))

    def test_unassigned_invigilator_denied(self) -> None:
        user = _user(["invigilator"], user_id="inv-1")
        with pytest.raises(HTTPException) as exc_info:
            require_transition_role(user, "armed", frozenset({"inv-99"}))
        assert exc_info.value.status_code == 403

    def test_student_denied_invigilator_transition(self) -> None:
        user = _user(["student"], user_id="stu-1")
        with pytest.raises(HTTPException) as exc_info:
            require_transition_role(user, "armed", frozenset({"inv-1"}))
        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# U-RBAC-06: require_transition_role — evaluator+ transitions
# ---------------------------------------------------------------------------


class TestTransitionRoleEvaluator:
    def test_evaluator_can_finalize(self) -> None:
        user = _user(["evaluator"])
        require_transition_role(user, "finalized")  # should not raise

    def test_invigilator_cannot_finalize(self) -> None:
        user = _user(["invigilator"])
        with pytest.raises(HTTPException) as exc_info:
            require_transition_role(user, "finalized")
        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# U-RBAC-07: super_admin bypasses all transition checks
# ---------------------------------------------------------------------------


class TestSuperAdminTransition:
    def test_super_admin_can_arm(self) -> None:
        user = _user(["super_admin"])
        require_transition_role(user, "armed", frozenset())

    def test_super_admin_can_finalize(self) -> None:
        user = _user(["super_admin"])
        require_transition_role(user, "finalized")


# ---------------------------------------------------------------------------
# U-RBAC-08: parent role is always denied mutating actions
# ---------------------------------------------------------------------------


class TestParentDenied:
    def test_parent_cannot_create_exam(self) -> None:
        user = _user(["parent"], stoody_role="parent")
        with pytest.raises(HTTPException) as exc_info:
            require_role(user, "super_admin", "principal", "hod", "evaluator")
        assert exc_info.value.status_code == 403

    def test_parent_cannot_transition(self) -> None:
        user = _user(["parent"], stoody_role="parent")
        with pytest.raises(HTTPException) as exc_info:
            require_transition_role(user, "armed", frozenset())
        assert exc_info.value.status_code == 403
