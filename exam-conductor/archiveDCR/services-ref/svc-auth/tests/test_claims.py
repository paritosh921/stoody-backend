"""Unit tests for domain/claims.py — ZERO I/O, pure logic.

Test IDs: U-AUTH-CL-01 through U-AUTH-CL-06
"""

from src.domain.claims import NormalizedClaims, Profile, normalize_claims


# -- Helpers ---------------------------------------------------------------

_BASE_JWT = {
    "sub": "user-123",
    "tenant_id": "tenant-abc",
    "role": "tutor",
    "name": "Alice Tutor",
    "email": "alice@school.edu",
}

_STUDENT_JWT = {
    "sub": "student-456",
    "tenant_id": "tenant-abc",
    "role": "student",
    "name": "Bob Student",
}

_PARENT_JWT = {
    "sub": "parent-789",
    "tenant_id": "tenant-abc",
    "role": "parent",
    "name": "Carol Parent",
}

_STOODY_PROFILE = {
    "name": "Alice M. Tutor",
    "email": "alice.tutor@school.edu",
    "phone": "+91-9876543210",
    "institute_name": "Springfield High",
    "subject_ids": ["math-101", "math-201"],
    "class_ids": ["class-10a", "class-10b"],
}


# -- U-AUTH-CL-01: Basic normalization without profile enrichment ----------

def test_basic_normalization():
    """U-AUTH-CL-01: JWT-only normalization produces valid claims."""
    claims = normalize_claims(_BASE_JWT)
    assert claims.user_id == "user-123"
    assert claims.tenant_id == "tenant-abc"
    assert claims.stoody_role == "tutor"
    assert claims.exampen_roles == ["evaluator"]
    assert claims.token_source == "stoody_jwt"
    assert claims.token_status == "valid"
    assert claims.profile.display_name == "Alice Tutor"
    assert claims.profile.email == "alice@school.edu"


# -- U-AUTH-CL-02: Profile enrichment from Stoody API ---------------------

def test_profile_enrichment():
    """U-AUTH-CL-02: Stoody profile overrides JWT claims for display_name."""
    claims = normalize_claims(_BASE_JWT, stoody_profile=_STOODY_PROFILE)
    assert claims.profile.display_name == "Alice M. Tutor"
    assert claims.profile.email == "alice.tutor@school.edu"
    assert claims.profile.phone == "+91-9876543210"
    assert claims.profile.institute_name == "Springfield High"
    assert claims.subject_ids == ["math-101", "math-201"]
    assert claims.class_ids == ["class-10a", "class-10b"]


# -- U-AUTH-CL-03: Graceful degradation without profile -------------------

def test_graceful_degradation_no_profile():
    """U-AUTH-CL-03: Without Stoody profile, JWT name used as display_name."""
    claims = normalize_claims(_STUDENT_JWT, stoody_profile=None)
    assert claims.profile.display_name == "Bob Student"
    assert claims.profile.email is None


# -- U-AUTH-CL-04: Role overrides applied ---------------------------------

def test_role_overrides():
    """U-AUTH-CL-04: DB role overrides change exampen_roles."""
    overrides = {"tutor": ["invigilator", "evaluator"]}
    claims = normalize_claims(_BASE_JWT, role_overrides=overrides)
    assert claims.exampen_roles == ["invigilator", "evaluator"]


# -- U-AUTH-CL-05: Parent with child IDs ----------------------------------

def test_parent_with_children():
    """U-AUTH-CL-05: Parent claims include child_student_ids."""
    claims = normalize_claims(
        _PARENT_JWT,
        child_ids=["student-100", "student-101"],
    )
    assert claims.exampen_roles == ["parent"]
    assert claims.child_student_ids == ["student-100", "student-101"]


# -- U-AUTH-CL-06: to_dict matches OpenAPI schema -------------------------

def test_to_dict_structure():
    """U-AUTH-CL-06: to_dict output matches NormalizedClaims OpenAPI schema."""
    claims = normalize_claims(
        _BASE_JWT,
        stoody_profile=_STOODY_PROFILE,
        child_ids=["child-1"],
    )
    d = claims.to_dict()
    assert d["user_id"] == "user-123"
    assert d["tenant_id"] == "tenant-abc"
    assert d["stoody_role"] == "tutor"
    assert d["exampen_roles"] == ["evaluator"]
    assert d["token_source"] == "stoody_jwt"
    assert d["token_status"] == "valid"
    assert d["profile"]["display_name"] == "Alice M. Tutor"
    assert d["profile"]["email"] == "alice.tutor@school.edu"
    assert d["subject_ids"] == ["math-101", "math-201"]
    assert d["child_student_ids"] == ["child-1"]


def test_to_dict_omits_empty_optionals():
    """U-AUTH-CL-06b: to_dict omits empty optional arrays."""
    claims = normalize_claims(_STUDENT_JWT)
    d = claims.to_dict()
    assert "subject_ids" not in d
    assert "class_ids" not in d
    assert "child_student_ids" not in d
    assert "phone" not in d["profile"]
    assert "institute_name" not in d["profile"]


def test_missing_sub_falls_back_to_user_id_key():
    """U-AUTH-CL-06c: If JWT has user_id instead of sub, it still works."""
    jwt_payload = {"user_id": "u-alt", "tenant_id": "t-1", "role": "student"}
    claims = normalize_claims(jwt_payload)
    assert claims.user_id == "u-alt"
