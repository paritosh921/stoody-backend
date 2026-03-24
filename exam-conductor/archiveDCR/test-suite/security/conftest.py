"""
Shared fixtures for ExamPen security tests (L3/L4).

Provides mock JWT generation, mock service clients, and tenant isolation
helpers. Tests are designed to run against mocked services, not a full stack.

Usage:
    pytest test-suite/security/ -m security
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest
import pytest_asyncio

# Allow imports from the sibling stoody-mock package.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "stoody-mock"),
)

from keys import get_jwks_dict, get_private_key, make_token  # noqa: E402


# ---------------------------------------------------------------------------
# Environment defaults
# ---------------------------------------------------------------------------

EXAM_ORCH_URL = os.getenv("EXAMPEN_EXAM_ORCH_URL", "http://localhost:8001")
SCORE_ENGINE_URL = os.getenv("EXAMPEN_SCORE_ENGINE_URL", "http://localhost:8003")
REVIEW_URL = os.getenv("EXAMPEN_REVIEW_URL", "http://localhost:8005")
ANALYTICS_URL = os.getenv("EXAMPEN_ANALYTICS_URL", "http://localhost:8007")
PLAGIARISM_URL = os.getenv("EXAMPEN_PLAGIARISM_URL", "http://localhost:8008")
TEACHER_BFF_URL = os.getenv("EXAMPEN_TEACHER_BFF_URL", "http://localhost:8010")
STUDENT_BFF_URL = os.getenv("EXAMPEN_STUDENT_BFF_URL", "http://localhost:8011")
CHAT_URL = os.getenv("EXAMPEN_CHAT_URL", "http://localhost:8012")
AUTH_URL = os.getenv("EXAMPEN_AUTH_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# pytest markers
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "security: Security audit test")
    config.addinivalue_line("markers", "rbac: RBAC matrix enforcement test")
    config.addinivalue_line("markers", "rls: Row-level security / tenant isolation test")
    config.addinivalue_line("markers", "dpdpa: DPDPA data-retention compliance test")


# ---------------------------------------------------------------------------
# Event loop
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# JWKS fixture (for tests that validate token verification)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def jwks_dict() -> dict[str, Any]:
    """Return the JWKS dict from the stoody-mock key module."""
    return get_jwks_dict()


# ---------------------------------------------------------------------------
# Token factory
# ---------------------------------------------------------------------------

# The 7 roles from the RBAC matrix in STOODY_INTEGRATION_SPEC section 6.
RBAC_ROLES = [
    "super_admin",
    "principal",
    "hod",
    "evaluator",
    "invigilator",
    "student",
    "parent",
]


@dataclass
class TokenFactory:
    """Generate signed JWTs for each role, with optional tenant override."""

    _cache: dict[str, str] = field(default_factory=dict)

    def for_role(
        self,
        role: str,
        *,
        tenant_id: str = "tenant-001",
        user_id: str | None = None,
        extra_claims: dict[str, Any] | None = None,
    ) -> str:
        uid = user_id or f"{role}-{uuid.uuid4().hex[:8]}"
        return make_token(
            user_id=uid,
            tenant_id=tenant_id,
            role=role,
            name=f"Test {role.replace('_', ' ').title()}",
            email=f"{role}@test.exampen.local",
            extra_claims=extra_claims,
        )

    def bearer(self, role: str, **kwargs: Any) -> dict[str, str]:
        """Return an Authorization header dict."""
        return {"Authorization": f"Bearer {self.for_role(role, **kwargs)}"}


@pytest.fixture(scope="session")
def token_factory() -> TokenFactory:
    return TokenFactory()


# ---------------------------------------------------------------------------
# HTTP session (aiohttp)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="session")
async def http_session():
    import aiohttp

    session = aiohttp.ClientSession()
    yield session
    await session.close()


# ---------------------------------------------------------------------------
# Service URL helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def service_urls() -> dict[str, str]:
    return {
        "exam_orch": EXAM_ORCH_URL,
        "score_engine": SCORE_ENGINE_URL,
        "review": REVIEW_URL,
        "analytics": ANALYTICS_URL,
        "plagiarism": PLAGIARISM_URL,
        "teacher_bff": TEACHER_BFF_URL,
        "student_bff": STUDENT_BFF_URL,
        "chat": CHAT_URL,
        "auth": AUTH_URL,
    }
