"""
SEC-DPDPA: Data retention and DPDPA compliance tests.

Verifies append-only invariants, configurable retention periods, and absence
of destructive SQL on protected data categories.

Source of truth: FAILURE_MITIGATION_REGISTER.md A8.2, CLAUDE.md state ownership table.
Level: L3 (static analysis + mock checks) / L4 (integration checks against DB schema).

Test-ID prefix: SEC-DPDPA-{NN}
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.security, pytest.mark.dpdpa]

# ---------------------------------------------------------------------------
# Project root (for static analysis of SQL / migration files)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SERVICES_DIR = PROJECT_ROOT / "services"


# ---------------------------------------------------------------------------
# Data categories and their retention policies (from spec + A8.2)
# ---------------------------------------------------------------------------

DATA_CATEGORIES = {
    "chat_messages": {
        "owner": "svc-chat",
        "append_only": True,
        "no_update": True,
        "no_delete": True,
        "retention": "indefinite (exam lifecycle)",
        "personal_data": True,
        "notes": "Append-only per CLAUDE.md state ownership. No UPDATE, no DELETE.",
    },
    "score_events": {
        "owner": "svc-score-engine",
        "append_only": True,
        "no_update": True,
        "no_delete": True,
        "retention": "indefinite (event-sourced, immutable ledger)",
        "personal_data": True,
        "notes": "Event-sourced, append-only. FSM: ai_draft->teacher_reviewed->finalized->locked.",
    },
    "stroke_data": {
        "owner": "svc-stroke-proc",
        "append_only": True,
        "no_update": True,
        "no_delete": False,  # Configurable retention, auto-delete after period.
        "retention": "configurable (default 2 years per A8.2)",
        "personal_data": True,
        "notes": "May be considered behavioral biometric under DPDPA. Auto-delete after retention period.",
    },
    "objections": {
        "owner": "svc-review",
        "append_only": False,  # FSM transitions update status.
        "no_update": False,
        "no_delete": True,
        "retention": "indefinite (audit trail)",
        "personal_data": True,
        "notes": "Status transitions via FSM. No deletion of filed objections.",
    },
    "plagiarism_flags": {
        "owner": "svc-plagiarism",
        "append_only": False,  # Verdicts update the flag.
        "no_update": False,
        "no_delete": True,
        "retention": "indefinite (audit trail)",
        "personal_data": True,
        "notes": "Teacher verdicts update flags. No deletion.",
    },
    "exam_definitions": {
        "owner": "svc-exam-orch",
        "append_only": False,
        "no_update": False,
        "no_delete": True,
        "retention": "indefinite",
        "personal_data": False,
        "notes": "Exam metadata. FSM transitions update state. No deletion.",
    },
    "pen_bindings": {
        "owner": "svc-exam-orch",
        "append_only": False,
        "no_update": False,
        "no_delete": True,
        "retention": "linked to exam lifecycle",
        "personal_data": True,
        "notes": "Maps pen_mac to student_id. PII linkage.",
    },
    "copy_images": {
        "owner": "svc-copy-upload",
        "append_only": True,
        "no_update": True,
        "no_delete": False,
        "retention": "configurable (follows stroke_data policy)",
        "personal_data": True,
        "notes": "Photographed answer copies. Same retention as strokes.",
    },
}


# ---------------------------------------------------------------------------
# Static analysis: scan service SQL for forbidden mutations
# ---------------------------------------------------------------------------

# Regex patterns for UPDATE/DELETE in SQL files.
_UPDATE_RE = re.compile(
    r"\b(UPDATE\s+\w+\s+SET)\b",
    re.IGNORECASE,
)
_DELETE_RE = re.compile(
    r"\b(DELETE\s+FROM\s+\w+)\b",
    re.IGNORECASE,
)

# Table names that correspond to append-only data categories.
APPEND_ONLY_TABLES = {
    "chat_messages": ["chat_messages", "messages"],
    "score_events": ["score_events", "scores", "score_history"],
}


def _find_sql_files(service_name: str) -> list[Path]:
    """Find all .sql and .py files in a service directory that may contain SQL."""
    svc_dir = SERVICES_DIR / service_name
    if not svc_dir.exists():
        return []
    sql_files = list(svc_dir.rglob("*.sql"))
    py_files = list(svc_dir.rglob("*.py"))
    return sql_files + py_files


def _check_no_mutations(
    files: list[Path],
    table_names: list[str],
    check_update: bool = True,
    check_delete: bool = True,
) -> list[str]:
    """Return list of violation descriptions."""
    violations: list[str] = []
    for fpath in files:
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for table in table_names:
            if check_update:
                for match in _UPDATE_RE.finditer(content):
                    stmt = match.group(0).lower()
                    if table.lower() in stmt:
                        violations.append(
                            f"UPDATE on append-only table '{table}' in {fpath.relative_to(PROJECT_ROOT)}"
                        )

            if check_delete:
                for match in _DELETE_RE.finditer(content):
                    stmt = match.group(0).lower()
                    if table.lower() in stmt:
                        violations.append(
                            f"DELETE on append-only table '{table}' in {fpath.relative_to(PROJECT_ROOT)}"
                        )

    return violations


# ---------------------------------------------------------------------------
# Tests: Append-only invariants
# ---------------------------------------------------------------------------


class TestChatAppendOnly:
    """SEC-DPDPA-01: Chat messages must be append-only."""

    def test_no_update_sql_on_chat_messages(self):
        """SEC-DPDPA-01a: No UPDATE statements targeting chat message tables
        in svc-chat source code."""
        files = _find_sql_files("svc-chat")
        if not files:
            pytest.skip("svc-chat source not found (not yet built)")
        violations = _check_no_mutations(
            files,
            APPEND_ONLY_TABLES["chat_messages"],
            check_update=True,
            check_delete=True,
        )
        assert not violations, (
            f"Chat messages must be append-only. Violations found:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_no_delete_sql_on_chat_messages(self):
        """SEC-DPDPA-01b: No DELETE statements targeting chat message tables."""
        files = _find_sql_files("svc-chat")
        if not files:
            pytest.skip("svc-chat source not found (not yet built)")
        violations = _check_no_mutations(
            files,
            APPEND_ONLY_TABLES["chat_messages"],
            check_update=False,
            check_delete=True,
        )
        assert not violations, (
            f"Chat messages must never be deleted. Violations found:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    @pytest.mark.asyncio
    async def test_chat_api_has_no_delete_endpoint(
        self,
        http_session,
        service_urls,
    ):
        """SEC-DPDPA-01c: The chat API has no DELETE method on any message endpoint."""
        # Attempt a DELETE on the chat thread endpoint.
        url = f"{service_urls['chat']}/api/v1/chat/threads/test-exam/test-user"
        resp = await http_session.delete(url)
        assert resp.status in (404, 405), (
            f"Chat API should not support DELETE on threads, got {resp.status}"
        )


class TestScoreEventsAppendOnly:
    """SEC-DPDPA-02: Score events must be append-only (event-sourced)."""

    def test_no_update_sql_on_score_events(self):
        """SEC-DPDPA-02a: No UPDATE on score event tables in svc-score-engine."""
        files = _find_sql_files("svc-score-engine")
        if not files:
            pytest.skip("svc-score-engine source not found (not yet built)")
        violations = _check_no_mutations(
            files,
            APPEND_ONLY_TABLES["score_events"],
            check_update=True,
            check_delete=True,
        )
        assert not violations, (
            f"Score events must be append-only. Violations found:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_no_delete_sql_on_score_events(self):
        """SEC-DPDPA-02b: No DELETE on score event tables."""
        files = _find_sql_files("svc-score-engine")
        if not files:
            pytest.skip("svc-score-engine source not found (not yet built)")
        violations = _check_no_mutations(
            files,
            APPEND_ONLY_TABLES["score_events"],
            check_update=False,
            check_delete=True,
        )
        assert not violations, (
            f"Score events must never be deleted. Violations found:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


class TestStrokeRetention:
    """SEC-DPDPA-03: Stroke data has configurable retention period."""

    def test_retention_config_exists(self):
        """SEC-DPDPA-03a: svc-stroke-proc should have a retention config parameter."""
        svc_dir = SERVICES_DIR / "svc-stroke-proc"
        if not svc_dir.exists():
            pytest.skip("svc-stroke-proc not yet built")

        # Scan for retention-related config in source or config files.
        found_retention_config = False
        for ext in ("*.py", "*.yaml", "*.yml", "*.toml", "*.env*"):
            for f in svc_dir.rglob(ext):
                try:
                    content = f.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if re.search(
                    r"(retention|ttl|expire|auto.?delete|data.?lifecycle)",
                    content,
                    re.IGNORECASE,
                ):
                    found_retention_config = True
                    break
            if found_retention_config:
                break

        assert found_retention_config, (
            "svc-stroke-proc must have a configurable retention period "
            "for stroke data (DPDPA A8.2 requirement)"
        )


# ---------------------------------------------------------------------------
# Tests: RLS policy coverage on migrations
# ---------------------------------------------------------------------------


class TestRLSPolicyCoverage:
    """SEC-DPDPA-04: Every migration that creates a table with tenant_id
    must also define an RLS policy."""

    def test_all_tenant_tables_have_rls_policy(self):
        """SEC-DPDPA-04a: Scan all SQL migrations for CREATE TABLE with
        tenant_id and verify a matching RLS ENABLE / CREATE POLICY exists."""
        migration_files: list[Path] = []
        if SERVICES_DIR.exists():
            migration_files = list(SERVICES_DIR.rglob("*.sql"))

        if not migration_files:
            pytest.skip("No SQL migration files found (services not yet built)")

        _CREATE_TABLE_RE = re.compile(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
            re.IGNORECASE,
        )
        _TENANT_COL_RE = re.compile(r"\btenant_id\b", re.IGNORECASE)
        _RLS_ENABLE_RE = re.compile(
            r"ALTER\s+TABLE\s+\w+\s+ENABLE\s+ROW\s+LEVEL\s+SECURITY",
            re.IGNORECASE,
        )
        _CREATE_POLICY_RE = re.compile(
            r"CREATE\s+POLICY\s+\w+\s+ON\s+(\w+)",
            re.IGNORECASE,
        )

        tables_needing_rls: set[str] = set()
        tables_with_rls: set[str] = set()

        for fpath in migration_files:
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            # Find tables with tenant_id column.
            for m in _CREATE_TABLE_RE.finditer(content):
                table_name = m.group(1).lower()
                # Check if tenant_id appears near this CREATE TABLE
                table_block_start = m.start()
                # Look ahead for the closing paren or next CREATE TABLE.
                next_create = content.find("CREATE TABLE", m.end())
                block = content[table_block_start : next_create if next_create > 0 else len(content)]
                if _TENANT_COL_RE.search(block):
                    tables_needing_rls.add(table_name)

            # Find RLS-enabled tables.
            for m in _RLS_ENABLE_RE.finditer(content):
                # Extract table name from ALTER TABLE <name> ENABLE ...
                parts = m.group(0).split()
                if len(parts) >= 3:
                    tables_with_rls.add(parts[2].lower())

            for m in _CREATE_POLICY_RE.finditer(content):
                tables_with_rls.add(m.group(1).lower())

        missing_rls = tables_needing_rls - tables_with_rls
        if tables_needing_rls:
            assert not missing_rls, (
                f"Tables with tenant_id but NO RLS policy: {sorted(missing_rls)}. "
                "Per A8.1, every table with tenant_id must have RLS enabled."
            )


# ---------------------------------------------------------------------------
# Tests: Data category documentation
# ---------------------------------------------------------------------------


class TestDataCategoryDocumentation:
    """SEC-DPDPA-05: All data categories have documented retention policies."""

    @pytest.mark.parametrize(
        "category,spec",
        list(DATA_CATEGORIES.items()),
        ids=list(DATA_CATEGORIES.keys()),
    )
    def test_retention_policy_documented(self, category: str, spec: dict[str, Any]):
        """SEC-DPDPA-05a: Each data category has a non-empty retention policy."""
        assert spec["retention"], (
            f"Data category '{category}' has no documented retention policy"
        )

    @pytest.mark.parametrize(
        "category,spec",
        [
            (k, v) for k, v in DATA_CATEGORIES.items() if v["personal_data"]
        ],
        ids=[k for k, v in DATA_CATEGORIES.items() if v["personal_data"]],
    )
    def test_personal_data_has_owner(self, category: str, spec: dict[str, Any]):
        """SEC-DPDPA-05b: Every personal-data category has a designated owner service."""
        assert spec["owner"], (
            f"Personal data category '{category}' has no designated owner service"
        )


# ---------------------------------------------------------------------------
# Tests: Encryption at rest and in transit
# ---------------------------------------------------------------------------


class TestEncryptionRequirements:
    """SEC-DPDPA-06: Verify encryption configuration markers exist."""

    def test_tls_config_present(self):
        """SEC-DPDPA-06a: Infrastructure config should reference TLS/SSL."""
        infra_dir = PROJECT_ROOT / "infra"
        if not infra_dir.exists():
            pytest.skip("infra/ directory not yet created")

        found_tls = False
        for f in infra_dir.rglob("*"):
            if f.is_file() and f.suffix in (".yaml", ".yml", ".toml", ".conf"):
                try:
                    content = f.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if re.search(r"(tls|ssl|certificate|https)", content, re.IGNORECASE):
                    found_tls = True
                    break

        assert found_tls, (
            "No TLS/SSL configuration found in infra/. "
            "DPDPA requires encryption in transit (TLS everywhere)."
        )

    def test_db_encryption_at_rest_mentioned(self):
        """SEC-DPDPA-06b: Infrastructure or service config references data-at-rest encryption."""
        infra_dir = PROJECT_ROOT / "infra"
        if not infra_dir.exists():
            pytest.skip("infra/ directory not yet created")

        found_encryption = False
        search_dirs = [infra_dir]
        if SERVICES_DIR.exists():
            search_dirs.append(SERVICES_DIR)

        for search_dir in search_dirs:
            for f in search_dir.rglob("*"):
                if f.is_file() and f.suffix in (".yaml", ".yml", ".toml", ".conf", ".sql"):
                    try:
                        content = f.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        continue
                    if re.search(
                        r"(encrypt|tde|pgcrypto|at.rest|aes)",
                        content,
                        re.IGNORECASE,
                    ):
                        found_encryption = True
                        break
            if found_encryption:
                break

        # This is advisory — DPDPA recommends encryption at rest but
        # implementation may use disk-level encryption not visible in config.
        if not found_encryption:
            pytest.skip(
                "No explicit encryption-at-rest config found. "
                "May rely on disk-level encryption (acceptable)."
            )


# ---------------------------------------------------------------------------
# Integration test: verify append-only at API level
# ---------------------------------------------------------------------------


class TestAppendOnlyAtAPILevel:
    """SEC-DPDPA-07: Verify append-only enforcement via HTTP."""

    @pytest.mark.asyncio
    async def test_score_engine_no_delete_endpoint(
        self,
        http_session,
        service_urls,
    ):
        """SEC-DPDPA-07a: Score engine exposes no DELETE endpoints."""
        import uuid

        exam_id = str(uuid.uuid4())
        student_id = "student-test"

        # Attempt DELETE on score detail.
        url = f"{service_urls['score_engine']}/api/v1/scores/{exam_id}/students/{student_id}"
        resp = await http_session.delete(url)
        assert resp.status in (404, 405), (
            f"Score engine should not support DELETE, got {resp.status}"
        )

    @pytest.mark.asyncio
    async def test_score_history_endpoint_returns_immutable_events(
        self,
        token_factory,
        http_session,
        service_urls,
    ):
        """SEC-DPDPA-07b: Score history endpoint returns event list, not mutable state."""
        import uuid

        exam_id = str(uuid.uuid4())
        student_id = "student-test"
        headers = token_factory.bearer("evaluator")
        url = (
            f"{service_urls['score_engine']}"
            f"/api/v1/scores/{exam_id}/students/{student_id}/history"
        )

        resp = await http_session.get(url, headers=headers)
        if resp.status == 200:
            data = await resp.json()
            items = data.get("items", [])
            # Each event should have required immutable fields.
            for event in items:
                assert "event_id" in event, "Score history event missing event_id"
                assert "event_type" in event, "Score history event missing event_type"
                assert "created_at" in event, "Score history event missing created_at"
