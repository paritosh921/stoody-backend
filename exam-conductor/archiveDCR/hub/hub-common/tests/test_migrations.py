"""Tests for hub SQLite schema migration runner.

Covers:
  - All 10 spec tables are created
  - WAL journal mode is active
  - Foreign-key constraints are enforced
  - CHECK constraints are enforced
  - Idempotency (running twice causes no error)
  - schema_version bookkeeping table records the migration
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hub_common.migrations.runner import run_migrations

# ── Fixtures ────────────────────────────────────────────────────────

EXPECTED_TABLES = {
    "hub_config",
    "invig_codes",
    "pen_inventory",
    "exam_sessions",
    "pen_bindings",
    "pen_sync_status",
    "upload_ledger",
    "dongle_registry",
    "interaction_log",
    "active_timer",
}


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "hub_test.db"


@pytest.fixture()
def conn(db_path: Path) -> sqlite3.Connection:
    """Run migrations and yield the connection; close on teardown."""
    connection = run_migrations(db_path)
    yield connection
    connection.close()


# ── Helpers ─────────────────────────────────────────────────────────

def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {row[0] for row in rows}


# ── Tests ───────────────────────────────────────────────────────────

class TestTableCreation:
    """All 10 tables from the spec must exist after migration."""

    def test_all_tables_created(self, conn: sqlite3.Connection) -> None:
        tables = _table_names(conn)
        assert EXPECTED_TABLES.issubset(tables), (
            f"Missing tables: {EXPECTED_TABLES - tables}"
        )

    def test_table_count(self, conn: sqlite3.Connection) -> None:
        tables = _table_names(conn)
        # 10 spec tables + schema_version = 11
        assert len(tables) == 11


class TestWalMode:
    """WAL journal mode must be enabled."""

    def test_wal_enabled(self, conn: sqlite3.Connection) -> None:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"


class TestForeignKeys:
    """Foreign-key constraints must be enforced."""

    def test_fk_enforcement_enabled(self, conn: sqlite3.Connection) -> None:
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_invalid_fk_raises(self, conn: sqlite3.Connection) -> None:
        """Inserting a pen_binding referencing a non-existent exam must fail."""
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO pen_bindings (exam_id, pen_mac, bound_at)
                VALUES ('nonexistent_exam', 'AA:BB:CC:DD:EE:FF', '2026-01-01T00:00:00Z')
                """
            )


class TestCheckConstraints:
    """CHECK constraints from the spec must reject invalid values."""

    def test_invalid_exam_state(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO exam_sessions (exam_id, invig_id, duration_min, state, created_at)
                VALUES ('e1', 'inv1', 60, 'INVALID_STATE', '2026-01-01T00:00:00Z')
                """
            )

    def test_invalid_uplink_mode(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO hub_config (hub_id, backend_url, uplink_mode, provisioned_at)
                VALUES ('h1', 'https://example.com', 'satellite', '2026-01-01T00:00:00Z')
                """
            )

    def test_invalid_dongle_status(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO dongle_registry (dongle_mac, first_seen, status)
                VALUES ('AA:BB:CC:DD:EE:01', '2026-01-01T00:00:00Z', 'broken')
                """
            )

    def test_invalid_severity(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO interaction_log (timestamp, source, event_type, severity)
                VALUES ('2026-01-01T00:00:00.000Z', 'test', 'test_event', 'fatal')
                """
            )

    def test_invalid_pen_binding_status(self, conn: sqlite3.Connection) -> None:
        # First create a valid exam so FK passes
        conn.execute(
            """
            INSERT INTO exam_sessions (exam_id, invig_id, duration_min, state, created_at)
            VALUES ('e_chk', 'inv1', 60, 'created', '2026-01-01T00:00:00Z')
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO pen_bindings (exam_id, pen_mac, status, bound_at)
                VALUES ('e_chk', 'AA:BB:CC:DD:EE:FF', 'invalid_status', '2026-01-01T00:00:00Z')
                """
            )

    def test_invalid_pen_binding_source(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT OR IGNORE INTO exam_sessions (exam_id, invig_id, duration_min, state, created_at)
            VALUES ('e_src', 'inv1', 60, 'created', '2026-01-01T00:00:00Z')
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO pen_bindings (exam_id, pen_mac, source, bound_at)
                VALUES ('e_src', 'AA:BB:CC:DD:EE:FF', 'bluetooth', '2026-01-01T00:00:00Z')
                """
            )

    def test_invalid_sync_status(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO pen_sync_status (exam_id, pen_mac, status)
                VALUES ('e1', 'AA:BB:CC:DD:EE:FF', 'unknown')
                """
            )

    def test_invalid_upload_path(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO upload_ledger (exam_id, pen_mac, total_chunks, upload_path)
                VALUES ('e1', 'AA:BB:CC:DD:EE:FF', 10, 'ethernet')
                """
            )


class TestIdempotency:
    """Running migrations twice must not raise errors."""

    def test_run_twice_no_error(self, db_path: Path) -> None:
        conn1 = run_migrations(db_path)
        conn1.close()

        conn2 = run_migrations(db_path)
        tables = _table_names(conn2)
        assert EXPECTED_TABLES.issubset(tables)
        conn2.close()

    def test_schema_version_not_duplicated(self, db_path: Path) -> None:
        conn1 = run_migrations(db_path)
        conn1.close()

        conn2 = run_migrations(db_path)
        rows = conn2.execute("SELECT version FROM schema_version").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 1
        conn2.close()


class TestSchemaVersion:
    """The schema_version bookkeeping table must record applied migrations."""

    def test_version_recorded(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT version, filename FROM schema_version"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0] == (1, "001_initial.sql")

    def test_applied_at_populated(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT applied_at FROM schema_version WHERE version = 1"
        ).fetchone()
        assert row is not None
        assert row[0] is not None  # ISO 8601 timestamp
