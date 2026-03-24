"""Unit tests for interaction_log.py — forensic audit logger.

All tests use in-memory SQLite (no disk I/O).
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from src.interaction_log import InteractionLog, LogEntry


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture()
def db_conn() -> sqlite3.Connection:
    """In-memory SQLite connection."""
    return sqlite3.connect(":memory:")


@pytest.fixture()
def ilog(db_conn: sqlite3.Connection) -> InteractionLog:
    log = InteractionLog()
    log.open(conn=db_conn)
    return log


# ===================================================================
# Table creation
# ===================================================================

class TestSchema:

    def test_table_created(self, db_conn: sqlite3.Connection) -> None:
        log = InteractionLog()
        log.open(conn=db_conn)
        # Verify the table exists.
        cur = db_conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='interaction_log'"
        )
        assert cur.fetchone() is not None

    def test_idempotent_open(self, db_conn: sqlite3.Connection) -> None:
        """Opening twice does not error (CREATE IF NOT EXISTS)."""
        log = InteractionLog()
        log.open(conn=db_conn)
        log.open(conn=db_conn)  # second open — should not raise


# ===================================================================
# Append entries
# ===================================================================

class TestAppend:

    def test_append_returns_log_id(self, ilog: InteractionLog) -> None:
        lid = ilog.append(LogEntry(
            source="hub-supervisor",
            event_type="hub_boot",
        ))
        assert isinstance(lid, int)
        assert lid >= 1

    def test_append_multiple(self, ilog: InteractionLog) -> None:
        id1 = ilog.append(LogEntry(source="a", event_type="e1"))
        id2 = ilog.append(LogEntry(source="b", event_type="e2"))
        assert id2 > id1

    def test_append_with_all_fields(self, ilog: InteractionLog) -> None:
        lid = ilog.append(LogEntry(
            source="hub-ble-mgr",
            event_type="pen_discovered",
            severity="info",
            exam_id="exam-42",
            pen_mac="AA:BB:CC:DD:EE:FF",
            invig_id="invig-7",
            detail={"rssi": -45, "dongle": "hci0"},
            timestamp="2026-03-18T10:20:30.123Z",
        ))
        assert lid >= 1

    def test_severity_default_is_info(
        self, ilog: InteractionLog, db_conn: sqlite3.Connection,
    ) -> None:
        ilog.append(LogEntry(source="x", event_type="y"))
        row = db_conn.execute(
            "SELECT severity FROM interaction_log WHERE log_id=1"
        ).fetchone()
        assert row[0] == "info"

    def test_severity_values(self, ilog: InteractionLog) -> None:
        for sev in ("debug", "info", "warn", "error", "critical"):
            ilog.append(LogEntry(source="x", event_type="y", severity=sev))
        assert ilog.count() == 5

    def test_invalid_severity_rejected(
        self, ilog: InteractionLog,
    ) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            ilog.append(LogEntry(
                source="x", event_type="y", severity="banana",
            ))


# ===================================================================
# Detail JSON serialization
# ===================================================================

class TestDetail:

    def test_detail_stored_as_json(
        self, ilog: InteractionLog, db_conn: sqlite3.Connection,
    ) -> None:
        detail = {"key": "value", "count": 42}
        ilog.append(LogEntry(
            source="s", event_type="e", detail=detail,
        ))
        row = db_conn.execute(
            "SELECT detail FROM interaction_log WHERE log_id=1"
        ).fetchone()
        assert json.loads(row[0]) == detail

    def test_none_detail_stored_as_null(
        self, ilog: InteractionLog, db_conn: sqlite3.Connection,
    ) -> None:
        ilog.append(LogEntry(source="s", event_type="e", detail=None))
        row = db_conn.execute(
            "SELECT detail FROM interaction_log WHERE log_id=1"
        ).fetchone()
        assert row[0] is None


# ===================================================================
# Timestamp
# ===================================================================

class TestTimestamp:

    def test_explicit_timestamp(
        self, ilog: InteractionLog, db_conn: sqlite3.Connection,
    ) -> None:
        ts = "2026-01-15T08:30:00.000Z"
        ilog.append(LogEntry(
            source="s", event_type="e", timestamp=ts,
        ))
        row = db_conn.execute(
            "SELECT timestamp FROM interaction_log WHERE log_id=1"
        ).fetchone()
        assert row[0] == ts

    def test_auto_timestamp(
        self, ilog: InteractionLog, db_conn: sqlite3.Connection,
    ) -> None:
        ilog.append(LogEntry(source="s", event_type="e"))
        row = db_conn.execute(
            "SELECT timestamp FROM interaction_log WHERE log_id=1"
        ).fetchone()
        assert row[0] is not None
        assert row[0].endswith("Z")


# ===================================================================
# Read helpers
# ===================================================================

class TestRead:

    def test_recent_returns_latest_first(self, ilog: InteractionLog) -> None:
        ilog.append(LogEntry(source="a", event_type="first"))
        ilog.append(LogEntry(source="b", event_type="second"))
        rows = ilog.recent(10)
        assert rows[0]["event_type"] == "second"
        assert rows[1]["event_type"] == "first"

    def test_recent_respects_limit(self, ilog: InteractionLog) -> None:
        for i in range(20):
            ilog.append(LogEntry(source="s", event_type=f"e{i}"))
        rows = ilog.recent(5)
        assert len(rows) == 5

    def test_count(self, ilog: InteractionLog) -> None:
        assert ilog.count() == 0
        ilog.append(LogEntry(source="s", event_type="e"))
        assert ilog.count() == 1


# ===================================================================
# Lifecycle errors
# ===================================================================

class TestLifecycle:

    def test_append_before_open_raises(self) -> None:
        log = InteractionLog()
        with pytest.raises(RuntimeError, match="not open"):
            log.append(LogEntry(source="s", event_type="e"))

    def test_recent_before_open_raises(self) -> None:
        log = InteractionLog()
        with pytest.raises(RuntimeError, match="not open"):
            log.recent()
