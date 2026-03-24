"""Migration runner for hub SQLite database.

Opens (or creates) the hub database, enables WAL mode and foreign keys,
then applies SQL migration files in order.  A ``schema_version`` table
tracks which migrations have already been applied so the runner is
idempotent — calling it twice is safe.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_MIGRATIONS_DIR = Path(__file__).parent

# Ordered list of migration files to apply.
_MIGRATION_FILES: list[tuple[int, str]] = [
    (1, "001_initial.sql"),
]


def _ensure_schema_version_table(conn: sqlite3.Connection) -> None:
    """Create the bookkeeping table if it does not exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version     INTEGER PRIMARY KEY,
            filename    TEXT NOT NULL,
            applied_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
        """
    )


def _applied_versions(conn: sqlite3.Connection) -> set[int]:
    """Return the set of migration versions already applied."""
    rows = conn.execute("SELECT version FROM schema_version").fetchall()
    return {row[0] for row in rows}


def run_migrations(db_path: Path) -> sqlite3.Connection:
    """Apply all pending migrations to the SQLite database at *db_path*.

    * Enables WAL journal mode for crash safety.
    * Enables foreign-key enforcement.
    * Returns the open :class:`sqlite3.Connection` for caller use.

    The function is idempotent: running it multiple times against the
    same database will not re-apply migrations or raise errors.
    """
    conn = sqlite3.connect(str(db_path))

    # --- Pragmas (must be set before any schema work) ---
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    _ensure_schema_version_table(conn)
    applied = _applied_versions(conn)

    for version, filename in _MIGRATION_FILES:
        if version in applied:
            continue

        sql_path = _MIGRATIONS_DIR / filename
        sql_text = sql_path.read_text(encoding="utf-8")

        conn.executescript(sql_text)

        # Re-enable foreign keys — executescript may reset pragmas.
        conn.execute("PRAGMA foreign_keys = ON")

        conn.execute(
            "INSERT INTO schema_version (version, filename) VALUES (?, ?)",
            (version, filename),
        )
        conn.commit()

    return conn
