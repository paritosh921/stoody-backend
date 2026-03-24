"""Software diagnostic tests S1-S5.

Each test is an async function returning (TestStatus, detail_dict).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.diagnostics.runner import TestCase, TestCategory, TestStatus

HUB_DB_PATH = Path("/var/lib/exampen/hub.db")

# Hub modules and their IPC socket paths (unix domain sockets).
HUB_MODULES = [
    ("hub-supervisor", "/run/exampen/supervisor.sock"),
    ("hub-ble-mgr", "/run/exampen/ble-mgr.sock"),
    ("hub-pen-sync", "/run/exampen/pen-sync.sock"),
    ("hub-timer", "/run/exampen/timer.sock"),
    ("hub-store", "/run/exampen/store.sock"),
    ("hub-uplink", "/run/exampen/uplink.sock"),
    ("hub-invig-ble", "/run/exampen/invig-ble.sock"),
]

BACKEND_HEALTH_TIMEOUT = 10  # seconds


async def _run_cmd(cmd: str) -> tuple[int, str, str]:
    """Run a shell command, return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return proc.returncode or 0, out.decode(errors="replace").strip(), err.decode(errors="replace").strip()


# -- S1: SQLite integrity ------------------------------------------------

async def s1_sqlite_integrity() -> tuple[TestStatus, dict[str, Any]]:
    """Run PRAGMA integrity_check, foreign_key_check, and verify WAL mode."""
    if not HUB_DB_PATH.exists():
        return TestStatus.FAIL, {"error": f"Database not found: {HUB_DB_PATH}"}

    detail: dict[str, Any] = {}
    try:
        conn = sqlite3.connect(str(HUB_DB_PATH), timeout=5)
        cur = conn.cursor()

        cur.execute("PRAGMA integrity_check;")
        integrity = cur.fetchone()
        detail["integrity_check"] = integrity[0] if integrity else "no result"
        if not integrity or integrity[0] != "ok":
            conn.close()
            return TestStatus.FAIL, {**detail, "error": f"integrity_check: {detail['integrity_check']}"}

        cur.execute("PRAGMA foreign_key_check;")
        fk_errors = cur.fetchall()
        detail["foreign_key_errors"] = len(fk_errors)
        if fk_errors:
            conn.close()
            return TestStatus.FAIL, {**detail, "error": f"foreign_key_check found {len(fk_errors)} violations"}

        cur.execute("PRAGMA journal_mode;")
        mode = cur.fetchone()
        detail["journal_mode"] = mode[0] if mode else "unknown"
        if not mode or mode[0].lower() != "wal":
            conn.close()
            return TestStatus.FAIL, {**detail, "error": f"Expected WAL mode, got: {detail['journal_mode']}"}

        detail["db_size_kb"] = HUB_DB_PATH.stat().st_size // 1024
        conn.close()
    except sqlite3.Error as exc:
        return TestStatus.FAIL, {"error": f"SQLite error: {exc}"}
    except OSError as exc:
        return TestStatus.FAIL, {"error": f"OS error: {exc}"}

    return TestStatus.PASS, detail


# -- S2: Service health --------------------------------------------------

async def _ipc_health_check(name: str, sock_path: str) -> tuple[str, bool, str]:
    """Send a health request to a module's IPC socket."""
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(sock_path), timeout=3)
        writer.write((json.dumps({"cmd": "health"}) + "\n").encode())
        await writer.drain()
        data = await asyncio.wait_for(reader.readline(), timeout=3)
        writer.close()
        await writer.wait_closed()
        resp = json.loads(data.decode())
        return (name, True, "healthy") if resp.get("status") == "ok" else (name, False, f"unhealthy: {resp}")
    except FileNotFoundError:
        return name, False, f"socket not found: {sock_path}"
    except asyncio.TimeoutError:
        return name, False, "timeout"
    except Exception as exc:
        return name, False, str(exc)


async def s2_service_health() -> tuple[TestStatus, dict[str, Any]]:
    """Check each hub module responds to IPC health request."""
    results = await asyncio.gather(*[_ipc_health_check(n, s) for n, s in HUB_MODULES])
    services = {n: {"healthy": ok, "detail": msg} for n, ok, msg in results}
    all_ok = all(ok for _, ok, _ in results)
    detail: dict[str, Any] = {"services": services}
    if not all_ok:
        detail["error"] = f"Unhealthy services: {', '.join(n for n, ok, _ in results if not ok)}"
    return TestStatus.PASS if all_ok else TestStatus.FAIL, detail


# -- S3: IPC connectivity ------------------------------------------------

async def _ipc_ping(name: str, sock_path: str) -> tuple[str, bool, str]:
    """Ping a module's IPC socket (connection test only)."""
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(sock_path), timeout=3)
        writer.close()
        await writer.wait_closed()
        return name, True, "reachable"
    except FileNotFoundError:
        return name, False, f"socket not found: {sock_path}"
    except asyncio.TimeoutError:
        return name, False, "timeout"
    except Exception as exc:
        return name, False, str(exc)


async def s3_ipc_connectivity() -> tuple[TestStatus, dict[str, Any]]:
    """Ping each module's IPC socket to verify connectivity."""
    results = await asyncio.gather(*[_ipc_ping(n, s) for n, s in HUB_MODULES])
    sockets = {n: {"reachable": ok, "detail": msg} for n, ok, msg in results}
    all_ok = all(ok for _, ok, _ in results)
    detail: dict[str, Any] = {"sockets": sockets}
    if not all_ok:
        detail["error"] = f"Unreachable sockets: {', '.join(n for n, ok, _ in results if not ok)}"
    return TestStatus.PASS if all_ok else TestStatus.FAIL, detail


# -- S4: Backend reachability ---------------------------------------------

async def _read_backend_url() -> str | None:
    """Read backend URL from hub config."""
    conf = Path("/etc/exampen/hub.conf")
    if not conf.exists():
        return None
    try:
        for line in conf.read_text().splitlines():
            if line.startswith("backend_url="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


async def s4_backend_reachability() -> tuple[TestStatus, dict[str, Any]]:
    """HTTP HEAD to backend health endpoint."""
    url = await _read_backend_url()
    if not url:
        return TestStatus.FAIL, {"error": "backend_url not configured in hub.conf"}

    health_url = f"{url.rstrip('/')}/health"
    rc, stdout, stderr = await _run_cmd(
        f"curl -s -o /dev/null -w '%{{http_code}}' --max-time {BACKEND_HEALTH_TIMEOUT} --head {health_url}"
    )
    if rc != 0:
        return TestStatus.FAIL, {"url": health_url, "error": f"curl failed (rc={rc}): {stderr}"}

    http_code = stdout.strip().strip("'")
    detail: dict[str, Any] = {"url": health_url, "http_status": http_code}
    if http_code == "200":
        return TestStatus.PASS, detail
    return TestStatus.FAIL, {**detail, "error": f"Backend returned HTTP {http_code}"}


# -- S5: Invigilator code cache ------------------------------------------

async def s5_invigilator_code_cache() -> tuple[TestStatus, dict[str, Any]]:
    """Check invig_codes table has non-expired entries."""
    if not HUB_DB_PATH.exists():
        return TestStatus.FAIL, {"error": f"Database not found: {HUB_DB_PATH}"}

    try:
        conn = sqlite3.connect(str(HUB_DB_PATH), timeout=5)
        cur = conn.cursor()
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cur.execute("SELECT COUNT(*) FROM invig_codes WHERE valid_until > ?;", (now_iso,))
        valid_count = (cur.fetchone() or (0,))[0]
        cur.execute("SELECT COUNT(*) FROM invig_codes;")
        total_count = (cur.fetchone() or (0,))[0]
        conn.close()
    except sqlite3.OperationalError as exc:
        return TestStatus.FAIL, {"error": f"SQLite error: {exc}"}

    detail: dict[str, Any] = {"total_codes": total_count, "valid_codes": valid_count}
    if valid_count == 0:
        return TestStatus.FAIL, {**detail, "error": "No valid (non-expired) invigilator codes cached"}
    return TestStatus.PASS, detail


# -- Registry -------------------------------------------------------------

def build_software_tests() -> list[TestCase]:
    """Return the full set of software test cases S1-S5."""
    return [
        TestCase(id="S1", name="SQLite integrity", category=TestCategory.SOFTWARE, run_fn=s1_sqlite_integrity),
        TestCase(id="S2", name="Service health", category=TestCategory.SOFTWARE, run_fn=s2_service_health),
        TestCase(id="S3", name="IPC connectivity", category=TestCategory.SOFTWARE, run_fn=s3_ipc_connectivity),
        TestCase(id="S4", name="Backend reachability", category=TestCategory.SOFTWARE, run_fn=s4_backend_reachability),
        TestCase(id="S5", name="Invigilator code cache", category=TestCategory.SOFTWARE, run_fn=s5_invigilator_code_cache),
    ]
