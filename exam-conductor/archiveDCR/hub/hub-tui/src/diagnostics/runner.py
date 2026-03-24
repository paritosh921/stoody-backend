"""Diagnostics runner engine — executes test cases, tracks results, exports JSON.

Provides TestCase dataclass, DiagnosticsRunner for orchestrating runs by
category or selection, and JSON export per TEST_SUITE_SPEC section 3.3.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine

DIAGNOSTICS_DIR = Path("/var/lib/exampen/diagnostics")
HUB_DB_PATH = Path("/var/lib/exampen/hub.db")


class TestStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


class TestCategory(Enum):
    HARDWARE = "hardware"
    SOFTWARE = "software"
    BLE = "ble"


# Type alias for async test functions.
# Must return (TestStatus, detail_dict).
TestFn = Callable[[], Coroutine[Any, Any, tuple[TestStatus, dict[str, Any]]]]


@dataclass
class TestCase:
    """Single diagnostic test case."""

    id: str
    name: str
    category: TestCategory
    run_fn: TestFn
    status: TestStatus = TestStatus.PENDING
    duration_ms: int = 0
    detail: dict[str, Any] = field(default_factory=dict)
    manual: bool = False

    def reset(self) -> None:
        self.status = TestStatus.PENDING
        self.duration_ms = 0
        self.detail = {}


@dataclass
class RunResult:
    """Aggregate result of a diagnostics run."""

    timestamp: str
    tests: list[TestCase]
    hub_id: str = ""
    sw_version: str = "0.4.2"
    os_version: str = ""

    @property
    def pass_count(self) -> int:
        return sum(1 for t in self.tests if t.status == TestStatus.PASS)

    @property
    def fail_count(self) -> int:
        return sum(1 for t in self.tests if t.status == TestStatus.FAIL)

    @property
    def skip_count(self) -> int:
        return sum(1 for t in self.tests if t.status == TestStatus.SKIP)

    @property
    def pending_count(self) -> int:
        return sum(
            1 for t in self.tests
            if t.status in (TestStatus.PENDING, TestStatus.RUNNING)
        )


class DiagnosticsRunner:
    """Orchestrates diagnostic test execution and result export."""

    def __init__(self, tests: list[TestCase] | None = None) -> None:
        self._tests: list[TestCase] = tests or []
        self._on_update: Callable[[], Any] | None = None
        self._last_result: RunResult | None = None

    @property
    def tests(self) -> list[TestCase]:
        return list(self._tests)

    @property
    def last_result(self) -> RunResult | None:
        return self._last_result

    def set_update_callback(self, cb: Callable[[], Any]) -> None:
        """Set a callback invoked after each test status change."""
        self._on_update = cb

    def register(self, test: TestCase) -> None:
        self._tests.append(test)

    def get_by_id(self, test_id: str) -> TestCase | None:
        for t in self._tests:
            if t.id == test_id:
                return t
        return None

    def get_by_category(self, category: TestCategory) -> list[TestCase]:
        return [t for t in self._tests if t.category == category]

    async def _run_one(self, tc: TestCase) -> None:
        """Execute a single test case, updating its status and duration."""
        if tc.manual:
            tc.status = TestStatus.SKIP
            tc.detail = {"reason": "manual test — skipped in auto-run"}
            self._notify()
            return

        tc.status = TestStatus.RUNNING
        tc.detail = {}
        self._notify()

        start = time.monotonic()
        try:
            status, detail = await tc.run_fn()
            tc.status = status
            tc.detail = detail
        except Exception as exc:
            tc.status = TestStatus.FAIL
            tc.detail = {"error": str(exc)}
        finally:
            elapsed = time.monotonic() - start
            tc.duration_ms = int(elapsed * 1000)
            self._notify()

    async def run_all(self) -> RunResult:
        """Run every non-manual test sequentially."""
        for tc in self._tests:
            tc.reset()
        self._notify()

        for tc in self._tests:
            await self._run_one(tc)

        return self._build_result()

    async def run_selected(self, test_ids: list[str]) -> RunResult:
        """Run only the specified tests by ID."""
        targets = [t for t in self._tests if t.id in test_ids]
        for tc in targets:
            tc.reset()
        self._notify()

        for tc in targets:
            await self._run_one(tc)

        return self._build_result()

    async def run_category(self, category: TestCategory) -> RunResult:
        """Run all tests in a given category."""
        targets = self.get_by_category(category)
        for tc in targets:
            tc.reset()
        self._notify()

        for tc in targets:
            await self._run_one(tc)

        return self._build_result()

    def _build_result(self) -> RunResult:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        result = RunResult(
            timestamp=ts,
            tests=list(self._tests),
            hub_id=self._read_hub_id(),
            os_version=self._read_os_version(),
        )
        self._last_result = result
        return result

    def _notify(self) -> None:
        if self._on_update is not None:
            self._on_update()

    @staticmethod
    def _read_hub_id() -> str:
        conf = Path("/etc/exampen/hub.conf")
        if conf.exists():
            try:
                for line in conf.read_text().splitlines():
                    if line.startswith("hub_id="):
                        return line.split("=", 1)[1].strip()
            except OSError:
                pass
        return "UNKNOWN"

    @staticmethod
    def _read_os_version() -> str:
        try:
            return f"{platform.system()} {platform.release()}"
        except Exception:
            return "unknown"

    def export_json(self, result: RunResult | None = None) -> Path:
        """Write results JSON to diagnostics directory. Returns the file path."""
        result = result or self._last_result
        if result is None:
            raise RuntimeError("No results to export — run tests first")

        DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

        ts_file = result.timestamp.replace(":", "-").replace("T", "_").rstrip("Z")
        out_path = DIAGNOSTICS_DIR / f"{ts_file}.json"

        payload = {
            "hub_id": result.hub_id,
            "timestamp": result.timestamp,
            "sw_version": result.sw_version,
            "os_version": result.os_version,
            "tests": [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "status": tc.status.value,
                    "duration_ms": tc.duration_ms,
                    "detail": tc.detail,
                }
                for tc in result.tests
            ],
        }

        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path
