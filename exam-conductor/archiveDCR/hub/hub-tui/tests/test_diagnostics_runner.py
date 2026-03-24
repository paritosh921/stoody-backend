"""Tests for the diagnostics runner engine.

Uses mock test functions to verify runner logic without real hardware.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.diagnostics.runner import (
    DiagnosticsRunner, RunResult, TestCase, TestCategory, TestStatus,
)

# -- Mock factories -------------------------------------------------------

def _make_pass_fn() -> AsyncMock:
    return AsyncMock(return_value=(TestStatus.PASS, {"info": "ok"}))

def _make_fail_fn() -> AsyncMock:
    return AsyncMock(return_value=(TestStatus.FAIL, {"error": "something broke"}))

def _make_skip_fn() -> AsyncMock:
    return AsyncMock(return_value=(TestStatus.SKIP, {"reason": "no hardware"}))

def _make_exception_fn() -> AsyncMock:
    return AsyncMock(side_effect=RuntimeError("unexpected crash"))

def _build_test(tid: str, name: str, cat: TestCategory, fn: AsyncMock, manual: bool = False) -> TestCase:
    return TestCase(id=tid, name=name, category=cat, run_fn=fn, manual=manual)


@pytest.fixture
def sample_tests() -> list[TestCase]:
    """A suite of 6 tests across 3 categories."""
    return [
        _build_test("H1", "Hw pass", TestCategory.HARDWARE, _make_pass_fn()),
        _build_test("H2", "Hw manual", TestCategory.HARDWARE, _make_pass_fn(), manual=True),
        _build_test("S1", "Sw pass", TestCategory.SOFTWARE, _make_pass_fn()),
        _build_test("S2", "Sw fail", TestCategory.SOFTWARE, _make_fail_fn()),
        _build_test("B1", "Ble skip", TestCategory.BLE, _make_skip_fn()),
        _build_test("B2", "Ble crash", TestCategory.BLE, _make_exception_fn()),
    ]

@pytest.fixture
def runner(sample_tests: list[TestCase]) -> DiagnosticsRunner:
    return DiagnosticsRunner(sample_tests)

# -- TestCase dataclass ---------------------------------------------------

def test_testcase_defaults() -> None:
    tc = TestCase(id="X1", name="Test", category=TestCategory.HARDWARE, run_fn=_make_pass_fn())
    assert tc.status == TestStatus.PENDING
    assert tc.duration_ms == 0 and tc.detail == {} and tc.manual is False

def test_testcase_reset() -> None:
    tc = TestCase(id="X1", name="Test", category=TestCategory.HARDWARE, run_fn=_make_pass_fn())
    tc.status, tc.duration_ms, tc.detail = TestStatus.PASS, 42, {"key": "val"}
    tc.reset()
    assert tc.status == TestStatus.PENDING and tc.duration_ms == 0 and tc.detail == {}

# -- Registration and lookup ----------------------------------------------

def test_register_and_lookup(runner: DiagnosticsRunner) -> None:
    assert len(runner.tests) == 6
    assert runner.get_by_id("H1") is not None
    assert runner.get_by_id("Z9") is None

def test_get_by_category(runner: DiagnosticsRunner) -> None:
    assert len(runner.get_by_category(TestCategory.HARDWARE)) == 2
    assert len(runner.get_by_category(TestCategory.SOFTWARE)) == 2
    assert len(runner.get_by_category(TestCategory.BLE)) == 2

# -- run_all --------------------------------------------------------------

async def test_run_all(runner: DiagnosticsRunner) -> None:
    """run_all executes non-manual tests and skips manual ones."""
    result = await runner.run_all()

    assert runner.get_by_id("H1").status == TestStatus.PASS
    assert runner.get_by_id("H1").duration_ms >= 0
    assert runner.get_by_id("H2").status == TestStatus.SKIP  # manual
    assert runner.get_by_id("S1").status == TestStatus.PASS
    assert runner.get_by_id("S2").status == TestStatus.FAIL
    assert runner.get_by_id("B1").status == TestStatus.SKIP  # mock returns SKIP
    assert runner.get_by_id("B2").status == TestStatus.FAIL  # exception
    assert "unexpected crash" in runner.get_by_id("B2").detail.get("error", "")

    assert result.pass_count == 2  # H1, S1
    assert result.fail_count == 2  # S2, B2
    assert result.skip_count == 2  # H2, B1

async def test_run_all_sets_timestamp(runner: DiagnosticsRunner) -> None:
    result = await runner.run_all()
    assert result.timestamp.endswith("Z") and "T" in result.timestamp

# -- run_selected ---------------------------------------------------------

async def test_run_selected(runner: DiagnosticsRunner) -> None:
    await runner.run_selected(["H1", "S2"])
    assert runner.get_by_id("H1").status == TestStatus.PASS
    assert runner.get_by_id("S2").status == TestStatus.FAIL
    assert runner.get_by_id("B1").status == TestStatus.PENDING  # not selected

async def test_run_selected_nonexistent_id(runner: DiagnosticsRunner) -> None:
    await runner.run_selected(["NOPE"])
    for t in runner.tests:
        assert t.status == TestStatus.PENDING

# -- run_category ---------------------------------------------------------

async def test_run_category(runner: DiagnosticsRunner) -> None:
    await runner.run_category(TestCategory.SOFTWARE)
    assert runner.get_by_id("S1").status == TestStatus.PASS
    assert runner.get_by_id("S2").status == TestStatus.FAIL
    assert runner.get_by_id("H1").status == TestStatus.PENDING  # wrong category

# -- Update callback ------------------------------------------------------

async def test_update_callback_invoked(runner: DiagnosticsRunner) -> None:
    calls: list[int] = []
    runner.set_update_callback(lambda: calls.append(1))
    await runner.run_all()
    assert len(calls) >= 6  # at least one per test

# -- Exception handling ---------------------------------------------------

async def test_exception_in_test_fn() -> None:
    fn = AsyncMock(side_effect=ValueError("boom"))
    tc = TestCase(id="X1", name="Boom", category=TestCategory.HARDWARE, run_fn=fn)
    r = DiagnosticsRunner([tc])
    await r.run_all()
    assert tc.status == TestStatus.FAIL
    assert "boom" in tc.detail.get("error", "")

# -- RunResult properties -------------------------------------------------

async def test_run_result_properties(runner: DiagnosticsRunner) -> None:
    result = await runner.run_all()
    assert result.pass_count + result.fail_count + result.skip_count == 6
    assert result.pending_count == 0

# -- JSON export ----------------------------------------------------------

async def test_export_json(runner: DiagnosticsRunner) -> None:
    """export_json writes valid JSON matching TEST_SUITE_SPEC section 3.3."""
    await runner.run_all()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "diagnostics"
        import src.diagnostics.runner as runner_mod
        original = runner_mod.DIAGNOSTICS_DIR
        runner_mod.DIAGNOSTICS_DIR = tmp_path
        try:
            out = runner.export_json()
        finally:
            runner_mod.DIAGNOSTICS_DIR = original

        assert out.exists()
        data = json.loads(out.read_text())

        for key in ("hub_id", "timestamp", "sw_version", "os_version", "tests"):
            assert key in data
        assert len(data["tests"]) == 6
        for t in data["tests"]:
            for key in ("id", "name", "status", "duration_ms", "detail"):
                assert key in t
            assert t["status"] in ("PASS", "FAIL", "SKIP", "PENDING", "RUNNING")

def test_export_json_no_results() -> None:
    with pytest.raises(RuntimeError, match="No results to export"):
        DiagnosticsRunner([]).export_json()

# -- Manual test handling -------------------------------------------------

async def test_manual_tests_skipped() -> None:
    fn = _make_pass_fn()
    tc = _build_test("M1", "Manual", TestCategory.HARDWARE, fn, manual=True)
    r = DiagnosticsRunner([tc])
    await r.run_all()
    assert tc.status == TestStatus.SKIP
    fn.assert_not_called()

# -- last_result property -------------------------------------------------

def test_last_result_initially_none() -> None:
    assert DiagnosticsRunner([]).last_result is None

async def test_last_result_after_run(runner: DiagnosticsRunner) -> None:
    await runner.run_all()
    assert runner.last_result is not None
    assert isinstance(runner.last_result, RunResult)
