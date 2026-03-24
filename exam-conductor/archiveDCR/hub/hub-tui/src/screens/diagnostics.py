"""Diagnostics screen — hardware, software, and BLE test runner.

Wires real test implementations from src.diagnostics into the TUI.
[R] Run All, [S] Run Selected, [E] Export JSON, [Q] Back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.containers import Vertical
from textual.widgets import Header, Static

from src.diagnostics.ble_tests import build_ble_tests
from src.diagnostics.hardware_tests import build_hardware_tests
from src.diagnostics.runner import (
    DiagnosticsRunner,
    TestCase,
    TestCategory,
    TestStatus,
)
from src.diagnostics.software_tests import build_software_tests
from src.widgets.footer import HubFooter


_STATUS_ICONS: dict[TestStatus, str] = {
    TestStatus.PASS: "[green]\u25cf PASS[/green]",
    TestStatus.FAIL: "[red]\u2717 FAIL[/red]",
    TestStatus.RUNNING: "[yellow]\u25d0 RUNNING[/yellow]",
    TestStatus.PENDING: "[dim]\u25cb PENDING[/dim]",
    TestStatus.SKIP: "[dim]\u25cb SKIP[/dim]",
}


class DiagnosticsScreen(Screen):
    """Test runner — shows H1-H7, S1-S5, B1-B4 with live status icons."""

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("q", "pop_screen", "Back", show=True),
        Binding("r", "run_all", "Run All", show=True),
        Binding("s", "run_selected", "Run Selected", show=True),
        Binding("e", "export_results", "Export", show=True),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("space", "toggle_select", "Toggle", show=False),
    ]

    DEFAULT_CSS = """
    DiagnosticsScreen #diag-container {
        padding: 1 2;
    }
    DiagnosticsScreen .screen-title {
        text-style: bold;
        margin-bottom: 1;
    }
    DiagnosticsScreen .section-title {
        text-style: bold;
        margin-top: 1;
    }
    DiagnosticsScreen .test-line {
        margin: 0 1;
    }
    DiagnosticsScreen .divider {
        margin-top: 1;
        color: $text-muted;
    }
    DiagnosticsScreen .summary-line {
        margin-top: 0;
    }
    DiagnosticsScreen .action-hint {
        color: $text-muted;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        hw = build_hardware_tests()
        sw = build_software_tests()
        ble = build_ble_tests()
        all_tests = hw + sw + ble

        self._runner = DiagnosticsRunner(all_tests)
        self._runner.set_update_callback(self._refresh_display)

        self._hw_tests = hw
        self._sw_tests = sw
        self._ble_tests = ble

        # For cursor-based selection
        self._all_tests = all_tests
        self._cursor: int = 0
        self._selected: set[str] = set()

        # Compat: expose _state-like properties for existing test_app.py
        self._state = _DiagStateCompat(self._runner, all_tests)

        self._running = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="diag-container"):
            yield Static("[6] Hub Diagnostics", classes="screen-title")

            yield Static("Hardware Tests", classes="section-title")
            yield Static(id="hw-tests")

            yield Static("Software Tests", classes="section-title")
            yield Static(id="sw-tests")

            yield Static(
                "BLE Tests (requires pens/simulator)", classes="section-title"
            )
            yield Static(id="ble-tests")

            yield Static("------", classes="divider")
            yield Static(
                "[R] Run all    [S] Run selected    [E] Export results    [Q] Back",
                classes="action-hint",
            )
            yield Static(id="summary", classes="summary-line")
            yield Static("", id="diag-action-result")
        yield HubFooter()

    def on_mount(self) -> None:
        self._refresh_display()

    # ── Display helpers ──────────────────────────────────────────────────

    def _render_group(self, tests: list[TestCase]) -> str:
        lines: list[str] = []
        for t in tests:
            icon = _STATUS_ICONS[t.status]
            cursor = ">" if self._all_tests[self._cursor].id == t.id else " "
            sel = "*" if t.id in self._selected else " "
            lines.append(f" {cursor}{sel}[{t.id}] {t.name:<28}{icon}")
        return "\n".join(lines)

    def _refresh_display(self) -> None:
        try:
            self.query_one("#hw-tests", Static).update(
                self._render_group(self._hw_tests)
            )
            self.query_one("#sw-tests", Static).update(
                self._render_group(self._sw_tests)
            )
            self.query_one("#ble-tests", Static).update(
                self._render_group(self._ble_tests)
            )
        except Exception:
            # Screen not yet mounted — silently ignore.
            return

        result = self._runner.last_result
        last_run = result.timestamp if result else "Never"
        p = sum(1 for t in self._all_tests if t.status == TestStatus.PASS)
        f = sum(1 for t in self._all_tests if t.status == TestStatus.FAIL)
        s = sum(1 for t in self._all_tests if t.status == TestStatus.SKIP)
        pend = sum(
            1 for t in self._all_tests
            if t.status in (TestStatus.PENDING, TestStatus.RUNNING)
        )
        self.query_one("#summary", Static).update(
            f"Last run: {last_run}    "
            f"Pass: {p}  Fail: {f}  Skip: {s}  Pending: {pend}"
        )

    # ── Cursor / selection actions ───────────────────────────────────────

    def action_cursor_up(self) -> None:
        if self._cursor > 0:
            self._cursor -= 1
            self._refresh_display()

    def action_cursor_down(self) -> None:
        if self._cursor < len(self._all_tests) - 1:
            self._cursor += 1
            self._refresh_display()

    def action_toggle_select(self) -> None:
        tid = self._all_tests[self._cursor].id
        if tid in self._selected:
            self._selected.discard(tid)
        else:
            self._selected.add(tid)
        self._refresh_display()

    # ── Test execution actions ───────────────────────────────────────────

    async def action_run_all(self) -> None:
        """Execute all non-manual tests."""
        if self._running:
            return
        self._running = True
        result_widget = self.query_one("#diag-action-result", Static)
        result_widget.update("Running all tests...")

        try:
            result = await self._runner.run_all()
            result_widget.update(
                f"Run All complete — "
                f"Pass: {result.pass_count}  Fail: {result.fail_count}  "
                f"Skip: {result.skip_count}"
            )
        except Exception as exc:
            result_widget.update(f"Run All failed: {exc}")
        finally:
            self._running = False
            self._refresh_display()

    async def action_run_selected(self) -> None:
        """Execute the selected tests (or the highlighted test if none selected)."""
        if self._running:
            return

        ids = list(self._selected) if self._selected else [
            self._all_tests[self._cursor].id
        ]

        self._running = True
        result_widget = self.query_one("#diag-action-result", Static)
        result_widget.update(f"Running {len(ids)} test(s): {', '.join(ids)}...")

        try:
            result = await self._runner.run_selected(ids)
            result_widget.update(
                f"Run Selected complete — "
                f"Pass: {result.pass_count}  Fail: {result.fail_count}  "
                f"Skip: {result.skip_count}"
            )
        except Exception as exc:
            result_widget.update(f"Run Selected failed: {exc}")
        finally:
            self._running = False
            self._refresh_display()

    def action_export_results(self) -> None:
        """Write JSON results to /var/lib/exampen/diagnostics/."""
        result_widget = self.query_one("#diag-action-result", Static)

        if self._runner.last_result is None:
            result_widget.update("No results to export — run tests first")
            return

        try:
            path = self._runner.export_json()
            result_widget.update(f"Exported to {path}")
        except Exception as exc:
            result_widget.update(f"Export failed: {exc}")

    def action_pop_screen(self) -> None:
        self.app.pop_screen()


class _DiagStateCompat:
    """Backward-compatible shim for test_app.py that accesses _state."""

    def __init__(
        self, runner: DiagnosticsRunner, tests: list[TestCase]
    ) -> None:
        self._runner = runner
        self._tests = tests

    @property
    def pass_count(self) -> int:
        return sum(1 for t in self._tests if t.status == TestStatus.PASS)

    @property
    def fail_count(self) -> int:
        return sum(1 for t in self._tests if t.status == TestStatus.FAIL)

    @property
    def skip_count(self) -> int:
        return sum(1 for t in self._tests if t.status == TestStatus.SKIP)

    @property
    def pending_count(self) -> int:
        return sum(
            1 for t in self._tests
            if t.status in (TestStatus.PENDING, TestStatus.RUNNING)
        )

    @property
    def last_run(self) -> str:
        r = self._runner.last_result
        return r.timestamp if r else "Never"
