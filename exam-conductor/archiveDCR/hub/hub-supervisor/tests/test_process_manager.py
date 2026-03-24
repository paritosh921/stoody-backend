"""Unit tests for process_manager.py — spawn, stop, crash restart, watchdog.

All subprocess interactions are mocked via ``unittest.mock``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.process_manager import (
    ModuleInfo,
    ModuleStatus,
    ProcessManager,
)


# ===================================================================
# Helpers
# ===================================================================

def _make_module(
    name: str = "hub-timer",
    optional: bool = False,
) -> ModuleInfo:
    return ModuleInfo(
        name=name,
        socket_path=f"/run/exampen/{name}.sock",
        command=[f"/opt/exampen/bin/{name}"],
        optional=optional,
    )


def _fake_process(returncode: int | None = None) -> MagicMock:
    """Create a mock asyncio.subprocess.Process."""
    proc = MagicMock()
    proc.pid = 12345
    proc.returncode = returncode
    proc.wait = AsyncMock(return_value=returncode)
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    return proc


# ===================================================================
# Spawn tests
# ===================================================================

class TestSpawn:

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.mod = _make_module()
        self.pm = ProcessManager([self.mod], max_restarts=3)

    async def test_spawn_module(self) -> None:
        fake_proc = _fake_process()
        fake_proc.returncode = None
        # Make wait() hang forever so we don't trigger crash handler
        fake_proc.wait = AsyncMock(side_effect=asyncio.CancelledError)
        with patch("src.process_manager.asyncio.create_subprocess_exec",
                    new_callable=AsyncMock, return_value=fake_proc):
            await self.pm.spawn_module("hub-timer")
        assert self.mod.status == ModuleStatus.RUNNING
        assert self.mod.process is fake_proc

    async def test_spawn_unknown_module_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown module"):
            await self.pm.spawn_module("nonexistent")

    async def test_spawn_all_skips_optional(self) -> None:
        opt_mod = _make_module("hub-tui", optional=True)
        req_mod = _make_module("hub-store")
        pm = ProcessManager([opt_mod, req_mod], max_restarts=3)
        fake_proc = _fake_process()
        fake_proc.returncode = None
        fake_proc.wait = AsyncMock(side_effect=asyncio.CancelledError)
        with patch("src.process_manager.asyncio.create_subprocess_exec",
                    new_callable=AsyncMock, return_value=fake_proc):
            await pm.spawn_all()
        assert opt_mod.status == ModuleStatus.STOPPED
        assert req_mod.status == ModuleStatus.RUNNING


# ===================================================================
# Stop tests
# ===================================================================

class TestStop:

    async def test_stop_running_module(self) -> None:
        mod = _make_module()
        pm = ProcessManager([mod], max_restarts=3)
        fake_proc = _fake_process()
        fake_proc.returncode = None
        fake_proc.wait = AsyncMock(side_effect=asyncio.CancelledError)
        with patch("src.process_manager.asyncio.create_subprocess_exec",
                    new_callable=AsyncMock, return_value=fake_proc):
            await pm.spawn_module("hub-timer")
        # Now stop it — simulate process.wait() completing after terminate
        fake_proc.wait = AsyncMock(return_value=0)
        fake_proc.returncode = 0
        await pm.stop_module("hub-timer")
        assert mod.status == ModuleStatus.STOPPED
        assert mod.process is None
        fake_proc.terminate.assert_called_once()

    async def test_stop_all(self) -> None:
        mod = _make_module()
        pm = ProcessManager([mod], max_restarts=3)
        fake_proc = _fake_process()
        fake_proc.returncode = None
        fake_proc.wait = AsyncMock(side_effect=asyncio.CancelledError)
        with patch("src.process_manager.asyncio.create_subprocess_exec",
                    new_callable=AsyncMock, return_value=fake_proc):
            await pm.spawn_module("hub-timer")
        fake_proc.wait = AsyncMock(return_value=0)
        fake_proc.returncode = 0
        await pm.stop_all()
        assert mod.status == ModuleStatus.STOPPED


# ===================================================================
# Crash restart tests
# ===================================================================

class TestCrashRestart:

    async def test_crash_triggers_restart(self) -> None:
        mod = _make_module()
        status_changes: list[tuple[str, ModuleStatus]] = []

        async def on_change(name: str, status: ModuleStatus) -> None:
            status_changes.append((name, status))

        pm = ProcessManager(
            [mod], max_restarts=3, on_status_change=on_change,
        )
        call_count = 0

        async def fake_create(*args: object, **kw: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            p = _fake_process()
            if call_count == 1:
                # First spawn: crash immediately
                p.returncode = None
                p.wait = AsyncMock(return_value=1)
            else:
                # Second spawn: stay alive
                p.returncode = None
                p.wait = AsyncMock(side_effect=asyncio.CancelledError)
            return p

        with patch("src.process_manager.asyncio.create_subprocess_exec",
                    side_effect=fake_create):
            await pm.spawn_module("hub-timer")
            # Give the waiter task time to run
            await asyncio.sleep(0.05)

        assert call_count == 2
        assert mod.restart_count == 1
        assert mod.status == ModuleStatus.RUNNING

    async def test_max_restarts_marks_failed(self) -> None:
        mod = _make_module()
        pm = ProcessManager([mod], max_restarts=1)
        call_count = 0

        async def fake_create(*args: object, **kw: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            p = _fake_process()
            p.returncode = None
            p.wait = AsyncMock(return_value=1)  # always crash
            return p

        with patch("src.process_manager.asyncio.create_subprocess_exec",
                    side_effect=fake_create):
            await pm.spawn_module("hub-timer")
            # Give time for crash + restart + second crash
            await asyncio.sleep(0.1)

        assert mod.status == ModuleStatus.FAILED


# ===================================================================
# Watchdog tests
# ===================================================================

class TestWatchdog:

    async def test_watchdog_calls_health_check(self) -> None:
        mod = _make_module()
        mod.status = ModuleStatus.RUNNING
        health_called: list[str] = []

        async def mock_health(name: str, sock: str) -> bool:
            health_called.append(name)
            return True

        pm = ProcessManager(
            [mod], health_check_fn=mock_health, watchdog_interval=0.05,
        )
        pm.start_watchdog()
        await asyncio.sleep(0.12)
        pm._stop_watchdog()
        assert "hub-timer" in health_called

    async def test_watchdog_skips_stopped_modules(self) -> None:
        mod = _make_module()
        mod.status = ModuleStatus.STOPPED
        health_called: list[str] = []

        async def mock_health(name: str, sock: str) -> bool:
            health_called.append(name)
            return True

        pm = ProcessManager(
            [mod], health_check_fn=mock_health, watchdog_interval=0.05,
        )
        pm.start_watchdog()
        await asyncio.sleep(0.12)
        pm._stop_watchdog()
        assert "hub-timer" not in health_called


# ===================================================================
# Status summary
# ===================================================================

class TestStatusSummary:

    def test_get_status_summary(self) -> None:
        m1 = _make_module("hub-timer")
        m2 = _make_module("hub-store")
        m2.status = ModuleStatus.RUNNING
        pm = ProcessManager([m1, m2])
        summary = pm.get_status_summary()
        assert summary["hub-timer"] == "stopped"
        assert summary["hub-store"] == "running"
