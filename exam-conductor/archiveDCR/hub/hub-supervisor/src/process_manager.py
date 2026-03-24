"""Child module lifecycle manager.

Spawns hub child processes (hub-ble-mgr, hub-pen-sync, hub-timer, etc.),
monitors them for crashes, and restarts failed children up to a
configurable maximum.

Uses ``asyncio.create_subprocess_exec`` for spawning.  IPC health checks
are performed on a periodic watchdog interval.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

from src.config import (
    HEALTH_CHECK_TIMEOUT_SEC,
    MAX_RESTART_COUNT,
    WATCHDOG_INTERVAL_SEC,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module status
# ---------------------------------------------------------------------------

class ModuleStatus(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    CRASHED = "crashed"
    FAILED = "failed"  # exceeded max restarts


# ---------------------------------------------------------------------------
# ModuleInfo dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModuleInfo:
    """Tracks a single child module's state."""

    name: str
    socket_path: str
    command: list[str]
    process: asyncio.subprocess.Process | None = None
    status: ModuleStatus = ModuleStatus.STOPPED
    restart_count: int = 0
    optional: bool = False

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process else None


# Type alias for the health check callback.
HealthCheckFn = Callable[[str, str], Coroutine[Any, Any, bool]]

# Type alias for crash notification callback.
CrashCallbackFn = Callable[[str, ModuleStatus], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# ProcessManager
# ---------------------------------------------------------------------------

class ProcessManager:
    """Manages child module processes with crash restart and watchdog.

    Parameters
    ----------
    modules:
        List of :class:`ModuleInfo` to manage.
    health_check_fn:
        Async callback ``(module_name, socket_path) -> bool`` used by the
        watchdog to verify module liveness via IPC.
    on_status_change:
        Optional async callback invoked when a module's status changes.
    """

    def __init__(
        self,
        modules: list[ModuleInfo],
        *,
        health_check_fn: HealthCheckFn | None = None,
        on_status_change: CrashCallbackFn | None = None,
        max_restarts: int = MAX_RESTART_COUNT,
        watchdog_interval: float = WATCHDOG_INTERVAL_SEC,
    ) -> None:
        self._modules: dict[str, ModuleInfo] = {m.name: m for m in modules}
        self._health_check_fn = health_check_fn
        self._on_status_change = on_status_change
        self._max_restarts = max_restarts
        self._watchdog_interval = watchdog_interval
        self._waiter_tasks: dict[str, asyncio.Task[None]] = {}
        self._watchdog_task: asyncio.Task[None] | None = None

    # -- public API ---------------------------------------------------------

    @property
    def modules(self) -> dict[str, ModuleInfo]:
        return dict(self._modules)

    async def spawn_module(self, name: str) -> None:
        """Start a child module process."""
        mod = self._modules.get(name)
        if mod is None:
            raise KeyError(f"Unknown module: {name}")
        if mod.status == ModuleStatus.RUNNING and mod.process is not None:
            logger.warning("Module %s is already running (pid=%s)", name, mod.pid)
            return
        logger.info("Spawning module %s: %s", name, mod.command)
        proc = await asyncio.create_subprocess_exec(
            *mod.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        mod.process = proc
        await self._set_status(mod, ModuleStatus.RUNNING)
        # Start a waiter task that detects when the process exits.
        self._waiter_tasks[name] = asyncio.create_task(
            self._wait_exit(name)
        )

    async def spawn_all(self) -> None:
        """Spawn all non-optional modules (and optional if desired)."""
        for mod in self._modules.values():
            if mod.optional:
                continue
            await self.spawn_module(mod.name)

    async def stop_module(self, name: str) -> None:
        """Gracefully stop a child module: SIGTERM, then wait."""
        mod = self._modules.get(name)
        if mod is None:
            raise KeyError(f"Unknown module: {name}")
        # Cancel the waiter task so we don't trigger crash handler.
        waiter = self._waiter_tasks.pop(name, None)
        if waiter is not None:
            waiter.cancel()
        if mod.process is not None and mod.process.returncode is None:
            logger.info("Stopping module %s (pid=%s)", name, mod.pid)
            mod.process.terminate()
            try:
                await asyncio.wait_for(mod.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Module %s did not exit, killing", name)
                mod.process.kill()
                await mod.process.wait()
        mod.process = None
        await self._set_status(mod, ModuleStatus.STOPPED)

    async def stop_all(self) -> None:
        """Stop all child modules and the watchdog."""
        self._stop_watchdog()
        for name in list(self._modules):
            await self.stop_module(name)

    def start_watchdog(self) -> None:
        """Start the periodic health-check watchdog loop."""
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    def _stop_watchdog(self) -> None:
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            self._watchdog_task = None

    def get_status_summary(self) -> dict[str, str]:
        """Return ``{module_name: status_value}`` for all modules."""
        return {name: mod.status.value for name, mod in self._modules.items()}

    # -- internal -----------------------------------------------------------

    async def _set_status(self, mod: ModuleInfo, status: ModuleStatus) -> None:
        old = mod.status
        mod.status = status
        if old != status and self._on_status_change is not None:
            await self._on_status_change(mod.name, status)

    async def _wait_exit(self, name: str) -> None:
        """Wait for a child to exit and handle crash restart."""
        mod = self._modules[name]
        assert mod.process is not None
        await mod.process.wait()
        rc = mod.process.returncode
        logger.warning("Module %s exited with code %s", name, rc)
        mod.process = None
        if mod.restart_count >= self._max_restarts:
            logger.error(
                "Module %s exceeded max restarts (%d), marking failed",
                name, self._max_restarts,
            )
            await self._set_status(mod, ModuleStatus.FAILED)
            return
        await self._set_status(mod, ModuleStatus.CRASHED)
        mod.restart_count += 1
        logger.info(
            "Restarting module %s (attempt %d/%d)",
            name, mod.restart_count, self._max_restarts,
        )
        await self.spawn_module(name)

    async def _watchdog_loop(self) -> None:
        """Periodically health-check each running module via IPC."""
        try:
            while True:
                await asyncio.sleep(self._watchdog_interval)
                await self._run_health_checks()
        except asyncio.CancelledError:
            pass

    async def _run_health_checks(self) -> None:
        if self._health_check_fn is None:
            return
        for name, mod in self._modules.items():
            if mod.status != ModuleStatus.RUNNING:
                continue
            try:
                healthy = await asyncio.wait_for(
                    self._health_check_fn(name, mod.socket_path),
                    timeout=HEALTH_CHECK_TIMEOUT_SEC,
                )
                if not healthy:
                    logger.warning("Module %s health check returned unhealthy", name)
            except (asyncio.TimeoutError, OSError) as exc:
                logger.warning("Module %s health check failed: %s", name, exc)
