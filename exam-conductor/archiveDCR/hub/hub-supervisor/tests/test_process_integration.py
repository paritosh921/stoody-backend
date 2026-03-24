"""L4 integration tests — process manager spawning real child processes.

Tests verify that hub-supervisor can spawn a real child process (a
simple echo Python script), detect crashes, restart children, and
perform graceful shutdown.  Uses TCP loopback (Windows compatible).

Test IDs: I-SUP-PROC-01 through I-SUP-PROC-03.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from hub_common.ipc_protocol import IpcClient, IpcEnvelope, IpcServer
from hub_common.message_types import SUPERVISOR_HEALTH_REQUEST

from src.process_manager import ModuleInfo, ModuleStatus, ProcessManager

# ===================================================================
# Fixtures — create temporary child scripts
# ===================================================================

_ECHO_CHILD_SCRIPT = textwrap.dedent("""\
    \"\"\"Minimal child process that listens on a TCP port and echoes IPC.\"\"\"
    import asyncio
    import json
    import sys

    PORT = int(sys.argv[1])

    async def handle(reader, writer):
        while True:
            line = await reader.readline()
            if not line:
                break
            data = json.loads(line)
            reply = dict(data)
            reply["msg_type"] = data["msg_type"].replace(".request", ".result")
            reply["correlation_id"] = data["msg_id"]
            reply["source"], reply["target"] = reply["target"], reply["source"]
            writer.write(json.dumps(reply).encode() + b"\\n")
            await writer.drain()
        writer.close()

    async def main():
        server = await asyncio.start_server(handle, "127.0.0.1", PORT)
        # Signal readiness by printing port
        print(f"READY:{PORT}", flush=True)
        await server.serve_forever()

    asyncio.run(main())
""")

_CRASH_CHILD_SCRIPT = textwrap.dedent("""\
    \"\"\"Child process that exits immediately with code 1.\"\"\"
    import sys
    sys.exit(1)
""")

_GRACEFUL_CHILD_SCRIPT = textwrap.dedent("""\
    \"\"\"Child that runs until terminated, exits 0 on SIGTERM / KeyboardInterrupt.\"\"\"
    import asyncio
    import signal
    import sys

    async def main():
        stop = asyncio.Event()
        if sys.platform != "win32":
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGTERM, stop.set)
        try:
            await stop.wait()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass

    asyncio.run(main())
""")


@pytest.fixture()
def echo_script(tmp_path: Path) -> Path:
    p = tmp_path / "echo_child.py"
    p.write_text(_ECHO_CHILD_SCRIPT, encoding="utf-8")
    return p


@pytest.fixture()
def crash_script(tmp_path: Path) -> Path:
    p = tmp_path / "crash_child.py"
    p.write_text(_CRASH_CHILD_SCRIPT, encoding="utf-8")
    return p


@pytest.fixture()
def graceful_script(tmp_path: Path) -> Path:
    p = tmp_path / "graceful_child.py"
    p.write_text(_GRACEFUL_CHILD_SCRIPT, encoding="utf-8")
    return p


def _free_port() -> int:
    """Find a free TCP port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ===================================================================
# I-SUP-PROC-01: Spawn real child, communicate via IPC
# ===================================================================

class TestSpawnAndCommunicate:
    async def test_spawn_echo_child_ipc(self, echo_script: Path) -> None:
        """I-SUP-PROC-01: Supervisor spawns a child and communicates over TCP."""
        port = _free_port()
        socket_addr = f"127.0.0.1:{port}"
        mod = ModuleInfo(
            name="echo-child",
            socket_path=socket_addr,
            command=[sys.executable, str(echo_script), str(port)],
        )
        pm = ProcessManager([mod], max_restarts=0)
        await pm.spawn_module("echo-child")
        assert pm.modules["echo-child"].status == ModuleStatus.RUNNING
        assert pm.modules["echo-child"].pid is not None

        # Give child time to start TCP server
        await asyncio.sleep(0.5)

        # Send an IPC message and verify it was received
        client = IpcClient(socket_addr, source_id="test")
        await client.connect()
        try:
            req = IpcEnvelope(
                msg_type=SUPERVISOR_HEALTH_REQUEST,
                source="test",
                target="echo-child",
                expects_reply=True,
                payload={"check": "alive"},
            )
            reply = await client.request(req, timeout=5.0)
            assert reply.correlation_id == req.msg_id
            assert reply.payload.get("check") == "alive"
        finally:
            await client.close()
            await pm.stop_all()


# ===================================================================
# I-SUP-PROC-02: Crash detection and restart
# ===================================================================

class TestCrashAndRestart:
    async def test_crash_detected_and_restarted(
        self, crash_script: Path,
    ) -> None:
        """I-SUP-PROC-02: ProcessManager detects child crash, restarts it."""
        status_changes: list[tuple[str, ModuleStatus]] = []

        async def on_change(name: str, status: ModuleStatus) -> None:
            status_changes.append((name, status))

        socket_addr = f"127.0.0.1:{_free_port()}"
        mod = ModuleInfo(
            name="crash-child",
            socket_path=socket_addr,
            command=[sys.executable, str(crash_script)],
        )
        pm = ProcessManager(
            [mod], max_restarts=2, on_status_change=on_change,
        )
        await pm.spawn_module("crash-child")

        # Wait for the child to crash and be restarted
        await asyncio.sleep(2.0)

        # Check status changes: should see RUNNING, CRASHED, RUNNING, ...
        running_count = sum(
            1 for _, s in status_changes if s == ModuleStatus.RUNNING
        )
        crashed_count = sum(
            1 for _, s in status_changes if s == ModuleStatus.CRASHED
        )
        assert crashed_count >= 1, f"Expected at least 1 crash, got {status_changes}"
        assert running_count >= 2, f"Expected at least 2 starts, got {status_changes}"
        await pm.stop_all()

    async def test_max_restarts_marks_failed(
        self, crash_script: Path,
    ) -> None:
        """I-SUP-PROC-02b: Exceeding max restarts marks module FAILED."""
        status_changes: list[tuple[str, ModuleStatus]] = []

        async def on_change(name: str, status: ModuleStatus) -> None:
            status_changes.append((name, status))

        socket_addr = f"127.0.0.1:{_free_port()}"
        mod = ModuleInfo(
            name="fail-child",
            socket_path=socket_addr,
            command=[sys.executable, str(crash_script)],
        )
        pm = ProcessManager(
            [mod], max_restarts=1, on_status_change=on_change,
        )
        await pm.spawn_module("fail-child")

        # Wait for crash cycles to exhaust
        await asyncio.sleep(3.0)

        final_status = pm.modules["fail-child"].status
        assert final_status == ModuleStatus.FAILED
        assert any(s == ModuleStatus.FAILED for _, s in status_changes)
        await pm.stop_all()


# ===================================================================
# I-SUP-PROC-03: Graceful shutdown sequence
# ===================================================================

class TestGracefulShutdown:
    async def test_stop_all_terminates_children(
        self, graceful_script: Path,
    ) -> None:
        """I-SUP-PROC-03: stop_all sends SIGTERM and waits for exit."""
        modules = []
        for i in range(2):
            port = _free_port()
            modules.append(ModuleInfo(
                name=f"child-{i}",
                socket_path=f"127.0.0.1:{port}",
                command=[sys.executable, str(graceful_script)],
            ))
        pm = ProcessManager(modules, max_restarts=0)
        for mod in modules:
            await pm.spawn_module(mod.name)
        # Verify all running
        for mod in modules:
            assert pm.modules[mod.name].status == ModuleStatus.RUNNING
        await asyncio.sleep(0.2)

        # Graceful shutdown
        await pm.stop_all()
        for mod in modules:
            assert pm.modules[mod.name].status == ModuleStatus.STOPPED
            assert pm.modules[mod.name].process is None

    async def test_status_summary_after_shutdown(
        self, graceful_script: Path,
    ) -> None:
        """I-SUP-PROC-03b: get_status_summary reflects stopped state."""
        port = _free_port()
        mod = ModuleInfo(
            name="child-sum",
            socket_path=f"127.0.0.1:{port}",
            command=[sys.executable, str(graceful_script)],
        )
        pm = ProcessManager([mod], max_restarts=0)
        await pm.spawn_module("child-sum")
        summary = pm.get_status_summary()
        assert summary["child-sum"] == "running"

        await pm.stop_all()
        summary = pm.get_status_summary()
        assert summary["child-sum"] == "stopped"
