"""
Shared fixtures for ExamPen hub hardware-in-loop tests (L6).

Provides SSH connection to the hub RPi, BLE dongle detection, pen simulator
setup, and hub database access.

Usage:
    pytest test-suite/hub-tests/ -m hardware -v

Task:   W6.A4 (hub hardware tests)
Spec:   TEST_SUITE_SPEC.md section 2.4
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Environment defaults
# ---------------------------------------------------------------------------

HUB_SSH_HOST = os.getenv("HUB_SSH_HOST", "exampen-hub.local")
HUB_SSH_USER = os.getenv("HUB_SSH_USER", "exampen")
HUB_SSH_KEY = os.getenv("HUB_SSH_KEY", str(Path.home() / ".ssh" / "id_rsa"))
HUB_DB_PATH = os.getenv("HUB_DB_PATH", "/var/lib/exampen/hub.db")
PEN_SIM_ADAPTER = os.getenv("PEN_SIM_ADAPTER", "hci1")
PEN_SIM_COUNT = int(os.getenv("PEN_SIM_COUNT", "8"))

# Results output directory.
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# pytest markers
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "hardware: Requires physical hub hardware (L6)")
    config.addinivalue_line("markers", "ble: Requires BLE dongles and/or pen simulators")
    config.addinivalue_line("markers", "power: Requires switchable power supply")
    config.addinivalue_line("markers", "wifi: Requires WiFi access point")


# ---------------------------------------------------------------------------
# SSH helper
# ---------------------------------------------------------------------------


@dataclass
class HubSSH:
    """Run commands on the hub via SSH."""

    host: str
    user: str
    key_path: str

    def run(self, cmd: str, *, timeout: int = 30) -> subprocess.CompletedProcess:
        """Execute *cmd* on the hub and return the result."""
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-i", self.key_path,
            f"{self.user}@{self.host}",
            cmd,
        ]
        return subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def is_reachable(self) -> bool:
        """Return True if the hub responds to SSH."""
        try:
            result = self.run("echo ok", timeout=15)
            return result.returncode == 0 and "ok" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def read_file(self, remote_path: str) -> str:
        """Read a file from the hub via SSH."""
        result = self.run(f"cat {remote_path}")
        if result.returncode != 0:
            raise FileNotFoundError(
                f"Cannot read {remote_path}: {result.stderr.strip()}"
            )
        return result.stdout

    def query_sqlite(self, query: str, db_path: str | None = None) -> str:
        """Run a SQLite query on the hub and return the output."""
        db = db_path or HUB_DB_PATH
        result = self.run(f'sqlite3 -json "{db}" "{query}"')
        if result.returncode != 0:
            raise RuntimeError(f"SQLite query failed: {result.stderr.strip()}")
        return result.stdout


@pytest.fixture(scope="session")
def hub_ssh() -> HubSSH:
    """Session-scoped SSH connection to the hub."""
    return HubSSH(host=HUB_SSH_HOST, user=HUB_SSH_USER, key_path=HUB_SSH_KEY)


@pytest.fixture(scope="session")
def hub_reachable(hub_ssh: HubSSH) -> bool:
    """Check if the hub is reachable via SSH. Used for skip conditions."""
    return hub_ssh.is_reachable()


# ---------------------------------------------------------------------------
# BLE dongle detection
# ---------------------------------------------------------------------------


@dataclass
class BLEDongleInfo:
    """Information about a detected BLE dongle on the hub."""

    hci_name: str
    mac_address: str
    is_up: bool


@pytest.fixture(scope="session")
def hub_dongles(hub_ssh: HubSSH, hub_reachable: bool) -> list[BLEDongleInfo]:
    """Detect BLE dongles connected to the hub."""
    if not hub_reachable:
        return []

    result = hub_ssh.run("hciconfig -a")
    if result.returncode != 0:
        return []

    dongles: list[BLEDongleInfo] = []
    current_hci: str | None = None
    current_mac: str | None = None
    current_up = False

    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("hci"):
            # Save previous dongle if any.
            if current_hci and current_mac:
                dongles.append(
                    BLEDongleInfo(
                        hci_name=current_hci,
                        mac_address=current_mac,
                        is_up=current_up,
                    )
                )
            parts = stripped.split(":")
            current_hci = parts[0] if parts else None
            current_mac = None
            current_up = "UP" in stripped
        elif "BD Address:" in stripped:
            current_mac = stripped.split("BD Address:")[1].strip().split()[0]
        elif "UP" in stripped:
            current_up = True

    # Don't forget the last dongle.
    if current_hci and current_mac:
        dongles.append(
            BLEDongleInfo(
                hci_name=current_hci,
                mac_address=current_mac,
                is_up=current_up,
            )
        )

    return dongles


# ---------------------------------------------------------------------------
# Pen simulator
# ---------------------------------------------------------------------------


@dataclass
class PenSimulator:
    """Control a BLE pen simulator (software-based)."""

    adapter: str
    pen_count: int
    _process: subprocess.Popen | None = None

    def start(self) -> None:
        """Start the pen simulator in the background."""
        sim_script = (
            Path(__file__).resolve().parent / "ble_pen_sim.py"
        )
        if not sim_script.exists():
            pytest.skip("ble_pen_sim.py not found; pen simulator unavailable")

        self._process = subprocess.Popen(
            [
                "python3",
                str(sim_script),
                "--pens", str(self.pen_count),
                "--adapter", self.adapter,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def stop(self) -> None:
        """Stop the pen simulator."""
        if self._process:
            self._process.terminate()
            self._process.wait(timeout=10)
            self._process = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None


@pytest.fixture(scope="session")
def pen_simulator() -> PenSimulator:
    """Session-scoped pen simulator (not started by default)."""
    sim = PenSimulator(adapter=PEN_SIM_ADAPTER, pen_count=PEN_SIM_COUNT)
    yield sim
    sim.stop()


# ---------------------------------------------------------------------------
# Results export
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def ensure_results_dir():
    """Create the results output directory if it does not exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def export_result():
    """Callable fixture to export a single test result as JSON."""

    def _export(test_id: str, name: str, status: str, duration_ms: int,
                detail: dict[str, Any] | None = None) -> None:
        result = {
            "id": test_id,
            "name": name,
            "status": status,
            "duration_ms": duration_ms,
            "detail": detail or {},
        }
        out_file = RESULTS_DIR / f"{test_id}.json"
        out_file.write_text(json.dumps(result, indent=2))

    return _export
