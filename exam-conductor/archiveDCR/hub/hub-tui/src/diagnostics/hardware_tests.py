"""Hardware diagnostic tests H1-H7.

Each test is an async function returning (TestStatus, detail_dict).
Subprocess calls are used for system commands so they can be mocked in tests.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from src.diagnostics.runner import TestCase, TestCategory, TestStatus

EXPECTED_DONGLE_COUNT = 5
USB_BACKUP_MOUNT = Path("/mnt/exampen-backup")
SD_DATA_DIR = Path("/var/lib/exampen")
MIN_FREE_MB = 500  # Minimum free space on SD partition


async def _run_cmd(cmd: str) -> tuple[int, str, str]:
    """Run a shell command, return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout.decode(errors="replace").strip(),
        stderr.decode(errors="replace").strip(),
    )


# ── H1: Dongle enumeration ──────────────────────────────────────────────


async def h1_dongle_enumeration() -> tuple[TestStatus, dict[str, Any]]:
    """Enumerate BLE dongles via hciconfig -a, verify 5 with stable MACs."""
    rc, stdout, stderr = await _run_cmd("hciconfig -a")
    if rc != 0:
        return TestStatus.FAIL, {"error": f"hciconfig failed: {stderr}"}

    # Parse adapter blocks — each starts with "hciN:"
    adapters: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in stdout.splitlines():
        hci_match = re.match(r"^(hci\d+):", line)
        if hci_match:
            if current is not None:
                adapters.append(current)
            current = {"hci": hci_match.group(1), "mac": ""}
        if current and "BD Address:" in line:
            mac_match = re.search(r"BD Address:\s+([0-9A-Fa-f:]{17})", line)
            if mac_match:
                current["mac"] = mac_match.group(1).upper()
    if current is not None:
        adapters.append(current)

    found = len(adapters)
    macs = [a["mac"] for a in adapters]

    if found < EXPECTED_DONGLE_COUNT:
        return TestStatus.FAIL, {
            "dongles_found": found,
            "expected": EXPECTED_DONGLE_COUNT,
            "macs": macs,
            "error": f"Expected {EXPECTED_DONGLE_COUNT} dongles, found {found}",
        }

    # Check for duplicates (unstable MACs)
    if len(set(macs)) != len(macs):
        return TestStatus.FAIL, {
            "dongles_found": found,
            "macs": macs,
            "error": "Duplicate MACs detected — unstable enumeration",
        }

    return TestStatus.PASS, {"dongles_found": found, "macs": macs}


# ── H2: Dongle hot-plug (manual) ────────────────────────────────────────


async def h2_dongle_hotplug() -> tuple[TestStatus, dict[str, Any]]:
    """Manual test — requires physical dongle unplug/replug."""
    return TestStatus.SKIP, {"reason": "manual test — requires physical action"}


# ── H3: USB storage mount ───────────────────────────────────────────────


async def h3_usb_storage_mount() -> tuple[TestStatus, dict[str, Any]]:
    """Check /mnt/exampen-backup is mounted and writable."""
    if not USB_BACKUP_MOUNT.is_dir():
        return TestStatus.FAIL, {
            "error": f"{USB_BACKUP_MOUNT} does not exist",
        }

    # Check it is actually a mountpoint
    rc, stdout, _ = await _run_cmd(f"mountpoint -q {USB_BACKUP_MOUNT}; echo $?")
    # mountpoint -q exits 0 if it is a mountpoint
    rc2, _, _ = await _run_cmd(f"mountpoint -q {USB_BACKUP_MOUNT}")
    if rc2 != 0:
        return TestStatus.FAIL, {
            "error": f"{USB_BACKUP_MOUNT} is not a mount point",
        }

    # Writable test
    test_file = USB_BACKUP_MOUNT / ".diag_write_test"
    try:
        test_file.write_text("diag", encoding="utf-8")
        test_file.unlink()
    except OSError as exc:
        return TestStatus.FAIL, {
            "error": f"Write test failed: {exc}",
        }

    return TestStatus.PASS, {"mount": str(USB_BACKUP_MOUNT), "writable": True}


# ── H4: SD card health ──────────────────────────────────────────────────


async def h4_sd_card_health() -> tuple[TestStatus, dict[str, Any]]:
    """Check /var/lib/exampen free space and write test."""
    if not SD_DATA_DIR.is_dir():
        return TestStatus.FAIL, {"error": f"{SD_DATA_DIR} does not exist"}

    # Free space via os.statvfs
    try:
        stat = os.statvfs(str(SD_DATA_DIR))
        free_mb = (stat.f_bavail * stat.f_frsize) // (1024 * 1024)
    except OSError as exc:
        return TestStatus.FAIL, {"error": f"statvfs failed: {exc}"}

    # Write test
    test_file = SD_DATA_DIR / ".diag_write_test"
    try:
        test_file.write_text("diag", encoding="utf-8")
        test_file.unlink()
    except OSError as exc:
        return TestStatus.FAIL, {
            "free_mb": free_mb,
            "error": f"Write test failed: {exc}",
        }

    if free_mb < MIN_FREE_MB:
        return TestStatus.FAIL, {
            "free_mb": free_mb,
            "min_required_mb": MIN_FREE_MB,
            "error": f"Low disk space: {free_mb} MB < {MIN_FREE_MB} MB",
        }

    return TestStatus.PASS, {"free_mb": free_mb, "writable": True}


# ── H5: NTP sync status ─────────────────────────────────────────────────


async def h5_ntp_sync() -> tuple[TestStatus, dict[str, Any]]:
    """Check chronyc tracking for leap status Normal."""
    rc, stdout, stderr = await _run_cmd("chronyc tracking")
    if rc != 0:
        return TestStatus.FAIL, {"error": f"chronyc failed: {stderr}"}

    leap_match = re.search(r"Leap status\s*:\s*(.+)", stdout)
    if not leap_match:
        return TestStatus.FAIL, {
            "error": "Could not parse leap status from chronyc output",
        }

    leap_status = leap_match.group(1).strip()
    if leap_status.lower() != "normal":
        return TestStatus.FAIL, {
            "leap_status": leap_status,
            "error": f"NTP not synced — leap status: {leap_status}",
        }

    # Extract system time offset if available
    offset_match = re.search(r"System time\s*:\s*(.+)", stdout)
    offset = offset_match.group(1).strip() if offset_match else "unknown"

    return TestStatus.PASS, {"leap_status": leap_status, "system_time_offset": offset}


# ── H6: WiFi connectivity ───────────────────────────────────────────────


async def h6_wifi_connectivity() -> tuple[TestStatus, dict[str, Any]]:
    """Check WiFi is associated via nmcli."""
    rc, stdout, stderr = await _run_cmd(
        "nmcli -t -f GENERAL.STATE,GENERAL.CONNECTION device show wlan0"
    )
    if rc != 0:
        return TestStatus.FAIL, {"error": f"nmcli failed: {stderr}"}

    connected = False
    connection_name = ""
    for line in stdout.splitlines():
        if "GENERAL.STATE" in line and "100" in line:
            connected = True
        if "GENERAL.CONNECTION" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                connection_name = parts[1].strip()

    if not connected:
        return TestStatus.FAIL, {
            "error": "WiFi not associated",
            "connection": connection_name,
        }

    return TestStatus.PASS, {"connected": True, "connection": connection_name}


# ── H7: WiFi band check ─────────────────────────────────────────────────


async def h7_wifi_band() -> tuple[TestStatus, dict[str, Any]]:
    """Check WiFi band via iw dev wlan0 info, prefer 5 GHz."""
    rc, stdout, stderr = await _run_cmd("iw dev wlan0 info")
    if rc != 0:
        return TestStatus.FAIL, {"error": f"iw failed: {stderr}"}

    freq_match = re.search(r"channel\s+\d+\s+\((\d+)\s+MHz\)", stdout)
    if not freq_match:
        return TestStatus.FAIL, {"error": "Could not parse frequency from iw output"}

    freq_mhz = int(freq_match.group(1))

    # 5 GHz band: 5180-5825 MHz
    is_5ghz = 5180 <= freq_mhz <= 5825
    band = "5GHz" if is_5ghz else "2.4GHz" if freq_mhz < 3000 else "unknown"

    detail: dict[str, Any] = {"frequency_mhz": freq_mhz, "band": band}

    if not is_5ghz:
        detail["warning"] = "Not on 5 GHz — consider switching for less BLE interference"
        # Still PASS, but with warning — 2.4 GHz is functional
        return TestStatus.PASS, detail

    return TestStatus.PASS, detail


# ── Registry ─────────────────────────────────────────────────────────────


def build_hardware_tests() -> list[TestCase]:
    """Return the full set of hardware test cases H1-H7."""
    return [
        TestCase(
            id="H1",
            name="Dongle enumeration",
            category=TestCategory.HARDWARE,
            run_fn=h1_dongle_enumeration,
        ),
        TestCase(
            id="H2",
            name="Dongle hot-plug",
            category=TestCategory.HARDWARE,
            run_fn=h2_dongle_hotplug,
            manual=True,
        ),
        TestCase(
            id="H3",
            name="USB storage mount",
            category=TestCategory.HARDWARE,
            run_fn=h3_usb_storage_mount,
        ),
        TestCase(
            id="H4",
            name="SD card health",
            category=TestCategory.HARDWARE,
            run_fn=h4_sd_card_health,
        ),
        TestCase(
            id="H5",
            name="NTP sync status",
            category=TestCategory.HARDWARE,
            run_fn=h5_ntp_sync,
        ),
        TestCase(
            id="H6",
            name="WiFi connectivity",
            category=TestCategory.HARDWARE,
            run_fn=h6_wifi_connectivity,
        ),
        TestCase(
            id="H7",
            name="WiFi band check",
            category=TestCategory.HARDWARE,
            run_fn=h7_wifi_band,
        ),
    ]
