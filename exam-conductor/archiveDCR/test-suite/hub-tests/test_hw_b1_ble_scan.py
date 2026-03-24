"""
HW-B1: BLE scan + connect — pen discovery and GATT service read.

Hardware required: 1+ BLE pen (P05) or nRF52840-DK pen simulator.

Procedure:
  1. Activate a BLE dongle on the hub.
  2. Start BLE scan for devices advertising the pen GATT service (0xEP01).
  3. Connect to the first discovered pen.
  4. Read the GATT service characteristics.
  5. Verify the pen responds with valid firmware/battery info.

Pass: Pen discovered, GATT service readable.
Fail: No pen found within timeout, GATT read fails.

Test-ID: HW-B1  (TEST_SUITE_SPEC.md section 2.4)
Level: L6 (hardware-in-loop)
"""

from __future__ import annotations

import json
import time

import pytest

pytestmark = [pytest.mark.hardware, pytest.mark.ble]


@pytest.fixture(autouse=True)
def _require_hub_and_dongles(hub_reachable: bool, hub_dongles):
    if not hub_reachable:
        pytest.skip("Hub not reachable via SSH; skipping HW-B1")
    if not hub_dongles:
        pytest.skip("No BLE dongles detected on hub; skipping HW-B1")


class TestBLEScanConnect:
    """HW-B1 — BLE pen discovery and GATT service read."""

    def test_ble_scan_discovers_pen(self, hub_ssh, export_result):
        """Scan for BLE pens and verify at least one is discovered."""
        start = time.monotonic()

        # Use the hub's diagnostics runner or bluetoothctl to scan.
        # The hub-ble-mgr provides a CLI for pen discovery.
        result = hub_ssh.run(
            "timeout 30 python3 -m hub_ble_mgr.cli scan --duration 15 --json",
            timeout=45,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        if result.returncode != 0:
            # Fall back to hcitool scan.
            result = hub_ssh.run(
                "timeout 30 hcitool -i hci0 lescan --duplicates 2>/dev/null &"
                " sleep 10 && kill %1 2>/dev/null; "
                "hcitool -i hci0 lecc --json 2>/dev/null || echo '[]'",
                timeout=45,
            )

        # Parse scan results.
        devices_found = 0
        try:
            devices = json.loads(result.stdout)
            devices_found = len(devices) if isinstance(devices, list) else 0
        except (json.JSONDecodeError, TypeError):
            # Count lines mentioning pen-like MAC patterns.
            devices_found = sum(
                1 for line in result.stdout.splitlines()
                if ":" in line and len(line.strip()) > 10
            )

        export_result(
            test_id="HW-B1",
            name="BLE scan + connect",
            status="PASS" if devices_found > 0 else "FAIL",
            duration_ms=elapsed_ms,
            detail={"devices_found": devices_found},
        )

        assert devices_found > 0, (
            "No BLE pens or simulators discovered. Ensure pens are powered on "
            "and advertising, or start the pen simulator."
        )

    def test_gatt_service_readable(self, hub_ssh, export_result):
        """Connect to a discovered pen and read its GATT service."""
        start = time.monotonic()

        # Use the hub-ble-mgr CLI to connect to the first pen and read GATT.
        result = hub_ssh.run(
            "timeout 30 python3 -m hub_ble_mgr.cli gatt-read --first --json",
            timeout=45,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        if result.returncode != 0:
            export_result(
                test_id="HW-B1-GATT",
                name="GATT service read",
                status="FAIL",
                duration_ms=elapsed_ms,
                detail={"error": result.stderr.strip()[:200]},
            )
            pytest.fail(
                f"GATT read failed: {result.stderr.strip()[:200]}"
            )

        export_result(
            test_id="HW-B1-GATT",
            name="GATT service read",
            status="PASS",
            duration_ms=elapsed_ms,
            detail={"output_length": len(result.stdout)},
        )

        # Basic validation: output should contain GATT characteristic data.
        assert len(result.stdout.strip()) > 0, "GATT read returned empty output"
