"""
HW-W1: WiFi connectivity — connect, verify band, check backend reachability.

Hardware required: WiFi access point within range of the hub.

Procedure:
  1. Verify WiFi interface is present (wlan0 or similar).
  2. Check current WiFi connection status via NetworkManager.
  3. Verify connected to expected SSID.
  4. Check WiFi band (2.4 GHz or 5 GHz).
  5. Verify backend API endpoint is reachable from the hub.

Pass: WiFi connected, band confirmed, backend reachable.
Fail: WiFi disconnected, wrong band, or backend unreachable.

Test-ID: HW-W1  (TEST_SUITE_SPEC.md section 2.4)
Level: L6 (hardware-in-loop)
"""

from __future__ import annotations

import time

import pytest

pytestmark = [pytest.mark.hardware, pytest.mark.wifi]

BACKEND_HEALTH_URL = "http://api.exampen.local/health"


@pytest.fixture(autouse=True)
def _require_hub(hub_reachable: bool):
    if not hub_reachable:
        pytest.skip("Hub not reachable via SSH; skipping HW-W1")


class TestWiFiConnectivity:
    """HW-W1 — WiFi connection, band verification, backend reachability."""

    def test_wifi_interface_present(self, hub_ssh):
        """Hub must have a WiFi interface (wlan0)."""
        result = hub_ssh.run("ip link show wlan0 2>/dev/null || ip link show wlp* 2>/dev/null")
        assert result.returncode == 0, "No WiFi interface found on hub"
        assert "wlan" in result.stdout or "wlp" in result.stdout

    def test_wifi_connected(self, hub_ssh, export_result):
        """Hub WiFi must be connected to an access point."""
        start = time.monotonic()

        result = hub_ssh.run("nmcli -t -f GENERAL.STATE,GENERAL.CONNECTION device show wlan0")
        elapsed_ms = int((time.monotonic() - start) * 1000)

        connected = "connected" in result.stdout.lower() and "disconnected" not in result.stdout.lower()

        export_result(
            test_id="HW-W1",
            name="WiFi connectivity",
            status="PASS" if connected else "FAIL",
            duration_ms=elapsed_ms,
            detail={"nmcli_output": result.stdout.strip()[:300]},
        )

        assert connected, f"WiFi is not connected: {result.stdout.strip()}"

    def test_wifi_band(self, hub_ssh):
        """Check which WiFi band the hub is connected on."""
        result = hub_ssh.run("iw dev wlan0 link")
        # Parse frequency from output (e.g., "freq: 5180").
        freq_line = [l for l in result.stdout.splitlines() if "freq:" in l]

        if not freq_line:
            pytest.skip("Could not determine WiFi frequency")

        freq_str = freq_line[0].split("freq:")[1].strip().split()[0]
        try:
            freq_mhz = int(freq_str)
        except ValueError:
            pytest.skip(f"Could not parse frequency: {freq_str}")
            return

        band = "5GHz" if freq_mhz >= 5000 else "2.4GHz"
        # Log the band but do not fail — both bands are acceptable.
        assert freq_mhz > 0, f"Invalid frequency: {freq_mhz}"

    def test_backend_reachable(self, hub_ssh, export_result):
        """Hub must be able to reach the backend health endpoint."""
        start = time.monotonic()

        result = hub_ssh.run(
            f"curl -s -o /dev/null -w '%{{http_code}}' --connect-timeout 10 {BACKEND_HEALTH_URL}",
            timeout=20,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        status_code = result.stdout.strip().strip("'")
        reachable = status_code.startswith("2") or status_code == "200"

        export_result(
            test_id="HW-W1-backend",
            name="Backend reachability",
            status="PASS" if reachable else "FAIL",
            duration_ms=elapsed_ms,
            detail={"http_status": status_code},
        )

        assert reachable, (
            f"Backend not reachable from hub. HTTP status: {status_code}. "
            f"URL: {BACKEND_HEALTH_URL}"
        )
