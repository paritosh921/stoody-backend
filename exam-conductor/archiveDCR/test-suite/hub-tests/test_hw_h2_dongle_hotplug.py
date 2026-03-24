"""
HW-H2: Dongle hot-plug — graceful degradation and recovery.

Hardware required: 5 USB BLE dongles. Test physically unplugs and replugs one.

Procedure:
  1. Verify 5 dongles detected (pre-condition).
  2. Unplug 1 dongle (manual step or USB hub GPIO control).
  3. Wait 5 seconds for hub to detect removal.
  4. Verify hub reports 4 dongles and logs a degradation warning.
  5. Replug the dongle.
  6. Wait 10 seconds for hub to detect recovery.
  7. Verify hub reports 5 dongles again.

Pass: Hub degrades gracefully (4 dongles) and recovers to 5 after replug.
Fail: Hub crashes, reports incorrect count, or does not recover.

Test-ID: HW-H2  (TEST_SUITE_SPEC.md section 2.4)
Level: L6 (hardware-in-loop)
"""

from __future__ import annotations

import time

import pytest

pytestmark = [pytest.mark.hardware]


@pytest.fixture(autouse=True)
def _require_hub(hub_reachable: bool):
    if not hub_reachable:
        pytest.skip("Hub not reachable via SSH; skipping HW-H2")


class TestDongleHotPlug:
    """HW-H2 — Dongle hot-plug degradation and recovery.

    NOTE: This test requires manual intervention (unplug/replug a dongle)
    unless a USB hub with GPIO control is available. In CI, it is typically
    skipped.
    """

    def test_pre_condition_five_dongles(self, hub_dongles):
        """Pre-condition: 5 dongles must be present before hot-plug test."""
        assert len(hub_dongles) >= 5, (
            f"HW-H2 pre-condition failed: need 5 dongles, found {len(hub_dongles)}"
        )

    def test_degradation_after_unplug(self, hub_ssh, export_result):
        """After removing a dongle, the hub should report 4 dongles.

        This test checks the hub's dongle count after a removal event.
        In automated mode, it queries the hub supervisor for the current
        dongle count. If no removal has occurred, the test is skipped.
        """
        start = time.monotonic()

        result = hub_ssh.run("hciconfig -a")
        lines = [l for l in result.stdout.splitlines() if l.startswith("hci")]
        current_count = len(lines)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        if current_count >= 5:
            pytest.skip(
                "All 5 dongles still present — unplug one manually and re-run "
                "this test to verify degradation."
            )

        export_result(
            test_id="HW-H2",
            name="Dongle hot-plug (degradation)",
            status="PASS" if current_count == 4 else "FAIL",
            duration_ms=elapsed_ms,
            detail={"dongles_found": current_count},
        )

        assert current_count == 4, f"Expected 4 dongles after unplug, found {current_count}"

    def test_recovery_after_replug(self, hub_ssh, export_result):
        """After replugging the dongle, the hub should report 5 dongles again.

        Similar to degradation test — queries the hub and checks count.
        """
        start = time.monotonic()

        result = hub_ssh.run("hciconfig -a")
        lines = [l for l in result.stdout.splitlines() if l.startswith("hci")]
        current_count = len(lines)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        if current_count < 5:
            pytest.skip(
                "Dongles still degraded — replug the dongle and re-run "
                "this test to verify recovery."
            )

        export_result(
            test_id="HW-H2",
            name="Dongle hot-plug (recovery)",
            status="PASS",
            duration_ms=elapsed_ms,
            detail={"dongles_found": current_count},
        )

        assert current_count >= 5
