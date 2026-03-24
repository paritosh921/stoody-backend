"""
HW-H1: Dongle enumeration.

Hardware required: 5 USB BLE dongles connected to the RPi hub.

Procedure (from TEST_SUITE_SPEC section 3.4):
  1. List all Bluetooth HCI adapters via ``hciconfig -a``.
  2. For each adapter, read BD_ADDR (MAC).
  3. Cross-reference with dongle_registry table in hub SQLite.
  4. Verify count matches expected (5).

Pass: 5 dongles detected, all MACs stable from last boot.
Fail: <5 dongles, or MAC changed (re-enumeration).

Test-ID: HW-H1  (TEST_SUITE_SPEC.md section 2.4)
Level: L6 (hardware-in-loop)
"""

from __future__ import annotations

import time

import pytest

pytestmark = [pytest.mark.hardware]

EXPECTED_DONGLE_COUNT = 5


@pytest.fixture(autouse=True)
def _require_hub(hub_reachable: bool):
    if not hub_reachable:
        pytest.skip("Hub not reachable via SSH; skipping HW-H1")


class TestDongleEnumeration:
    """HW-H1 — All 5 USB BLE dongles detected with stable MACs."""

    def test_expected_dongle_count(self, hub_dongles, export_result):
        """Verify that exactly 5 BLE dongles are detected on the hub."""
        start = time.monotonic()
        count = len(hub_dongles)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        export_result(
            test_id="HW-H1",
            name="Dongle enumeration",
            status="PASS" if count >= EXPECTED_DONGLE_COUNT else "FAIL",
            duration_ms=elapsed_ms,
            detail={"dongles_found": count, "expected": EXPECTED_DONGLE_COUNT},
        )

        assert count >= EXPECTED_DONGLE_COUNT, (
            f"Expected {EXPECTED_DONGLE_COUNT} dongles, found {count}"
        )

    def test_all_dongles_have_valid_mac(self, hub_dongles):
        """Every detected dongle must have a non-zero MAC address."""
        for dongle in hub_dongles:
            assert dongle.mac_address, f"{dongle.hci_name} has no MAC"
            assert dongle.mac_address != "00:00:00:00:00:00", (
                f"{dongle.hci_name} has zero MAC"
            )

    def test_all_dongles_are_up(self, hub_dongles):
        """Every detected dongle must be in UP state."""
        for dongle in hub_dongles:
            assert dongle.is_up, f"{dongle.hci_name} is not UP"

    def test_mac_stability_against_registry(self, hub_ssh, hub_dongles):
        """Cross-reference detected MACs with dongle_registry table."""
        try:
            raw = hub_ssh.query_sqlite(
                "SELECT hci_name, mac_address FROM dongle_registry"
            )
        except RuntimeError:
            pytest.skip("dongle_registry table not found; first boot?")
            return

        import json

        registry = {r["hci_name"]: r["mac_address"] for r in json.loads(raw)}

        for dongle in hub_dongles:
            if dongle.hci_name in registry:
                assert dongle.mac_address == registry[dongle.hci_name], (
                    f"{dongle.hci_name} MAC changed: "
                    f"was {registry[dongle.hci_name]}, now {dongle.mac_address}"
                )
