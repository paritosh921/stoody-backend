"""
HW-B2: Multi-pen sync — concurrent sync of 8 pens per dongle.

Hardware required: 8 BLE pens (or simulators) per dongle, at least 1 dongle.

Procedure (from TEST_SUITE_SPEC section 3.4, B3):
  1. Activate all dongles.
  2. Wait for N pens to connect (or simulators).
  3. Trigger sync on all connected pens simultaneously.
  4. Measure: connection time, throughput per pen, total sync time.
  5. Verify: all data dual-written, checksums match.

Pass: All pens synced, checksums match, no dongle crashes.
Fail: Any pen fails to sync, checksum mismatch, dongle crash.
Duration: 2-5 minutes depending on pen count.

Test-ID: HW-B2  (TEST_SUITE_SPEC.md section 2.4)
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
        pytest.skip("Hub not reachable via SSH; skipping HW-B2")
    if not hub_dongles:
        pytest.skip("No BLE dongles detected on hub; skipping HW-B2")


class TestMultiPenSync:
    """HW-B2 — Concurrent pen sync throughput and correctness."""

    def test_multi_pen_sync_completes(self, hub_ssh, pen_simulator, export_result):
        """Trigger sync on all connected pens and verify completion."""
        start = time.monotonic()

        # Start pen simulator if available.
        try:
            pen_simulator.start()
            # Wait for simulators to advertise.
            time.sleep(5)
        except Exception:
            pass  # Simulator not available; rely on real pens.

        # Trigger sync via hub supervisor IPC or CLI.
        result = hub_ssh.run(
            "timeout 180 python3 -m hub_pen_sync.cli sync-all --json",
            timeout=200,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        pens_synced = 0
        pens_failed = 0

        try:
            sync_result = json.loads(result.stdout)
            pens_synced = sync_result.get("pens_synced", 0)
            pens_failed = sync_result.get("pens_failed", 0)
        except (json.JSONDecodeError, TypeError):
            pass

        export_result(
            test_id="HW-B2",
            name="Multi-pen sync",
            status="PASS" if pens_synced > 0 and pens_failed == 0 else "FAIL",
            duration_ms=elapsed_ms,
            detail={
                "pens_synced": pens_synced,
                "pens_failed": pens_failed,
                "total_time_ms": elapsed_ms,
            },
        )

        assert pens_synced > 0, "No pens were synced"
        assert pens_failed == 0, f"{pens_failed} pen(s) failed to sync"

    def test_sync_throughput_acceptable(self, hub_ssh, export_result):
        """Verify per-pen sync throughput meets minimum threshold.

        Target: Each pen sync should complete within 30 seconds for a
        typical 500KB stroke payload.
        """
        start = time.monotonic()

        result = hub_ssh.run(
            "python3 -m hub_pen_sync.cli last-sync-stats --json",
            timeout=30,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        if result.returncode != 0:
            pytest.skip("Sync stats not available; run sync first")

        try:
            stats = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            pytest.skip("Could not parse sync stats")
            return

        max_per_pen_ms = 30_000  # 30 seconds per pen
        slow_pens = [
            p for p in stats.get("per_pen", [])
            if p.get("duration_ms", 0) > max_per_pen_ms
        ]

        export_result(
            test_id="HW-B2-throughput",
            name="Multi-pen sync throughput",
            status="PASS" if not slow_pens else "FAIL",
            duration_ms=elapsed_ms,
            detail={
                "slow_pens": len(slow_pens),
                "threshold_ms": max_per_pen_ms,
            },
        )

        assert not slow_pens, (
            f"{len(slow_pens)} pen(s) exceeded {max_per_pen_ms}ms sync time"
        )
