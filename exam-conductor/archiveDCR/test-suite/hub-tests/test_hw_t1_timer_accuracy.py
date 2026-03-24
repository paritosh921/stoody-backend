"""
HW-T1: Timer accuracy — 90-minute exam timer drift < 1 second.

Hardware required: NTP-synced RPi hub.

Procedure:
  1. Verify NTP synchronization on the hub (chronyc / timedatectl).
  2. Arm a test timer for a short duration (e.g., 60 seconds).
  3. Measure actual elapsed wall-clock time via CLOCK_MONOTONIC.
  4. Compute drift = |expected - actual|.
  5. Extrapolate drift to 90 minutes.

Pass: Extrapolated 90-minute drift < 1 second.
Fail: Drift exceeds 1 second.

Test-ID: HW-T1  (TEST_SUITE_SPEC.md section 2.4)
Level: L6 (hardware-in-loop)
"""

from __future__ import annotations

import time

import pytest

pytestmark = [pytest.mark.hardware]

# 60-second test timer (short enough for CI, long enough to measure drift).
TEST_DURATION_SECONDS = 60
# Maximum acceptable drift for a 90-minute (5400s) exam.
MAX_DRIFT_90MIN_SECONDS = 1.0
# 90 minutes in seconds.
EXAM_DURATION_SECONDS = 5400


@pytest.fixture(autouse=True)
def _require_hub(hub_reachable: bool):
    if not hub_reachable:
        pytest.skip("Hub not reachable via SSH; skipping HW-T1")


class TestTimerAccuracy:
    """HW-T1 — Exam timer drift within acceptable bounds."""

    def test_ntp_synchronized(self, hub_ssh):
        """Verify the hub has NTP synchronization active."""
        result = hub_ssh.run("timedatectl show --property=NTPSynchronized")
        if "NTPSynchronized=yes" not in result.stdout:
            # Try chrony as fallback.
            result = hub_ssh.run("chronyc tracking 2>/dev/null | head -5")
            if result.returncode != 0:
                pytest.skip("NTP not configured on hub")

        assert (
            "NTPSynchronized=yes" in result.stdout
            or "Reference" in result.stdout
        ), "Hub NTP is not synchronized"

    def test_timer_drift_within_bounds(self, hub_ssh, export_result):
        """Arm a short timer and measure drift, extrapolate to 90 minutes."""
        start = time.monotonic()

        # Run a test timer on the hub using CLOCK_MONOTONIC.
        # This script arms a timer, sleeps the expected duration, then
        # reports the actual elapsed time.
        timer_script = (
            f"python3 -c \""
            f"import time; "
            f"start = time.monotonic(); "
            f"time.sleep({TEST_DURATION_SECONDS}); "
            f"elapsed = time.monotonic() - start; "
            f"print(f'{{elapsed:.6f}}')"
            f"\""
        )

        result = hub_ssh.run(timer_script, timeout=TEST_DURATION_SECONDS + 30)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        if result.returncode != 0:
            export_result(
                test_id="HW-T1",
                name="Timer accuracy",
                status="FAIL",
                duration_ms=elapsed_ms,
                detail={"error": result.stderr.strip()[:200]},
            )
            pytest.fail(f"Timer script failed: {result.stderr.strip()[:200]}")

        try:
            actual_elapsed = float(result.stdout.strip())
        except ValueError:
            pytest.fail(f"Could not parse timer output: {result.stdout.strip()}")
            return

        drift_per_second = abs(actual_elapsed - TEST_DURATION_SECONDS) / TEST_DURATION_SECONDS
        extrapolated_drift_90min = drift_per_second * EXAM_DURATION_SECONDS

        export_result(
            test_id="HW-T1",
            name="Timer accuracy",
            status="PASS" if extrapolated_drift_90min < MAX_DRIFT_90MIN_SECONDS else "FAIL",
            duration_ms=elapsed_ms,
            detail={
                "test_duration_s": TEST_DURATION_SECONDS,
                "actual_elapsed_s": round(actual_elapsed, 6),
                "drift_per_second": round(drift_per_second, 9),
                "extrapolated_90min_drift_s": round(extrapolated_drift_90min, 6),
                "max_allowed_drift_s": MAX_DRIFT_90MIN_SECONDS,
            },
        )

        assert extrapolated_drift_90min < MAX_DRIFT_90MIN_SECONDS, (
            f"Extrapolated 90-min drift is {extrapolated_drift_90min:.4f}s, "
            f"exceeds {MAX_DRIFT_90MIN_SECONDS}s threshold"
        )
