"""
HW-B3: Dual-write integrity — SD card + USB drive byte-identical.

Hardware required: SD card (primary) + USB flash drive (secondary) on hub.

Procedure:
  1. After a successful pen sync, read the sync data file from SD storage.
  2. Read the same file from USB storage.
  3. Compute checksums (SHA-256) for both copies.
  4. Verify they are byte-identical.

Pass: Both copies are byte-identical (SHA-256 match).
Fail: Checksum mismatch, missing file, or write failure.

Test-ID: HW-B3  (TEST_SUITE_SPEC.md section 2.4)
Level: L6 (hardware-in-loop)
"""

from __future__ import annotations

import json
import time

import pytest

pytestmark = [pytest.mark.hardware]

SD_STORE_PATH = "/var/lib/exampen/store/sd"
USB_STORE_PATH = "/var/lib/exampen/store/usb"


@pytest.fixture(autouse=True)
def _require_hub(hub_reachable: bool):
    if not hub_reachable:
        pytest.skip("Hub not reachable via SSH; skipping HW-B3")


class TestDualWriteIntegrity:
    """HW-B3 — SD and USB copies must be byte-identical."""

    def test_both_storage_paths_exist(self, hub_ssh):
        """Verify that both SD and USB store directories are present."""
        sd_result = hub_ssh.run(f"test -d {SD_STORE_PATH} && echo ok")
        usb_result = hub_ssh.run(f"test -d {USB_STORE_PATH} && echo ok")

        assert "ok" in sd_result.stdout, f"SD store path missing: {SD_STORE_PATH}"
        assert "ok" in usb_result.stdout, (
            f"USB store path missing: {USB_STORE_PATH}. "
            "Hub may be in degraded (SD-only) mode."
        )

    def test_dual_write_checksums_match(self, hub_ssh, export_result):
        """Compare SHA-256 checksums of all files in SD vs USB store."""
        start = time.monotonic()

        # List files in SD store and compute checksums.
        sd_result = hub_ssh.run(
            f"find {SD_STORE_PATH} -type f -exec sha256sum {{}} \\; | sort",
            timeout=60,
        )
        usb_result = hub_ssh.run(
            f"find {USB_STORE_PATH} -type f -exec sha256sum {{}} \\; | sort",
            timeout=60,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        if not sd_result.stdout.strip():
            pytest.skip("No files in SD store; run a sync first")

        # Parse checksums into {relative_path: hash} dicts.
        def parse_checksums(output: str, base: str) -> dict[str, str]:
            result = {}
            for line in output.strip().splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    checksum, path = parts
                    rel = path.replace(base, "").lstrip("/")
                    result[rel] = checksum
            return result

        sd_checksums = parse_checksums(sd_result.stdout, SD_STORE_PATH)
        usb_checksums = parse_checksums(usb_result.stdout, USB_STORE_PATH)

        mismatches: list[str] = []
        missing_on_usb: list[str] = []

        for rel_path, sd_hash in sd_checksums.items():
            usb_hash = usb_checksums.get(rel_path)
            if usb_hash is None:
                missing_on_usb.append(rel_path)
            elif usb_hash != sd_hash:
                mismatches.append(rel_path)

        all_ok = not mismatches and not missing_on_usb

        export_result(
            test_id="HW-B3",
            name="Dual-write integrity",
            status="PASS" if all_ok else "FAIL",
            duration_ms=elapsed_ms,
            detail={
                "sd_files": len(sd_checksums),
                "usb_files": len(usb_checksums),
                "mismatches": len(mismatches),
                "missing_on_usb": len(missing_on_usb),
            },
        )

        assert not missing_on_usb, (
            f"{len(missing_on_usb)} file(s) missing on USB: {missing_on_usb[:5]}"
        )
        assert not mismatches, (
            f"{len(mismatches)} file(s) with checksum mismatch: {mismatches[:5]}"
        )
