"""BLE diagnostic tests B1-B4.

Each test is an async function returning (TestStatus, detail_dict).
BLE tests require physical hardware (pens or nRF52840-DK simulator).
When hardware is unavailable, tests return SKIP.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

from src.diagnostics.runner import TestCase, TestCategory, TestStatus

PEN_SERVICE_UUID = "0000ae30-0000-1000-8000-00805f9b34fb"
DEVICE_INFO_CHAR_UUID = "00002a29-0000-1000-8000-00805f9b34fb"
BLE_SCAN_TIMEOUT = 30
BLE_CONNECT_TIMEOUT = 10
SD_DATA_DIR = Path("/var/lib/exampen/data")
USB_DATA_DIR = Path("/mnt/exampen-backup/data")


def _bleak_available() -> bool:
    """Check if the bleak BLE library is importable."""
    try:
        import bleak  # noqa: F401
        return True
    except ImportError:
        return False


async def _ensure_connected(client: Any) -> bool:
    """Ensure BleakClient is connected, return True if successful."""
    if not client.is_connected:
        await client.connect()
    return client.is_connected


# -- B1: Pen discovery ----------------------------------------------------

async def b1_pen_discovery() -> tuple[TestStatus, dict[str, Any]]:
    """Scan for BLE pens advertising the ExamPen GATT service UUID."""
    if not _bleak_available():
        return TestStatus.SKIP, {"reason": "bleak library not installed"}

    try:
        from bleak import BleakScanner
        scanner = BleakScanner(service_uuids=[PEN_SERVICE_UUID])
        devices = await asyncio.wait_for(
            scanner.discover(timeout=BLE_SCAN_TIMEOUT), timeout=BLE_SCAN_TIMEOUT + 5,
        )
        pens = [
            {"name": d.name or "unknown", "address": d.address}
            for d in devices
            if PEN_SERVICE_UUID in [str(u).lower() for u in (d.metadata.get("uuids") or [])]
        ]
        # Accept all discovered if service_uuids filter was applied at scanner level
        if not pens and devices:
            pens = [{"name": d.name or "unknown", "address": d.address} for d in devices]
        if not pens:
            return TestStatus.FAIL, {"pens_found": 0, "error": f"No pens found in {BLE_SCAN_TIMEOUT}s scan"}
        return TestStatus.PASS, {"pens_found": len(pens), "pens": pens}
    except asyncio.TimeoutError:
        return TestStatus.FAIL, {"error": f"Timeout: scan exceeded {BLE_SCAN_TIMEOUT}s"}
    except Exception as exc:
        return TestStatus.FAIL, {"error": f"BLE scan error: {exc}"}


# -- B2: GATT read test --------------------------------------------------

async def b2_gatt_read() -> tuple[TestStatus, dict[str, Any]]:
    """Connect to a pen and read the device info characteristic."""
    if not _bleak_available():
        return TestStatus.SKIP, {"reason": "bleak library not installed"}

    try:
        from bleak import BleakClient, BleakScanner
        scanner = BleakScanner(service_uuids=[PEN_SERVICE_UUID])
        devices = await asyncio.wait_for(
            scanner.discover(timeout=BLE_SCAN_TIMEOUT), timeout=BLE_SCAN_TIMEOUT + 5,
        )
        if not devices:
            return TestStatus.FAIL, {"error": "No BLE pens found for GATT read test"}

        target = devices[0]
        detail: dict[str, Any] = {"pen_address": target.address, "pen_name": target.name or "unknown"}

        async with BleakClient(target.address) as client:
            connected = await asyncio.wait_for(_ensure_connected(client), timeout=BLE_CONNECT_TIMEOUT)
            if not connected:
                return TestStatus.FAIL, {**detail, "error": "Failed to connect to pen"}
            try:
                value = await client.read_gatt_char(DEVICE_INFO_CHAR_UUID)
                detail["device_info"] = value.decode(errors="replace")
                return TestStatus.PASS, detail
            except Exception as exc:
                return TestStatus.FAIL, {**detail, "error": f"GATT read failed: {exc}"}
    except asyncio.TimeoutError:
        return TestStatus.FAIL, {"error": "Timeout during GATT read test"}
    except Exception as exc:
        return TestStatus.FAIL, {"error": f"BLE error: {exc}"}


# -- B3: Multi-pen stress -------------------------------------------------

async def b3_multi_pen_stress() -> tuple[TestStatus, dict[str, Any]]:
    """Concurrent sync simulation -- requires multiple pens or simulators."""
    if not _bleak_available():
        return TestStatus.SKIP, {"reason": "bleak library not installed"}

    try:
        from bleak import BleakClient, BleakScanner
        scanner = BleakScanner(service_uuids=[PEN_SERVICE_UUID])
        devices = await asyncio.wait_for(
            scanner.discover(timeout=BLE_SCAN_TIMEOUT), timeout=BLE_SCAN_TIMEOUT + 5,
        )
        if len(devices) < 2:
            return TestStatus.SKIP, {
                "pens_found": len(devices),
                "reason": "Multi-pen stress requires 2+ pens",
            }

        async def _connect_pen(addr: str, name: str) -> dict[str, Any]:
            try:
                async with BleakClient(addr) as client:
                    await asyncio.wait_for(_ensure_connected(client), timeout=BLE_CONNECT_TIMEOUT)
                    return {"address": addr, "name": name, "connected": True}
            except Exception as exc:
                return {"address": addr, "name": name, "connected": False, "error": str(exc)}

        results = await asyncio.gather(*[_connect_pen(d.address, d.name or "unknown") for d in devices])
        connected = sum(1 for r in results if r["connected"])
        failed = sum(1 for r in results if not r["connected"])
        detail: dict[str, Any] = {"total_pens": len(devices), "connected": connected, "failed": failed, "results": results}
        if failed > 0:
            detail["error"] = f"{failed}/{len(devices)} pens failed to connect"
            return TestStatus.FAIL, detail
        return TestStatus.PASS, detail
    except asyncio.TimeoutError:
        return TestStatus.FAIL, {"error": "Timeout during multi-pen stress"}
    except Exception as exc:
        return TestStatus.FAIL, {"error": f"BLE error: {exc}"}


# -- B4: Sync + dual-write -----------------------------------------------

def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


async def b4_sync_dual_write() -> tuple[TestStatus, dict[str, Any]]:
    """Verify both SD and USB copies match via checksum for the most recent exam."""
    if not SD_DATA_DIR.is_dir():
        return TestStatus.FAIL, {"error": f"SD data dir not found: {SD_DATA_DIR}"}
    if not USB_DATA_DIR.is_dir():
        return TestStatus.FAIL, {"error": f"USB data dir not found: {USB_DATA_DIR}"}

    exam_dirs = sorted(SD_DATA_DIR.iterdir(), reverse=True)
    if not exam_dirs:
        return TestStatus.SKIP, {"reason": "No exam data directories found"}

    exam_dir = exam_dirs[0]
    exam_id = exam_dir.name
    pen_dirs = [d for d in exam_dir.iterdir() if d.is_dir()]
    if not pen_dirs:
        return TestStatus.SKIP, {"reason": f"No pen data in exam {exam_id}"}

    mismatches: list[dict[str, str]] = []
    checked = 0
    for pen_dir in pen_dirs:
        pen_mac = pen_dir.name
        sd_raw = pen_dir / "strokes_raw.bin"
        usb_raw = USB_DATA_DIR / exam_id / pen_mac / "strokes_raw.bin"
        if not sd_raw.exists():
            continue
        checked += 1
        if not usb_raw.exists():
            mismatches.append({"pen_mac": pen_mac, "error": "USB copy missing"})
            continue
        sd_hash, usb_hash = _file_sha256(sd_raw), _file_sha256(usb_raw)
        if sd_hash != usb_hash:
            mismatches.append({"pen_mac": pen_mac, "error": f"Checksum mismatch: SD={sd_hash[:16]}... USB={usb_hash[:16]}..."})

    detail: dict[str, Any] = {"exam_id": exam_id, "pens_checked": checked, "mismatches": len(mismatches)}
    if mismatches:
        detail["mismatch_details"] = mismatches
        detail["error"] = f"{len(mismatches)} dual-write mismatches found"
        return TestStatus.FAIL, detail
    if checked == 0:
        return TestStatus.SKIP, {**detail, "reason": "No stroke files found to check"}
    return TestStatus.PASS, detail


# -- Registry -------------------------------------------------------------

def build_ble_tests() -> list[TestCase]:
    """Return the full set of BLE test cases B1-B4."""
    return [
        TestCase(id="B1", name="Pen discovery", category=TestCategory.BLE, run_fn=b1_pen_discovery),
        TestCase(id="B2", name="GATT read test", category=TestCategory.BLE, run_fn=b2_gatt_read),
        TestCase(id="B3", name="Multi-pen stress", category=TestCategory.BLE, run_fn=b3_multi_pen_stress),
        TestCase(id="B4", name="Sync + dual-write", category=TestCategory.BLE, run_fn=b4_sync_dual_write),
    ]
