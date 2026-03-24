"""Production BLE adapter implementations using bleak.

Provides concrete classes for the protocol interfaces defined in
``dongle_manager.BleAdapter``, ``pen_discovery.BleScanner``,
``connection_manager.BleConnector``, and ``health_monitor.HealthProbe``.

All operations accept a dongle identifier (MAC / hci path) so that
multi-dongle setups route BLE traffic through the correct USB adapter.

Falls back to stub implementations when bleak is not installed (e.g.,
Windows dev machines, CI runners without Bluetooth hardware).
"""

from __future__ import annotations

import asyncio
import logging
import shutil

from src.dongle_manager import DongleState
from src.health_monitor import ProbeResult
from src.pen_discovery import DiscoveryCallback, PenInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import bleak; fall back gracefully for dev/test environments.
# ---------------------------------------------------------------------------

try:
    from bleak import BleakClient, BleakScanner  # type: ignore[import-untyped]

    _HAS_BLEAK = True
except ImportError:
    _HAS_BLEAK = False
    BleakScanner = None  # type: ignore[assignment,misc]
    BleakClient = None  # type: ignore[assignment,misc]
    logger.info("bleak not available -- using stub BLE adapter")


# ---------------------------------------------------------------------------
# BlueZAdapter  (BleAdapter protocol: dongle enumeration)
# ---------------------------------------------------------------------------

class BlueZAdapter:
    """Enumerate HCI adapters via ``hciconfig`` or bleak backends.

    On Linux with BlueZ, parses ``hciconfig -a`` output to discover
    USB BLE dongles.  On other platforms returns an empty list (stubs).
    """

    async def enumerate(self) -> list[DongleState]:
        """Return currently visible HCI adapters."""
        if shutil.which("hciconfig") is None:
            logger.debug("hciconfig not found -- returning empty dongle list")
            return []
        return await self._enumerate_hciconfig()

    async def _enumerate_hciconfig(self) -> list[DongleState]:
        """Parse ``hciconfig -a`` output into DongleState objects."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "hciconfig", "-a",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except (OSError, asyncio.TimeoutError) as exc:
            logger.warning("hciconfig enumeration failed: %s", exc)
            return []

        return self._parse_hciconfig(stdout.decode(errors="replace"))

    @staticmethod
    def _parse_hciconfig(output: str) -> list[DongleState]:
        """Extract adapter info from hciconfig -a output."""
        dongles: list[DongleState] = []
        current_hci: str | None = None
        current_mac: str | None = None

        for line in output.splitlines():
            stripped = line.strip()
            # Lines like "hci0:   Type: Primary  Bus: USB"
            if line and not line[0].isspace() and ":" in line:
                if current_hci and current_mac:
                    dongles.append(DongleState(
                        mac=current_mac,
                        hci_path=current_hci,
                    ))
                token = line.split(":", 1)[0].strip()
                current_hci = token if token.startswith("hci") else None
                current_mac = None

            # Lines like "BD Address: AA:BB:CC:DD:EE:FF  ..."
            if "BD Address:" in stripped and current_hci:
                parts = stripped.split("BD Address:", 1)[1].strip().split()
                if parts:
                    current_mac = parts[0]

        # Flush last adapter.
        if current_hci and current_mac:
            dongles.append(DongleState(
                mac=current_mac,
                hci_path=current_hci,
            ))

        return dongles


# ---------------------------------------------------------------------------
# BleakScanner wrapper  (BleScanner protocol: pen discovery)
# ---------------------------------------------------------------------------

class BleakScannerAdapter:
    """Wraps ``bleak.BleakScanner`` to implement the ``BleScanner`` protocol."""

    async def start_scan(
        self,
        dongle: DongleState,
        callback: DiscoveryCallback,
        timeout_sec: int,
    ) -> None:
        if not _HAS_BLEAK:
            logger.warning("bleak not available -- scan is a no-op")
            return

        def _detection(device, adv_data):  # type: ignore[no-untyped-def]
            uuids = list(adv_data.service_uuids) if adv_data.service_uuids else []
            pen = PenInfo(
                mac=device.address,
                rssi=adv_data.rssi or -100,
                dongle_mac=dongle.mac,
                service_uuids=uuids,
            )
            asyncio.get_event_loop().create_task(callback(pen))

        scanner = BleakScanner(
            detection_callback=_detection,
            adapter=dongle.hci_path,
        )
        await scanner.start()
        try:
            await asyncio.sleep(timeout_sec)
        finally:
            await scanner.stop()

    async def stop_scan(self, dongle: DongleState) -> None:
        # Scanner stop is handled in start_scan's finally block.
        pass


# ---------------------------------------------------------------------------
# BleakClient wrapper  (BleConnector protocol: pen connection)
# ---------------------------------------------------------------------------

class BleakConnectorAdapter:
    """Wraps ``bleak.BleakClient`` to implement the ``BleConnector`` protocol."""

    def __init__(self) -> None:
        # pen_mac -> BleakClient (active connections).
        self._clients: dict[str, object] = {}

    async def connect(
        self, pen_mac: str, dongle_mac: str, timeout: float,
    ) -> str:
        if not _HAS_BLEAK:
            logger.warning("bleak not available -- returning stub connection")
            return f"stub-conn-{pen_mac}"

        client = BleakClient(pen_mac, adapter=dongle_mac, timeout=timeout)
        await client.connect()
        self._clients[pen_mac] = client
        return f"bleak-{pen_mac}"

    async def disconnect(self, pen_mac: str) -> None:
        client = self._clients.pop(pen_mac, None)
        if client is not None and _HAS_BLEAK:
            try:
                await client.disconnect()  # type: ignore[union-attr]
            except Exception:
                logger.debug("Disconnect error for %s (ignoring)", pen_mac)


# ---------------------------------------------------------------------------
# Health probe  (HealthProbe protocol)
# ---------------------------------------------------------------------------

class HciHealthProbe:
    """Probe dongle health by running ``hciconfig <hciX>``."""

    async def probe(self, dongle: DongleState) -> ProbeResult:
        if shutil.which("hciconfig") is None:
            return ProbeResult(
                dongle_mac=dongle.mac,
                responsive=True,
                response_time_sec=0.0,
                detail="hciconfig not available -- assuming healthy",
            )

        try:
            t0 = asyncio.get_event_loop().time()
            proc = await asyncio.create_subprocess_exec(
                "hciconfig", dongle.hci_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            elapsed = asyncio.get_event_loop().time() - t0
        except asyncio.TimeoutError:
            return ProbeResult(
                dongle_mac=dongle.mac,
                responsive=False,
                detail="hciconfig timed out",
            )
        except OSError as exc:
            return ProbeResult(
                dongle_mac=dongle.mac,
                responsive=False,
                detail=f"hciconfig error: {exc}",
            )

        text = stdout.decode(errors="replace")
        up = "UP" in text and "RUNNING" in text
        can_scan = "UP" in text

        return ProbeResult(
            dongle_mac=dongle.mac,
            responsive=up,
            response_time_sec=elapsed,
            can_scan=can_scan,
            detail="" if up else "adapter not UP RUNNING",
        )
