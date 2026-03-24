"""Network status probes — read-only per STATE_OWNERSHIP_MAP.md Section 2.1.

``check_wifi_status()`` and ``check_backend_reachable()`` MUST NOT mutate
any durable state.  They query NetworkManager and the backend health
endpoint respectively and return status structs.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class WifiStatus:
    """Read-only snapshot of the WiFi link state.

    Populated from ``nmcli`` output (HUB_DEPLOYMENT_SPEC.md Section 5).
    """

    connected: bool
    ssid: str
    signal_dbm: int  # -100 when unknown / not connected
    band: str  # "2.4GHz" | "5GHz" | "unknown"
    channel: int  # 0 when unknown
    ip_address: str


# -- WiFi ----------------------------------------------------------------

async def check_wifi_status() -> WifiStatus:
    """Query NetworkManager for current WiFi state.

    This function MUST NOT mutate state.  It runs ``nmcli`` in a
    subprocess and parses the terse output.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "nmcli", "-t", "-f",
            "GENERAL.STATE,GENERAL.CONNECTION,WIFI.FREQ,WIFI.SIGNAL,IP4.ADDRESS",
            "device", "show", "wlan0",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        return _parse_nmcli(stdout.decode("utf-8", errors="replace"))
    except (FileNotFoundError, asyncio.TimeoutError, OSError) as exc:
        logger.warning("WiFi status check failed: %s", exc)
        return _disconnected_status()


def _parse_nmcli(output: str) -> WifiStatus:
    """Parse ``nmcli -t device show wlan0`` terse output."""
    fields: dict[str, str] = {}
    for line in output.strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()

    connected = "connected" in fields.get("GENERAL.STATE", "").lower()
    ssid = fields.get("GENERAL.CONNECTION", "")
    ip_address = fields.get("IP4.ADDRESS[1]", fields.get("IP4.ADDRESS", ""))

    freq_str = fields.get("WIFI.FREQ", "0")
    freq_mhz = _extract_freq(freq_str)
    band = _freq_to_band(freq_mhz)
    channel = _freq_to_channel(freq_mhz)

    signal_raw = fields.get("WIFI.SIGNAL", "0")
    signal_pct = int(signal_raw) if signal_raw.isdigit() else 0
    signal_dbm = _pct_to_dbm(signal_pct) if connected else -100

    return WifiStatus(
        connected=connected,
        ssid=ssid,
        signal_dbm=signal_dbm,
        band=band,
        channel=channel,
        ip_address=ip_address,
    )


def _disconnected_status() -> WifiStatus:
    return WifiStatus(
        connected=False, ssid="", signal_dbm=-100,
        band="unknown", channel=0, ip_address="",
    )


# -- Backend reachability ------------------------------------------------

async def check_backend_reachable(
    backend_url: str,
    health_endpoint: str = "/health",
    timeout_sec: float = 5.0,
) -> bool:
    """HTTP HEAD to backend health endpoint.  Returns True on 2xx.

    This function MUST NOT mutate state — it is a read-only probe.
    """
    url = backend_url.rstrip("/") + health_endpoint
    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(url, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as resp:
                return 200 <= resp.status < 300
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        logger.debug("Backend unreachable at %s: %s", url, exc)
        return False


# -- Frequency helpers ---------------------------------------------------

def _extract_freq(freq_str: str) -> int:
    """Extract integer MHz from strings like '5180 MHz'."""
    digits = "".join(c for c in freq_str if c.isdigit())
    return int(digits) if digits else 0


def _freq_to_band(freq_mhz: int) -> str:
    if 2400 <= freq_mhz <= 2500:
        return "2.4GHz"
    if 5150 <= freq_mhz <= 5900:
        return "5GHz"
    return "unknown"


def _freq_to_channel(freq_mhz: int) -> int:
    if 2412 <= freq_mhz <= 2484:
        return 1 + (freq_mhz - 2412) // 5
    if 5180 <= freq_mhz <= 5825:
        return 36 + (freq_mhz - 5180) // 5
    return 0


def _pct_to_dbm(pct: int) -> int:
    """Rough conversion from nmcli signal percentage to dBm."""
    if pct <= 0:
        return -100
    if pct >= 100:
        return -30
    return -100 + int(pct * 0.7)
