"""Tests for connectivity module — WiFi check is read-only (no mutations).

Test IDs: U-UPL-24 .. U-UPL-29
Validation level: L3 (unit — mocked subprocess and aiohttp)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.connectivity import (
    WifiStatus,
    _freq_to_band,
    _freq_to_channel,
    _parse_nmcli,
    _pct_to_dbm,
    check_backend_reachable,
    check_wifi_status,
)


# -----------------------------------------------------------------------
# U-UPL-24: check_wifi_status is read-only (no mutations)
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wifi_check_is_read_only() -> None:
    """U-UPL-24: check_wifi_status calls nmcli with read-only flags.

    STATE_OWNERSHIP_MAP.md Section 2.1 requires this function never
    mutates durable state.  We verify the subprocess command does not
    contain any mutating keywords.
    """
    captured_args: list = []

    async def mock_create_subprocess(*args: object, **kwargs: object) -> MagicMock:
        captured_args.extend(args)
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(b"", b""))
        return proc

    with patch("src.connectivity.asyncio.create_subprocess_exec", side_effect=mock_create_subprocess):
        await check_wifi_status()

    # The command must be "nmcli ... device show wlan0" — a read-only op.
    cmd = " ".join(str(a) for a in captured_args)
    assert "nmcli" in cmd
    # Must NOT contain any mutating verbs.
    mutating = ["connect", "modify", "delete", "add", "up", "down"]
    for verb in mutating:
        assert verb not in cmd.split(), f"WiFi check must not use '{verb}'"


# -----------------------------------------------------------------------
# U-UPL-25: parse connected nmcli output
# -----------------------------------------------------------------------

def test_parse_nmcli_connected() -> None:
    """U-UPL-25: parse a connected WiFi nmcli output."""
    output = (
        "GENERAL.STATE:100 (connected)\n"
        "GENERAL.CONNECTION:SchoolWiFi\n"
        "WIFI.FREQ:5180 MHz\n"
        "WIFI.SIGNAL:75\n"
        "IP4.ADDRESS[1]:192.168.1.105/24\n"
    )
    status = _parse_nmcli(output)
    assert status.connected is True
    assert status.ssid == "SchoolWiFi"
    assert status.band == "5GHz"
    assert status.channel == 36
    assert status.ip_address == "192.168.1.105/24"
    assert status.signal_dbm > -100


# -----------------------------------------------------------------------
# U-UPL-26: parse disconnected nmcli output
# -----------------------------------------------------------------------

def test_parse_nmcli_disconnected() -> None:
    """U-UPL-26: disconnected output returns connected=False."""
    output = "GENERAL.STATE:30 (disconnected)\n"
    status = _parse_nmcli(output)
    assert status.connected is False
    assert status.signal_dbm == -100


# -----------------------------------------------------------------------
# U-UPL-27: check_backend_reachable returns True on 200
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backend_reachable_on_200() -> None:
    """U-UPL-27: HEAD to health endpoint returning 200 -> True."""
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.head = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("src.connectivity.aiohttp.ClientSession", return_value=mock_session):
        result = await check_backend_reachable("https://backend.test")

    assert result is True


# -----------------------------------------------------------------------
# U-UPL-28: check_backend_reachable returns False on timeout
# -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backend_unreachable_on_timeout() -> None:
    """U-UPL-28: connection timeout -> False (no exception raised)."""
    mock_session = AsyncMock()
    mock_session.head = MagicMock(side_effect=asyncio.TimeoutError())
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("src.connectivity.aiohttp.ClientSession", return_value=mock_session):
        result = await check_backend_reachable("https://backend.test")

    assert result is False


# -----------------------------------------------------------------------
# U-UPL-29: frequency helpers
# -----------------------------------------------------------------------

def test_freq_to_band() -> None:
    """U-UPL-29: frequency to band conversion."""
    assert _freq_to_band(2437) == "2.4GHz"
    assert _freq_to_band(5180) == "5GHz"
    assert _freq_to_band(0) == "unknown"


def test_freq_to_channel() -> None:
    """U-UPL-29b: frequency to channel conversion."""
    assert _freq_to_channel(2412) == 1
    assert _freq_to_channel(5180) == 36
    assert _freq_to_channel(5745) == 149
    assert _freq_to_channel(0) == 0


def test_pct_to_dbm_range() -> None:
    """U-UPL-29c: percentage to dBm stays in sane range."""
    assert _pct_to_dbm(0) == -100
    assert _pct_to_dbm(100) == -30
    assert -100 <= _pct_to_dbm(50) <= -30


# -----------------------------------------------------------------------
# U-UPL-29d: WifiStatus is frozen
# -----------------------------------------------------------------------

def test_wifi_status_immutable() -> None:
    """U-UPL-29d: WifiStatus is a frozen dataclass (read-only)."""
    ws = WifiStatus(
        connected=True, ssid="Test", signal_dbm=-50,
        band="5GHz", channel=36, ip_address="10.0.0.1",
    )
    with pytest.raises(AttributeError):
        ws.connected = False  # type: ignore[misc]
