"""Hub configuration loader.

Reads ``/etc/exampen/hub.conf`` (INI-style) and exposes a typed
:class:`HubConfig` dataclass.  Falls back to environment variables and
then to hardcoded defaults so that unit tests run without a real config
file.
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path

# Default paths (match HUB_DEPLOYMENT_SPEC.md)
DEFAULT_CONFIG_PATH = "/etc/exampen/hub.conf"
DEFAULT_SD_DATA_PATH = "/var/lib/exampen"
DEFAULT_USB_DATA_PATH = "/mnt/exampen-backup"
DEFAULT_SOCKET_DIR = "/run/exampen"

# Socket file names per module (ipc-protocol.md Section 1)
SOCKET_NAMES: dict[str, str] = {
    "hub-supervisor": "supervisor.sock",
    "hub-ble-mgr": "ble-mgr.sock",
    "hub-pen-sync": "pen-sync.sock",
    "hub-store": "store.sock",
    "hub-timer": "timer.sock",
    "hub-uplink": "uplink.sock",
    "hub-invig-ble": "invig-ble.sock",
    "hub-tui": "tui.sock",
}


@dataclass(slots=True)
class HubConfig:
    """Typed representation of hub configuration."""

    hub_id: str
    backend_url: str
    uplink_mode: str  # "wifi" | "mobile" | "auto"
    region: str

    sd_data_path: str
    usb_data_path: str
    socket_dir: str

    # Derived: absolute socket path for a given module
    def socket_path(self, module_id: str) -> str:
        name = SOCKET_NAMES.get(module_id)
        if name is None:
            raise ValueError(f"Unknown module id: {module_id}")
        return os.path.join(self.socket_dir, name)


def load_hub_config(
    config_path: str | None = None,
) -> HubConfig:
    """Load :class:`HubConfig` from file, env-vars, or defaults.

    Resolution order for each field:
    1. INI file value (if *config_path* exists).
    2. Environment variable (``EXAMPEN_*``).
    3. Hardcoded default.
    """
    cp = configparser.ConfigParser()
    path = config_path or os.environ.get("EXAMPEN_CONFIG", DEFAULT_CONFIG_PATH)
    if Path(path).is_file():
        cp.read(path)

    def _get(section: str, key: str, env_key: str, default: str) -> str:
        try:
            return cp.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return os.environ.get(env_key, default)

    return HubConfig(
        hub_id=_get("hub", "hub_id", "EXAMPEN_HUB_ID", "UNPROVISIONED"),
        backend_url=_get("hub", "backend_url", "EXAMPEN_BACKEND_URL", ""),
        uplink_mode=_get("hub", "uplink_mode", "EXAMPEN_UPLINK_MODE", "wifi"),
        region=_get("hub", "region", "EXAMPEN_REGION", "US"),
        sd_data_path=_get("paths", "sd_data", "EXAMPEN_DATA", DEFAULT_SD_DATA_PATH),
        usb_data_path=_get("paths", "usb_data", "EXAMPEN_BACKUP", DEFAULT_USB_DATA_PATH),
        socket_dir=_get("paths", "socket_dir", "EXAMPEN_SOCKET_DIR", DEFAULT_SOCKET_DIR),
    )
