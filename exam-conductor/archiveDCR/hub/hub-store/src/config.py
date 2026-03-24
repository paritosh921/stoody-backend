"""Environment configuration for hub-store.

Paths match HUB_DEPLOYMENT_SPEC.md Section 3.2:
  - SD primary:  /var/lib/exampen
  - USB backup:  /mnt/exampen-backup
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SD_PATH = "/var/lib/exampen"
DEFAULT_USB_PATH = "/mnt/exampen-backup"


@dataclass(slots=True)
class StoreConfig:
    """Resolved storage paths and flags."""

    sd_base: Path
    usb_base: Path

    @property
    def sd_data(self) -> Path:
        return self.sd_base / "data"

    @property
    def usb_data(self) -> Path:
        return self.usb_base / "data"

    @property
    def db_path(self) -> Path:
        """SQLite ledger database (WAL mode) — ``hub.db`` under SD base."""
        return self.sd_base / "hub.db"

    def usb_available(self) -> bool:
        """Return True if the USB mount point exists and is a directory."""
        return self.usb_base.is_dir()


def load_store_config() -> StoreConfig:
    """Build :class:`StoreConfig` from environment or defaults."""
    sd = os.environ.get("EXAMPEN_DATA", DEFAULT_SD_PATH)
    usb = os.environ.get("EXAMPEN_BACKUP", DEFAULT_USB_PATH)
    return StoreConfig(sd_base=Path(sd), usb_base=Path(usb))
