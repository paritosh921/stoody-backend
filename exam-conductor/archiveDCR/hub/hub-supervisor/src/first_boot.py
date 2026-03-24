"""First-boot detection and provisioning.

On first power-on the hub has no ``/etc/exampen/hub.conf``.  The
supervisor detects this and forces the TUI Setup screen.  After WiFi is
connected and the backend is verified, a provisioning request is sent:

    POST /api/v1/hubs/provision {hub_code}

The response (hub_id, institute_id, invig codes, pen inventory) is
cached to SQLite and the config file is written.

See ``HUB_DEPLOYMENT_SPEC.md`` Section 7 for the full sequence.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import HUB_CONF_PATH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def is_first_boot(config_path: str | None = None) -> bool:
    """Return True if the hub has not been provisioned.

    Detection: ``/etc/exampen/hub.conf`` does not exist.
    """
    path = config_path or HUB_CONF_PATH
    return not Path(path).is_file()


# ---------------------------------------------------------------------------
# Provisioning response
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ProvisionResponse:
    """Parsed response from ``POST /api/v1/hubs/provision``."""

    hub_id: str
    institute_id: str
    invig_codes: list[dict[str, str]]
    pen_inventory: list[dict[str, str]]


# ---------------------------------------------------------------------------
# Provision data persistence
# ---------------------------------------------------------------------------

_INVIG_CODES_INSERT = (
    "INSERT OR REPLACE INTO invig_codes "
    "(code, valid_from, valid_until, fetched_at) "
    "VALUES (?, ?, ?, ?)"
)

_PEN_INVENTORY_INSERT = (
    "INSERT OR REPLACE INTO pen_inventory "
    "(pen_mac, pen_serial, fw_version, registered_at) "
    "VALUES (?, ?, ?, ?)"
)


def store_provision_data(
    conn: sqlite3.Connection,
    response: ProvisionResponse,
    provisioned_at: str,
) -> None:
    """Persist provisioning response to SQLite tables.

    Populates ``hub_config``, ``invig_codes``, and ``pen_inventory``.
    """
    conn.execute(
        "INSERT OR REPLACE INTO hub_config "
        "(hub_id, backend_url, uplink_mode, region, provisioned_at) "
        "VALUES (?, '', 'wifi', 'US', ?)",
        (response.hub_id, provisioned_at),
    )
    for code_entry in response.invig_codes:
        conn.execute(
            _INVIG_CODES_INSERT,
            (
                code_entry.get("code", ""),
                code_entry.get("valid_from", ""),
                code_entry.get("valid_until", ""),
                provisioned_at,
            ),
        )
    for pen in response.pen_inventory:
        conn.execute(
            _PEN_INVENTORY_INSERT,
            (
                pen.get("pen_mac", ""),
                pen.get("pen_serial", ""),
                pen.get("fw_version", ""),
                provisioned_at,
            ),
        )
    conn.commit()
    logger.info(
        "Stored provision data: hub_id=%s, %d invig codes, %d pens",
        response.hub_id,
        len(response.invig_codes),
        len(response.pen_inventory),
    )


# ---------------------------------------------------------------------------
# Config file writer
# ---------------------------------------------------------------------------

def write_hub_conf(
    hub_id: str,
    backend_url: str,
    uplink_mode: str = "wifi",
    config_path: str | None = None,
) -> None:
    """Write ``/etc/exampen/hub.conf`` (INI format).

    After this file exists, subsequent boots are NOT first-boot.
    """
    path = Path(config_path or HUB_CONF_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "[hub]",
        f"hub_id = {hub_id}",
        f"backend_url = {backend_url}",
        f"uplink_mode = {uplink_mode}",
        "region = US",
        "",
        "[paths]",
        "sd_data = /var/lib/exampen",
        "usb_data = /mnt/exampen-backup",
        "socket_dir = /run/exampen",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote hub config to %s", path)


# ---------------------------------------------------------------------------
# Provision orchestrator (HTTP call is injected for testability)
# ---------------------------------------------------------------------------

async def run_provisioning(
    hub_code: str,
    backend_url: str,
    uplink_mode: str,
    conn: sqlite3.Connection,
    provisioned_at: str,
    *,
    http_post_fn: Any | None = None,
    config_path: str | None = None,
) -> ProvisionResponse:
    """Execute the full first-boot provisioning sequence.

    Parameters
    ----------
    hub_code:
        The unique hub code entered by the operator.
    backend_url:
        The ExamPen backend base URL.
    uplink_mode:
        ``"wifi"``, ``"mobile"``, or ``"auto"``.
    conn:
        Open SQLite connection for persisting results.
    provisioned_at:
        ISO 8601 timestamp.
    http_post_fn:
        Injected ``async (url, payload) -> dict`` for the HTTP call.
        In production this would use ``aiohttp``; in tests a mock.
    config_path:
        Override for hub.conf path (tests).
    """
    if http_post_fn is None:
        raise RuntimeError("http_post_fn must be provided")

    url = f"{backend_url.rstrip('/')}/api/v1/hubs/provision"
    raw: dict[str, Any] = await http_post_fn(url, {"hub_code": hub_code})

    response = ProvisionResponse(
        hub_id=raw["hub_id"],
        institute_id=raw["institute_id"],
        invig_codes=raw.get("invig_codes", []),
        pen_inventory=raw.get("pen_inventory", []),
    )

    store_provision_data(conn, response, provisioned_at)
    write_hub_conf(
        response.hub_id,
        backend_url,
        uplink_mode,
        config_path=config_path,
    )
    return response
