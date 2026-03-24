"""Environment configuration for hub-uplink.

Backend URL and retry parameters are sourced from the hub config file
or environment variables.  Timeouts and batch sizes match the
operational profile described in HUB_DEPLOYMENT_SPEC.md Section 5
and FAILURE_MITIGATION_REGISTER.md U1/U4.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Defaults tuned for RPi 4B on school WiFi.
DEFAULT_UPLOAD_TIMEOUT_SEC = 30
DEFAULT_RETRY_INTERVAL_SEC = 5
DEFAULT_CHUNK_BATCH_SIZE = 4
DEFAULT_HEALTH_ENDPOINT = "/health"
DEFAULT_INGEST_ENDPOINT = "/api/v1/strokes/ingest"

# Upload time estimates (seconds per chunk) used by path_selector.
WIFI_SEC_PER_CHUNK = 0.5
MOBILE_BLE_SEC_PER_CHUNK = 6.0


@dataclass(slots=True)
class UplinkConfig:
    """Resolved uplink configuration."""

    backend_url: str
    ingest_endpoint: str
    health_endpoint: str
    upload_timeout_sec: int
    retry_interval_sec: int
    chunk_batch_size: int


def load_uplink_config(backend_url: str) -> UplinkConfig:
    """Build :class:`UplinkConfig` from environment or defaults.

    *backend_url* is sourced from :func:`hub_common.config.load_hub_config`
    so it is always passed explicitly by the caller.
    """
    return UplinkConfig(
        backend_url=backend_url,
        ingest_endpoint=os.environ.get(
            "EXAMPEN_INGEST_ENDPOINT", DEFAULT_INGEST_ENDPOINT,
        ),
        health_endpoint=os.environ.get(
            "EXAMPEN_HEALTH_ENDPOINT", DEFAULT_HEALTH_ENDPOINT,
        ),
        upload_timeout_sec=int(
            os.environ.get("EXAMPEN_UPLOAD_TIMEOUT", DEFAULT_UPLOAD_TIMEOUT_SEC),
        ),
        retry_interval_sec=int(
            os.environ.get("EXAMPEN_RETRY_INTERVAL", DEFAULT_RETRY_INTERVAL_SEC),
        ),
        chunk_batch_size=int(
            os.environ.get("EXAMPEN_CHUNK_BATCH_SIZE", DEFAULT_CHUNK_BATCH_SIZE),
        ),
    )
