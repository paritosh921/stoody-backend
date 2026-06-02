"""Firmware release metadata helpers for Stoody desktop clients."""
from __future__ import annotations

import os
from typing import Any


DEFAULT_OWNER = "ashuein"
DEFAULT_REPO = "stoody-ble-agent"
DEFAULT_PROD_PREFIX = "firmware-"
DEFAULT_DEV_PREFIX = "dev-firmware-"


def github_owner() -> str:
    return (
        os.getenv("FIRMWARE_RELEASES_GITHUB_OWNER", "")
        or os.getenv("GITHUB_FIRMWARE_OWNER", "")
        or DEFAULT_OWNER
    ).strip()


def github_repo() -> str:
    return (
        os.getenv("FIRMWARE_RELEASES_GITHUB_REPO", "")
        or os.getenv("GITHUB_FIRMWARE_REPO", "")
        or DEFAULT_REPO
    ).strip()


def github_token(channel: str = "prod") -> str:
    if channel == "dev":
        return os.getenv("DEV_FIRMWARE_GIT_TOKEN", "").strip()
    return (
        os.getenv("GITHUB_FIRMWARE_TOKEN", "")
        or os.getenv("FIRMWARE_RELEASES_GITHUB_TOKEN", "")
    ).strip()


def public_base_url() -> str:
    return (
        os.getenv("FIRMWARE_UPDATE_PUBLIC_BASE_URL", "")
        or os.getenv("DESKTOP_UPDATE_PUBLIC_BASE_URL", "")
    ).strip().rstrip("/")


def release_prefix(channel: str = "prod") -> str:
    if channel == "dev":
        return os.getenv("GITHUB_FIRMWARE_DEV_RELEASE_PREFIX", DEFAULT_DEV_PREFIX).strip() or DEFAULT_DEV_PREFIX
    return os.getenv("GITHUB_FIRMWARE_PROD_RELEASE_PREFIX", DEFAULT_PROD_PREFIX).strip() or DEFAULT_PROD_PREFIX


def releases_url(limit: int = 20) -> str:
    per_page = max(1, min(limit, 100))
    return f"https://api.github.com/repos/{github_owner()}/{github_repo()}/releases?per_page={per_page}"


def asset_api_url(asset_id: int | str) -> str:
    return f"https://api.github.com/repos/{github_owner()}/{github_repo()}/releases/assets/{asset_id}"


def github_headers(channel: str = "prod", *, accept: str = "application/vnd.github+json") -> dict[str, str]:
    headers = {
        "Accept": accept,
        "User-Agent": "StoodyFirmwareUpdateBackend",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = github_token(channel)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def normalize_channel(channel: str | None) -> str:
    value = str(channel or "prod").strip().lower()
    return "dev" if value == "dev" else "prod"


def normalize_firmware_version(version: str) -> str:
    return str(version or "").strip().casefold()


def asset_by_name(release: dict[str, Any], name: str) -> dict[str, Any] | None:
    for asset in release.get("assets") or []:
        if asset.get("name") == name:
            return asset
    return None


def compatible_entry(manifest: dict[str, Any], current_version: str, channel: str) -> dict[str, Any] | None:
    manifest_channel = str(manifest.get("channel") or "").strip()
    if manifest_channel and manifest_channel != channel:
        return None

    normalized_current = normalize_firmware_version(current_version)
    for entry in manifest.get("firmwares") or []:
        if not entry.get("enabled", True):
            continue

        entry_channel = str(entry.get("channel") or manifest_channel or channel).strip()
        if entry_channel != channel:
            continue

        to_version = str(entry.get("toVersion") or "").strip()
        from_version = str(entry.get("fromVersion") or "").strip()
        force_update = bool(entry.get("forceUpdate"))
        if not to_version:
            continue
        if normalize_firmware_version(to_version) == normalized_current:
            continue
        if from_version and normalize_firmware_version(from_version) != normalized_current and not force_update:
            continue
        return entry

    return None


def firmware_payload(
    *,
    release: dict[str, Any],
    firmware: dict[str, Any],
    binary_asset: dict[str, Any],
    download_url: str,
) -> dict[str, Any]:
    return {
        "ota_enabled": True,
        "latest_version": str(firmware.get("toVersion") or ""),
        "file_size": int(firmware.get("size") or binary_asset.get("size") or 0),
        "file_md5": str(firmware.get("md5") or ""),
        "download_url": download_url,
        "download_headers": {},
        "release_tag": str(release.get("tag_name") or ""),
        "asset_name": str(firmware.get("assetName") or binary_asset.get("name") or "update-ota.ufw"),
    }
