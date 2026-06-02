"""Unauthenticated desktop firmware update endpoints.

The installed desktop client must not ship or read GitHub firmware tokens.
These endpoints use the backend's GitHub token and expose sanitized firmware
metadata plus backend-proxied firmware downloads.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.v1.auth_async import get_current_user
from services import desktop_firmware_updates

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

GITHUB_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)


def _download_url(request: Request, asset_id: int | str, channel: str) -> str:
    base_url = desktop_firmware_updates.public_base_url()
    channel_query = f"?channel={channel}"
    if base_url:
        return f"{base_url}/desktop/firmware/download/{asset_id}{channel_query}"
    return f"{request.url_for('download_desktop_firmware', asset_id=str(asset_id))}{channel_query}"


def _safe_download_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "")
    return name[:150] or "update-ota.ufw"


async def _download_asset_bytes(client: httpx.AsyncClient, asset_id: int | str, channel: str) -> bytes:
    response = await client.get(
        desktop_firmware_updates.asset_api_url(asset_id),
        headers=desktop_firmware_updates.github_headers(channel, accept="application/octet-stream"),
    )
    if response.status_code >= 400:
        logger.warning("Firmware asset lookup returned HTTP %s", response.status_code)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Firmware update service returned an error",
        )
    return response.content


async def _valid_firmware_asset_ids(client: httpx.AsyncClient, channel: str) -> set[str]:
    prefix = desktop_firmware_updates.release_prefix(channel)
    response = await client.get(
        desktop_firmware_updates.releases_url(20),
        headers=desktop_firmware_updates.github_headers(channel),
    )
    if response.status_code >= 400:
        logger.warning("Firmware release validation returned HTTP %s", response.status_code)
        return set()

    releases = response.json()
    if not isinstance(releases, list):
        return set()

    valid_asset_ids: set[str] = set()
    for release in releases:
        tag = str(release.get("tag_name") or "")
        if not tag.startswith(prefix):
            continue

        manifest_asset = desktop_firmware_updates.asset_by_name(release, "firmware.json")
        if not manifest_asset or not manifest_asset.get("id"):
            continue

        try:
            manifest_bytes = await _download_asset_bytes(client, manifest_asset["id"], channel)
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (HTTPException, UnicodeDecodeError, json.JSONDecodeError):
            continue

        manifest_channel = str(manifest.get("channel") or "").strip()
        if manifest_channel and manifest_channel != channel:
            continue

        for firmware in manifest.get("firmwares") or []:
            if not firmware.get("enabled", True):
                continue
            entry_channel = str(firmware.get("channel") or manifest_channel or channel).strip()
            if entry_channel != channel:
                continue
            asset_name = str(firmware.get("assetName") or "update-ota.ufw")
            binary_asset = desktop_firmware_updates.asset_by_name(release, asset_name)
            if binary_asset and binary_asset.get("id"):
                valid_asset_ids.add(str(binary_asset["id"]))

    return valid_asset_ids


@router.get("/firmware/check")
@limiter.limit("60/minute")
async def check_desktop_firmware_update(
    request: Request,
    current_version: str = Query(default=""),
    channel: str = Query(default="prod"),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Return firmware metadata without exposing GitHub credentials."""
    _ = current_user
    selected_channel = desktop_firmware_updates.normalize_channel(channel)
    prefix = desktop_firmware_updates.release_prefix(selected_channel)

    try:
        async with httpx.AsyncClient(timeout=GITHUB_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(
                desktop_firmware_updates.releases_url(20),
                headers=desktop_firmware_updates.github_headers(selected_channel),
            )
            if response.status_code == 404:
                return {"ota_enabled": False}
            if response.status_code >= 400:
                logger.warning("Firmware release lookup returned HTTP %s", response.status_code)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Firmware update service returned an error",
                )

            releases = response.json()
            if not isinstance(releases, list):
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Firmware update service returned invalid data",
                )

            for release in releases:
                tag = str(release.get("tag_name") or "")
                if not tag.startswith(prefix):
                    continue

                manifest_asset = desktop_firmware_updates.asset_by_name(release, "firmware.json")
                if not manifest_asset or not manifest_asset.get("id"):
                    continue

                manifest_bytes = await _download_asset_bytes(client, manifest_asset["id"], selected_channel)
                try:
                    manifest = json.loads(manifest_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    logger.warning("Firmware manifest in %s is invalid JSON", tag)
                    continue

                firmware = desktop_firmware_updates.compatible_entry(
                    manifest,
                    current_version,
                    selected_channel,
                )
                if firmware is None:
                    continue

                asset_name = str(firmware.get("assetName") or "update-ota.ufw")
                binary_asset = desktop_firmware_updates.asset_by_name(release, asset_name)
                if not binary_asset or not binary_asset.get("id"):
                    continue

                return desktop_firmware_updates.firmware_payload(
                    release=release,
                    firmware=firmware,
                    binary_asset=binary_asset,
                    download_url=_download_url(request, binary_asset["id"], selected_channel),
                )
    except httpx.RequestError as exc:
        logger.warning("Firmware update GitHub lookup failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firmware update service temporarily unavailable",
        ) from exc

    return {"ota_enabled": False}


@router.get("/firmware/download/{asset_id}", name="download_desktop_firmware")
@limiter.limit("20/minute")
async def download_desktop_firmware(
    request: Request,
    asset_id: int,
    channel: str = Query(default="prod"),
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Stream a firmware release asset through the backend."""
    _ = current_user
    selected_channel = desktop_firmware_updates.normalize_channel(channel)
    client = httpx.AsyncClient(timeout=GITHUB_TIMEOUT, follow_redirects=True)
    try:
        valid_asset_ids = await _valid_firmware_asset_ids(client, selected_channel)
    except httpx.RequestError as exc:
        await client.aclose()
        logger.warning("Firmware asset validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firmware download temporarily unavailable",
        ) from exc

    if str(asset_id) not in valid_asset_ids:
        await client.aclose()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Firmware asset not found",
        )

    req = client.build_request(
        "GET",
        desktop_firmware_updates.asset_api_url(asset_id),
        headers=desktop_firmware_updates.github_headers(selected_channel, accept="application/octet-stream"),
    )
    try:
        response = await client.send(req, stream=True)
    except httpx.RequestError as exc:
        await client.aclose()
        logger.warning("Firmware asset download failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firmware download temporarily unavailable",
        ) from exc

    if response.status_code >= 400:
        body = await response.aread()
        await response.aclose()
        await client.aclose()
        logger.warning(
            "Firmware asset download returned HTTP %s: %s",
            response.status_code,
            body[:200],
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Firmware download returned an error",
        )

    content_type = response.headers.get("content-type", "application/octet-stream")
    content_length = response.headers.get("content-length")
    disposition = response.headers.get("content-disposition", "")
    filename = "update-ota.ufw"
    match = re.search(r'filename="?([^";]+)"?', disposition)
    if match:
        filename = _safe_download_name(match.group(1))

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    if content_length:
        headers["Content-Length"] = content_length

    async def body_iter():
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(body_iter(), media_type=content_type, headers=headers)
