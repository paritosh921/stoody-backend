"""Unauthenticated desktop client update endpoints.

The desktop app must not ship GitHub credentials. These endpoints let the
backend use its private GitHub token and expose only sanitized update metadata
plus a backend download URL to the installed EXE.
"""
from __future__ import annotations

import logging
import re

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from services import desktop_updates

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

GITHUB_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)


def _download_url(request: Request, asset_id: int | str) -> str:
    base_url = desktop_updates.public_base_url()
    if base_url:
        return f"{base_url}/desktop/download/{asset_id}"
    return str(request.url_for("download_desktop_update", asset_id=str(asset_id)))


def _safe_download_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "")
    return name[:150] or "Stoody_Client_Update.exe"


async def _latest_desktop_asset(client: httpx.AsyncClient) -> dict | None:
    response = await client.get(
        desktop_updates.latest_release_url(),
        headers=desktop_updates.github_headers(),
    )
    if response.status_code >= 400:
        logger.warning("Desktop update latest release validation returned HTTP %s", response.status_code)
        return None
    release = response.json()
    latest_version = desktop_updates.normalize_release_version(str(release.get("tag_name") or ""))
    if not desktop_updates.is_semver(latest_version):
        return None
    return desktop_updates.select_windows_asset(release)


async def _validated_latest_desktop_asset(client: httpx.AsyncClient) -> dict:
    try:
        allowed_asset = await _latest_desktop_asset(client)
    except httpx.RequestError as exc:
        await client.aclose()
        logger.warning("Desktop update asset validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Update download temporarily unavailable",
        ) from exc

    if not allowed_asset or not allowed_asset.get("id"):
        await client.aclose()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Desktop update asset not found",
        )

    return allowed_asset


async def _stream_desktop_asset(client: httpx.AsyncClient, asset_id: int | str) -> StreamingResponse:
    req = client.build_request(
        "GET",
        desktop_updates.asset_api_url(asset_id),
        headers=desktop_updates.github_headers(accept="application/octet-stream"),
    )
    try:
        response = await client.send(req, stream=True)
    except httpx.RequestError as exc:
        await client.aclose()
        logger.warning("Desktop update asset download failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Update download temporarily unavailable",
        ) from exc

    if response.status_code >= 400:
        body = await response.aread()
        await response.aclose()
        await client.aclose()
        logger.warning(
            "Desktop update asset download returned HTTP %s: %s",
            response.status_code,
            body[:200],
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Update download returned an error",
        )

    content_type = response.headers.get("content-type", "application/octet-stream")
    content_length = response.headers.get("content-length")
    disposition = response.headers.get("content-disposition", "")
    filename = "Stoody_Client_Update.exe"
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


@router.get("/latest")
@limiter.limit("60/minute")
async def latest_desktop_update(request: Request):
    """Return latest desktop release metadata without exposing GitHub secrets."""
    try:
        async with httpx.AsyncClient(timeout=GITHUB_TIMEOUT) as client:
            response = await client.get(
                desktop_updates.latest_release_url(),
                headers=desktop_updates.github_headers(),
            )
    except httpx.RequestError as exc:
        logger.warning("Desktop update GitHub lookup failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Update service temporarily unavailable",
        ) from exc

    if response.status_code == 404:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Desktop release not found")
    if response.status_code >= 400:
        logger.warning("Desktop update GitHub lookup returned HTTP %s", response.status_code)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Update service returned an error",
        )

    release = response.json()
    asset = desktop_updates.select_windows_asset(release)
    if not asset or not asset.get("id"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No desktop installer asset found in latest release",
        )

    payload = desktop_updates.latest_payload(release, _download_url(request, asset["id"]))
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Latest release is not a valid desktop app release",
        )
    return payload


@router.get("/releases")
@limiter.limit("30/minute")
async def desktop_release_notes(request: Request, limit: int = Query(default=10, ge=1, le=25)):
    """Return recent desktop release notes without exposing GitHub credentials."""
    try:
        async with httpx.AsyncClient(timeout=GITHUB_TIMEOUT) as client:
            response = await client.get(
                desktop_updates.releases_url(limit),
                headers=desktop_updates.github_headers(),
            )
    except httpx.RequestError as exc:
        logger.warning("Desktop release notes lookup failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Release notes temporarily unavailable",
        ) from exc

    if response.status_code >= 400:
        logger.warning("Desktop release notes lookup returned HTTP %s", response.status_code)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Release notes service returned an error",
        )

    releases = response.json()
    if not isinstance(releases, list):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Release notes service returned invalid data",
        )

    return {
        "success": True,
        "releases": desktop_updates.release_notes_payload(releases, limit),
    }


@router.get("/latest/download", name="download_latest_desktop_update")
@limiter.limit("20/minute")
async def download_latest_desktop_update(request: Request):
    """Stream the latest GitHub desktop release asset through the backend."""
    client = httpx.AsyncClient(timeout=GITHUB_TIMEOUT, follow_redirects=True)
    allowed_asset = await _validated_latest_desktop_asset(client)
    return await _stream_desktop_asset(client, allowed_asset["id"])


@router.get("/download/{asset_id}", name="download_desktop_update")
@limiter.limit("20/minute")
async def download_desktop_update(request: Request, asset_id: int):
    """Stream a GitHub release asset through the backend."""
    client = httpx.AsyncClient(timeout=GITHUB_TIMEOUT, follow_redirects=True)
    allowed_asset = await _validated_latest_desktop_asset(client)

    if not allowed_asset or str(allowed_asset.get("id") or "") != str(asset_id):
        await client.aclose()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Desktop update asset not found",
        )

    return await _stream_desktop_asset(client, asset_id)
