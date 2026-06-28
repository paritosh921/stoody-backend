"""Desktop release metadata helpers for Stoody client updates."""
from __future__ import annotations

import os
import re
from typing import Any


DEFAULT_OWNER = "ashuein"
DEFAULT_REPO = "stoody-ble-agent"
PLATFORM_WINDOWS = "windows"
PLATFORM_MACOS = "macos"
DEFAULT_WINDOWS_ASSET_PATTERN = r"\.exe$"
DEFAULT_MACOS_ASSET_PATTERN = r"\.dmg$"


def github_owner() -> str:
    return os.getenv("DESKTOP_RELEASES_GITHUB_OWNER", DEFAULT_OWNER).strip() or DEFAULT_OWNER


def github_repo() -> str:
    return os.getenv("DESKTOP_RELEASES_GITHUB_REPO", DEFAULT_REPO).strip() or DEFAULT_REPO


def github_token() -> str:
    return (
        os.getenv("DESKTOP_RELEASES_GITHUB_TOKEN", "")
        or os.getenv("GITHUB_TOKEN", "")
        or os.getenv("GH_TOKEN", "")
    ).strip()


def public_base_url() -> str:
    return os.getenv("DESKTOP_UPDATE_PUBLIC_BASE_URL", "").strip().rstrip("/")


def normalize_platform(platform: str | None = None) -> str:
    value = (platform or "").strip().lower()
    if value in {"mac", "macos", "darwin", "osx"}:
        return PLATFORM_MACOS
    return PLATFORM_WINDOWS


def asset_pattern(platform: str | None = None) -> str:
    normalized = normalize_platform(platform)
    if normalized == PLATFORM_MACOS:
        return (
            os.getenv("DESKTOP_UPDATE_MACOS_ASSET_PATTERN", DEFAULT_MACOS_ASSET_PATTERN).strip()
            or DEFAULT_MACOS_ASSET_PATTERN
        )
    return (
        os.getenv(
            "DESKTOP_UPDATE_WINDOWS_ASSET_PATTERN",
            os.getenv("DESKTOP_UPDATE_ASSET_PATTERN", DEFAULT_WINDOWS_ASSET_PATTERN),
        ).strip()
        or DEFAULT_WINDOWS_ASSET_PATTERN
    )


def latest_release_url() -> str:
    return f"https://api.github.com/repos/{github_owner()}/{github_repo()}/releases/latest"


def releases_url(limit: int) -> str:
    per_page = max(1, min(limit, 100))
    return f"https://api.github.com/repos/{github_owner()}/{github_repo()}/releases?per_page={per_page}"


def asset_api_url(asset_id: int | str) -> str:
    return f"https://api.github.com/repos/{github_owner()}/{github_repo()}/releases/assets/{asset_id}"


def github_headers(*, accept: str = "application/vnd.github+json") -> dict[str, str]:
    headers = {
        "Accept": accept,
        "User-Agent": "StoodyDesktopUpdateBackend",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def normalize_release_version(tag_name: str, platform: str | None = None) -> str:
    value = (tag_name or "").strip()
    if normalize_platform(platform) == PLATFORM_MACOS:
        value = re.sub(r"^mac-v", "", value, flags=re.IGNORECASE)
    else:
        value = re.sub(r"^v", "", value, flags=re.IGNORECASE)
    return value


def is_semver(value: str) -> bool:
    return bool(re.match(r"^\d+\.\d+\.\d+$", value or ""))


def release_matches_platform(release: dict[str, Any], platform: str | None = None) -> bool:
    version = normalize_release_version(str(release.get("tag_name") or ""), platform)
    return is_semver(version)


def select_platform_asset(release: dict[str, Any], platform: str | None = None) -> dict[str, Any] | None:
    pattern = re.compile(asset_pattern(platform), re.IGNORECASE)
    for asset in release.get("assets") or []:
        name = str(asset.get("name") or "")
        if pattern.search(name):
            return asset
    return None


def select_windows_asset(release: dict[str, Any]) -> dict[str, Any] | None:
    return select_platform_asset(release, PLATFORM_WINDOWS)


def select_latest_release(releases: list[dict[str, Any]], platform: str | None = None) -> tuple[dict[str, Any], dict[str, Any]] | None:
    for release in releases:
        if not isinstance(release, dict):
            continue
        if not release_matches_platform(release, platform):
            continue
        asset = select_platform_asset(release, platform)
        if asset and asset.get("id"):
            return release, asset
    return None


def latest_payload(release: dict[str, Any], download_url: str, platform: str | None = None) -> dict[str, Any] | None:
    latest_version = normalize_release_version(str(release.get("tag_name") or ""), platform)
    if not is_semver(latest_version):
        return None

    asset = select_platform_asset(release, platform)
    if not asset:
        return None

    return {
        "success": True,
        "platform": normalize_platform(platform),
        "latest_version": latest_version,
        "version": latest_version,
        "download_url": download_url,
        "asset_id": asset.get("id"),
        "asset_name": asset.get("name") or "",
        "asset_size": asset.get("size") or 0,
        "release_body": release.get("body") or "",
        "published_at": release.get("published_at"),
        "tag_name": release.get("tag_name"),
    }


def release_notes_payload(releases: list[dict[str, Any]], limit: int, platform: str | None = None) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []
    for release in releases:
        if not isinstance(release, dict):
            continue
        version = normalize_release_version(str(release.get("tag_name") or ""), platform)
        if not is_semver(version):
            continue
        notes.append(
            {
                "version": version,
                "body": str(release.get("body") or ""),
                "tag_name": str(release.get("tag_name") or ""),
                "published_at": str(release.get("published_at") or ""),
            }
        )
        if len(notes) >= max(1, min(limit, 25)):
            break
    return notes
