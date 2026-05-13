"""Desktop release metadata helpers for Stoody client updates."""
from __future__ import annotations

import os
import re
from typing import Any


DEFAULT_OWNER = "ashuein"
DEFAULT_REPO = "stoody-ble-agent"
DEFAULT_ASSET_PATTERN = r"\.exe$"


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


def asset_pattern() -> str:
    return os.getenv("DESKTOP_UPDATE_ASSET_PATTERN", DEFAULT_ASSET_PATTERN).strip() or DEFAULT_ASSET_PATTERN


def latest_release_url() -> str:
    return f"https://api.github.com/repos/{github_owner()}/{github_repo()}/releases/latest"


def releases_url(limit: int) -> str:
    per_page = max(1, min(limit, 25))
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


def normalize_release_version(tag_name: str) -> str:
    return (tag_name or "").strip().lstrip("vV")


def is_semver(value: str) -> bool:
    return bool(re.match(r"^\d+\.\d+\.\d+$", value or ""))


def select_windows_asset(release: dict[str, Any]) -> dict[str, Any] | None:
    pattern = re.compile(asset_pattern(), re.IGNORECASE)
    for asset in release.get("assets") or []:
        name = str(asset.get("name") or "")
        if pattern.search(name):
            return asset
    return None


def latest_payload(release: dict[str, Any], download_url: str) -> dict[str, Any] | None:
    latest_version = normalize_release_version(str(release.get("tag_name") or ""))
    if not is_semver(latest_version):
        return None

    asset = select_windows_asset(release)
    if not asset:
        return None

    return {
        "success": True,
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


def release_notes_payload(releases: list[dict[str, Any]], limit: int) -> list[dict[str, str]]:
    notes: list[dict[str, str]] = []
    for release in releases[: max(1, min(limit, 25))]:
        version = normalize_release_version(str(release.get("tag_name") or ""))
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
    return notes
