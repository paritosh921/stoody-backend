import pytest

from services import desktop_updates


def test_latest_payload_uses_backend_download_url():
    release = {
        "tag_name": "v1.2.21",
        "body": "Release notes",
        "published_at": "2026-05-13T00:00:00Z",
        "assets": [
            {"id": 101, "name": "notes.txt", "size": 10},
            {"id": 202, "name": "Stoody_Client_1.2.21_x64.exe", "size": 12345},
        ],
    }

    payload = desktop_updates.latest_payload(release, "https://api.stoody.in/desktop/download/202")

    assert payload is not None
    assert payload["platform"] == "windows"
    assert payload["latest_version"] == "1.2.21"
    assert payload["download_url"] == "https://api.stoody.in/desktop/download/202"
    assert payload["asset_id"] == 202
    assert payload["asset_name"] == "Stoody_Client_1.2.21_x64.exe"


def test_latest_payload_supports_macos_release_tags_and_dmg_assets():
    release = {
        "tag_name": "mac-v1.3.30",
        "body": "Mac release notes",
        "published_at": "2026-06-29T00:00:00Z",
        "assets": [
            {"id": 303, "name": "Stoody_Client_1.3.30_macOS.dmg", "size": 45678},
            {"id": 404, "name": "Stoody_Client_1.3.30_x64.exe", "size": 12345},
        ],
    }

    payload = desktop_updates.latest_payload(
        release,
        "https://api.stoody.in/desktop/download/303?platform=macos",
        "macos",
    )

    assert payload is not None
    assert payload["platform"] == "macos"
    assert payload["latest_version"] == "1.3.30"
    assert payload["download_url"] == "https://api.stoody.in/desktop/download/303?platform=macos"
    assert payload["asset_id"] == 303
    assert payload["asset_name"] == "Stoody_Client_1.3.30_macOS.dmg"


def test_latest_payload_rejects_non_semver_release():
    release = {
        "tag_name": "firmware-p05-v2.7.16_260430",
        "assets": [{"id": 202, "name": "Stoody_Client_1.2.21_x64.exe", "size": 12345}],
    }

    assert desktop_updates.latest_payload(release, "https://api.stoody.in/desktop/download/202") is None


def test_release_notes_payload_filters_non_desktop_versions():
    releases = [
        {"tag_name": "mac-v1.3.30", "body": "Mac desktop release"},
        {"tag_name": "v1.2.21", "body": "Desktop release", "published_at": "2026-05-13T00:00:00Z"},
        {"tag_name": "firmware-p05-v2.7.16_260430", "body": "Firmware release"},
        {"tag_name": "1.2.20", "body": "Previous desktop release"},
    ]

    notes = desktop_updates.release_notes_payload(releases, 10)

    assert [note["version"] for note in notes] == ["1.2.21", "1.2.20"]


def test_release_notes_payload_filters_macos_versions():
    releases = [
        {"tag_name": "mac-v1.3.30", "body": "Mac desktop release"},
        {"tag_name": "v1.2.21", "body": "Windows desktop release"},
        {"tag_name": "firmware-p05-v2.7.16_260430", "body": "Firmware release"},
        {"tag_name": "mac-v1.3.22", "body": "Previous Mac desktop release"},
    ]

    notes = desktop_updates.release_notes_payload(releases, 10, "darwin")

    assert [note["version"] for note in notes] == ["1.3.30", "1.3.22"]


def test_select_latest_release_keeps_windows_and_macos_separate():
    releases = [
        {
            "tag_name": "mac-v1.3.30",
            "assets": [{"id": 303, "name": "Stoody_Client_1.3.30_macOS.dmg", "size": 45678}],
        },
        {
            "tag_name": "v1.3.11",
            "assets": [{"id": 202, "name": "Stoody_Client_1.3.11_x64.exe", "size": 12345}],
        },
    ]

    windows = desktop_updates.select_latest_release(releases, "windows")
    macos = desktop_updates.select_latest_release(releases, "macos")

    assert windows is not None
    assert windows[0]["tag_name"] == "v1.3.11"
    assert windows[1]["id"] == 202
    assert macos is not None
    assert macos[0]["tag_name"] == "mac-v1.3.30"
    assert macos[1]["id"] == 303


def test_select_windows_asset_rejects_non_exe_assets():
    release = {
        "tag_name": "v1.2.21",
        "assets": [
            {"id": 101, "name": "firmware.json", "size": 10},
            {"id": 102, "name": "update-ota.ufw", "size": 100},
        ],
    }

    assert desktop_updates.select_windows_asset(release) is None


def test_router_exposes_plain_latest_download_route():
    from api.v1.desktop_updates_async import router

    paths = {(route.path, tuple(sorted(route.methods))) for route in router.routes}

    assert ("/latest/download", ("GET",)) in paths


@pytest.mark.asyncio
async def test_latest_download_route_streams_selected_latest_asset(monkeypatch):
    from api.v1 import desktop_updates_async

    seen = {}

    async def fake_validated_latest_desktop_asset(client, platform="windows"):
        seen["validated_client"] = client
        seen["platform"] = platform
        return {"id": 202}

    async def fake_stream_desktop_asset(client, asset_id):
        seen["stream_client"] = client
        seen["asset_id"] = asset_id
        return {"streamed": True}

    monkeypatch.setattr(
        desktop_updates_async,
        "_validated_latest_desktop_asset",
        fake_validated_latest_desktop_asset,
    )
    monkeypatch.setattr(
        desktop_updates_async,
        "_stream_desktop_asset",
        fake_stream_desktop_asset,
    )

    handler = desktop_updates_async.download_latest_desktop_update.__wrapped__
    result = await handler(request=None, platform="macos")  # type: ignore[arg-type]

    assert result == {"streamed": True}
    assert seen["asset_id"] == 202
    assert seen["platform"] == "macos"
    assert seen["stream_client"] is seen["validated_client"]
