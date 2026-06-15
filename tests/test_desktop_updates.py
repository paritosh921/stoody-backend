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
    assert payload["latest_version"] == "1.2.21"
    assert payload["download_url"] == "https://api.stoody.in/desktop/download/202"
    assert payload["asset_id"] == 202
    assert payload["asset_name"] == "Stoody_Client_1.2.21_x64.exe"


def test_latest_payload_rejects_non_semver_release():
    release = {
        "tag_name": "firmware-p05-v2.7.16_260430",
        "assets": [{"id": 202, "name": "Stoody_Client_1.2.21_x64.exe", "size": 12345}],
    }

    assert desktop_updates.latest_payload(release, "https://api.stoody.in/desktop/download/202") is None


def test_release_notes_payload_filters_non_desktop_versions():
    releases = [
        {"tag_name": "v1.2.21", "body": "Desktop release", "published_at": "2026-05-13T00:00:00Z"},
        {"tag_name": "firmware-p05-v2.7.16_260430", "body": "Firmware release"},
        {"tag_name": "1.2.20", "body": "Previous desktop release"},
    ]

    notes = desktop_updates.release_notes_payload(releases, 10)

    assert [note["version"] for note in notes] == ["1.2.21", "1.2.20"]


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

    async def fake_validated_latest_desktop_asset(client):
        seen["validated_client"] = client
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
    result = await handler(request=None)  # type: ignore[arg-type]

    assert result == {"streamed": True}
    assert seen["asset_id"] == 202
    assert seen["stream_client"] is seen["validated_client"]
