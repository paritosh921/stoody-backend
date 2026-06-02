from services import desktop_firmware_updates


def test_firmware_routes_require_user_authentication():
    import inspect

    from api.v1 import desktop_firmware_updates_async

    check_source = inspect.getsource(desktop_firmware_updates_async.check_desktop_firmware_update)
    download_source = inspect.getsource(desktop_firmware_updates_async.download_desktop_firmware)

    assert "Depends(get_current_user)" in check_source
    assert "Depends(get_current_user)" in download_source


def test_firmware_payload_uses_backend_download_url_without_headers():
    payload = desktop_firmware_updates.firmware_payload(
        release={"tag_name": "firmware-p05-v2.7.16_260430"},
        firmware={
            "toVersion": "V2.7.16_260430",
            "size": "123",
            "md5": "abc",
            "assetName": "update-ota.ufw",
        },
        binary_asset={"id": 303, "name": "update-ota.ufw", "size": 123},
        download_url="https://api.stoody.in/api/v1/desktop/firmware/download/303",
    )

    assert payload["ota_enabled"] is True
    assert payload["latest_version"] == "V2.7.16_260430"
    assert payload["download_url"] == "https://api.stoody.in/api/v1/desktop/firmware/download/303"
    assert payload["download_headers"] == {}


def test_compatible_entry_suppresses_same_version_even_when_forced():
    manifest = {
        "channel": "prod",
        "firmwares": [
            {
                "channel": "prod",
                "enabled": True,
                "forceUpdate": True,
                "fromVersion": "V2.7.15_260124",
                "toVersion": "V2.7.16_260430",
            }
        ],
    }

    result = desktop_firmware_updates.compatible_entry(manifest, "V2.7.16_260430", "prod")

    assert result is None


def test_compatible_entry_allows_forced_different_version():
    manifest = {
        "channel": "prod",
        "firmwares": [
            {
                "channel": "prod",
                "enabled": True,
                "forceUpdate": True,
                "fromVersion": "V2.7.14",
                "toVersion": "V2.7.16_260430",
            }
        ],
    }

    result = desktop_firmware_updates.compatible_entry(manifest, "V2.7.15_260124", "prod")

    assert result is not None
    assert result["toVersion"] == "V2.7.16_260430"


def test_backend_firmware_token_selection_uses_expected_env_names(monkeypatch):
    monkeypatch.setenv("GITHUB_FIRMWARE_TOKEN", "prod-secret")
    monkeypatch.setenv("DEV_FIRMWARE_GIT_TOKEN", "dev-secret")

    assert desktop_firmware_updates.github_token("prod") == "prod-secret"
    assert desktop_firmware_updates.github_token("dev") == "dev-secret"
