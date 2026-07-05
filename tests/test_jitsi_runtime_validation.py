from scripts.validate_jitsi_runtime import (
    resolve_jitsi_domain,
    validate_runtime_config,
)


def test_resolves_domain_from_base_url():
    env = {
        "ONLINE_CLASS_JITSI_DOMAIN": "",
        "ONLINE_CLASS_JITSI_BASE_URL": "https://class.stoody.in",
    }

    assert resolve_jitsi_domain(env) == "class.stoody.in"


def test_requires_jitsi_host():
    errors = validate_runtime_config({}, require_jwt=False)

    assert "Set ONLINE_CLASS_JITSI_DOMAIN or ONLINE_CLASS_JITSI_BASE_URL" in errors


def test_require_jwt_blocks_public_room_config():
    errors = validate_runtime_config(
        {
            "ONLINE_CLASS_JITSI_DOMAIN": "class.stoody.in",
            "ONLINE_CLASS_JITSI_JWT_ENABLED": "false",
        },
        require_jwt=True,
    )

    assert "Set ONLINE_CLASS_JITSI_JWT_ENABLED=true for private online-class rooms" in errors
    assert "Set ONLINE_CLASS_JITSI_JWT_SECRET when Jitsi JWT is enabled or required" in errors


def test_jwt_enabled_requires_secret_even_when_not_forced():
    errors = validate_runtime_config(
        {
            "ONLINE_CLASS_JITSI_DOMAIN": "class.stoody.in",
            "ONLINE_CLASS_JITSI_JWT_ENABLED": "true",
        },
        require_jwt=False,
    )

    assert errors == ["Set ONLINE_CLASS_JITSI_JWT_SECRET when Jitsi JWT is enabled or required"]


def test_accepts_private_jitsi_config_with_positive_ttl():
    errors = validate_runtime_config(
        {
            "ONLINE_CLASS_JITSI_DOMAIN": "class.stoody.in",
            "ONLINE_CLASS_JITSI_JWT_ENABLED": "true",
            "ONLINE_CLASS_JITSI_JWT_SECRET": "not-printed-by-validator",
            "ONLINE_CLASS_JITSI_JWT_TTL_SECONDS": "7200",
        },
        require_jwt=True,
    )

    assert errors == []


def test_rejects_invalid_jwt_ttl():
    errors = validate_runtime_config(
        {
            "ONLINE_CLASS_JITSI_DOMAIN": "class.stoody.in",
            "ONLINE_CLASS_JITSI_JWT_ENABLED": "true",
            "ONLINE_CLASS_JITSI_JWT_SECRET": "not-printed-by-validator",
            "ONLINE_CLASS_JITSI_JWT_TTL_SECONDS": "abc",
        },
        require_jwt=True,
    )

    assert errors == ["ONLINE_CLASS_JITSI_JWT_TTL_SECONDS must be an integer"]
