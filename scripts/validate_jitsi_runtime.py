#!/usr/bin/env python3
"""Validate online-class Jitsi runtime configuration without exposing secrets."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen


JITSI_REQUIRED_ASSETS = (
    "/external_api.js",
    "/config.js",
    "/interface_config.js",
    "/libs/lib-jitsi-meet.min.js",
    "/http-bind",
)


def _parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def load_runtime_env(env_file: str | None) -> dict[str, str]:
    merged = dict(os.environ)
    candidate = Path(env_file) if env_file else Path(".env")
    merged.update(_read_env_file(candidate))
    return merged


def resolve_jitsi_domain(env: Mapping[str, str]) -> str:
    domain = str(env.get("ONLINE_CLASS_JITSI_DOMAIN") or "").strip()
    if domain:
        return domain.replace("https://", "").replace("http://", "").split("/", 1)[0]

    base_url = str(env.get("ONLINE_CLASS_JITSI_BASE_URL") or "").strip()
    if not base_url:
        return ""
    parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
    return parsed.hostname or ""


def validate_runtime_config(
    env: Mapping[str, str],
    *,
    require_jwt: bool,
) -> list[str]:
    errors: list[str] = []
    domain = resolve_jitsi_domain(env)
    if not domain:
        errors.append("Set ONLINE_CLASS_JITSI_DOMAIN or ONLINE_CLASS_JITSI_BASE_URL")

    jwt_enabled = _parse_bool(env.get("ONLINE_CLASS_JITSI_JWT_ENABLED"))
    jwt_secret = str(env.get("ONLINE_CLASS_JITSI_JWT_SECRET") or "").strip()
    if require_jwt and not jwt_enabled:
        errors.append("Set ONLINE_CLASS_JITSI_JWT_ENABLED=true for private online-class rooms")
    if (require_jwt or jwt_enabled) and not jwt_secret:
        errors.append("Set ONLINE_CLASS_JITSI_JWT_SECRET when Jitsi JWT is enabled or required")

    ttl = str(env.get("ONLINE_CLASS_JITSI_JWT_TTL_SECONDS") or "").strip()
    if ttl:
        try:
            if int(ttl) <= 0:
                errors.append("ONLINE_CLASS_JITSI_JWT_TTL_SECONDS must be positive")
        except ValueError:
            errors.append("ONLINE_CLASS_JITSI_JWT_TTL_SECONDS must be an integer")

    return errors


def _check_url(url: str, timeout_seconds: int) -> None:
    request = Request(url, method="HEAD", headers={"User-Agent": "StoodyDeployCheck/1.0"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = response.status
    except Exception:
        request = Request(url, method="GET", headers={"User-Agent": "StoodyDeployCheck/1.0"})
        with urlopen(request, timeout=timeout_seconds) as response:
            status = response.status

    if status >= 400:
        raise RuntimeError(f"{url} returned HTTP {status}")


def validate_public_assets(domain: str, timeout_seconds: int) -> list[str]:
    base = f"https://{domain.strip().rstrip('/')}"
    errors: list[str] = []
    for path in JITSI_REQUIRED_ASSETS:
        url = f"{base}{path}"
        try:
            _check_url(url, timeout_seconds)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stoody online-class Jitsi runtime settings")
    parser.add_argument("--env-file", default=".env", help="Runtime env file to inspect")
    parser.add_argument("--require-jwt", action="store_true", help="Require JWT for meeting and canvas rooms")
    parser.add_argument("--check-public-assets", action="store_true", help="Fetch required Jitsi public assets")
    parser.add_argument("--timeout-seconds", type=int, default=10)
    args = parser.parse_args()

    env = load_runtime_env(args.env_file)
    errors = validate_runtime_config(env, require_jwt=args.require_jwt)
    domain = resolve_jitsi_domain(env)
    if args.check_public_assets and domain:
        errors.extend(validate_public_assets(domain, max(1, args.timeout_seconds)))

    if errors:
        print("Jitsi runtime validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    jwt_enabled = _parse_bool(env.get("ONLINE_CLASS_JITSI_JWT_ENABLED"))
    print(f"Jitsi runtime validation passed for {domain} (jwt_enabled={jwt_enabled})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
