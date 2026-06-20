"""Post-deploy validation gate for upload-security runtime controls."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


EICAR = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"


CommandRunner = Callable[[Sequence[str], int], tuple[int, str, str]]


@dataclass(frozen=True)
class DeployValidationResult:
    ok: bool
    passed_checks: list[str]
    failed_checks: list[str]
    details: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_upload_security_deploy_validation(
    *,
    env: Mapping[str, str] | None = None,
    runner: CommandRunner | None = None,
    nginx_config_paths: Sequence[Path] | None = None,
) -> DeployValidationResult:
    environment = env or os.environ
    run_command = runner or _default_runner
    checks = _CheckAccumulator()

    checks.require_env_true("UPLOAD_SCAN_REQUIRED", environment.get("UPLOAD_SCAN_REQUIRED"))
    checks.require_env_true("UPLOAD_AV_FAIL_CLOSED", environment.get("UPLOAD_AV_FAIL_CLOSED"))

    socket_path = Path(environment.get("CLAMAV_SOCKET") or environment.get("CLAMD_SOCKET") or "")
    checks.record("CLAMAV_SOCKET_EXISTS", bool(str(socket_path)) and socket_path.exists(), str(socket_path))
    checks.record("CLAMAV_SOCKET_NOT_WORLD_ACCESSIBLE", _socket_mode_is_private(socket_path), str(socket_path))

    _check_systemd_active(checks, run_command, "clamav-daemon")
    _check_systemd_active(checks, run_command, "clamav-freshclam")
    _check_eicar_detection(checks, run_command)
    _check_private_upload_dirs(checks, environment)
    _check_nginx_private_root(checks, environment, nginx_config_paths or _default_nginx_configs())
    _check_backend_health(checks, environment, run_command)

    return DeployValidationResult(
        ok=not checks.failed,
        passed_checks=checks.passed,
        failed_checks=checks.failed,
        details=checks.details,
    )


class _CheckAccumulator:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.details: dict[str, str] = {}

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        (self.passed if ok else self.failed).append(name)
        if detail:
            self.details[name] = detail

    def require_env_true(self, name: str, value: str | None) -> None:
        normalized = (value or "").strip().lower()
        self.record(name, normalized == "true", value or "")


def _check_systemd_active(checks: _CheckAccumulator, runner: CommandRunner, unit: str) -> None:
    code, stdout, stderr = runner(["systemctl", "is-active", unit], 10)
    checks.record(f"SYSTEMD_{unit}", code == 0 and stdout.strip() == "active", (stdout or stderr).strip())


def _socket_mode_is_private(path: Path) -> bool:
    if not path.exists():
        return False
    if os.name == "nt":
        return True
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        return False
    return mode & 0o007 == 0


def _check_eicar_detection(checks: _CheckAccumulator, runner: CommandRunner) -> None:
    code, stdout, stderr = runner(["clamdscan", "--fdpass", "--stream"], 20)
    output = f"{stdout}\n{stderr}"
    checks.record("EICAR_DETECTED", code == 1 and "FOUND" in output, output.strip())


def _check_private_upload_dirs(checks: _CheckAccumulator, env: Mapping[str, str]) -> None:
    root = Path(env.get("UPLOAD_PRIVATE_LOCAL_DIR") or "/var/lib/stoody/uploads")
    for prefix in ("quarantine", "rejected", "clean"):
        path = root / prefix
        exists = path.exists() and path.is_dir()
        mode_ok = True
        if exists and os.name != "nt":
            mode = path.stat().st_mode & 0o777
            mode_ok = mode & 0o007 == 0
        checks.record(f"PRIVATE_UPLOAD_DIR_{prefix.upper()}", exists and mode_ok, str(path))


def _check_nginx_private_root(
    checks: _CheckAccumulator,
    env: Mapping[str, str],
    nginx_config_paths: Sequence[Path],
) -> None:
    root = str(Path(env.get("UPLOAD_PRIVATE_LOCAL_DIR") or "/var/lib/stoody/uploads"))
    offending: list[str] = []
    for config in nginx_config_paths:
        try:
            text = config.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if root in text:
            offending.append(str(config))
    checks.record("NGINX_PRIVATE_UPLOAD_NOT_SERVED", not offending, ",".join(offending))


def _check_backend_health(checks: _CheckAccumulator, env: Mapping[str, str], runner: CommandRunner) -> None:
    health_url = env.get("BACKEND_HEALTH_URL")
    if not health_url:
        checks.record("BACKEND_HEALTH_SCANNER_AVAILABLE", False, "BACKEND_HEALTH_URL missing")
        return
    code, stdout, stderr = runner(["curl", "-fsS", health_url], 15)
    if code != 0:
        checks.record("BACKEND_HEALTH_SCANNER_AVAILABLE", False, stderr.strip())
        return
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        checks.record("BACKEND_HEALTH_SCANNER_AVAILABLE", False, str(exc))
        return
    scanner = payload.get("upload_malware_scanner") or payload.get("services", {}).get("upload_malware_scanner") or {}
    checks.record("BACKEND_HEALTH_SCANNER_AVAILABLE", bool(scanner.get("available")), json.dumps(scanner, sort_keys=True))


def _default_nginx_configs() -> list[Path]:
    roots = [Path("/etc/nginx/nginx.conf"), Path("/etc/nginx/sites-enabled")]
    configs: list[Path] = []
    for root in roots:
        if root.is_file():
            configs.append(root)
        elif root.is_dir():
            configs.extend(path for path in root.rglob("*") if path.is_file())
    return configs


def _default_runner(command: Sequence[str], timeout: int = 10) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            list(command),
            input=EICAR if command[:2] == ["clamdscan", "--fdpass"] else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return 127, "", str(exc)
    return (
        completed.returncode,
        completed.stdout.decode("utf-8", errors="replace"),
        completed.stderr.decode("utf-8", errors="replace"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate deployed upload-security runtime controls.")
    parser.add_argument("--nginx-config", action="append", type=Path, default=[])
    args = parser.parse_args(argv)
    result = run_upload_security_deploy_validation(nginx_config_paths=args.nginx_config)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
