"""Post-deploy validation gate for upload-security runtime controls."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
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

    _check_clamav_socket_permissions(checks, environment)
    _check_systemd_active(checks, run_command, "clamav-daemon")
    _check_systemd_active(checks, run_command, "clamav-freshclam")
    _check_eicar_detection(checks, run_command)
    _check_private_upload_dirs(checks, environment)
    _check_upload_cleanup_timer(checks, environment, run_command)
    _check_upload_cleanup_service_not_failed(checks, environment, run_command)
    _check_upload_cleanup_script_dry_run(checks, environment, run_command)
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


def _check_clamav_socket_permissions(checks: _CheckAccumulator, env: Mapping[str, str]) -> None:
    socket_path = Path(env.get("CLAMAV_SOCKET") or env.get("CLAMD_SOCKET") or "")
    expected_owner = env.get("CLAMAV_SOCKET_OWNER") or env.get("STOODY_CLAMAV_OWNER") or "clamav"
    expected_group = env.get("CLAMAV_SOCKET_GROUP") or env.get("STOODY_CLAMAV_GROUP") or "clamav"
    expected_mode_text = env.get("CLAMAV_SOCKET_MODE") or env.get("STOODY_CLAMAV_SOCKET_MODE") or "660"
    backend_user = env.get("BACKEND_SERVICE_USER") or env.get("STOODY_UPLOAD_OWNER") or ""

    exists = bool(str(socket_path)) and socket_path.exists()
    checks.record("CLAMAV_SOCKET_EXISTS", exists, str(socket_path))
    if not exists:
        return

    if os.name == "nt":
        checks.record("CLAMAV_SOCKET_NOT_WORLD_ACCESSIBLE", True, "POSIX-only check skipped on Windows")
        checks.record("CLAMAV_SOCKET_MODE_660", True, "POSIX-only check skipped on Windows")
        checks.record("CLAMAV_SOCKET_OWNER_GROUP", True, "POSIX-only check skipped on Windows")
        checks.record("BACKEND_SERVICE_USER_IN_CLAMAV_GROUP", True, "POSIX-only check skipped on Windows")
        return

    try:
        expected_mode = int(expected_mode_text, 8)
    except ValueError:
        checks.record("CLAMAV_SOCKET_MODE_660", False, expected_mode_text)
        return

    try:
        import grp
        import pwd

        stat_result = socket_path.stat()
        actual_mode = stat_result.st_mode & 0o777
        actual_owner = pwd.getpwuid(stat_result.st_uid).pw_name
        actual_group = grp.getgrgid(stat_result.st_gid).gr_name
    except OSError as exc:
        checks.record("CLAMAV_SOCKET_NOT_WORLD_ACCESSIBLE", False, str(exc))
        return

    checks.record("CLAMAV_SOCKET_NOT_WORLD_ACCESSIBLE", actual_mode & 0o007 == 0, oct(actual_mode))
    checks.record("CLAMAV_SOCKET_MODE_660", actual_mode == expected_mode, oct(actual_mode))
    checks.record(
        "CLAMAV_SOCKET_OWNER_GROUP",
        actual_owner == expected_owner and actual_group == expected_group,
        f"{actual_owner}:{actual_group}",
    )

    if not backend_user:
        checks.record("BACKEND_SERVICE_USER_IN_CLAMAV_GROUP", False, "BACKEND_SERVICE_USER missing")
        return

    try:
        user_entry = pwd.getpwnam(backend_user)
        user_groups = {grp.getgrgid(user_entry.pw_gid).gr_name}
        user_groups.update(group.gr_name for group in grp.getgrall() if backend_user in group.gr_mem)
    except KeyError as exc:
        checks.record("BACKEND_SERVICE_USER_IN_CLAMAV_GROUP", False, str(exc))
        return
    checks.record(
        "BACKEND_SERVICE_USER_IN_CLAMAV_GROUP",
        expected_group in user_groups,
        ",".join(sorted(user_groups)),
    )


def _check_eicar_detection(checks: _CheckAccumulator, runner: CommandRunner) -> None:
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(prefix="eicar-upload-security-", delete=False) as temp_file:
            temp_file.write(EICAR)
            temp_path = temp_file.name
        code, stdout, stderr = runner(["clamdscan", "--fdpass", temp_path], 20)
        output = f"{stdout}\n{stderr}"
        checks.record("EICAR_DETECTED", code == 1 and "FOUND" in output, output.strip())
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink()
            except OSError:
                pass


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


def _check_upload_cleanup_timer(
    checks: _CheckAccumulator,
    env: Mapping[str, str],
    runner: CommandRunner,
) -> None:
    timer_unit = env.get("UPLOAD_CLEANUP_TIMER_UNIT") or "stoody-upload-cleanup.timer"
    code, stdout, stderr = runner(["systemctl", "is-enabled", timer_unit], 10)
    checks.record(
        "UPLOAD_CLEANUP_TIMER_ENABLED",
        code == 0 and stdout.strip() == "enabled",
        (stdout or stderr).strip(),
    )
    code, stdout, stderr = runner(["systemctl", "is-active", timer_unit], 10)
    checks.record(
        "UPLOAD_CLEANUP_TIMER_ACTIVE",
        code == 0 and stdout.strip() == "active",
        (stdout or stderr).strip(),
    )


def _check_upload_cleanup_script_dry_run(
    checks: _CheckAccumulator,
    env: Mapping[str, str],
    runner: CommandRunner,
) -> None:
    root = Path(env.get("UPLOAD_PRIVATE_LOCAL_DIR") or "/var/lib/stoody/uploads")
    code, stdout, stderr = runner(
        [sys.executable, "scripts/cleanup_upload_storage.py", "--root", str(root)],
        60,
    )
    output = (stdout or stderr).strip()
    ok = False
    if code == 0:
        try:
            payload = json.loads(stdout)
            ok = payload.get("dry_run") is True
        except json.JSONDecodeError:
            ok = False
    checks.record("UPLOAD_CLEANUP_SCRIPT_DRY_RUN", ok, output)


def _check_upload_cleanup_service_not_failed(
    checks: _CheckAccumulator,
    env: Mapping[str, str],
    runner: CommandRunner,
) -> None:
    timer_unit = env.get("UPLOAD_CLEANUP_TIMER_UNIT") or "stoody-upload-cleanup.timer"
    service_unit = env.get("UPLOAD_CLEANUP_SERVICE_UNIT") or _service_unit_for_timer(timer_unit)
    code, stdout, stderr = runner(["systemctl", "is-failed", service_unit], 10)
    state = (stdout or stderr).strip()
    normalized = state.lower()
    ok = code != 0 and normalized not in {"failed", "not-found"} and "could not be found" not in normalized
    checks.record("UPLOAD_CLEANUP_SERVICE_NOT_FAILED", ok, state)


def _service_unit_for_timer(timer_unit: str) -> str:
    if timer_unit.endswith(".timer"):
        return f"{timer_unit[:-6]}.service"
    return f"{timer_unit}.service"


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
    parser.add_argument("--status-output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_upload_security_deploy_validation(nginx_config_paths=args.nginx_config)
    if args.status_output is not None:
        _write_status_output(args.status_output, result)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.ok else 1


def _write_status_output(path: Path, result: DeployValidationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_epoch": time.time(),
        "result": result.to_dict(),
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
