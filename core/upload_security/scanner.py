"""Malware scanner abstraction for quarantined uploads."""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from config_async import settings


@dataclass(frozen=True)
class ScanResult:
    status: str
    scanner_name: str = "clamav"
    scanner_version: str | None = None
    signature: str | None = None
    error: str | None = None

    @classmethod
    def clean(cls, *, scanner_name: str = "clamav", scanner_version: str | None = None) -> "ScanResult":
        return cls(status="clean", scanner_name=scanner_name, scanner_version=scanner_version)

    @classmethod
    def rejected(
        cls,
        signature: str,
        *,
        scanner_name: str = "clamav",
        scanner_version: str | None = None,
    ) -> "ScanResult":
        return cls(
            status="rejected",
            scanner_name=scanner_name,
            scanner_version=scanner_version,
            signature=signature,
        )

    @classmethod
    def scan_failed(
        cls,
        error: str,
        *,
        scanner_name: str = "clamav",
        scanner_version: str | None = None,
    ) -> "ScanResult":
        return cls(
            status="scan_failed",
            scanner_name=scanner_name,
            scanner_version=scanner_version,
            error=error,
        )


class ClamAVScanner:
    """ClamAV scanner using clamdscan against a quarantined filesystem path."""

    def _build_clamdscan_command(self, path: str | Path) -> tuple[list[str], Callable[[], None]]:
        config_path: Path | None = None
        command = ["clamdscan", "--fdpass", "--no-summary"]

        if settings.CLAMAV_SOCKET:
            config_path = _write_clamdscan_config(f"LocalSocket {settings.CLAMAV_SOCKET}\n")
        elif os.getenv("CLAMD_HOST") or os.getenv("CLAMD_PORT"):
            config_path = _write_clamdscan_config(
                f"TCPSocket {int(settings.CLAMD_PORT)}\nTCPAddr {settings.CLAMD_HOST}\n"
            )

        if config_path is not None:
            command.append(f"--config-file={config_path}")

        command.append(str(path))

        def cleanup() -> None:
            if config_path is not None:
                try:
                    config_path.unlink(missing_ok=True)
                except Exception:
                    pass

        return command, cleanup

    async def health(self) -> dict[str, object]:
        result = await self.scan_bytes(b"stoody-upload-scanner-health", "health.txt", "scanner_health")
        return {
            "enabled": settings.UPLOAD_AV_ENABLED,
            "fail_closed": settings.UPLOAD_AV_FAIL_CLOSED,
            "available": result.status == "clean",
            "status": result.status,
            "scanner_name": result.scanner_name,
            "scanner_version": result.scanner_version,
            "freshclam_age_seconds": clamav_signature_age_seconds(),
            "error": result.error,
        }

    async def scan_bytes(self, data: bytes, filename: str, policy_id: str) -> ScanResult:
        suffix = Path(filename or "upload.bin").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            return await self.scan_path(tmp_path, filename=filename, policy_id=policy_id)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    async def scan_path(self, path: str | Path, *, filename: str, policy_id: str) -> ScanResult:
        if not settings.UPLOAD_AV_ENABLED:
            if settings.DEBUG_MODE:
                return ScanResult.clean(scanner_name="disabled-dev")
            if settings.UPLOAD_AV_FAIL_CLOSED:
                return ScanResult.scan_failed("Malware scanner disabled outside DEBUG_MODE")
            return ScanResult.clean(scanner_name="disabled-nonprod")

        path = Path(path)
        if not path.exists():
            return ScanResult.scan_failed(f"Quarantine file missing: {path}")

        command, cleanup = self._build_clamdscan_command(path)
        env = os.environ.copy()
        version = await _get_clamdscan_version()
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=settings.UPLOAD_SCANNER_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ScanResult.scan_failed(
                    f"clamdscan timed out after {settings.UPLOAD_SCANNER_TIMEOUT_SECONDS:g} seconds",
                    scanner_name="clamav",
                    scanner_version=version,
                )
        except Exception as exc:
            return ScanResult.scan_failed(str(exc), scanner_name="clamav", scanner_version=version)
        finally:
            cleanup()
        output = (stdout + stderr).decode("utf-8", errors="replace").strip()

        if process.returncode == 0:
            return ScanResult.clean(scanner_name="clamav", scanner_version=version)
        if process.returncode == 1:
            signature = output.rsplit(":", 1)[-1].replace("FOUND", "").strip() or "malware"
            return ScanResult.rejected(signature, scanner_name="clamav", scanner_version=version)
        return ScanResult.scan_failed(
            output or f"clamdscan exited {process.returncode}",
            scanner_name="clamav",
            scanner_version=version,
        )


def _write_clamdscan_config(contents: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", delete=False, suffix=".conf", encoding="utf-8")
    try:
        handle.write(contents)
        return Path(handle.name)
    finally:
        handle.close()


async def _get_clamdscan_version() -> str | None:
    try:
        process = await asyncio.create_subprocess_exec(
            "clamdscan",
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
    except Exception:
        return None
    if process.returncode != 0:
        return None
    output = (stdout + stderr).decode("utf-8", errors="replace").strip()
    return output or None


def clamav_signature_age_seconds(
    database_dir: str | Path = "/var/lib/clamav",
    *,
    now: datetime | None = None,
) -> int | None:
    signature_dir = Path(database_dir)
    signature_files = [
        signature_dir / name
        for name in ("daily.cvd", "daily.cld", "main.cvd", "main.cld", "bytecode.cvd", "bytecode.cld")
    ]
    mtimes = []
    for path in signature_files:
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            continue
    if not mtimes:
        return None
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    newest_signature = datetime.fromtimestamp(max(mtimes), tz=timezone.utc)
    return max(int((current_time.astimezone(timezone.utc) - newest_signature).total_seconds()), 0)
