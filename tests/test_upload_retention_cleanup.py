import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.upload_security.cleanup import cleanup_private_upload_storage


def _write_file(path: Path, data: bytes, *, mtime: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    timestamp = mtime.timestamp()
    os.utime(path, (timestamp, timestamp))


def test_cleanup_dry_run_reports_expired_files_without_deleting(tmp_path):
    now = datetime(2026, 6, 20, tzinfo=timezone.utc)
    rejected = tmp_path / "rejected" / "skb_ciel" / "upload-1" / "bad.pdf"
    quarantine = tmp_path / "quarantine" / "skb_ciel" / "upload-2" / "pending.pdf"
    clean = tmp_path / "clean" / "skb_ciel" / "pdf_document" / "upload-3" / "paper.pdf"
    _write_file(rejected, b"bad", mtime=now - timedelta(days=31))
    _write_file(quarantine, b"pending", mtime=now - timedelta(hours=25))
    _write_file(clean, b"clean", mtime=now - timedelta(days=365))

    result = cleanup_private_upload_storage(
        tmp_path,
        now=now,
        rejected_retention_days=30,
        quarantine_retention_hours=24,
        dry_run=True,
    )

    assert result.candidate_files == 2
    assert result.deleted_files == 0
    assert result.reclaimed_bytes == 0
    assert rejected.exists()
    assert quarantine.exists()
    assert clean.exists()


def test_cleanup_deletes_expired_quarantine_and_rejected_but_not_clean(tmp_path):
    now = datetime(2026, 6, 20, tzinfo=timezone.utc)
    old_rejected = tmp_path / "rejected" / "skb_ciel" / "upload-1" / "bad.pdf"
    old_rejected_sidecar = Path(f"{old_rejected}.metadata.json")
    fresh_rejected = tmp_path / "rejected" / "skb_ciel" / "upload-2" / "new.pdf"
    old_quarantine = tmp_path / "quarantine" / "skb_ciel" / "upload-3" / "pending.pdf"
    clean = tmp_path / "clean" / "skb_ciel" / "pdf_document" / "upload-4" / "paper.pdf"
    _write_file(old_rejected, b"bad", mtime=now - timedelta(days=31))
    _write_file(old_rejected_sidecar, b"{}", mtime=now - timedelta(days=31))
    _write_file(fresh_rejected, b"new", mtime=now - timedelta(days=2))
    _write_file(old_quarantine, b"pending", mtime=now - timedelta(hours=25))
    _write_file(clean, b"clean", mtime=now - timedelta(days=365))

    result = cleanup_private_upload_storage(
        tmp_path,
        now=now,
        rejected_retention_days=30,
        quarantine_retention_hours=24,
        dry_run=False,
    )

    assert result.deleted_files == 3
    assert result.reclaimed_bytes == len(b"bad") + len(b"{}") + len(b"pending")
    assert not old_rejected.exists()
    assert not old_rejected_sidecar.exists()
    assert not old_quarantine.exists()
    assert fresh_rejected.exists()
    assert clean.exists()


def test_cleanup_refuses_clean_deletion_without_explicit_configuration(tmp_path):
    now = datetime(2026, 6, 20, tzinfo=timezone.utc)
    clean = tmp_path / "clean" / "skb_ciel" / "pdf_document" / "upload-1" / "paper.pdf"
    _write_file(clean, b"clean", mtime=now - timedelta(days=365))

    result = cleanup_private_upload_storage(
        tmp_path,
        now=now,
        rejected_retention_days=0,
        quarantine_retention_hours=0,
        dry_run=False,
    )

    assert result.deleted_files == 0
    assert clean.exists()


def test_cleanup_script_runs_directly_from_repo_root(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/cleanup_upload_storage.py",
            "--root",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"dry_run": true' in result.stdout
