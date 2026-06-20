from pathlib import Path

from core.upload_security.legacy_audit import (
    audit_local_uploads,
    migrate_verified_legacy_uploads,
    write_legacy_audit_report,
)


def test_legacy_audit_marks_files_unverified(tmp_path):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    legacy_pdf = upload_root / "paper.pdf"
    legacy_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    records = audit_local_uploads(upload_root)

    assert len(records) == 1
    assert records[0].detected_magic_type == "pdf"
    assert records[0].likely_policy_id == "pdf_document"
    assert records[0].status == "legacy_unverified"
    assert len(records[0].sha256) == 64


def test_legacy_audit_writes_json_report(tmp_path):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    (upload_root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\nabc")
    report_path = tmp_path / "legacy-report.json"

    records = audit_local_uploads(upload_root)
    write_legacy_audit_report(records, report_path)

    assert "legacy_unverified" in report_path.read_text(encoding="utf-8")


def test_legacy_migration_only_plans_clean_hashes(tmp_path):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    (upload_root / "paper.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    records = audit_local_uploads(upload_root)

    no_clean_plan = migrate_verified_legacy_uploads(records, clean_sha256=set(), destination_root=tmp_path / "private")
    clean_plan = migrate_verified_legacy_uploads(
        records,
        clean_sha256={records[0].sha256},
        destination_root=tmp_path / "private",
    )

    assert no_clean_plan == []
    assert clean_plan[0]["destination"].endswith(str(Path("paper.pdf")))
    assert str(Path("clean")) in clean_plan[0]["destination"]
