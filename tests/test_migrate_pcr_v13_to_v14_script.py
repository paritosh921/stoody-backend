from __future__ import annotations

import argparse

import pytest

from scripts import migrate_pcr_v13_to_v14 as migration_script


def _args(**overrides):
    values = {
        "tenant_db": "skb_test",
        "exam_id": "EXAM-1",
        "apply": False,
        "confirm": None,
        "requested_by": "OPS-1",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.asyncio
async def test_dry_run_is_default_and_never_calls_apply(monkeypatch, capsys):
    calls: list[str] = []

    async def initialize(_self):
        calls.append("initialize")

    async def close(_self):
        calls.append("close")

    async def get_tenant_db(_self, _db_name):
        return object()

    async def inspect(_db, *, db_name, exam_id=None):
        calls.append("inspect")
        return [{"db_name": db_name, "exam_id": exam_id, "eligible": True}]

    async def apply(*_args, **_kwargs):
        calls.append("apply")
        raise AssertionError("dry-run must not invoke the migration")

    monkeypatch.setattr(migration_script.DatabaseManager, "initialize", initialize)
    monkeypatch.setattr(migration_script.DatabaseManager, "close", close)
    monkeypatch.setattr(migration_script.DatabaseManager, "get_tenant_db", get_tenant_db)
    monkeypatch.setattr(migration_script, "inspect_v13_contracts", inspect)
    monkeypatch.setattr(migration_script, "migrate_v13_exam_to_v14", apply)

    result = await migration_script.run(_args())

    assert result == 0
    assert calls == ["initialize", "inspect", "close"]
    assert '"mode": "dry-run"' in capsys.readouterr().out


@pytest.mark.asyncio
async def test_apply_requires_exact_confirmation_before_database_initialization(
    monkeypatch,
):
    initialized = False

    async def fail_if_initialized(_self):
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(migration_script.DatabaseManager, "initialize", fail_if_initialized)

    result = await migration_script.run(_args(apply=True, confirm="WRONG"))

    assert result == 2
    assert initialized is False


@pytest.mark.asyncio
async def test_apply_with_exact_confirmation_reaches_mocked_migration(monkeypatch, capsys):
    calls: list[tuple] = []
    tenant_db = object()

    async def initialize(_self):
        calls.append(("initialize",))

    async def close(_self):
        calls.append(("close",))

    async def get_tenant_db(_self, db_name):
        calls.append(("get_tenant_db", db_name))
        return tenant_db

    async def inspect(_db, *, db_name, exam_id=None):
        calls.append(("inspect", db_name, exam_id))
        return [{"db_name": db_name, "exam_id": exam_id, "eligible": True}]

    async def apply(
        _db,
        *,
        db_name,
        exam_id,
        requested_by,
        confirmation_token,
    ):
        calls.append(
            (
                "apply",
                db_name,
                exam_id,
                requested_by,
                confirmation_token,
            )
        )
        return {
            "db_name": db_name,
            "exam_id": exam_id,
            "status": "migrated",
            "queued_job_count": 1,
        }

    monkeypatch.setattr(migration_script.DatabaseManager, "initialize", initialize)
    monkeypatch.setattr(migration_script.DatabaseManager, "close", close)
    monkeypatch.setattr(migration_script.DatabaseManager, "get_tenant_db", get_tenant_db)
    monkeypatch.setattr(migration_script, "inspect_v13_contracts", inspect)
    monkeypatch.setattr(migration_script, "migrate_v13_exam_to_v14", apply)

    result = await migration_script.run(
        _args(
            apply=True,
            confirm=migration_script.CONFIRMATION_TOKEN,
            requested_by="OPS-AUTHORIZED",
        )
    )

    assert result == 0
    assert calls == [
        ("initialize",),
        ("get_tenant_db", "skb_test"),
        ("inspect", "skb_test", "EXAM-1"),
        ("get_tenant_db", "skb_test"),
        (
            "apply",
            "skb_test",
            "EXAM-1",
            "OPS-AUTHORIZED",
            migration_script.CONFIRMATION_TOKEN,
        ),
        ("close",),
    ]
    assert '"mode": "apply"' in capsys.readouterr().out


def test_parser_is_read_only_by_default():
    args = migration_script.build_parser().parse_args([])

    assert args.apply is False
    assert args.confirm is None
    assert args.requested_by == "operations:pcr-v13-v14-migration"
