from __future__ import annotations

import argparse

import pytest

from scripts import migrate_pcr_v12_contracts as migration_script


@pytest.mark.asyncio
async def test_apply_requires_exact_confirmation_before_database_initialization(
    monkeypatch,
):
    initialized = False

    async def fail_if_initialized(_self):
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(migration_script.DatabaseManager, "initialize", fail_if_initialized)
    args = argparse.Namespace(
        tenant_db="skb_test",
        exam_id="EXAM-1",
        apply=True,
        confirm="WRONG",
        requested_by="OPS-1",
    )

    result = await migration_script.run(args)

    assert result == 2
    assert initialized is False


def test_parser_is_read_only_by_default():
    args = migration_script.build_parser().parse_args([])

    assert args.apply is False
    assert args.confirm is None
    assert args.requested_by == "operations:pcr-v12-v13-migration"
