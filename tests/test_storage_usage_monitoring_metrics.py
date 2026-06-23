import asyncio
from datetime import datetime, timezone

from core.ai_usage.metrics_exporter import public_identity_ref
from core.storage_usage.metrics_exporter import build_storage_usage_metric_rows
from core import observability


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def limit(self, _length):
        return self

    async def to_list(self, length=None):
        return self._rows[:length] if length else list(self._rows)


class FakeCollection:
    def __init__(self, rows):
        self._rows = rows

    def find(self, *_args, **_kwargs):
        return FakeCursor(self._rows)


class FakeDatabase:
    def __init__(self, *, stats=None, tenants=None):
        self._stats = stats or {}
        self._tenants = tenants or []

    async def command(self, command_name, **_kwargs):
        if command_name != "dbStats":
            raise AssertionError(f"unexpected command {command_name}")
        return self._stats

    def __getitem__(self, name):
        if name != "tenants":
            raise KeyError(name)
        return FakeCollection(self._tenants)


class FakeDbManager:
    def __init__(self, *, master_db=None, b2c_db=None, tenant_dbs=None):
        self._master_db = master_db
        self._b2c_db = b2c_db
        self._tenant_dbs = tenant_dbs or {}

    async def get_master_db(self):
        return self._master_db

    async def get_b2c_db(self):
        return self._b2c_db

    async def get_tenant_db(self, db_name):
        return self._tenant_dbs.get(db_name)


def test_private_upload_rows_group_bytes_by_prefix_and_hashed_tenant(tmp_path):
    (tmp_path / "clean" / "tenant-alpha-db" / "pdf_document" / "u1").mkdir(parents=True)
    (tmp_path / "clean" / "tenant-alpha-db" / "pdf_document" / "u1" / "doc.pdf").write_bytes(b"a" * 10)
    (tmp_path / "clean" / "tenant-alpha-db" / "pdf_document" / "u1" / "doc.pdf.metadata.json").write_bytes(b"{}")
    (tmp_path / "rejected" / "tenant-alpha-db" / "u2").mkdir(parents=True)
    (tmp_path / "rejected" / "tenant-alpha-db" / "u2" / "bad.bin").write_bytes(b"b" * 5)
    (tmp_path / "derived" / "tenant-beta-db" / "templates").mkdir(parents=True)
    (tmp_path / "derived" / "tenant-beta-db" / "templates" / "template.pdf").write_bytes(b"c" * 11)

    rows = asyncio.run(
        build_storage_usage_metric_rows(
            FakeDbManager(),
            local_root=tmp_path,
            now=datetime(2026, 6, 24, tzinfo=timezone.utc),
            cache_ttl_seconds=0,
        )
    )

    alpha_ref = public_identity_ref("tenant-alpha-db", prefix="tenant")
    beta_ref = public_identity_ref("tenant-beta-db", prefix="tenant")
    assert {
        "metric": "tenant_storage",
        "labels": {
            "tenant_ref": alpha_ref,
            "storage": "private_uploads",
            "prefix": "clean",
            "kind": "objects_used",
        },
        "value": 12.0,
    } in rows
    assert {
        "metric": "tenant_storage",
        "labels": {
            "tenant_ref": alpha_ref,
            "storage": "private_uploads",
            "prefix": "rejected",
            "kind": "objects_used",
        },
        "value": 5.0,
    } in rows
    assert {
        "metric": "tenant_storage",
        "labels": {
            "tenant_ref": beta_ref,
            "storage": "private_uploads",
            "prefix": "derived",
            "kind": "objects_used",
        },
        "value": 11.0,
    } in rows
    assert any(
        row["metric"] == "backend_storage"
        and row["labels"] == {"storage": "private_uploads", "kind": "objects_used"}
        and row["value"] == 28.0
        for row in rows
    )
    assert any(
        row["metric"] == "backend_storage"
        and row["labels"] == {"storage": "private_uploads", "kind": "filesystem_capacity"}
        and row["value"] > 0
        for row in rows
    )
    rendered = repr(rows)
    assert "tenant-alpha-db" not in rendered
    assert "tenant-beta-db" not in rendered


def test_mongodb_storage_rows_include_total_and_per_tenant_without_raw_db_names(tmp_path):
    master = FakeDatabase(
        stats={"dataSize": 100, "storageSize": 200, "indexSize": 50},
        tenants=[
            {"tenant_id": "TENANT-A", "db_name": "tenant_a_private_db"},
            {"tenant_id": "TENANT-B", "db_name": "tenant_b_private_db"},
        ],
    )
    b2c = FakeDatabase(stats={"dataSize": 10, "storageSize": 20, "indexSize": 5})
    tenants = {
        "tenant_a_private_db": FakeDatabase(stats={"dataSize": 1000, "storageSize": 2000, "indexSize": 500}),
        "tenant_b_private_db": FakeDatabase(stats={"dataSize": 3000, "storageSize": 5000, "indexSize": 700}),
    }

    rows = asyncio.run(
        build_storage_usage_metric_rows(
            FakeDbManager(master_db=master, b2c_db=b2c, tenant_dbs=tenants),
            local_root=tmp_path,
            now=datetime(2026, 6, 24, tzinfo=timezone.utc),
            cache_ttl_seconds=0,
        )
    )

    tenant_a_ref = public_identity_ref("TENANT-A", prefix="tenant")
    assert {
        "metric": "mongodb_storage",
        "labels": {
            "database_role": "tenant",
            "tenant_ref": tenant_a_ref,
            "kind": "total",
        },
        "value": 2500.0,
    } in rows
    assert {
        "metric": "mongodb_storage",
        "labels": {
            "database_role": "master",
            "tenant_ref": "platform",
            "kind": "total",
        },
        "value": 250.0,
    } in rows
    assert {
        "metric": "mongodb_storage",
        "labels": {
            "database_role": "b2c",
            "tenant_ref": "b2c",
            "kind": "total",
        },
        "value": 25.0,
    } in rows
    rendered = repr(rows)
    assert "tenant_a_private_db" not in rendered
    assert "tenant_b_private_db" not in rendered


def test_mongodb_storage_rows_export_cluster_capacity_when_db_stats_exposes_it(tmp_path):
    master = FakeDatabase(
        stats={
            "dataSize": 100,
            "storageSize": 200,
            "indexSize": 50,
            "fsUsedSize": 4096,
            "fsTotalSize": 8192,
        },
        tenants=[],
    )

    rows = asyncio.run(
        build_storage_usage_metric_rows(
            FakeDbManager(master_db=master),
            local_root=tmp_path,
            now=datetime(2026, 6, 24, tzinfo=timezone.utc),
            cache_ttl_seconds=0,
        )
    )

    assert {
        "metric": "mongodb_storage",
        "labels": {
            "database_role": "cluster",
            "tenant_ref": "platform",
            "kind": "filesystem_used",
        },
        "value": 4096.0,
    } in rows
    assert {
        "metric": "mongodb_storage",
        "labels": {
            "database_role": "cluster",
            "tenant_ref": "platform",
            "kind": "filesystem_capacity",
        },
        "value": 8192.0,
    } in rows
    assert {
        "metric": "mongodb_storage",
        "labels": {
            "database_role": "cluster",
            "tenant_ref": "platform",
            "kind": "filesystem_free",
        },
        "value": 4096.0,
    } in rows


def test_storage_usage_gauges_can_be_cleared_between_scrapes():
    observability.set_storage_usage_metric(
        "tenant_storage",
        {
            "tenant_ref": "tenant_abc",
            "storage": "private_uploads",
            "prefix": "clean",
            "kind": "objects_used",
        },
        50.0,
    )
    assert _has_sample(
        observability.TENANT_STORAGE_BYTES,
        {
            "tenant_ref": "tenant_abc",
            "storage": "private_uploads",
            "prefix": "clean",
            "kind": "objects_used",
        },
    )

    observability.clear_storage_usage_metrics()

    assert not _has_sample(
        observability.TENANT_STORAGE_BYTES,
        {
            "tenant_ref": "tenant_abc",
            "storage": "private_uploads",
            "prefix": "clean",
            "kind": "objects_used",
        },
    )


def _has_sample(metric, labels):
    for family in metric.collect():
        for sample in family.samples:
            if all(sample.labels.get(key) == value for key, value in labels.items()):
                return True
    return False
