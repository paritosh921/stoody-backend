"""
Shared fixtures for ExamPen E2E pipeline tests (L5).

Provides Docker Compose startup/teardown (or connection to an already-running
stack), plus client fixtures for NATS JetStream, PostgreSQL/TimescaleDB,
and MinIO (S3-compatible object store).

Usage:
    pytest test-suite/pipeline-tests/ -m e2e
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from factories import (
    AIResultFactory,
    ExamFactory,
    ScoreFactory,
    StudentFactory,
    StrokeFactory,
)

# ---------------------------------------------------------------------------
# Environment defaults — override via env vars when running against a custom
# Docker Compose deployment.
# ---------------------------------------------------------------------------

NATS_URL = os.getenv("EXAMPEN_NATS_URL", "nats://localhost:4222")
PG_DSN = os.getenv(
    "EXAMPEN_PG_DSN",
    "postgresql+asyncpg://exampen:exampen@localhost:5432/exampen",
)
MINIO_ENDPOINT = os.getenv("EXAMPEN_MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("EXAMPEN_MINIO_ACCESS_KEY", "exampen")
MINIO_SECRET_KEY = os.getenv("EXAMPEN_MINIO_SECRET_KEY", "exampen123")
MINIO_BUCKET = os.getenv("EXAMPEN_MINIO_BUCKET", "exampen-pages")

SCORE_ENGINE_URL = os.getenv(
    "EXAMPEN_SCORE_ENGINE_URL", "http://localhost:8003"
)
REVIEW_URL = os.getenv("EXAMPEN_REVIEW_URL", "http://localhost:8005")
COPY_UPLOAD_URL = os.getenv(
    "EXAMPEN_COPY_UPLOAD_URL", "http://localhost:8006"
)
TEACHER_BFF_URL = os.getenv(
    "EXAMPEN_TEACHER_BFF_URL", "http://localhost:8010"
)
STUDENT_BFF_URL = os.getenv(
    "EXAMPEN_STUDENT_BFF_URL", "http://localhost:8011"
)
STOODY_WEBHOOK_URL = os.getenv(
    "EXAMPEN_STOODY_WEBHOOK_URL", "http://localhost:9090"
)
EXAM_ORCH_URL = os.getenv(
    "EXAMPEN_EXAM_ORCH_URL", "http://localhost:8001"
)
ANALYTICS_URL = os.getenv(
    "EXAMPEN_ANALYTICS_URL", "http://localhost:8007"
)

# Default timeout (seconds) when waiting for a downstream NATS event.
EVENT_TIMEOUT = int(os.getenv("EXAMPEN_EVENT_TIMEOUT", "30"))

# Fixtures directory relative to this file.
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


# ── pytest markers ────────────────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "e2e: End-to-end pipeline test (L5)")


# ── Event loop ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop shared by all async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── NATS client ───────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="session")
async def nats_client():
    """Session-scoped NATS connection. Auto-drains on teardown."""
    import nats as nats_pkg

    nc = await nats_pkg.connect(NATS_URL)
    yield nc
    await nc.drain()


@pytest_asyncio.fixture(scope="session")
async def nats_js(nats_client):
    """JetStream context bound to the session NATS connection."""
    js = nats_client.jetstream()
    yield js


# ── PostgreSQL / TimescaleDB ──────────────────────────────────────────────

@pytest_asyncio.fixture(scope="session")
async def pg_pool():
    """Session-scoped asyncpg connection pool."""
    import asyncpg

    dsn = PG_DSN.replace("postgresql+asyncpg://", "postgresql://")
    pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
    yield pool
    await pool.close()


# ── MinIO / S3 ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def minio_client():
    """Session-scoped MinIO client."""
    from minio import Minio

    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)
    yield client


# ── HTTP helper ───────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="session")
async def http_session():
    """Session-scoped aiohttp ClientSession for REST calls."""
    import aiohttp

    session = aiohttp.ClientSession()
    yield session
    await session.close()


# ── NATS event helper ─────────────────────────────────────────────────────

@dataclass
class NatsEventWaiter:
    """Subscribe to a NATS subject and wait for a matching event."""

    nats_client: Any

    async def wait_for_event(
        self,
        subject: str,
        *,
        filter_fn: Any | None = None,
        timeout: float = EVENT_TIMEOUT,
    ) -> dict:
        """Wait for a single event on *subject*.

        Raises ``asyncio.TimeoutError`` if no match within *timeout*.
        """
        fut: asyncio.Future[dict] = asyncio.get_event_loop().create_future()

        async def _handler(msg):
            data = json.loads(msg.data.decode())
            if filter_fn is None or filter_fn(data):
                if not fut.done():
                    fut.set_result(data)

        sub = await self.nats_client.subscribe(subject, cb=_handler)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            await sub.unsubscribe()

    async def collect_events(
        self,
        subject: str,
        *,
        count: int,
        timeout: float = EVENT_TIMEOUT,
    ) -> list[dict]:
        """Collect *count* events from *subject* within *timeout*."""
        collected: list[dict] = []
        done = asyncio.Event()

        async def _handler(msg):
            data = json.loads(msg.data.decode())
            collected.append(data)
            if len(collected) >= count:
                done.set()

        sub = await self.nats_client.subscribe(subject, cb=_handler)
        try:
            await asyncio.wait_for(done.wait(), timeout=timeout)
            return collected
        finally:
            await sub.unsubscribe()


@pytest_asyncio.fixture
async def event_waiter(nats_client) -> NatsEventWaiter:
    return NatsEventWaiter(nats_client=nats_client)


# ── Factory fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def exam_factory() -> ExamFactory:
    return ExamFactory()


@pytest.fixture
def student_factory() -> StudentFactory:
    return StudentFactory()


@pytest.fixture
def stroke_factory() -> StrokeFactory:
    return StrokeFactory()


@pytest.fixture
def ai_result_factory() -> AIResultFactory:
    return AIResultFactory()


@pytest.fixture
def score_factory() -> ScoreFactory:
    return ScoreFactory()


# ── Fixture-file loaders ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def fixture_students() -> list[dict]:
    return json.loads((FIXTURES_DIR / "students.json").read_text())


@pytest.fixture(scope="session")
def fixture_exam_math() -> dict:
    return json.loads(
        (FIXTURES_DIR / "exams" / "exam_mathematics_01.json").read_text()
    )


@pytest.fixture(scope="session")
def fixture_ai_results() -> list[dict]:
    return json.loads((FIXTURES_DIR / "ai_results.json").read_text())


@pytest.fixture(scope="session")
def fixture_scores() -> list[dict]:
    return json.loads((FIXTURES_DIR / "scores.json").read_text())


@pytest.fixture(scope="session")
def fixture_objections() -> list[dict]:
    return json.loads((FIXTURES_DIR / "objections.json").read_text())


# ── NATS publish helper ─────────────────────────────────────────────────

@pytest_asyncio.fixture
async def publish_event(nats_client):
    """Async callable: ``await publish_event(subject, payload_dict)``."""

    async def _publish(subject: str, payload: dict) -> None:
        await nats_client.publish(
            subject, json.dumps(payload).encode()
        )

    return _publish
