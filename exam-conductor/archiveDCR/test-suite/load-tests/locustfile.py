"""ExamPen load test scenarios — Locust.

Scenarios:
  1. Stroke ingestion burst   — 10K students x 336KB each = 3.3GB burst
  2. Score query load          — 500 concurrent teachers viewing class scores
  3. Student portal load       — 5000 concurrent students checking scores
  4. Mixed workload            — combination of all above

Usage:
  # Interactive (web UI)
  locust -f locustfile.py --host http://localhost:8080

  # Headless (CI)
  locust -f locustfile.py --host http://localhost:8080 \
    --headless -u 500 -r 50 --run-time 5m \
    --csv results/run

Reference: CLAUDE.md §Testing Strategy, FAILURE_MITIGATION_REGISTER.md §A8.4
"""

from __future__ import annotations

import base64
import json
import os
import random
import string
import uuid
from pathlib import Path

from locust import HttpUser, between, events, tag, task
from locust.runners import MasterRunner

# ---------------------------------------------------------------------------
# Configuration — override via environment variables
# ---------------------------------------------------------------------------
BASE_URL = os.getenv("EXAMPEN_BASE_URL", "http://localhost:8080")
FIXTURE_DIR = Path(os.getenv(
    "EXAMPEN_FIXTURE_DIR",
    str(Path(__file__).resolve().parent.parent / "fixtures"),
))

# Performance budget thresholds (seconds)
STROKE_INGEST_P95 = float(os.getenv("THRESHOLD_STROKE_P95", "2.0"))
SCORE_QUERY_P95 = float(os.getenv("THRESHOLD_SCORE_P95", "2.0"))
STUDENT_QUERY_P95 = float(os.getenv("THRESHOLD_STUDENT_P95", "2.0"))

# Burst sizing — matches A8.4: 10K students x 336KB = 3.3GB
CHUNK_SIZE_BYTES = 8_400  # 600 coordinate frames x 14 bytes = 8400
CHUNKS_PER_PEN = 40       # 336KB / 8.4KB = 40 chunks
TOTAL_STUDENTS = int(os.getenv("LOAD_TOTAL_STUDENTS", "10000"))
TOTAL_TEACHERS = int(os.getenv("LOAD_TOTAL_TEACHERS", "500"))

# Simulated exam / entity pool sizes
EXAM_POOL_SIZE = 50
STUDENT_POOL_SIZE = 200
TEACHER_POOL_SIZE = 20
QUESTION_POOL_SIZE = 10

# ---------------------------------------------------------------------------
# Seed fixtures loader
# ---------------------------------------------------------------------------
_exam_ids: list[str] = []
_student_ids: list[str] = []
_pen_macs: list[str] = []
_teacher_tokens: list[str] = []
_student_tokens: list[str] = []


def _load_fixtures() -> None:
    """Load IDs from seed fixtures if available, otherwise generate."""
    global _exam_ids, _student_ids, _pen_macs

    exam_dir = FIXTURE_DIR / "exams"
    if exam_dir.exists():
        for p in sorted(exam_dir.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            _exam_ids.append(data["id"])

    student_file = FIXTURE_DIR / "students.json"
    if student_file.exists():
        students = json.loads(student_file.read_text(encoding="utf-8"))
        _student_ids.extend(s["id"] for s in students)

    # Pad pools to required size with synthetic IDs
    while len(_exam_ids) < EXAM_POOL_SIZE:
        _exam_ids.append(str(uuid.uuid4()))
    while len(_student_ids) < STUDENT_POOL_SIZE:
        _student_ids.append(str(uuid.uuid4()))

    # Generate MAC addresses for pens
    for i in range(STUDENT_POOL_SIZE):
        _pen_macs.append(
            ":".join(f"{b:02X}" for b in [0xAA, 0xBB, 0xCC,
                                           (i >> 16) & 0xFF,
                                           (i >> 8) & 0xFF,
                                           i & 0xFF])
        )


def _generate_mock_jwt(role: str, user_id: str) -> str:
    """Generate a mock JWT for load testing (not cryptographically valid).

    In a real run the test harness should provide valid tokens via
    EXAMPEN_TEACHER_TOKENS / EXAMPEN_STUDENT_TOKENS env vars or a
    token-provisioning script.
    """
    header = base64.urlsafe_b64encode(
        b'{"alg":"HS256","typ":"JWT"}'
    ).rstrip(b"=").decode()
    payload_obj = {
        "sub": user_id,
        "role": role,
        "tenant_id": "load-test-tenant",
        "exp": 9999999999,
    }
    payload = base64.urlsafe_b64encode(
        json.dumps(payload_obj).encode()
    ).rstrip(b"=").decode()
    sig = base64.urlsafe_b64encode(b"mock-signature").rstrip(b"=").decode()
    return f"{header}.{payload}.{sig}"


def _init_tokens() -> None:
    """Initialize bearer tokens from env or generate mocks."""
    global _teacher_tokens, _student_tokens

    env_teacher = os.getenv("EXAMPEN_TEACHER_TOKENS", "")
    env_student = os.getenv("EXAMPEN_STUDENT_TOKENS", "")

    if env_teacher:
        _teacher_tokens = env_teacher.split(",")
    else:
        for i in range(TEACHER_POOL_SIZE):
            _teacher_tokens.append(
                _generate_mock_jwt("tutor", f"tutor_{i:04d}")
            )

    if env_student:
        _student_tokens = env_student.split(",")
    else:
        for i in range(STUDENT_POOL_SIZE):
            _student_tokens.append(
                _generate_mock_jwt("student", _student_ids[i])
            )


# ---------------------------------------------------------------------------
# Payload generators
# ---------------------------------------------------------------------------
def _random_chunk_payload() -> bytes:
    """Generate a realistic binary chunk: 600 coordinate frames x 14 bytes.

    P05 coordinate frame (14 bytes):
      bookType(1) + pageNo(1) + X(2) + Y(2) + pressure(2)
      + penProp(1) + reserved(1) + timestamp(4)
    """
    frames = []
    for _ in range(600):
        frame = bytearray(14)
        frame[0] = 0x01                           # bookType: exam
        frame[1] = random.randint(1, 8)            # pageNo
        frame[2:4] = random.randint(0, 2100).to_bytes(2, "big")   # X
        frame[4:6] = random.randint(0, 2970).to_bytes(2, "big")   # Y
        frame[6:8] = random.randint(100, 2048).to_bytes(2, "big") # pressure
        frame[8] = 0x00                            # penProp: down
        frame[9] = 0x00                            # reserved
        frame[10:14] = random.getrandbits(32).to_bytes(4, "big")  # timestamp
        frames.append(bytes(frame))
    return b"".join(frames)


def _make_ingest_request(
    exam_id: str,
    pen_mac: str,
    chunk_index: int,
    total_chunks: int = CHUNKS_PER_PEN,
) -> dict:
    """Build a StrokeChunkUploadRequest matching the OpenAPI contract."""
    raw = _random_chunk_payload()
    return {
        "exam_id": exam_id,
        "pen_mac": pen_mac,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "payload_base64": base64.b64encode(raw).decode("ascii"),
        "checksum_crc32": f"{random.getrandbits(32):08x}",
        "upload_path": "wifi",
        "idempotency_key": f"{exam_id}:{pen_mac}:{chunk_index}",
        "binding_status": "confirmed",
    }


# ---------------------------------------------------------------------------
# Locust event hooks
# ---------------------------------------------------------------------------
@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Load fixtures once at startup."""
    _load_fixtures()
    _init_tokens()


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    """Check performance budgets on exit."""
    stats = environment.stats
    failures = []

    for name, threshold in [
        ("POST /api/v1/strokes/ingest", STROKE_INGEST_P95),
        ("GET /api/v1/teacher/exams/[exam_id]/scores", SCORE_QUERY_P95),
        ("GET /api/v1/student/exams/[exam_id]/scores", STUDENT_QUERY_P95),
    ]:
        entry = stats.get(name, "GET") or stats.get(name, "POST")
        if entry and entry.get_response_time_percentile(0.95):
            p95 = entry.get_response_time_percentile(0.95) / 1000.0
            if p95 > threshold:
                failures.append(
                    f"  {name}: p95={p95:.2f}s > threshold={threshold}s"
                )

    total = stats.total
    if total.num_requests > 0:
        error_rate = total.num_failures / total.num_requests
        if error_rate > 0.01:
            failures.append(
                f"  Error rate: {error_rate:.2%} > 1% threshold"
            )

    if failures:
        print("\n=== PERFORMANCE BUDGET VIOLATIONS ===")
        for f in failures:
            print(f)
        print("=====================================\n")
        environment.process_exit_code = 1


# ===================================================================
# Scenario 1: Stroke Ingestion Burst
# ===================================================================
class StrokeIngestUser(HttpUser):
    """Simulates a hub uploading stroke chunks for its pens.

    Each virtual user represents one hub uploading chunks for one pen
    at a time. For the 10K-student burst, run with -u 250 (250 hubs,
    each cycling through 40 pens).
    """

    wait_time = between(0.05, 0.2)
    weight = 5

    def on_start(self) -> None:
        self._pen_idx = random.randint(0, len(_pen_macs) - 1)
        self._exam_id = random.choice(_exam_ids)
        self._chunk_idx = 0
        self._token = random.choice(_teacher_tokens)

    @tag("stroke", "ingest", "burst")
    @task
    def upload_chunk(self) -> None:
        """Upload a single stroke chunk to svc-stroke-ingest."""
        pen_mac = _pen_macs[self._pen_idx % len(_pen_macs)]
        payload = _make_ingest_request(
            exam_id=self._exam_id,
            pen_mac=pen_mac,
            chunk_index=self._chunk_idx,
        )

        with self.client.post(
            "/api/v1/strokes/ingest",
            json=payload,
            headers={"Authorization": f"Bearer {self._token}"},
            name="POST /api/v1/strokes/ingest",
            catch_response=True,
        ) as resp:
            if resp.status_code == 202:
                resp.success()
            elif resp.status_code == 409:
                # Deduplicated — acceptable
                resp.success()
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

        self._chunk_idx += 1
        if self._chunk_idx >= CHUNKS_PER_PEN:
            # Move to next pen
            self._chunk_idx = 0
            self._pen_idx += 1
            self._exam_id = random.choice(_exam_ids)


# ===================================================================
# Scenario 2: Score Query Load (Teachers)
# ===================================================================
class TeacherScoreUser(HttpUser):
    """Simulates 500 concurrent teachers viewing class score overviews.

    Targets: GET /api/v1/teacher/exams/{exam_id}/scores
    """

    wait_time = between(1, 5)
    weight = 2

    def on_start(self) -> None:
        self._token = random.choice(_teacher_tokens)

    @tag("teacher", "scores", "read")
    @task(3)
    def view_class_scores(self) -> None:
        """GET class score overview for a random exam."""
        exam_id = random.choice(_exam_ids)
        with self.client.get(
            f"/api/v1/teacher/exams/{exam_id}/scores",
            headers={"Authorization": f"Bearer {self._token}"},
            name="GET /api/v1/teacher/exams/[exam_id]/scores",
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 404):
                resp.success()
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @tag("teacher", "scores", "read")
    @task(1)
    def view_student_detail(self) -> None:
        """GET per-student drill-down."""
        exam_id = random.choice(_exam_ids)
        student_id = random.choice(_student_ids)
        with self.client.get(
            f"/api/v1/teacher/exams/{exam_id}/scores/{student_id}",
            headers={"Authorization": f"Bearer {self._token}"},
            name="GET /api/v1/teacher/exams/[exam_id]/scores/[student_id]",
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 404):
                resp.success()
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @tag("teacher", "exams", "read")
    @task(1)
    def list_exams(self) -> None:
        """GET teacher exam list."""
        with self.client.get(
            "/api/v1/teacher/exams",
            headers={"Authorization": f"Bearer {self._token}"},
            name="GET /api/v1/teacher/exams",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Unexpected status {resp.status_code}")


# ===================================================================
# Scenario 3: Student Portal Load
# ===================================================================
class StudentScoreUser(HttpUser):
    """Simulates 5000 concurrent students checking their scores.

    Targets: GET /api/v1/student/exams/{exam_id}/scores
    """

    wait_time = between(2, 8)
    weight = 10

    def on_start(self) -> None:
        idx = random.randint(0, len(_student_tokens) - 1)
        self._token = _student_tokens[idx]
        self._student_id = _student_ids[idx % len(_student_ids)]

    @tag("student", "scores", "read")
    @task(5)
    def view_score_summary(self) -> None:
        """GET score summary for a random exam."""
        exam_id = random.choice(_exam_ids)
        with self.client.get(
            f"/api/v1/student/exams/{exam_id}/scores",
            headers={"Authorization": f"Bearer {self._token}"},
            name="GET /api/v1/student/exams/[exam_id]/scores",
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 404):
                resp.success()
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @tag("student", "exams", "read")
    @task(2)
    def list_exams(self) -> None:
        """GET student exam list."""
        with self.client.get(
            "/api/v1/student/exams",
            headers={"Authorization": f"Bearer {self._token}"},
            name="GET /api/v1/student/exams",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @tag("student", "answers", "read")
    @task(1)
    def view_answer_detail(self) -> None:
        """GET answer insight for a specific question."""
        exam_id = random.choice(_exam_ids)
        q_num = random.randint(1, QUESTION_POOL_SIZE)
        question_id = f"q{q_num:02d}"
        with self.client.get(
            f"/api/v1/student/exams/{exam_id}/answers/{question_id}",
            headers={"Authorization": f"Bearer {self._token}"},
            name="GET /api/v1/student/exams/[exam_id]/answers/[question_id]",
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 404):
                resp.success()
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @tag("student", "performance", "read")
    @task(1)
    def view_performance(self) -> None:
        """GET historical performance."""
        with self.client.get(
            "/api/v1/student/performance",
            headers={"Authorization": f"Bearer {self._token}"},
            name="GET /api/v1/student/performance",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Unexpected status {resp.status_code}")


# ===================================================================
# Scenario 4: Mixed Workload
# ===================================================================
# The mixed workload is achieved by running all three user classes
# simultaneously. Weights control the ratio:
#   StrokeIngestUser  weight=5   (~29%)
#   TeacherScoreUser  weight=2   (~12%)
#   StudentScoreUser  weight=10  (~59%)
#
# This mirrors a realistic post-exam scenario where:
# - Hubs are uploading stroke data
# - Teachers are reviewing scores
# - Students are checking results
#
# To run ONLY one scenario, use tags:
#   locust -f locustfile.py --tags stroke
#   locust -f locustfile.py --tags teacher
#   locust -f locustfile.py --tags student
