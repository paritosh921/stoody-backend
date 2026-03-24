/**
 * ExamPen load test scenarios — k6 (CI-friendly).
 *
 * Same scenarios as Locust but expressed in k6 JavaScript for
 * lightweight CI pipeline integration.
 *
 * Usage:
 *   k6 run k6_script.js                              # default (mixed)
 *   k6 run k6_script.js --env SCENARIO=stroke_burst   # single scenario
 *   k6 run k6_script.js --env SCENARIO=teacher_scores
 *   k6 run k6_script.js --env SCENARIO=student_scores
 *   k6 run k6_script.js --out json=results.json       # export results
 *
 * Environment variables:
 *   BASE_URL           — target host (default: http://localhost:8080)
 *   SCENARIO           — run a single scenario instead of mixed
 *   TEACHER_TOKEN      — valid JWT for teacher role
 *   STUDENT_TOKEN      — valid JWT for student role
 *   HUB_TOKEN          — valid JWT for hub upload role
 *
 * Reference: CLAUDE.md §Testing Strategy, FAILURE_MITIGATION_REGISTER.md §A8.4
 */

import http from "k6/http";
import { check, group, sleep } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";
import { randomItem, randomIntBetween } from "https://jslib.k6.io/k6-utils/1.4.0/index.js";
import encoding from "k6/encoding";
import { uuidv4 } from "https://jslib.k6.io/k6-utils/1.4.0/index.js";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const BASE_URL = __ENV.BASE_URL || "http://localhost:8080";
const SELECTED_SCENARIO = __ENV.SCENARIO || "mixed";

// Mock JWT (override with real tokens in CI)
function mockJwt(role, userId) {
  const header = encoding.b64encode('{"alg":"HS256","typ":"JWT"}', "rawurl");
  const payload = encoding.b64encode(
    JSON.stringify({
      sub: userId,
      role: role,
      tenant_id: "load-test-tenant",
      exp: 9999999999,
    }),
    "rawurl"
  );
  const sig = encoding.b64encode("mock-signature", "rawurl");
  return `${header}.${payload}.${sig}`;
}

const TEACHER_TOKEN = __ENV.TEACHER_TOKEN || mockJwt("tutor", "tutor_0001");
const STUDENT_TOKEN = __ENV.STUDENT_TOKEN || mockJwt("student", "stu_0001");
const HUB_TOKEN = __ENV.HUB_TOKEN || mockJwt("hub", "hub_0001");

// Pre-generated ID pools
const EXAM_IDS = Array.from({ length: 50 }, () => uuidv4());
const STUDENT_IDS = Array.from({ length: 200 }, () => uuidv4());
const PEN_MACS = Array.from({ length: 200 }, (_, i) =>
  `AA:BB:CC:${((i >> 16) & 0xff).toString(16).padStart(2, "0")}:${((i >> 8) & 0xff).toString(16).padStart(2, "0")}:${(i & 0xff).toString(16).padStart(2, "0")}`.toUpperCase()
);

// Chunk constants (matches A8.4: 600 frames x 14 bytes = 8400 bytes per chunk)
const CHUNK_SIZE_BYTES = 8400;
const CHUNKS_PER_PEN = 40;

// ---------------------------------------------------------------------------
// Custom metrics
// ---------------------------------------------------------------------------
const strokeIngestDuration = new Trend("stroke_ingest_duration", true);
const strokeIngestErrors = new Rate("stroke_ingest_errors");
const teacherScoreDuration = new Trend("teacher_score_duration", true);
const teacherScoreErrors = new Rate("teacher_score_errors");
const studentScoreDuration = new Trend("student_score_duration", true);
const studentScoreErrors = new Rate("student_score_errors");
const chunksUploaded = new Counter("chunks_uploaded");

// ---------------------------------------------------------------------------
// Thresholds — performance budgets
// ---------------------------------------------------------------------------
export const options = {
  thresholds: {
    // Global
    http_req_failed: ["rate<0.01"],            // <1% error rate
    http_req_duration: ["p(95)<2000"],         // p95 < 2s across all

    // Per-scenario
    stroke_ingest_duration: ["p(95)<2000", "p(99)<5000"],
    stroke_ingest_errors: ["rate<0.01"],
    teacher_score_duration: ["p(95)<2000", "p(99)<3000"],
    teacher_score_errors: ["rate<0.01"],
    student_score_duration: ["p(95)<2000", "p(99)<3000"],
    student_score_errors: ["rate<0.01"],
  },

  scenarios: _buildScenarios(),
};

function _buildScenarios() {
  if (SELECTED_SCENARIO === "stroke_burst") {
    return {
      stroke_burst: {
        executor: "ramping-vus",
        exec: "strokeBurst",
        startVUs: 0,
        stages: [
          { duration: "30s", target: 50 },   // ramp up
          { duration: "2m", target: 250 },   // full burst (250 hubs)
          { duration: "1m", target: 250 },   // sustain
          { duration: "30s", target: 0 },    // ramp down
        ],
      },
    };
  }

  if (SELECTED_SCENARIO === "teacher_scores") {
    return {
      teacher_scores: {
        executor: "constant-vus",
        exec: "teacherScores",
        vus: 500,
        duration: "3m",
      },
    };
  }

  if (SELECTED_SCENARIO === "student_scores") {
    return {
      student_scores: {
        executor: "constant-vus",
        exec: "studentScores",
        vus: 500,
        duration: "3m",
      },
    };
  }

  // Mixed workload (default)
  return {
    stroke_burst: {
      executor: "ramping-vus",
      exec: "strokeBurst",
      startVUs: 0,
      stages: [
        { duration: "30s", target: 25 },
        { duration: "3m", target: 100 },
        { duration: "1m", target: 100 },
        { duration: "30s", target: 0 },
      ],
    },
    teacher_scores: {
      executor: "ramping-vus",
      exec: "teacherScores",
      startVUs: 0,
      stages: [
        { duration: "30s", target: 50 },
        { duration: "3m", target: 200 },
        { duration: "1m", target: 200 },
        { duration: "30s", target: 0 },
      ],
    },
    student_scores: {
      executor: "ramping-vus",
      exec: "studentScores",
      startVUs: 0,
      stages: [
        { duration: "30s", target: 100 },
        { duration: "3m", target: 500 },
        { duration: "1m", target: 500 },
        { duration: "30s", target: 0 },
      ],
    },
  };
}

// ---------------------------------------------------------------------------
// Payload generators
// ---------------------------------------------------------------------------

/**
 * Generate a random binary chunk payload (base64).
 *
 * 600 coordinate frames x 14 bytes = 8400 bytes, matching the P05 pen
 * coordinate frame layout.
 */
function generateChunkPayload() {
  // k6 doesn't have crypto.getRandomValues in all versions,
  // so we build a pseudo-random buffer
  const buf = new ArrayBuffer(CHUNK_SIZE_BYTES);
  const view = new Uint8Array(buf);
  for (let i = 0; i < CHUNK_SIZE_BYTES; i++) {
    view[i] = Math.floor(Math.random() * 256);
  }
  // Stamp bookType and a reasonable pageNo in each frame header
  for (let f = 0; f < 600; f++) {
    const offset = f * 14;
    view[offset] = 0x01;     // bookType: exam
    view[offset + 1] = (f % 8) + 1; // pageNo: 1-8
  }
  return encoding.b64encode(new Uint8Array(buf));
}

function makeIngestPayload(examId, penMac, chunkIndex) {
  return JSON.stringify({
    exam_id: examId,
    pen_mac: penMac,
    chunk_index: chunkIndex,
    total_chunks: CHUNKS_PER_PEN,
    payload_base64: generateChunkPayload(),
    checksum_crc32: Math.floor(Math.random() * 0xffffffff)
      .toString(16)
      .padStart(8, "0"),
    upload_path: "wifi",
    idempotency_key: `${examId}:${penMac}:${chunkIndex}`,
    binding_status: "confirmed",
  });
}

// ---------------------------------------------------------------------------
// Shared HTTP params
// ---------------------------------------------------------------------------
function teacherHeaders() {
  return {
    headers: {
      Authorization: `Bearer ${TEACHER_TOKEN}`,
      "Content-Type": "application/json",
    },
  };
}

function studentHeaders() {
  return {
    headers: {
      Authorization: `Bearer ${STUDENT_TOKEN}`,
      "Content-Type": "application/json",
    },
  };
}

function hubHeaders() {
  return {
    headers: {
      Authorization: `Bearer ${HUB_TOKEN}`,
      "Content-Type": "application/json",
    },
  };
}

// ---------------------------------------------------------------------------
// Scenario 1: Stroke Ingestion Burst
// ---------------------------------------------------------------------------
export function strokeBurst() {
  const examId = randomItem(EXAM_IDS);
  const penIdx = randomIntBetween(0, PEN_MACS.length - 1);
  const penMac = PEN_MACS[penIdx];

  group("Stroke Ingestion — upload chunk", () => {
    for (let chunk = 0; chunk < 3; chunk++) {
      // Upload 3 chunks per iteration to simulate sustained burst
      const chunkIndex = randomIntBetween(0, CHUNKS_PER_PEN - 1);
      const body = makeIngestPayload(examId, penMac, chunkIndex);

      const res = http.post(
        `${BASE_URL}/api/v1/strokes/ingest`,
        body,
        hubHeaders()
      );

      strokeIngestDuration.add(res.timings.duration);
      const ok = check(res, {
        "status is 202 or 409": (r) =>
          r.status === 202 || r.status === 409,
      });
      strokeIngestErrors.add(!ok);
      if (ok) chunksUploaded.add(1);
    }
  });

  sleep(randomIntBetween(50, 200) / 1000); // 50-200ms between batches
}

// ---------------------------------------------------------------------------
// Scenario 2: Teacher Score Queries
// ---------------------------------------------------------------------------
export function teacherScores() {
  const examId = randomItem(EXAM_IDS);

  group("Teacher — class score overview", () => {
    const res = http.get(
      `${BASE_URL}/api/v1/teacher/exams/${examId}/scores`,
      teacherHeaders()
    );
    teacherScoreDuration.add(res.timings.duration);
    const ok = check(res, {
      "status is 200 or 404": (r) => r.status === 200 || r.status === 404,
    });
    teacherScoreErrors.add(!ok);
  });

  sleep(randomIntBetween(1, 3));

  group("Teacher — student detail drill-down", () => {
    const studentId = randomItem(STUDENT_IDS);
    const res = http.get(
      `${BASE_URL}/api/v1/teacher/exams/${examId}/scores/${studentId}`,
      teacherHeaders()
    );
    teacherScoreDuration.add(res.timings.duration);
    const ok = check(res, {
      "status is 200 or 404": (r) => r.status === 200 || r.status === 404,
    });
    teacherScoreErrors.add(!ok);
  });

  sleep(randomIntBetween(2, 5));

  group("Teacher — exam list", () => {
    const res = http.get(
      `${BASE_URL}/api/v1/teacher/exams`,
      teacherHeaders()
    );
    const ok = check(res, {
      "status is 200": (r) => r.status === 200,
    });
    teacherScoreErrors.add(!ok);
  });

  sleep(randomIntBetween(1, 3));
}

// ---------------------------------------------------------------------------
// Scenario 3: Student Score Queries
// ---------------------------------------------------------------------------
export function studentScores() {
  const examId = randomItem(EXAM_IDS);

  group("Student — score summary", () => {
    const res = http.get(
      `${BASE_URL}/api/v1/student/exams/${examId}/scores`,
      studentHeaders()
    );
    studentScoreDuration.add(res.timings.duration);
    const ok = check(res, {
      "status is 200 or 404": (r) => r.status === 200 || r.status === 404,
    });
    studentScoreErrors.add(!ok);
  });

  sleep(randomIntBetween(2, 5));

  group("Student — exam list", () => {
    const res = http.get(
      `${BASE_URL}/api/v1/student/exams`,
      studentHeaders()
    );
    const ok = check(res, {
      "status is 200": (r) => r.status === 200,
    });
    studentScoreErrors.add(!ok);
  });

  sleep(randomIntBetween(2, 5));

  group("Student — answer detail", () => {
    const qNum = randomIntBetween(1, 10);
    const questionId = `q${qNum.toString().padStart(2, "0")}`;
    const res = http.get(
      `${BASE_URL}/api/v1/student/exams/${examId}/answers/${questionId}`,
      studentHeaders()
    );
    const ok = check(res, {
      "status is 200 or 404": (r) => r.status === 200 || r.status === 404,
    });
    studentScoreErrors.add(!ok);
  });

  sleep(randomIntBetween(3, 8));

  // 20% of students also check performance history
  if (Math.random() < 0.2) {
    group("Student — performance history", () => {
      const res = http.get(
        `${BASE_URL}/api/v1/student/performance`,
        studentHeaders()
      );
      const ok = check(res, {
        "status is 200": (r) => r.status === 200,
      });
      studentScoreErrors.add(!ok);
    });
    sleep(randomIntBetween(1, 3));
  }
}
