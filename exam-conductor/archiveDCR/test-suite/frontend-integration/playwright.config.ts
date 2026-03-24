/**
 * Playwright configuration for ExamPen frontend <-> BFF integration tests.
 *
 * Covers all three web frontends:
 *   - teacher-dashboard  (default: http://localhost:5173)
 *   - student-portal     (default: http://localhost:5174)
 *   - invigilator-console (default: http://localhost:5175)
 *
 * Run against a live Docker Compose stack with seeded data.
 *
 * Task: W6.A2 (TEST_SUITE_SPEC.md -- L5 frontend integration)
 */

import { defineConfig, devices } from "@playwright/test";

const TEACHER_BASE_URL =
  process.env.TEACHER_BASE_URL ?? "http://localhost:5173";
const STUDENT_BASE_URL =
  process.env.STUDENT_BASE_URL ?? "http://localhost:5174";
const INVIG_BASE_URL =
  process.env.INVIG_BASE_URL ?? "http://localhost:5175";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 1 : 2,
  reporter: [["html", { open: "never" }], ["list"]],
  timeout: 60_000,
  expect: { timeout: 10_000 },

  use: {
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },

  projects: [
    // ── Teacher Dashboard ──────────────────────────────────────────────
    {
      name: "teacher-dashboard",
      testMatch: "teacher-dashboard.spec.ts",
      use: {
        ...devices["Desktop Chrome"],
        baseURL: TEACHER_BASE_URL,
      },
    },

    // ── Student Portal ─────────────────────────────────────────────────
    {
      name: "student-portal",
      testMatch: "student-portal.spec.ts",
      use: {
        ...devices["Desktop Chrome"],
        baseURL: STUDENT_BASE_URL,
      },
    },

    // ── Invigilator Console ────────────────────────────────────────────
    {
      name: "invigilator-console",
      testMatch: "invigilator-console.spec.ts",
      use: {
        ...devices["Desktop Chrome"],
        baseURL: INVIG_BASE_URL,
      },
    },

    // ── Cross-browser smoke (Firefox) ──────────────────────────────────
    {
      name: "teacher-dashboard-firefox",
      testMatch: "teacher-dashboard.spec.ts",
      use: {
        ...devices["Desktop Firefox"],
        baseURL: TEACHER_BASE_URL,
      },
    },
  ],
});
