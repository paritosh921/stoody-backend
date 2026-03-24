/**
 * Invigilator Console <-> svc-invig-console integration tests.
 *
 * Task:   W6.A2 (frontend integration)
 * Level:  L5
 * Spec:   TEST_SUITE_SPEC.md -- I-INVIG-01
 *
 * Prerequisites:
 *   - Full Docker Compose stack running (svc-invig-console + svc-exam-orch)
 *   - Seed data loaded (scripts/seed-data.sh)
 *   - invigilator-console dev server on INVIG_BASE_URL (default 5175)
 */

import { test, expect, Page } from "@playwright/test";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const INVIG_BFF_URL =
  process.env.INVIG_BFF_URL ?? "http://localhost:8090";

const INVIGILATOR_CREDS = {
  email: "invigilator@test.exampen.local",
  password: "test-invigilator-pass",
};

async function loginAsInvigilator(page: Page): Promise<void> {
  await page.goto("/login");
  await page.getByLabel(/email/i).fill(INVIGILATOR_CREDS.email);
  await page.getByLabel(/password/i).fill(INVIGILATOR_CREDS.password);
  await page.getByRole("button", { name: /sign in|log in|login/i }).click();
  await page.waitForURL((url) => !url.pathname.includes("/login"), {
    timeout: 15_000,
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test.describe("Invigilator Console — Session List", () => {
  test(
    "I-INVIG-01: Login -> session list loads with seeded sessions",
    async ({ page }) => {
      await loginAsInvigilator(page);

      // Expect session list / dashboard.
      await expect(
        page.getByRole("heading", { name: /session|exam|dashboard/i }),
      ).toBeVisible({ timeout: 10_000 });

      // At least one exam session should be visible from seeded data.
      const sessionRows = page.locator(
        '[data-testid="session-row"], .session-card, tbody tr',
      );
      await expect(sessionRows.first()).toBeVisible({ timeout: 10_000 });
    },
  );
});

test.describe("Invigilator Console — WebSocket Dashboard", () => {
  test(
    "I-INVIG-01: Dashboard opens WebSocket and displays timer",
    async ({ page }) => {
      await loginAsInvigilator(page);

      // Click the first session to enter its dashboard.
      const firstSession = page.locator(
        '[data-testid="session-row"], .session-card, tbody tr',
      );
      await firstSession.first().click();

      // The dashboard should contain a timer display.
      const timerDisplay = page.locator(
        '[data-testid="exam-timer"], .timer, .countdown',
      );
      await expect(timerDisplay).toBeVisible({ timeout: 10_000 });

      // Verify WebSocket connection is active by checking for a connection
      // status indicator or live data updates.
      const wsIndicator = page.locator(
        '[data-testid="ws-status"], .connection-status, .ws-connected',
      );

      // Wait briefly for WebSocket to establish and render status.
      await page.waitForTimeout(3_000);

      // Either the WS indicator shows connected or the dashboard shows live
      // data (pen count, sync status, etc.).
      const hasWsIndicator = await wsIndicator
        .isVisible()
        .catch(() => false);
      const hasLiveData = await page
        .locator(
          '[data-testid="pen-count"], [data-testid="sync-status"], .pen-grid',
        )
        .first()
        .isVisible()
        .catch(() => false);

      expect(
        hasWsIndicator || hasLiveData,
      ).toBeTruthy();
    },
  );

  test(
    "Dashboard shows exam state and hub connectivity",
    async ({ page }) => {
      await loginAsInvigilator(page);

      // Enter a session dashboard.
      const firstSession = page.locator(
        '[data-testid="session-row"], .session-card, tbody tr',
      );
      await firstSession.first().click();

      // Expect exam state to be displayed (e.g. armed, timer_running, etc.).
      const stateLabel = page.locator(
        '[data-testid="exam-state"], .exam-state, .status-badge',
      );
      await expect(stateLabel).toBeVisible({ timeout: 10_000 });

      // Hub connectivity indicator should be present.
      const hubStatus = page.locator(
        '[data-testid="hub-status"], .hub-connectivity, .hub-online',
      );
      await expect(hubStatus).toBeVisible({ timeout: 10_000 });
    },
  );
});

test.describe("Invigilator Console — Pen Grid", () => {
  test(
    "Pen grid shows per-pen sync progress (real-time)",
    async ({ page }) => {
      await loginAsInvigilator(page);

      // Enter a session dashboard.
      const firstSession = page.locator(
        '[data-testid="session-row"], .session-card, tbody tr',
      );
      await firstSession.first().click();

      // The pen grid should be visible, showing per-pen status.
      const penGrid = page.locator(
        '[data-testid="pen-grid"], .pen-grid, .pen-status-grid',
      );
      await expect(penGrid).toBeVisible({ timeout: 10_000 });

      // Each pen cell should show some status (MAC, sync %, student name, etc.)
      const penCells = penGrid.locator(
        '[data-testid="pen-cell"], .pen-cell, .pen-card',
      );
      const cellCount = await penCells.count();

      // At least one pen should be visible in the grid from seeded session.
      expect(cellCount).toBeGreaterThan(0);

      // First pen cell should contain identifiable information.
      const firstCell = penCells.first();
      await expect(firstCell).toBeVisible();

      // Wait for real-time update (WebSocket push). After a short delay,
      // the pen grid data should still be present (not disappeared due to WS
      // reconnection issues).
      await page.waitForTimeout(5_000);
      await expect(penGrid).toBeVisible();
    },
  );
});
